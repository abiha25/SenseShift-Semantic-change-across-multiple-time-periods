"""
Classical baseline (Baseline A) for SemEval-2027 Task 3.   Owner: Person 3.

Subtask 1 (1a + 1b): cluster TF-IDF context vectors of each target word with
    average-linkage agglomerative clustering, then fold clusters smaller than
    a minimum size into their nearest large cluster. The official scorer
    turns these labels into per-period sense distributions and JSD (1b).
Subtask 2: four usage-definition similarity features -> logistic regression.

Validation is leave-one-word-out (LOWO) everywhere:
  * Subtask 1 is unsupervised, so LOWO is used only to CHOOSE settings
    (features, window, threshold, minimum cluster size) on 9 words and apply
    them to the held-out word. The 10 held-out results are written into one
    file and scored once, so 1b is computed on all 60 scored period pairs.
  * Subtask 2: the classifier and its decision threshold are fitted on 9
    words and applied to the 10th.

All scores come from src/scoring.py (the official evaluation code).
Seed 42 everywhere (only K-means is random).
"""

from __future__ import annotations

import itertools
import json
import tempfile
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_mutual_info_score, f1_score, roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from data_loader import leave_one_word_out, write_subtask1, write_subtask2
from src.features import DEF_FEATURES, FeatureConfig, definition_features, word_matrix
from src.scoring import MIN_INSTANCES_1B, score_1a, score_1b, score_2
from evaluation import compute_bcubed, compute_spearman, filter_jsd_by_min_instances
from evaluation_utils import calculate_JSD_distances, load_subtask1_labels

SEED = 42
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='An input array is constant')

# The search space. Kept small enough to explain in the report.
KINDS = ['char', 'word', 'both']
WINDOWS = [3, 5, 10, None]
THRESHOLDS = [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99]
MIN_SIZES = [5, 10, 15, 20, 30]
DELTA_1A = 0.03   # 'combined' rule: allowed B3-F1 loss vs. one cluster


# ===========================================================================
# Subtask 1: clustering one word
# ===========================================================================

def cosine_distance_matrix(X) -> np.ndarray:
    """1 - cosine. Usages whose context vector is all zeros (no shared
    features) are put at distance 1 from everything."""
    S = (X @ X.T).toarray()
    D = np.clip(1.0 - S, 0.0, 2.0)
    zero = np.asarray(X.getnnz(axis=1)) == 0
    D[zero, :] = 1.0
    D[:, zero] = 1.0
    np.fill_diagonal(D, 0.0)
    return D


def merge_small_clusters(labels: np.ndarray, X, min_size: int) -> np.ndarray:
    """
    Clusters with fewer than min_size usages are dissolved; each of their
    usages moves to the large cluster whose centroid it is most similar to.
    If no cluster is large enough, everything becomes one cluster.
    Finally clusters are renumbered 0, 1, ... from largest to smallest.
    """
    sizes = Counter(labels)
    big = [c for c, n in sizes.items() if n >= min_size]
    if not big:
        return np.zeros(len(labels), dtype=int)
    out = labels.copy()
    if len(big) > 1:
        centroids = np.vstack([np.asarray(X[labels == c].mean(axis=0)) for c in big])
        centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
        sims = np.asarray(X @ centroids.T)
        for i in np.where(~np.isin(labels, big))[0]:
            out[i] = big[int(np.argmax(sims[i]))]
    else:
        out[~np.isin(labels, big)] = big[0]
    order = [c for c, _ in Counter(out).most_common()]
    remap = {c: k for k, c in enumerate(order)}
    return np.array([remap[c] for c in out])


def agglomerative_labels(X, thresholds, min_sizes) -> dict:
    """
    Build the average-linkage tree once, then cut it at every threshold and
    apply every minimum size. Returns {(threshold, min_size): labels}.
    A cut at distance t means: two groups are merged only while their
    average cosine distance is <= t (t near 1 = merge almost everything).
    """
    Z = linkage(squareform(cosine_distance_matrix(X), checks=False), method='average')
    out = {}
    for t in thresholds:
        raw = fcluster(Z, t=t, criterion='distance')
        for m in min_sizes:
            out[(t, m)] = merge_small_clusters(raw, X, m)
    return out


def kmeans_silhouette_labels(X, k_range=range(2, 11)) -> np.ndarray:
    """Comparison method of Giulianelli et al. (2020): K-means for K=2..10,
    keep the K with the best (cosine) silhouette. It can never return 1 sense."""
    Xd = X.toarray()
    best, best_s = None, -np.inf
    for k in k_range:
        lab = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(Xd)
        s = silhouette_score(Xd, lab, metric='cosine')
        if s > best_s:
            best, best_s = lab, s
    return best


# ===========================================================================
# Subtask 1: scoring helpers (all through src/scoring.py)
# ===========================================================================

def _write_labels(df, labels_by_word: dict, path) -> Path:
    col = np.zeros(len(df), dtype=int)
    for w, lab in labels_by_word.items():
        col[(df['word'] == w).values] = lab
    write_subtask1(df.assign(_pred=col), '_pred', path)
    return Path(path)


def score_labels(df, labels_by_word, gold_file, words=None, keep_file=None) -> dict:
    """B3 per word + macro, and Spearman rho for 1b (optionally on a subset)."""
    path = keep_file or tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False).name
    _write_labels(df, labels_by_word, path)
    a = score_1a(path, gold_file, words=words)
    b = score_1b(path, gold_file, words=words)
    return {'per_word_f1': {w: v['fcubed'] for w, v in a['per_word'].items()},
            'per_word': a['per_word'], 'macro': a['macro'],
            'rho': b['rho'], 'gold_jsd': b['gold_jsd'], 'pred_jsd': b['pred_jsd']}


@dataclass(frozen=True)
class ClusterConfig:
    feat: FeatureConfig
    threshold: float
    min_size: int

    def name(self):
        return f'{self.feat.name()}|t={self.threshold}|m={self.min_size}'


def build_grid(df, feature_configs, thresholds=THRESHOLDS, min_sizes=MIN_SIZES) -> dict:
    """{ClusterConfig: {word: labels}} for every setting. Uses no gold labels."""
    grid = {}
    for fc in feature_configs:
        per_word = {w: agglomerative_labels(word_matrix(g, fc), thresholds, min_sizes)
                    for w, g in df.groupby('word')}
        for t, m in itertools.product(thresholds, min_sizes):
            grid[ClusterConfig(fc, t, m)] = {w: per_word[w][(t, m)] for w in per_word}
    return grid


class FastScorer:
    """
    In-memory scorer for the hyper-parameter search ONLY.

    LOWO needs thousands of scores (every setting x every fold). Writing a file
    for each is too slow, so this class calls the SAME official functions
    (compute_bcubed, calculate_JSD_distances, filter_jsd_by_min_instances,
    compute_spearman) on in-memory dicts, and caches per-word results.
    Final reported numbers always come from files scored by src/scoring.py,
    and the notebook checks that both give identical values.
    """

    def __init__(self, df, gold_file):
        _, gold_counts, _, gold_labels = load_subtask1_labels(Path(gold_file), Path(gold_file))
        self.gold_labels = gold_labels
        gold_jsd = filter_jsd_by_min_instances(calculate_JSD_distances(gold_counts), gold_counts,
                                               min_instances=MIN_INSTANCES_1B)
        self.words = sorted(df['word'].unique())
        self.keys = {w: [f'{w};{s}' for s in g['sentence_id']] for w, g in df.groupby('word')}
        self.periods = {w: g['period_label'].tolist() for w, g in df.groupby('word')}
        self.gold_jsd = {w: {k: v for k, v in gold_jsd.items() if k.rsplit('_', 2)[0] == w}
                         for w in self.words}
        self._f1, self._jsd = {}, {}

    def f1(self, word, labels) -> float:
        key = (word, labels.tobytes())
        if key not in self._f1:
            gold = {k: self.gold_labels[k] for k in self.keys[word]}
            pred = {k: [int(l)] for k, l in zip(self.keys[word], labels)}
            self._f1[key] = compute_bcubed(gold, pred, '')['fcubed']
        return self._f1[key]

    def pred_jsd(self, word, labels) -> dict:
        key = (word, labels.tobytes())
        if key not in self._jsd:
            counts = {}
            for period, lab in zip(self.periods[word], labels):
                counts.setdefault(period, Counter())[int(lab)] += 1.0
            self._jsd[key] = calculate_JSD_distances({word: counts})
        return self._jsd[key]

    def rho(self, labels_by_word, words=None) -> float:
        words = self.words if words is None else words
        gold, pred = {}, {}
        for w in words:
            gold.update(self.gold_jsd[w])
            pred.update(self.pred_jsd(w, labels_by_word[w]))
        return compute_spearman(gold, pred, '')['rho']


def score_grid(df, grid, scorer: FastScorer) -> pd.DataFrame:
    """Per-word B3 F1 (and rho on all words) of every setting."""
    rows = []
    for cfg, labels in grid.items():
        f1 = {w: scorer.f1(w, labels[w]) for w in scorer.words}
        rows.append({'config': cfg, **{f'f1_{w}': v for w, v in f1.items()},
                     'f1_macro': float(np.mean(list(f1.values()))),
                     'rho_all': scorer.rho(labels)})
    return pd.DataFrame(rows)


def lowo_select(df, grid, grid_scores, scorer: FastScorer, objective='combined',
                single=None) -> dict:
    """
    For every held-out word, pick the setting that is best on the OTHER nine
    words, then use that setting's labels for the held-out word.

    objective
      '1a'        highest mean B3 F1 on the nine words
      '1b'        highest Spearman rho on the nine words' period pairs
      'combined'  highest rho among settings whose mean B3 F1 is at most
                  DELTA_1A below the one-cluster baseline on those words
                  (falls back to '1a' if no setting qualifies)
    Returns the out-of-fold labels and the chosen setting per word.
    """
    labels, chosen = {}, {}
    for train_words, test_word in leave_one_word_out(df):
        f1_train = grid_scores[[f'f1_{w}' for w in train_words]].mean(axis=1)
        if objective == '1a':
            best = int(np.argmax(f1_train.values))
        else:
            rhos = np.array([scorer.rho(grid[c], train_words) for c in grid_scores['config']])
            rhos = np.nan_to_num(rhos, nan=-1.0)   # constant output = no ranking
            if objective == '1b':
                best = int(np.argmax(rhos))
            else:
                floor = np.mean([single[w] for w in train_words]) - DELTA_1A
                ok = f1_train.values >= floor
                best = int(np.argmax(np.where(ok, rhos, -np.inf))) if ok.any() \
                    else int(np.argmax(f1_train.values))
        cfg = grid_scores['config'].iloc[best]
        labels[test_word] = grid[cfg][test_word]
        chosen[test_word] = cfg
    return {'labels': labels, 'chosen': chosen}


# ===========================================================================
# Subtask 1: diagnostics that link back to the EDA
# ===========================================================================

CROSS_ERA = '1912-1915_2006-2007'


def era_diagnostics(res: dict) -> dict:
    """
    EDA F2/F6: the only cross-era step is where gold change concentrates,
    and spelling makes that step easy to "detect" for the wrong reason.
    Reports rho on within-era pairs only, and how strongly the predicted
    JSD simply tracks the cross-era flag.
    """
    items = sorted(res['gold_jsd'])
    gold = np.array([res['gold_jsd'][i] for i in items])
    pred = np.array([res['pred_jsd'][i] for i in items])
    cross = np.array([CROSS_ERA in i for i in items])
    within = spearmanr(gold[~cross], pred[~cross])[0] if pred[~cross].std() > 0 else np.nan
    track = spearmanr(pred, cross)[0] if pred.std() > 0 else np.nan
    return {'n_pairs': len(items), 'n_cross_era': int(cross.sum()),
            'rho_within_era': float(within), 'rho_pred_vs_cross_era_flag': float(track),
            'mean_pred_jsd_cross': float(pred[cross].mean()),
            'mean_pred_jsd_within': float(pred[~cross].mean())}


def permutation_control(scorer: FastScorer, labels_by_word, n=200, seed=SEED) -> dict:
    """
    Is a 1b score real, or an artefact of HOW MANY clusters (and of what
    sizes) each word got? JSD between small samples is larger when a word
    is split into many small clusters, and gold JSD is also higher for
    many-sense words (skär, suga), so granularity alone can produce a
    positive rho. Control: keep every word's cluster sizes, shuffle which
    usage is in which cluster, and rescore. 'p' = share of shuffles that
    score at least as well as the system (small p = real signal).
    """
    observed = scorer.rho(labels_by_word)
    rng = np.random.default_rng(seed)
    null = np.array([scorer.rho({w: rng.permutation(l) for w, l in labels_by_word.items()})
                     for _ in range(n)])
    null = np.nan_to_num(null, nan=0.0)
    return {'rho': float(observed), 'shuffled_mean': float(null.mean()),
            'shuffled_sd': float(null.std()),
            'p': float((null >= observed).mean()) if not np.isnan(observed) else np.nan}


def cluster_alignment(df, labels_by_word) -> pd.DataFrame:
    """
    EDA F8 asked whether representations group usages by form, era or sense.
    Same question for our clusters: adjusted mutual information (AMI) of the
    predicted clusters with the gold first sense, the era and the target form.
    """
    rows = []
    for w, g in df.groupby('word'):
        lab = labels_by_word[w]
        era = (g['period_index'] >= 5).astype(int).values
        row = {'word': w, 'n_clusters': len(set(lab)),
               'n_gold_senses': len({s for l in g['senses'] for s in l}),
               'ami_sense': adjusted_mutual_info_score(g['senses'].str[0], lab),
               'ami_form': adjusted_mutual_info_score(g['target_fixed'].str.lower(), lab)}
        row['ami_era'] = adjusted_mutual_info_score(era, lab) if len(set(era)) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index('word')


# ===========================================================================
# Subtask 2: definition features + logistic regression
# ===========================================================================

RAW = DEF_FEATURES
RANK = [c + '_rank' for c in DEF_FEATURES]
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def _macro_f1(y, p, words) -> float:
    return float(np.mean([f1_score(y[words == w], p[words == w], zero_division=0)
                          for w in np.unique(words)]))


def subtask2_lowo(df, F, cols, tune_threshold=True, C=1.0) -> dict:
    """
    Leave-one-word-out logistic regression.
    Features are standardised on the nine training words; class_weight
    'balanced' stops the majority class of large words from dominating.
    The decision threshold is chosen on the nine training words by the
    same metric we are scored on (per-word F1, averaged).
    """
    y_all, w_all = df['target_sense'].values, df['word'].values
    pred = np.zeros(len(df), dtype=int)
    prob = np.zeros(len(df))
    coefs, thresholds = [], {}
    for train_words, test_word in leave_one_word_out(df):
        tr = np.isin(w_all, train_words)
        te = w_all == test_word
        scaler = StandardScaler().fit(F.loc[tr, cols])
        clf = LogisticRegression(C=C, class_weight='balanced', max_iter=1000, random_state=SEED)
        clf.fit(scaler.transform(F.loc[tr, cols]), y_all[tr])
        thr = 0.5
        if tune_threshold:
            p_tr = clf.predict_proba(scaler.transform(F.loc[tr, cols]))[:, 1]
            scores = [_macro_f1(y_all[tr], (p_tr >= t).astype(int), w_all[tr]) for t in THRESHOLD_GRID]
            thr = float(THRESHOLD_GRID[int(np.argmax(scores))])
        prob[te] = clf.predict_proba(scaler.transform(F.loc[te, cols]))[:, 1]
        pred[te] = (prob[te] >= thr).astype(int)
        coefs.append(dict(zip(cols, clf.coef_[0])))
        thresholds[test_word] = thr
    auc = {w: roc_auc_score(y_all[w_all == w], prob[w_all == w]) for w in np.unique(w_all)}
    return {'pred': pred, 'prob': prob, 'coefs': pd.DataFrame(coefs),
            'thresholds': thresholds, 'auc': auc}


def oracle_threshold_f1(df, score: np.ndarray) -> dict:
    """
    DIAGNOSTIC ONLY (uses gold labels): the best per-word F1 any threshold on
    this score could reach. If even this is close to 'predict all 1', the
    feature carries too little signal for any classifier.
    """
    out = {}
    for w, g in df.groupby('word'):
        s, y = score[g.index], g['target_sense'].values
        cands = np.unique(s)
        out[w] = max(f1_score(y, (s >= c).astype(int), zero_division=0) for c in cands)
    return out


def score_subtask2(df, pred, gold_file, path) -> dict:
    write_subtask2(df.assign(_pred=pred), '_pred', path)
    return score_2(path, gold_file)


def all_ones_f1(df) -> dict:
    """Per-word F1 of predicting 1 everywhere: 2p/(1+p) for positive share p."""
    return {w: f1_score(g['target_sense'], np.ones(len(g), int)) for w, g in df.groupby('word')}


# ===========================================================================
# Subtask 2: plain TF-IDF + logistic regression (the textbook baseline)
# ===========================================================================

def _context_tfidf(df, window=10, min_df=3):
    """Word-unigram TF-IDF of the +-window context (spelling-normalised,
    target removed), with ONE vocabulary shared by all words, so that a
    model trained on nine words can be applied to the tenth."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from src.features import contexts
    vec = TfidfVectorizer(token_pattern=r'(?u)\b\w\w+\b', min_df=min_df, sublinear_tf=True)
    X = vec.fit_transform(contexts(df, window))
    return X, np.array(vec.get_feature_names_out())


def tfidf_lr_lowo(df, C=1.0, window=10) -> dict:
    """
    TF-IDF context words -> logistic regression, leave-one-word-out, with the
    decision threshold tuned on the nine training words (as in subtask2_lowo).
    Also fits one model on all ten words, only to list the words with the
    largest positive and negative coefficients.
    """
    X, vocab = _context_tfidf(df, window)
    y, w_all = df['target_sense'].values, df['word'].values
    pred, prob, thresholds = np.zeros(len(df), int), np.zeros(len(df)), {}
    for train_words, test_word in leave_one_word_out(df):
        tr, te = np.isin(w_all, train_words), w_all == test_word
        clf = LogisticRegression(C=C, class_weight='balanced', max_iter=2000, random_state=SEED)
        clf.fit(X[tr], y[tr])
        p_tr = clf.predict_proba(X[tr])[:, 1]
        scores = [_macro_f1(y[tr], (p_tr >= t).astype(int), w_all[tr]) for t in THRESHOLD_GRID]
        thr = float(THRESHOLD_GRID[int(np.argmax(scores))])
        prob[te] = clf.predict_proba(X[te])[:, 1]
        pred[te] = (prob[te] >= thr).astype(int)
        thresholds[test_word] = thr
    auc = {w: roc_auc_score(y[w_all == w], prob[w_all == w]) for w in np.unique(w_all)}
    full = LogisticRegression(C=C, class_weight='balanced', max_iter=2000, random_state=SEED).fit(X, y)
    order = np.argsort(full.coef_[0])
    top = pd.DataFrame({'positive_word': vocab[order[::-1][:15]], 'pos_coef': full.coef_[0][order[::-1][:15]],
                        'negative_word': vocab[order[:15]], 'neg_coef': full.coef_[0][order[:15]]})
    return {'pred': pred, 'prob': prob, 'auc': auc, 'thresholds': thresholds,
            'top_coefficients': top, 'vocab_size': len(vocab)}


def tfidf_lr_within_word(df, n_splits=5, C=1.0, window=10) -> pd.DataFrame:
    """
    DIAGNOSTIC (not a valid system: the test words have no labels).
    5-fold stratified cross-validation INSIDE each word. If this works well
    while LOWO does not, the contexts do contain sense information but it is
    word-specific and does not transfer to new words.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.feature_extraction.text import TfidfVectorizer
    from src.features import contexts
    rows = []
    for w, g in df.groupby('word'):
        y = g['target_sense'].values
        docs = np.array(contexts(g, window))
        prob = np.zeros(len(g))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        for tr, te in skf.split(docs, y):
            vec = TfidfVectorizer(token_pattern=r'(?u)\b\w\w+\b', sublinear_tf=True)
            clf = LogisticRegression(C=C, class_weight='balanced', max_iter=2000, random_state=SEED)
            clf.fit(vec.fit_transform(docs[tr]), y[tr])
            prob[te] = clf.predict_proba(vec.transform(docs[te]))[:, 1]
        rows.append({'word': w, 'f1': f1_score(y, (prob >= 0.5).astype(int), zero_division=0),
                     'auc': roc_auc_score(y, prob)})
    return pd.DataFrame(rows).set_index('word')


# ===========================================================================
# Subtask 1b: export the change scores and sense shares (for error analysis)
# ===========================================================================

def jsd_table(res_by_system: dict) -> pd.DataFrame:
    """One row per scored (word, period pair): gold JSD and each system's JSD
    (Jensen-Shannon distance, exactly as the official scorer computes it)."""
    first = next(iter(res_by_system.values()))
    rows = []
    for item in sorted(first['gold_jsd']):
        word, p1, p2 = item.rsplit('_', 2)
        row = {'word': word, 'period_from': p1, 'period_to': p2,
               'cross_era': CROSS_ERA == f'{p1}_{p2}', 'gold_jsd': first['gold_jsd'][item]}
        for name, res in res_by_system.items():
            row[f'pred_jsd_{name}'] = res['pred_jsd'][item]
        rows.append(row)
    return pd.DataFrame(rows)


def sense_shares(df, labels_by_word) -> pd.DataFrame:
    """Share of each predicted cluster per word and period (what 1b compares)."""
    rows = []
    for w, g in df.groupby('word'):
        t = pd.crosstab(g['period_label'].values, labels_by_word[w], normalize='index')
        t.index.name, t.columns.name = 'period_label', 'cluster'
        t = t.stack().rename('share').reset_index()
        t.insert(0, 'word', w)
        rows.append(t)
    return pd.concat(rows, ignore_index=True)
