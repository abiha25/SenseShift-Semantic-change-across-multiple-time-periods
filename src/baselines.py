"""
Comparison systems (baselines) for Assignment 2.

Three baselines, all on the shared split (data/splits/SV_splits.json),
all with settings from configs/default.yaml, all scored by src/evaluate.py:

  1. SIMPLE      majority : one cluster per word (= the dominant sense) and
                            the majority 0/1 label of the training words.
                            (A constant change score has rho = 0 by the
                            official rule, the same as one cluster per word.)
                 random   : random clusters (K = median number of senses of
                            the training words), random 0/1 labels (at the
                            positive rate of the training words), random
                            change scores. Run with the 5 seeds in the config.
  2. CLASSICAL   Assignment 1 Baseline A, unchanged method: TF-IDF context
                 vectors + average-linkage clustering (1a/1b), logistic
                 regression on usage-definition similarity ranks (2).
  3. PRETRAINED  frozen XLM-R (no fine-tuning): target vectors from one
                 hidden layer, clustered (1a/1b); APD as a graded change
                 score; usage-definition cosine (2).

HOW THE SPLIT IS USED
    For every fold the test word is predicted ONCE. Every choice (clustering
    setting, decision threshold, majority label, K for random) is made on
    the other nine words = the fold's 8 training words + its validation
    word. The baselines have no training loop, so there is nothing to early-
    stop; the validation word is simply used together with the training
    words for these choices. The test word's gold labels are never used.
    The 10 test-word predictions are written into one file and scored once
    (so Subtask 1b uses all 60 scored period pairs), as Section 7 requires.

CHANGES TO THE ASSIGNMENT 1 CODE (method unchanged, objective updated)
    * Subtask 2 thresholds are tuned on the NEW official metric (macro F1
      over both classes per word). Assignment 1 tuned them on positive-class
      F1, which is no longer what the task scores.
    * A constant change prediction now scores rho = 0 in the official code
      (was NaN, which Assignment 1 treated as -1 during selection).

Run (repo root; Colab: see notebooks/baselines.ipynb):
    python -m src.baselines --which simple classical
    python -m src.baselines --which pretrained        # GPU recommended
Each system writes results/runs/<time>_<name>_seed<seed>/ with config.yaml,
run_info.json, SV_subtask1.jsonl, SV_subtask2.jsonl, [SV_graded_change.jsonl],
selection.json, scores.json, per_word.csv, subgroups.csv.
python -m src.baselines --table   collects every run into results/baselines/.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loader import load_and_prepare, period_start_year, write_subtask1, write_subtask2  # noqa: E402
from src.config import load_config, make_run_dir, save_run_config  # noqa: E402
from src.evaluate import evaluate_predictions, summarise, write_graded  # noqa: E402
from src.reproducibility import set_seed  # noqa: E402
from src.splits import load_splits  # noqa: E402

warnings.filterwarnings('ignore', message='An input array is constant')
warnings.filterwarnings('ignore', category=UserWarning)

DELTA_1A = 0.03                                     # A1 'combined' rule
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)
SUMMARY_DIR = REPO_ROOT / 'results' / 'baselines'


# ===========================================================================
# Shared set-up
# ===========================================================================

def setup(config=None, overrides=None):
    """Config, seeded RNGs, the prepared data and the verified split."""
    cfg = load_config(config or REPO_ROOT / 'configs' / 'default.yaml', overrides)
    set_seed(cfg['seed'], deterministic=cfg['training']['deterministic'])
    df = load_and_prepare(cfg['data']['data_dir'], cfg['data']['language'], cfg['preprocessing'])
    splits = load_splits(cfg['split']['file'], df, cfg['data']['data_dir'])
    folds = splits['folds']
    if cfg['split']['folds'] != 'all':
        folds = [f for f in folds if f['fold'] in set(cfg['split']['folds'])]
    gold1 = Path(cfg['data']['data_dir']) / cfg['data']['language'] / 'subtask1.jsonl'
    return cfg, df, folds, gold1


def selection_words(fold):
    """The words a baseline may use to make its choices in this fold."""
    return list(fold['train']) + list(fold['validation'])


def check_all_words_predicted(df, folds):
    covered = sorted(f['test'][0] for f in folds)
    if covered != sorted(df['word'].unique()):
        raise ValueError('Not every word is a test word in the chosen folds, so the official '
                         f'scorer would reject the file. Folds cover {covered}. Use split.folds=all.')


def save_system(cfg, df, name, description, s1_labels=None, s2_pred=None, graded=None,
                selection=None, seed=None, extra=None, verbose=True):
    """
    Write one run folder and score it with the shared evaluation script.
    s1_labels: {word: array of cluster ids (rows in df order for that word)}
    s2_pred:   array of 0/1 aligned with df
    graded:    list of {'word', 'period1', 'period2', 'score'}
    """
    run_cfg = copy.deepcopy(cfg)
    if seed is not None:
        run_cfg['seed'] = int(seed)
    run_dir = make_run_dir(run_cfg, f'baseline_{name}')
    save_run_config(run_cfg, run_dir, extra={'system': name, 'description': description,
                                             **(extra or {})})
    lang = cfg['data']['language']
    s1 = s2 = gr = None
    if s1_labels is not None:
        col = np.zeros(len(df), dtype=int)
        for w, g in df.groupby('word'):
            col[g.index.values] = np.asarray(s1_labels[w], dtype=int)
        s1 = run_dir / f'{lang}_subtask1.jsonl'
        write_subtask1(df.assign(_p=col), '_p', s1)
    if s2_pred is not None:
        s2 = run_dir / f'{lang}_subtask2.jsonl'
        write_subtask2(df.assign(_p=np.asarray(s2_pred, dtype=int)), '_p', s2)
    if graded is not None:
        gr = write_graded(graded, run_dir / f'{lang}_graded_change.jsonl')
    if selection is not None:
        (run_dir / 'selection.json').write_text(
            json.dumps(selection, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    res = evaluate_predictions(s1, s2, gr, name=name, out_dir=run_dir,
                               config=cfg.get('_config_file'), subgroups=True, verbose=verbose)
    return run_dir, res


def consecutive_pairs(df):
    """[(word, earlier period, later period), ...] for every word."""
    out = []
    for w, g in df.groupby('word'):
        periods = sorted(g['period_label'].unique(), key=period_start_year)
        out += [(w, a, b) for a, b in zip(periods, periods[1:])]
    return out


def official_macro_f1(y, p, words):
    """The NEW official Subtask 2 metric: macro F1 over both classes per
    word, averaged over words (same sklearn call as evaluation.py)."""
    from sklearn.metrics import f1_score
    return float(np.mean([f1_score(y[words == w], p[words == w], average='macro')
                          for w in np.unique(words)]))


def tune_threshold(y, score, words, grid=THRESHOLD_GRID):
    """Threshold on `score` that maximises the official metric on these words."""
    vals = [official_macro_f1(y, (score >= t).astype(int), words) for t in grid]
    return float(grid[int(np.argmax(vals))])


# ===========================================================================
# 1. Simple baselines
# ===========================================================================

def run_majority(cfg, df, folds, verbose=True):
    """
    One cluster per word and the majority 0/1 label of the 9 non-test words.
    No graded score: a constant change score has no ranking, so its Spearman
    rho is 0 by the official rule - exactly the 1b score of one cluster per
    word. (A per-fold mean would NOT be constant across words: each test word
    would get the mean of the other nine, which ranks the high-change words
    lowest - a leave-one-out artefact, not a baseline.)
    """
    y, w_all = df['target_sense'].values, df['word'].values
    s1 = {w: np.zeros(int((w_all == w).sum()), dtype=int) for w in sorted(set(w_all))}
    s2 = np.zeros(len(df), dtype=int)
    selection = {}
    for fold in folds:
        test, sel = fold['test'][0], selection_words(fold)
        rate = float(np.mean(y[np.isin(w_all, sel)]))
        s2[w_all == test] = int(rate >= 0.5)
        selection[test] = {'majority_label': int(rate >= 0.5),
                           'positive_rate_of_selection_words': rate}
    return save_system(cfg, df, 'majority',
                       'One cluster per word (= dominant sense); majority 0/1 label of the 9 '
                       'non-test words.', s1, s2, None, selection, verbose=verbose)


def run_random(cfg, df, folds, seed, verbose=True):
    rng = np.random.default_rng(seed)
    y, w_all = df['target_sense'].values, df['word'].values
    n_senses = df.groupby('word')['senses'].apply(lambda s: len({x for l in s for x in l})).to_dict()
    s1, s2, graded, selection = {}, np.zeros(len(df), dtype=int), [], {}
    pairs = consecutive_pairs(df)
    for fold in folds:
        test, sel = fold['test'][0], selection_words(fold)
        k = max(1, int(round(np.median([n_senses[w] for w in sel]))))
        p = float(np.mean(y[np.isin(w_all, sel)]))
        n = int((w_all == test).sum())
        s1[test] = rng.integers(0, k, size=n)
        s2[w_all == test] = (rng.random(n) < p).astype(int)
        graded += [{'word': w, 'period1': a, 'period2': b, 'score': float(rng.random())}
                   for w, a, b in pairs if w == test]
        selection[test] = {'k_clusters': k, 'positive_rate': p}
    return save_system(cfg, df, 'random',
                       'Random clusters (K = median gold senses of the 9 non-test words), '
                       'random labels at their positive rate, random change scores.',
                       s1, s2, graded, selection, seed=seed, verbose=verbose)


# ===========================================================================
# Fold-wise selection of a clustering setting (A1 lowo_select on the split)
# ===========================================================================

def select_clustering(grid, grid_scores, scorer, single, folds, objective='combined'):
    """
    For each fold, choose the clustering setting on the 9 non-test words and
    use that setting's labels for the test word.
      '1a'       highest mean B3 F1
      '1b'       highest Spearman rho on their period pairs
      'combined' highest rho among settings whose mean B3 F1 is at most
                 DELTA_1A below one cluster per word (else best B3 F1)
    """
    labels, chosen = {}, {}
    configs = list(grid_scores['config'])
    for fold in folds:
        test, sel = fold['test'][0], selection_words(fold)
        f1_sel = grid_scores[[f'f1_{w}' for w in sel]].mean(axis=1).values
        if objective == '1a':
            best = int(np.argmax(f1_sel))
        else:
            rhos = np.nan_to_num(np.array([scorer.rho(grid[c], sel) for c in configs]), nan=0.0)
            if objective == '1b':
                best = int(np.argmax(rhos))
            else:
                ok = f1_sel >= np.mean([single[w] for w in sel]) - DELTA_1A
                best = int(np.argmax(np.where(ok, rhos, -np.inf))) if ok.any() else int(np.argmax(f1_sel))
        cfg = configs[best]
        labels[test] = grid[cfg][test]
        chosen[test] = {'setting': cfg, 'mean_f1_on_selection_words': float(f1_sel[best])}
    return labels, chosen


# ===========================================================================
# 2. Classical baseline (Assignment 1 Baseline A)
# ===========================================================================

def run_classical(cfg, df, folds, gold1, verbose=True):
    from src import classical_baseline as cb
    from src.features import FeatureConfig, definition_features
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    t0 = time.time()
    # --- Subtask 1: same feature grid and clustering as Assignment 1 ----------
    feature_cfgs = [FeatureConfig(k, w) for k in cb.KINDS for w in cb.WINDOWS]
    grid = cb.build_grid(df, feature_cfgs)
    scorer = cb.FastScorer(df, gold1)
    grid_scores = cb.score_grid(df, grid, scorer)
    single = {w: scorer.f1(w, np.zeros(int((df['word'] == w).sum()), int)) for w in scorer.words}
    objective = cfg['clustering']['selection_objective']
    labels, chosen = select_clustering(grid, grid_scores, scorer, single, folds, objective)
    for c in chosen.values():
        c['setting'] = c['setting'].name()          # e.g. 'both-pm10|t=0.95|m=10'

    # --- Subtask 2: LR on within-word ranks of 4 usage-definition similarities
    F = definition_features(df)
    cols = cb.RANK
    y, w_all = df['target_sense'].values, df['word'].values
    pred = np.zeros(len(df), dtype=int)
    for fold in folds:
        test, sel = fold['test'][0], selection_words(fold)
        tr, te = np.isin(w_all, sel), w_all == test
        scaler = StandardScaler().fit(F.loc[tr, cols])
        clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000,
                                 random_state=cfg['baselines']['random_seed'])
        clf.fit(scaler.transform(F.loc[tr, cols]), y[tr])
        thr = tune_threshold(y[tr], clf.predict_proba(scaler.transform(F.loc[tr, cols]))[:, 1], w_all[tr])
        pred[te] = (clf.predict_proba(scaler.transform(F.loc[te, cols]))[:, 1] >= thr).astype(int)
        chosen[test]['subtask2_threshold'] = thr
        chosen[test]['subtask2_coefficients'] = dict(zip(cols, map(float, clf.coef_[0])))

    selection = {'objective_1': objective, 'n_settings': len(grid),
                 'features': [c.name() for c in feature_cfgs],
                 'thresholds': list(cb.THRESHOLDS), 'min_cluster_sizes': list(cb.MIN_SIZES),
                 'per_test_word': chosen, 'runtime_seconds': round(time.time() - t0, 1)}
    return save_system(cfg, df, 'classical',
                       'A1 Baseline A on the shared split: TF-IDF context vectors + average-'
                       'linkage clustering; LR on usage-definition similarity ranks, threshold '
                       'tuned on the new official macro F1.',
                       labels, pred, None, selection, verbose=verbose)


# ===========================================================================
# 3. Pretrained baseline: frozen XLM-R + APD
# ===========================================================================

def encode_frozen(items, model_name, layer, max_length=256, batch_size=32, pooling='target_mean'):
    """
    Frozen encoder, no gradient: one vector per (text, start, end) = mean of
    the hidden states (layer `layer`) of the sub-words that overlap the
    target characters. Same method as Assignment 1 (pretrained_baseline.py),
    settings from the config. Returns (vectors, n_truncated) where
    n_truncated counts targets cut off by max_length (their vector falls
    back to the mean of the whole sentence).
    """
    import torch
    from transformers import AutoModel, AutoTokenizer
    if pooling != 'target_mean':
        raise ValueError(f'pooling {pooling!r} is not implemented for the baseline')
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).eval()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(dev)
    vecs, n_trunc = [], 0
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        enc = tok([t for t, _, _ in chunk], return_offsets_mapping=True, return_tensors='pt',
                  padding=True, truncation=True, max_length=max_length)
        offs = enc.pop('offset_mapping')
        with torch.no_grad():
            h = model(**{k: v.to(dev) for k, v in enc.items()}).hidden_states[layer].float().cpu()
        for j, (_, s, e) in enumerate(chunk):
            m = (offs[j, :, 0] < e) & (offs[j, :, 1] > s) & (offs[j, :, 1] > offs[j, :, 0])
            if not m.any():
                n_trunc += 1
                m = enc['attention_mask'][j].bool()
            vecs.append(h[j][m].mean(0).numpy())
    return np.vstack(vecs), n_trunc


def cached_vectors(cfg, df, cache_dir, encoder=None):
    """Encode usages + definitions once; reuse the .npy files afterwards."""
    from src.splits import data_fingerprint
    m = cfg['model']
    pp = cfg['preprocessing']
    key = (f"{m['frozen_encoder'].replace('/', '-')}_L{m['layer']}_{m['pooling']}_len{m['max_length']}"
           f"_spell{int(pp['spelling_normalisation'])}_lower{int(pp['lowercase'])}"
           f"_win{pp['context_window']}_{data_fingerprint(cfg['data']['data_dir'], cfg['data']['language'])[:10]}")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fu, fd = cache_dir / f'{key}_usages.npy', cache_dir / f'{key}_definitions.json'
    if fu.exists() and fd.exists():
        info = json.loads(fd.read_text(encoding='utf-8'))
        return np.load(fu), {w: np.array(v) for w, v in info['definitions'].items()}, info['n_truncated'], key

    from data_loader import prepare_usage
    enc = encoder or (lambda items: encode_frozen(items, m['frozen_encoder'], m['layer'],
                                                  m['max_length'], m['embedding_batch_size'],
                                                  m['pooling']))
    items = list(zip(df['model_text'], df['model_start'].astype(int), df['model_end'].astype(int)))
    V, n_trunc = enc(items)
    # Definition as a pseudo-usage "word: definition", target = the word
    # (as in A1 Baseline B), with the same preprocessing as the usages.
    words = sorted(df['word'].unique())
    defs = []
    for w in words:
        d = df.loc[df['word'] == w, 'definition'].iloc[0]
        defs.append(prepare_usage(f'{w}: {d}', 0, len(w), cfg['data']['language'],
                                  pp['spelling_normalisation'], pp['lowercase'], None))
    D, _ = enc(defs)
    np.save(fu, V)
    fd.write_text(json.dumps({'definitions': {w: d.tolist() for w, d in zip(words, D)},
                              'n_truncated': n_trunc}), encoding='utf-8')
    return V, dict(zip(words, D)), n_trunc, key


def apd_scores(df, V):
    """
    Average pairwise cosine distance (APD) between the usage vectors of two
    consecutive periods: mean over all (u in earlier, v in later) of
    1 - cos(u, v). No clustering and no labels are involved.
    """
    X = V / np.linalg.norm(V, axis=1, keepdims=True)
    rows = []
    for w, a, b in consecutive_pairs(df):
        ia = df.index[(df['word'] == w) & (df['period_label'] == a)].values
        ib = df.index[(df['word'] == w) & (df['period_label'] == b)].values
        rows.append({'word': w, 'period1': a, 'period2': b,
                     'score': float(np.mean(1.0 - X[ia] @ X[ib].T))})
    return rows


def run_pretrained(cfg, df, folds, gold1, cache_dir=None, encoder=None, verbose=True):
    from src import classical_baseline as cb
    from src import pretrained_baseline as pb

    t0 = time.time()
    cache_dir = cache_dir or REPO_ROOT / 'cache'
    V, Dvec, n_trunc, key = cached_vectors(cfg, df, cache_dir, encoder)
    t_enc = time.time() - t0

    # --- Subtask 1: cluster the frozen vectors -------------------------------
    cl = cfg['clustering']
    thresholds = pb.quantile_thresholds(df, V, tuple(cl['threshold_percentiles']))
    grid = pb.cluster_all_words(df, V, thresholds, cl['min_cluster_sizes'])
    scorer = cb.FastScorer(df, gold1)
    configs = list(grid)
    gs = pd.DataFrame([{'config': c, **{f'f1_{w}': scorer.f1(w, grid[c][w]) for w in scorer.words}}
                       for c in configs])
    single = {w: scorer.f1(w, np.zeros(int((df['word'] == w).sum()), int)) for w in scorer.words}
    labels, chosen = select_clustering(grid, gs, scorer, single, folds, cl['selection_objective'])
    pct = dict(zip(thresholds, cl['threshold_percentiles']))
    for w, c in chosen.items():
        t, m = c['setting']
        c['setting'] = {'distance_threshold': t, 'threshold_percentile': pct.get(t), 'min_cluster_size': m}

    # --- Graded change: APD ---------------------------------------------------
    graded = apd_scores(df, V)

    # --- Subtask 2: cosine(usage, definition) -> within-word rank -> threshold
    X = V / np.linalg.norm(V, axis=1, keepdims=True)
    cos = np.array([X[i] @ (Dvec[w] / np.linalg.norm(Dvec[w])) for i, w in enumerate(df['word'])])
    rank = pd.Series(cos, index=df.index).groupby(df['word']).rank(pct=True).values
    y, w_all = df['target_sense'].values, df['word'].values
    pred = np.zeros(len(df), dtype=int)
    for fold in folds:
        test, sel = fold['test'][0], selection_words(fold)
        tr, te = np.isin(w_all, sel), w_all == test
        q = tune_threshold(y[tr], rank[tr], w_all[tr])
        pred[te] = (rank[te] >= q).astype(int)
        chosen[test]['subtask2_rank_threshold'] = q

    m = cfg['model']
    selection = {'encoder': m['frozen_encoder'], 'layer': m['layer'], 'pooling': m['pooling'],
                 'max_length': m['max_length'], 'vector_cache': key,
                 'n_targets_truncated': n_trunc, 'objective_1': cl['selection_objective'],
                 'distance_thresholds': thresholds,
                 'threshold_percentiles': cl['threshold_percentiles'],
                 'min_cluster_sizes': cl['min_cluster_sizes'], 'per_test_word': chosen,
                 'encoding_seconds': round(t_enc, 1), 'runtime_seconds': round(time.time() - t0, 1)}
    return save_system(cfg, df, 'pretrained_frozen_xlmr',
                       f"Frozen {m['frozen_encoder']} layer {m['layer']} ({m['pooling']}), no "
                       'fine-tuning: agglomerative clustering (1a/1b), APD graded change, '
                       'usage-definition cosine rank threshold (2).',
                       labels, pred, graded, selection, verbose=verbose)


# ===========================================================================
# Summary table of all baseline runs
# ===========================================================================

def collect_table(runs_dir=None, out_dir=SUMMARY_DIR):
    """
    Read every results/runs/*_baseline_*/scores.json and write
    results/baselines/summary.csv (+ .md). Random: mean +- sd over seeds,
    and the latest run of every other system.
    """
    runs_dir = Path(runs_dir or REPO_ROOT / 'results' / 'runs')
    by_system = {}
    for p in sorted(runs_dir.glob('*_baseline_*/scores.json')):
        r = json.loads(p.read_text(encoding='utf-8'))
        r['_dir'] = p.parent.name
        by_system.setdefault(r['name'], []).append(r)
    rows = []
    order = ['majority', 'random', 'classical', 'pretrained_frozen_xlmr']
    for name in sorted(by_system, key=lambda n: order.index(n) if n in order else 99):
        runs = by_system[name]
        if name == 'random':
            seeds = {}
            for r in runs:                                   # latest run per seed
                seeds[r['_dir'].rsplit('_seed', 1)[1]] = r
            s = summarise(list(seeds.values()))
            row = {'system': f'random ({len(seeds)} seeds, mean ± sd)'}
            for k in ('1a_bcubed_f1', '1b_spearman', '2_macro_f1', 'graded_spearman'):
                if k in s:
                    row[k] = f"{s[k]['mean']:.3f} ± {s[k]['sd']:.3f}"
            row['runs'] = ', '.join(sorted(r['_dir'] for r in seeds.values()))
        else:
            r = runs[-1]
            sc = r['scores']
            row = {'system': name}
            for k in ('1a_bcubed_f1', '1b_spearman', '2_macro_f1', 'graded_spearman'):
                if sc.get(k) is not None:
                    ci = sc.get(f'{k}_ci')
                    row[k] = f'{sc[k]:.3f}' + (f" [{ci['low']:.3f}, {ci['high']:.3f}]" if ci else '')
            ctrl = r.get('subtask1', {}).get('subtask1b', {}).get('shuffle_control')
            if ctrl:
                row['1b_shuffle_control'] = f"{ctrl['shuffled_mean']:.3f} ± {ctrl['shuffled_sd']:.3f}"
            row['runs'] = r['_dir']
        rows.append(row)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows).fillna('—')
    table.to_csv(out_dir / 'summary.csv', index=False)
    header = ('Baselines on the shared split (Swedish dev, 10 test words, official scorer of '
              '8 Nov 2026). 95% CI by bootstrap over words. graded_spearman is a LOCAL score '
              '(APD), not an official one.\n\n')
    cols = list(table.columns)
    md = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    md += ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in table.values]
    (out_dir / 'summary.md').write_text(header + '\n'.join(md) + '\n', encoding='utf-8')
    return table


# ===========================================================================
# Command line
# ===========================================================================

def main(argv=None):
    from src.config import add_cli_arguments
    ap = argparse.ArgumentParser(description='Run the comparison baselines on the shared split.')
    ap.add_argument('--which', nargs='+', choices=['simple', 'classical', 'pretrained'], default=[])
    ap.add_argument('--table', action='store_true', help='collect all baseline runs into a table')
    ap.add_argument('--cache', default=str(REPO_ROOT / 'cache'), help='folder for encoder vectors')
    add_cli_arguments(ap)
    a = ap.parse_args(argv)

    if a.which:
        cfg, df, folds, gold1 = setup(a.config, a.overrides)
        check_all_words_predicted(df, folds)
        if 'simple' in a.which:
            run_majority(cfg, df, folds)
            for seed in cfg['seeds_for_final_runs']:
                run_random(cfg, df, folds, seed)
        if 'classical' in a.which:
            run_classical(cfg, df, folds, gold1)
        if 'pretrained' in a.which:
            run_pretrained(cfg, df, folds, gold1, cache_dir=a.cache)
    if a.table or a.which:
        print(collect_table().to_string(index=False))


if __name__ == '__main__':
    main()
