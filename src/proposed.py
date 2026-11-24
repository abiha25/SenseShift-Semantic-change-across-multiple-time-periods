"""
Proposed system (Assignment 2).

    usages --> preprocessing (shared loader) --> fine-tuned XLM-R usage encoder
           --> one vector per usage --> agglomerative clustering per word (senses)
           --> sense distribution per period --> JSD between consecutive periods (1b)
           --> APD between consecutive periods (local graded score)
           --> usage-definition cosine (Subtask 2)

Training (src/usage_encoder.py), two stages:
    Stage A  synthetic English (organisers' set, ~2,300 words). Uses no Swedish
             data, so it is trained ONCE per seed and shared by all folds.
    Stage B  per fold: the 8 TRAINING words of the fold (Swedish gold senses),
             early stopping on the VALIDATION word (pair AUC). The test word is
             never seen in training or in any choice.

Everything is read from configs/default.yaml; override with --set key=value.

Run (repo root):
    python -m src.proposed run                                  # full system, all folds
    python -m src.proposed run --set split.folds=[0]            # one fold, quick test
    python -m src.proposed run --set training.enabled=false     # frozen encoder, same pipeline
    python -m src.proposed recluster --run results/runs/<dir> --set clustering.threshold_selection=fixed
          (re-uses the saved vectors: new clustering / Subtask 2 settings without retraining)
    python -m src.proposed recluster --run results/runs/<dir> --set clustering.method=kmeans
          (Assignment 3, E4: K-means, K in clustering.kmeans_k_range chosen by silhouette)
    python -m src.proposed run --name pool_first --set model.pooling=first_subword
          (Assignment 3, E6: pooling = target_mean | first_subword | sentence_mean)

Each run writes results/runs/<time>_proposed_<name>_seed<seed>/ with config.yaml,
run_info.json, SV_subtask1.jsonl, SV_subtask2.jsonl, SV_graded_change.jsonl,
selection.json, training_history.json, vectors/fold<k>.npz, scores.json, per_word.csv.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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

from data_loader import period_start_year, write_subtask1, write_subtask2  # noqa: E402
from src.baselines import (apd_scores, check_all_words_predicted, official_macro_f1,  # noqa: E402
                           select_clustering, selection_words, setup, tune_threshold)
from src.config import add_cli_arguments, load_config, make_run_dir, save_run_config  # noqa: E402
from src.evaluate import evaluate_predictions, write_graded  # noqa: E402
from src.reproducibility import set_seed  # noqa: E402

warnings.filterwarnings('ignore', message='An input array is constant')


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def _abs(path):
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


# ===========================================================================
# Encoder construction and training
# ===========================================================================

def build_encoder(cfg, path=None, frozen=False):
    from src.usage_encoder import UsageEncoder
    m, tr = cfg['model'], cfg['training']
    if frozen:
        return UsageEncoder(m['frozen_encoder'], layer=m['layer'], markers=False,
                            max_length=m['max_length'], freeze_embeddings=True,
                            pooling=m['pooling'])
    return UsageEncoder(path or m['base_encoder'], layer=m['finetuned_layer'],
                        markers=m['target_markers'], max_length=m['max_length'],
                        freeze_embeddings=tr['freeze_embeddings'], dropout=tr.get('dropout'),
                        pooling=m['pooling'])


def _sampler(cfg, items):
    from src.usage_encoder import PairSampler
    tr = cfg['training']
    return PairSampler(items, tr['positive_fraction'], tr['era_aware_sampling'],
                       tr['p_cross_era_positive'], tr['p_same_era_negative'], tr['soft_labels'])


def stage_a_key(cfg):
    m, tr = cfg['model'], cfg['training']
    keep = {'base': m['base_encoder'], 'layer': m['finetuned_layer'], 'markers': m['target_markers'],
            'max_length': m['max_length'], 'syn': tr['synthetic'], 'margin': tr['margin'],
            'soft': tr['soft_labels'], 'era': tr['era_aware_sampling'], 'pos': tr['positive_fraction'],
            'p': [tr['p_cross_era_positive'], tr['p_same_era_negative']], 'bs': tr['batch_size'],
            'frz': tr['freeze_embeddings'], 'drop': tr.get('dropout'), 'seed': cfg['seed']}
    if tr.get('stage_a_guard'):
        # Added in Assignment 3 after the seed-87 collapse. Off by default, so the
        # key of every existing Stage A checkpoint stays the same.
        keep['stage_a_guard'] = True
    if m.get('pooling', 'target_mean') != 'target_mean':
        # Added in Assignment 3 (E6). Only a non-default pooling changes the key, so
        # the Stage A checkpoints of the Assignment 2 settings stay valid.
        keep['pooling'] = m['pooling']
    return hashlib.sha1(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:10]


def train_stage_a(cfg, history):
    """
    Synthetic English pre-training (once per seed, cached in checkpoints_dir).
    The pair AUC on held-out synthetic words is measured before and after
    training. With training.stage_a_guard=false (Assignment 2) the final weights
    are always kept; with true, the pre-training weights are kept if training
    did not raise that AUC (added after the seed-87 collapse, Assignment 3).
    """
    from src.usage_encoder import load_synthetic, pair_auc, train_pairs
    tr, syn = cfg['training'], cfg['training']['synthetic']
    ckpt = Path(cfg['output']['checkpoints_dir']) / f'stageA_{stage_a_key(cfg)}'
    if (ckpt / 'usage_encoder.json').exists():
        log(f'Stage A: loading cached checkpoint {ckpt}')
        history['stage_a'] = json.loads((ckpt / 'history.json').read_text(encoding='utf-8'))
        return build_encoder(cfg, path=ckpt)
    log('Stage A: loading synthetic English data')
    items, hold, stats = load_synthetic(_abs(syn['data_dir']), syn['max_per_sense'], cfg['seed'],
                                        syn['holdout_fraction'], syn.get('max_words'))
    log(f'Stage A: {stats}')
    enc = build_encoder(cfg)
    sampler = _sampler(cfg, items)
    rng = np.random.default_rng(cfg['seed'])
    per_epoch = int(syn['n_pairs']) // int(syn['epochs'])

    hold_words = {}
    for k, it in enumerate(hold):
        hold_words.setdefault(it['word'], []).append(k)
    hold_words = dict(list(sorted(hold_words.items()))[:40])

    def eval_fn(e):
        aucs = []
        for w, ks in hold_words.items():
            V = e.encode([(hold[k]['text'], hold[k]['start'], hold[k]['end']) for k in ks],
                         cfg['model']['embedding_batch_size'])
            a = pair_auc(V, [hold[k]['senses'] for k in ks])
            if not np.isnan(a):
                aucs.append(a)
        return float(np.mean(aucs)) if aucs else float('nan')

    hist = train_pairs(enc, items, lambda ep: sampler.sample(per_epoch, rng), int(syn['epochs']),
                       float(syn['learning_rate']), tr['batch_size'], tr['warmup_ratio'],
                       tr['weight_decay'], tr['max_grad_norm'], tr['margin'], eval_fn=eval_fn,
                       patience=10 ** 6, eval_before=True, seed=cfg['seed'], log=log, tag='Stage A',
                       keep_best=bool(tr.get('stage_a_guard', False)))
    history['stage_a'] = {'data': stats, 'epochs': hist}
    if tr.get('stage_a_guard'):
        # keep_best=True restored the best of {before training, after each epoch}
        # on the held-out SYNTHETIC words (no Swedish data is involved).
        fired = bool(hist and hist[0].get('best'))
        history['stage_a']['guard'] = {'enabled': True, 'kept_pretraining_weights': fired}
        if fired:
            log('Stage A guard: training lowered the held-out pair AUC; keeping the pre-training weights')
    enc.save(ckpt)
    (ckpt / 'history.json').write_text(json.dumps(history['stage_a'], indent=2), encoding='utf-8')
    log(f'Stage A: saved {ckpt}')
    return enc


def train_stage_b(cfg, enc, df, fold, seed):
    """Fine-tune on the fold's training words; early stopping on the validation word."""
    from src.usage_encoder import pair_auc, swedish_items, train_pairs
    tr = cfg['training']
    held_out = set(fold['test']) | set(fold['validation'])
    if held_out & set(fold['train']):
        raise ValueError(f"Fold {fold['fold']}: test/validation word in the training words")
    items = swedish_items(df, fold['train'], tr['exclude_conflicting_duplicates'])
    assert not {it['word'] for it in items} & held_out, 'leakage: held-out word in training pairs'
    sampler = _sampler(cfg, items)
    rng = np.random.default_rng(seed)
    val = df[df['word'].isin(fold['validation'])]
    val_items = list(zip(val['model_text'], val['model_start'].astype(int), val['model_end'].astype(int)))
    val_senses = list(val['senses'])

    def eval_fn(e):
        return pair_auc(e.encode(val_items, cfg['model']['embedding_batch_size']), val_senses)

    return train_pairs(enc, items, lambda ep: sampler.sample(int(tr['pairs_per_epoch']), rng),
                       int(tr['epochs']), float(tr['learning_rate']), tr['batch_size'],
                       tr['warmup_ratio'], tr['weight_decay'], tr['max_grad_norm'], tr['margin'],
                       eval_fn=eval_fn, patience=int(tr['early_stopping_patience']),
                       eval_before=True, seed=seed, log=log, tag=f"Fold {fold['fold']} ({fold['test'][0]})")


# ===========================================================================
# The model functions of the work plan
# ===========================================================================

def extract_embeddings(encoder, df, cfg):
    """One vector per usage (rows aligned with df) and one per definition.
    The definition is encoded as a pseudo-usage 'word: definition' with the
    word as target, with the same preprocessing as the usages."""
    from data_loader import prepare_usage
    pp, bs = cfg['preprocessing'], cfg['model']['embedding_batch_size']
    items = list(zip(df['model_text'], df['model_start'].astype(int), df['model_end'].astype(int)))
    V = encoder.encode(items, bs)
    words = sorted(df['word'].unique())
    defs = []
    for w in words:
        d = df.loc[df['word'] == w, 'definition'].iloc[0]
        defs.append(prepare_usage(f'{w}: {d}', 0, len(w), cfg['data']['language'],
                                  pp['spelling_normalisation'], pp['lowercase'], None))
    D = encoder.encode(defs, bs)
    return V, dict(zip(words, D))


def _unit(X):
    return X / np.linalg.norm(X, axis=-1, keepdims=True).clip(1e-12)


def cosine_distances(Vw):
    X = _unit(Vw)
    return np.clip(1.0 - X @ X.T, 0.0, 2.0)


def cluster_senses(Vw, threshold, min_size):
    """Average-linkage agglomerative clustering on cosine distance, cut at
    `threshold`; clusters smaller than `min_size` join the nearest large one
    (A1 rule, so a word can end up with one sense). Labels 0.. by size."""
    from src.classical_baseline import agglomerative_labels
    from src.pretrained_baseline import DenseWrap
    if len(Vw) < 2:
        return np.zeros(len(Vw), dtype=int)
    return agglomerative_labels(DenseWrap(Vw), [float(threshold)], [int(min_size)])[(float(threshold), int(min_size))]


def kmeans_senses(Vw, k_range=(2, 10), seed=42):
    """
    K-means on unit-length vectors; K is the value in k_range with the highest
    silhouette (cosine) on this word's own vectors - label-free, so it uses no
    gold sense of any word (Giulianelli et al. 2020; ablation E4). Unlike the
    agglomerative rule, K-means always returns at least 2 clusters. Ties go to
    the smaller K. Labels 0.. by cluster size (largest first), like
    cluster_senses. Returns (labels, info).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    X = _unit(np.asarray(Vw, dtype=np.float64))
    n = len(X)
    k_lo, k_hi = int(k_range[0]), int(k_range[1])
    ks = [k for k in range(max(2, k_lo), min(k_hi, n - 1) + 1)]
    if not ks:
        return np.zeros(n, dtype=int), {'k': 1, 'silhouette': None, 'silhouettes': {}}
    best, sil = None, {}
    for k in ks:
        lab = KMeans(n_clusters=k, n_init=10, random_state=int(seed)).fit_predict(X)
        sil[k] = float(silhouette_score(X, lab, metric='cosine')) if len(set(lab)) > 1 else -1.0
        if best is None or sil[k] > sil[best[0]] + 1e-12:
            best = (k, lab)
    k, lab = best
    order = np.argsort(-np.bincount(lab, minlength=k), kind='stable')
    relabel = np.empty(k, dtype=int)
    relabel[order] = np.arange(k)
    return relabel[lab], {'k': int(k), 'silhouette': sil[k], 'silhouettes': sil}


def score_change(df, labels_by_word):
    """JSD (base 2, as distance = sqrt of the divergence, like the official
    code) between the sense distributions of consecutive periods."""
    from scipy.spatial.distance import jensenshannon
    rows = []
    for w, g in df.groupby('word'):
        lab = np.asarray(labels_by_word[w])
        periods = sorted(g['period_label'].unique(), key=period_start_year)
        k = int(lab.max()) + 1
        dist = {}
        for p in periods:
            c = np.bincount(lab[(g['period_label'] == p).values], minlength=k).astype(float)
            dist[p] = c / c.sum()
        for a, b in zip(periods, periods[1:]):
            rows.append({'word': w, 'period1': a, 'period2': b,
                         'jsd': float(jensenshannon(dist[a], dist[b], base=2))})
    return rows


# ===========================================================================
# Choices made without the test word
# ===========================================================================

def _same_sense_matrix(senses):
    return np.array([[bool(set(a) & set(b)) for b in senses] for a in senses])


def choose_clustering(cfg, df, V, fold, scorer):
    """Pick (threshold, min_size) for the test word of this fold, using only
    the validation word (or the 9 non-test words for 'nine_words')."""
    cl = cfg['clustering']
    mode = cl['threshold_selection']
    sizes = [int(m) for m in cl['min_cluster_sizes']]
    val = fold['validation'][0]
    if mode == 'fixed':
        return {'mode': mode, 'threshold': cfg['training']['margin'] / 2,
                'min_size': int(cl['fixed_min_cluster_size'])}
    if mode in ('validation_pairs', 'validation_bcubed'):
        g = df[df['word'] == val]
        Vw = V[g.index.values]
        D = cosine_distances(Vw)
        iu = np.triu_indices(len(Vw), 1)
        d = D[iu]
        cands = np.unique(np.round(np.percentile(d, np.arange(2, 99, 2)), 4))
        if mode == 'validation_pairs':
            same = _same_sense_matrix(list(g['senses']))[iu]
            bal = [0.5 * (np.mean(d[same] <= t) + np.mean(d[~same] > t)) if same.any() and (~same).any()
                   else 0.0 for t in cands]
            t = float(cands[int(np.argmax(bal))])
            f1s = {m: scorer.f1(val, cluster_senses(Vw, t, m)) for m in sizes}
            m = max(sizes, key=lambda s: (round(f1s[s], 6), s))
            return {'mode': mode, 'threshold': t, 'min_size': m, 'val_balanced_acc': float(max(bal)),
                    'val_f1': f1s[m]}
        best = max(((scorer.f1(val, cluster_senses(Vw, t, m)), m, float(t)) for t in cands for m in sizes))
        return {'mode': mode, 'threshold': best[2], 'min_size': best[1], 'val_f1': best[0]}
    if mode == 'nine_words':
        sel = selection_words(fold)
        dists = np.concatenate([cosine_distances(V[df.index[df['word'] == w].values])[
            np.triu_indices(int((df['word'] == w).sum()), 1)] for w in sel])
        thresholds = [round(float(np.percentile(dists, q)), 4) for q in cl['threshold_percentiles']]
        words = sel + fold['test']
        grid = {(t, m): {w: cluster_senses(V[df.index[df['word'] == w].values], t, m) for w in words}
                for t in thresholds for m in sizes}
        gs = pd.DataFrame([{'config': c, **{f'f1_{w}': scorer.f1(w, grid[c][w]) for w in sel}}
                           for c in grid])
        single = {w: scorer.f1(w, np.zeros(int((df['word'] == w).sum()), int)) for w in sel}
        _, chosen = select_clustering(grid, gs, scorer, single, [fold], cl['selection_objective'])
        t, m = chosen[fold['test'][0]]['setting']
        return {'mode': mode, 'threshold': float(t), 'min_size': int(m),
                'mean_f1_selection_words': chosen[fold['test'][0]]['mean_f1_on_selection_words']}
    raise ValueError(f'Unknown clustering.threshold_selection: {mode}')


def subtask2_predict(cfg, df, V, D, fold, test_labels):
    """0/1 for the usages of the test word. Modes:
       absolute  cosine(usage, definition) >= t, t tuned on the 9 non-test words
                 (official macro F1). The positive rate can differ per word.
       rank      within-word percentile rank >= q (A1 / baseline method)
       cluster   the definition is assigned to its nearest induced sense; that
                 sense's usages are 1 (ties Subtask 2 to the Subtask 1 clusters)
    Returns {mode: predictions} for all three, plus the chosen thresholds."""
    w_all, y = df['word'].values, df['target_sense'].values
    test, sel = fold['test'][0], selection_words(fold)
    cos = np.einsum('ij,ij->i', _unit(V), _unit(np.vstack([D[w] for w in w_all])))
    tr, te = np.isin(w_all, sel), w_all == test
    out, info = {}, {}
    grid = np.unique(np.round(np.percentile(cos[tr], np.arange(2, 99, 2)), 4))
    t_abs = tune_threshold(y[tr], cos[tr], w_all[tr], grid)
    out['absolute'] = (cos[te] >= t_abs).astype(int)
    info['absolute_threshold'] = t_abs
    rank = pd.Series(cos).groupby(w_all).rank(pct=True).values
    q = tune_threshold(y[tr], rank[tr], w_all[tr])
    out['rank'] = (rank[te] >= q).astype(int)
    info['rank_threshold'] = q
    Vt, lab = _unit(V[te]), np.asarray(test_labels)
    cents = np.vstack([_unit(Vt[lab == c].mean(0)) for c in range(lab.max() + 1)])
    best = int(np.argmax(cents @ _unit(D[test])))
    out['cluster'] = (lab == best).astype(int)
    info['cluster_chosen'] = best
    return out, info


# ===========================================================================
# Run
# ===========================================================================

def run_name(cfg):
    tr = cfg['training']
    if not tr['enabled']:
        return 'frozen'
    parts = ['ft']
    if tr['synthetic']['enabled']:
        parts.append('syn')
    if tr['swedish_stage']:
        parts.append('sv')
    return '_'.join(parts)


def predict(cfg, df, folds, vectors, scorer, run_dir, histories=None, extra=None, name=None):
    """
    Clustering, change scores and Subtask 2 for every test word from the saved
    vectors ({fold id: (V, D)}), then write the official files and score them.
    """
    labels, s2_main, s2_alt, chosen = {}, np.zeros(len(df), dtype=int), {}, {}
    V_pred = np.zeros_like(next(iter(vectors.values()))[0])
    mode2 = cfg['subtask2']['mode']
    for fold in folds:
        V, D = vectors[fold['fold']]
        test = fold['test'][0]
        idx = df.index[df['word'] == test].values
        if cfg['clustering']['method'] == 'kmeans':
            labels[test], c = kmeans_senses(V[idx], cfg['clustering']['kmeans_k_range'], cfg['seed'])
            c = {'mode': 'kmeans_silhouette', 'threshold': None, 'min_size': None, **c}
        elif cfg['clustering']['method'] == 'agglomerative':
            c = choose_clustering(cfg, df, V, fold, scorer)
            labels[test] = cluster_senses(V[idx], c['threshold'], c['min_size'])
        else:
            raise ValueError(f"Unknown clustering.method: {cfg['clustering']['method']}")
        c['n_clusters'] = int(labels[test].max()) + 1
        preds, info = subtask2_predict(cfg, df, V, D, fold, labels[test])
        s2_main[idx] = preds[mode2]
        for k, p in preds.items():
            s2_alt.setdefault(k, np.zeros(len(df), dtype=int))[idx] = p
        V_pred[idx] = V[idx]
        chosen[test] = {'fold': fold['fold'], 'validation_word': fold['validation'][0],
                        'clustering': c, 'subtask2': info}
        log(f"{test}: {c['n_clusters']} cluster(s), " + (f"K-means silhouette {c['silhouette']:.4f}"
            if c['mode'] == 'kmeans_silhouette' else f"t={c['threshold']:.4f}, m={c['min_size']}"))

    lang = cfg['data']['language']
    col = np.zeros(len(df), dtype=int)
    for w in labels:
        col[df.index[df['word'] == w].values] = labels[w]
    keep = df['word'].isin(list(labels)).values      # all words, or the test words of a quick run
    sub = df[keep].assign(_p=col[keep], _q=s2_main[keep])
    s1 = run_dir / f'{lang}_subtask1.jsonl'
    write_subtask1(sub, '_p', s1)
    s2 = run_dir / f'{lang}_subtask2.jsonl'
    write_subtask2(sub, '_q', s2)
    graded = write_graded(apd_scores(df[keep], V_pred),
                          run_dir / f'{lang}_graded_change.jsonl')
    pd.DataFrame(score_change(df[keep], labels)).to_csv(run_dir / 'predicted_jsd.csv', index=False)

    y, w_all = df['target_sense'].values[keep], df['word'].values[keep]
    s2_alt = {k: v[keep] for k, v in s2_alt.items()}
    selection = {'system': name, 'subtask2_mode': mode2,
                 'subtask2_all_modes_macro_f1 (test words; ablation only, not used for choices)':
                     {k: official_macro_f1(y, p, w_all) for k, p in s2_alt.items()},
                 'clustering_method': cfg['clustering']['method'],
                 'clustering_selection': (cfg['clustering']['threshold_selection']
                                          if cfg['clustering']['method'] == 'agglomerative'
                                          else 'kmeans_silhouette'),
                 'per_test_word': chosen, **(extra or {})}
    (run_dir / 'selection.json').write_text(json.dumps(selection, indent=2, ensure_ascii=False,
                                                       default=str), encoding='utf-8')
    if histories is not None:
        (run_dir / 'training_history.json').write_text(json.dumps(histories, indent=2, default=str),
                                                       encoding='utf-8')
    return evaluate_predictions(s1, s2, graded, name=name, out_dir=run_dir,
                                config=cfg.get('_config_file'), subgroups=True)


def run(cfg, df, folds, gold1, name=None):
    from src.classical_baseline import FastScorer
    from src.usage_encoder import UsageEncoder  # noqa: F401  (fails early if torch is missing)
    t_start = time.time()
    if not df.index.equals(pd.RangeIndex(len(df))):
        raise ValueError('df must have a 0..n-1 index: vector rows are matched to df rows by position')
    tr = cfg['training']
    name = name or run_name(cfg)
    run_dir = make_run_dir(cfg, f'proposed_{name}')
    save_run_config(cfg, run_dir, extra={'system': f'proposed_{name}'})
    (run_dir / 'vectors').mkdir()
    log(f'Run folder: {run_dir}')
    scorer = FastScorer(df, gold1)
    histories, vectors, timing = {}, {}, {}

    if not tr['enabled']:
        enc = build_encoder(cfg, frozen=True)
        shared = extract_embeddings(enc, df, cfg)
    else:
        t0 = time.time()
        if tr['synthetic']['enabled']:
            enc = train_stage_a(cfg, histories)
        else:
            enc = build_encoder(cfg)
        timing['stage_a_seconds'] = round(time.time() - t0, 1)
        shared = None if tr['swedish_stage'] else extract_embeddings(enc, df, cfg)
        base_state = enc.state_dict_cpu() if tr['swedish_stage'] else None

    for fold in folds:
        if shared is not None:
            V, D = shared
        else:
            t0 = time.time()
            enc.load_state_dict(base_state)
            histories[f"fold{fold['fold']}"] = train_stage_b(cfg, enc, df, fold, cfg['seed'] + fold['fold'])
            V, D = extract_embeddings(enc, df, cfg)
            timing[f"fold{fold['fold']}_seconds"] = round(time.time() - t0, 1)
        vectors[fold['fold']] = (V, D)
        np.savez_compressed(run_dir / 'vectors' / f"fold{fold['fold']}.npz", V=V,
                            D_words=np.array(sorted(D)), D=np.vstack([D[w] for w in sorted(D)]))
    timing['total_seconds'] = round(time.time() - t_start, 1)
    log(f"Training + encoding done in {timing['total_seconds']}s")
    return run_dir, predict(cfg, df, folds, vectors, scorer, run_dir, histories,
                            {'timing': timing}, name=f'proposed_{name}')


def recluster(cfg, df, folds, gold1, src_run):
    """New clustering / Subtask 2 settings on the vectors of an earlier run."""
    from src.classical_baseline import FastScorer
    src_run = Path(src_run)
    vectors = {}
    for fold in folds:
        z = np.load(src_run / 'vectors' / f"fold{fold['fold']}.npz")
        vectors[fold['fold']] = (z['V'], dict(zip([str(w) for w in z['D_words']], z['D'])))
    selection = ('kmeans' if cfg['clustering']['method'] == 'kmeans'
                 else cfg['clustering']['threshold_selection'])
    name = f"{src_run.name.split('_proposed_')[-1].rsplit('_seed', 1)[0]}_{selection}_{cfg['subtask2']['mode']}"
    run_dir = make_run_dir(cfg, f'proposed_{name}')
    save_run_config(cfg, run_dir, extra={'system': f'proposed_{name}', 'vectors_from': str(src_run)})
    return run_dir, predict(cfg, df, folds, vectors, FastScorer(df, gold1), run_dir,
                            extra={'vectors_from': str(src_run)}, name=f'proposed_{name}')


def main(argv=None):
    ap = argparse.ArgumentParser(description='Proposed system: fine-tuned usage encoder + sense clustering.')
    ap.add_argument('command', choices=['run', 'recluster'])
    ap.add_argument('--run', help='(recluster) run folder whose vectors are reused')
    ap.add_argument('--name', help='name of the run folder (default: from the settings)')
    add_cli_arguments(ap)
    a = ap.parse_args(argv)
    cfg, df, folds, gold1 = setup(a.config, a.overrides)
    if cfg['split']['folds'] == 'all':
        check_all_words_predicted(df, folds)
    if a.command == 'run':
        run(cfg, df, folds, gold1, a.name)
    else:
        if not a.run:
            ap.error('recluster needs --run <run folder>')
        recluster(cfg, df, folds, gold1, a.run)


if __name__ == '__main__':
    main()
