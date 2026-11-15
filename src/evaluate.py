"""
Shared evaluation script (team SenseShift, Assignment 2).

ONE command scores any system (baseline, proposed model, ablation) the same way:

    python -m src.evaluate score --name majority \
        --s1 results/runs/<run>/SV_subtask1.jsonl \
        --s2 results/runs/<run>/SV_subtask2.jsonl \
        [--graded results/runs/<run>/SV_graded_change.jsonl] [--subgroups]

What it does, in order:
  1. FORMAT CHECK of every prediction file (fields, types, duplicates) and a
     WARNING for missing or extra words / usages. A file the official scorer
     would reject is reported here with the reason.
  2. OFFICIAL METRICS, computed by the organisers' own code through
     src/scoring.py (never re-implemented):
       Subtask 1a  extended BCubed F1, per word, macro-averaged over words
       Subtask 1b  Spearman rho between predicted and gold JSD of consecutive
                   periods (pairs with < 10 gold usages in a period dropped)
       Subtask 2   macro F1 over the two classes per word, averaged over words
  3. SECONDARY METRICS (Section 6.2 of the report):
       1a  BCubed precision/recall, predicted vs gold number of senses, gain
           over the one-cluster reference (0.756)
       1b  Pearson r, rho on within-era pairs only, rho against the cross-era
           flag (does the system only detect the 90-year gap?), optional
           shuffle control (same cluster sizes, shuffled usages)
       2   accuracy, balanced accuracy, positive-class F1 (= the Assignment 1
           metric), predicted vs gold positive rate
       graded change (e.g. APD): Spearman/Pearson against gold JSD on the same
           60 pairs. This is a LOCAL score only - the official 1b score is
           always computed from the submitted cluster labels.
  4. 95% CONFIDENCE INTERVALS by bootstrap over WORDS (Section 10.2).
  5. Optional SUBGROUPS (--subgroups, Section 6.3): era/period, word
     frequency, part of speech, amount of gold change, time coverage.

Other commands:
    python -m src.evaluate check     --s1 FILE --s2 FILE       # format check only
    python -m src.evaluate compare   A/scores.json B/scores.json --metric 1a
    python -m src.evaluate summarise run_seed13/scores.json run_seed42/scores.json ...

From Python:
    from src.evaluate import evaluate_predictions
    res = evaluate_predictions(s1='...', s2='...', name='majority', out_dir=run_dir)

Outputs (in --out, default: the folder of the first prediction file):
    scores.json      everything above, machine-readable (used by compare/summarise)
    per_word.csv     one row per word, all per-word numbers
    subgroups.csv    only with --subgroups

Team rule: nobody changes this file or src/scoring.py without telling the team.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import tempfile
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import compute_bcubed, compute_f1_score, compute_spearman  # noqa: E402
from evaluation_utils import (calculate_JSD_distances, load_subtask1_labels,  # noqa: E402
                              load_subtask2_labels, period_sort_key)
from src.scoring import (MIN_INSTANCES_1B, _prepare_gold, score_1a, score_1b,  # noqa: E402
                         score_2)

ONE_CLUSTER_1A = 0.756          # reference: every usage of a word in one cluster
ERA_BOUNDARY = 1950             # periods starting before this year are "historical"
DEFAULTS = {'bootstrap_resamples': 10000, 'bootstrap_seed': 42, 'ci_level': 0.95,
            'shuffle_control_draws': 200, 'min_instances_1b': MIN_INSTANCES_1B}


# ===========================================================================
# Small helpers
# ===========================================================================

def read_jsonl(path):
    with open(path, encoding='utf-8') as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')
    return path


def write_graded(rows, path):
    """
    Save graded change scores (e.g. APD) for evaluation.
    rows: iterable of dicts {'word', 'period1', 'period2', 'score'} where
    period1/period2 are CONSECUTIVE period labels of that word (earlier first).
    """
    return write_jsonl(path, [{'word': r['word'], 'period1': r['period1'],
                               'period2': r['period2'], 'score': float(r['score'])}
                              for r in rows])


def _clean(x):
    """NaN/inf -> None and numpy -> Python, so the result is valid JSON."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return None if (x != x or x in (float('inf'), float('-inf'))) else x
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    return x


def _safe_corr(fn, a, b):
    """Spearman/Pearson that returns NaN (not a warning) for constant input."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float('nan')
    return float(fn(a, b)[0])


def is_historical(period_label):
    return period_sort_key(period_label)[0] < ERA_BOUNDARY


def _settings(config=None, overrides=None):
    """Data folder, language and evaluation settings from the shared config."""
    from src.config import load_config
    cfg = load_config(config or REPO_ROOT / 'configs' / 'default.yaml', overrides)
    ev = {**DEFAULTS, **(cfg.get('evaluation') or {})}
    if int(ev['min_instances_1b']) != MIN_INSTANCES_1B:
        raise ValueError('evaluation.min_instances_1b must stay 10 (the official value).')
    return cfg['data']['data_dir'], cfg['data']['language'], ev


# ===========================================================================
# 1. Format check
# ===========================================================================

def check_predictions(pred_file, gold_file, subtask):
    """
    Check one prediction file against the gold file WITHOUT scoring it.

    subtask: '1' (cluster labels, list of ints) or '2' (0/1 label).
    Returns {'errors': [...], 'warnings': [...], 'words_scored': [...],
             'complete': bool}. errors = the official scorer would reject the
    file; warnings = it can be scored, but maybe not as intended.
    """
    errors, warns = [], []
    pred_file, gold_file = Path(pred_file), Path(gold_file)
    if not pred_file.exists():
        return {'errors': [f'{pred_file} does not exist'], 'warnings': [],
                'words_scored': [], 'complete': False}

    gold = {(r['word'], r['sentence_id']): r for r in read_jsonl(gold_file)}
    gold_words = sorted({w for w, _ in gold})
    pred, seen_words = {}, set()
    with open(pred_file, encoding='utf-8') as fh:
        for n, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f'line {n}: not valid JSON ({exc})')
                continue
            missing = {'word', 'sentence_id', 'label'} - row.keys()
            if missing:
                errors.append(f'line {n}: missing field(s) {sorted(missing)}')
                continue
            label = row['label']
            if subtask == '1':
                if isinstance(label, int) and not isinstance(label, bool):
                    errors.append(f'line {n}: label must be a LIST of ints, e.g. [{label}], not {label}')
                elif (not isinstance(label, list) or not label
                      or any(not isinstance(v, int) or isinstance(v, bool) for v in label)):
                    errors.append(f'line {n}: label must be a non-empty list of ints, got {label!r}')
                elif len(label) != len(set(label)):
                    errors.append(f'line {n}: repeated cluster id in {label}')
            else:
                if not isinstance(label, int) or isinstance(label, bool) or label not in (0, 1):
                    errors.append(f'line {n}: label must be the integer 0 or 1, got {label!r}')
            key = (row['word'], row['sentence_id'])
            if key in pred:
                errors.append(f'line {n}: duplicate usage {key}')
            pred[key] = row
            seen_words.add(row['word'])
            if subtask == '1' and key in gold and 'period_label' in row \
                    and row['period_label'] != gold[key]['period_label']:
                warns.append(f'line {n}: period_label {row["period_label"]} differs from gold '
                             f'{gold[key]["period_label"]} (the scorer uses the gold period)')
            if len(errors) >= 20:
                errors.append('... stopped after 20 errors')
                break

    missing_words = [w for w in gold_words if w not in seen_words]
    extra_words = sorted(seen_words - set(gold_words))
    if missing_words:
        warns.append(f'MISSING WORDS (no predictions): {missing_words}. Only the other words are '
                     'scored, so the score is NOT comparable to a full run.')
    if extra_words:
        warns.append(f'EXTRA WORDS (not in gold): {extra_words}. They are ignored.')

    words_scored = [w for w in gold_words if w in seen_words]
    for w in words_scored:
        g_ids = {s for ww, s in gold if ww == w}
        p_ids = {s for ww, s in pred if ww == w}
        if g_ids - p_ids:
            errors.append(f'{w}: {len(g_ids - p_ids)} of {len(g_ids)} gold usages have no prediction '
                          f'(e.g. {sorted(g_ids - p_ids)[0]}). Predict EVERY usage.')
        if p_ids - g_ids:
            errors.append(f'{w}: {len(p_ids - g_ids)} predicted sentence_ids are not in gold '
                          f'(e.g. {sorted(p_ids - g_ids)[0]}).')
    if subtask == '1' and not errors:
        multi = sum(len(r['label']) > 1 for r in pred.values())
        if multi:
            warns.append(f'{multi} usages have more than one predicted cluster (allowed; '
                         'counts are split fractionally).')
    return {'errors': errors, 'warnings': warns, 'words_scored': words_scored,
            'complete': not missing_words and not errors, 'n_rows': len(pred)}


def check_graded(graded_file, gold_jsd):
    """Graded change file: one score for each scored consecutive period pair."""
    errors, warns, scores = [], [], {}
    for n, row in enumerate(read_jsonl(graded_file), start=1):
        missing = {'word', 'period1', 'period2', 'score'} - row.keys()
        if missing:
            errors.append(f'line {n}: missing field(s) {sorted(missing)}')
            continue
        try:
            value = float(row['score'])
        except (TypeError, ValueError):
            errors.append(f'line {n}: score must be a number, got {row["score"]!r}')
            continue
        if value != value:
            errors.append(f'line {n}: score is NaN')
        key = f"{row['word']}_{row['period1']}_{row['period2']}"
        if key in scores:
            errors.append(f'line {n}: duplicate pair {key}')
        scores[key] = value
    missing = sorted(set(gold_jsd) - set(scores))
    extra = sorted(set(scores) - set(gold_jsd))
    if missing:
        errors.append(f'{len(missing)} of {len(gold_jsd)} scored period pairs have no score '
                      f'(e.g. {missing[0]}). Pairs are word_<earlier>_<later> consecutive periods.')
    if extra:
        warns.append(f'{len(extra)} pairs are not scored by the task (fewer than 10 gold usages '
                     f'or not consecutive) and are ignored, e.g. {extra[0]}.')
    return {'errors': errors, 'warnings': warns, 'scores': scores}


# ===========================================================================
# 2-3. Official and secondary metrics
# ===========================================================================

def _gold_tables(data_dir, language):
    """Gold labels, gold JSD (filtered like the official scorer) and word facts."""
    g1 = Path(data_dir) / language / 'subtask1.jsonl'
    _, gold_counts, _, gold_labels = load_subtask1_labels(g1, g1)
    from evaluation import filter_jsd_by_min_instances
    gold_jsd = filter_jsd_by_min_instances(calculate_JSD_distances(gold_counts), gold_counts,
                                           min_instances=MIN_INSTANCES_1B)
    usages = read_jsonl(Path(data_dir) / language / 'usages.jsonl')
    pos = {u['word']: u.get('pos', 'NA') for u in usages}
    n_usages = Counter(u['word'] for u in usages)
    period = {f"{r['word']};{r['sentence_id']}": r['period_label'] for r in read_jsonl(g1)}

    # Amount of gold change per word: mean gold JSD over the word's SCORED
    # consecutive period pairs. Defined for every word (5 words have only
    # modern or only historical periods, so an era-vs-era measure would not be).
    per_word_jsd = defaultdict(list)
    for key, value in gold_jsd.items():
        per_word_jsd[key.rsplit('_', 2)[0]].append(float(value))
    change = {w: float(np.mean(v)) for w, v in per_word_jsd.items()}
    coverage = {}
    for word, per_period in gold_counts.items():
        eras = {is_historical(p) for p in per_period}
        coverage[word] = ('both eras' if len(eras) == 2 else
                          'historical only' if True in eras else 'modern only')
    return {'g1': g1, 'g2': Path(data_dir) / language / 'subtask2.jsonl',
            'gold_labels': gold_labels, 'gold_counts': gold_counts, 'gold_jsd': gold_jsd,
            'pos': pos, 'n_usages': dict(n_usages), 'period': period, 'change': change,
            'coverage': coverage}


def _by_word(keys):
    out = defaultdict(list)
    for k in keys:
        out[k.split(';', 1)[0]].append(k)
    return out


def _pred_jsd_from_labels(pred_labels, period_of, words):
    """Same fractional counting as the official loader, then official JSD."""
    counts = {}
    for key, labels in pred_labels.items():
        word = key.split(';', 1)[0]
        if word not in words:
            continue
        c = counts.setdefault(word, {}).setdefault(period_of[key], Counter())
        for lab in labels:
            c[lab] += 1.0 / len(labels)
    return calculate_JSD_distances(counts)


def evaluate_subtask1(pred_file, gold, words, shuffle_draws=0, seed=42):
    with tempfile.TemporaryDirectory() as tmp:
        gp, pp = _prepare_gold(gold['g1'], pred_file, words, False, tmp)
        _, _, pred_labels, gold_labels = load_subtask1_labels(pp, gp)
    a = score_1a(pred_file, gold['g1'], words=words)          # official numbers
    b = score_1b(pred_file, gold['g1'], words=words)

    per_word = {}
    for w, keys in sorted(_by_word(gold_labels).items()):
        one = compute_bcubed({k: gold_labels[k] for k in keys}, {k: [0] for k in keys}, '')['fcubed']
        per_word[w] = {**a['per_word'][w],
                       'n_pred_clusters': len({l for k in keys for l in pred_labels[k]}),
                       'n_gold_senses': len({l for k in keys for l in gold_labels[k]}),
                       'one_cluster_f1': one,
                       'gain_over_one_cluster': a['per_word'][w]['fcubed'] - one}

    items = sorted(b['gold_jsd'])
    g = np.array([b['gold_jsd'][i] for i in items])
    p = np.array([b['pred_jsd'][i] for i in items])
    cross = np.array([is_historical(i.rsplit('_', 2)[1]) and not is_historical(i.rsplit('_', 2)[2])
                      for i in items])
    sub1b = {
        'rho': b['rho'], 'pvalue': b['pvalue'], 'n_pairs': len(items), 'note': b['note'],
        'pearson_r': _safe_corr(pearsonr, g, p),
        'rho_within_era': _safe_corr(spearmanr, g[~cross], p[~cross]),
        'n_within_era_pairs': int((~cross).sum()),
        'rho_pred_vs_cross_era_flag': _safe_corr(spearmanr, p, cross.astype(float)),
        'pairs': {i: {'gold': float(gv), 'pred': float(pv)} for i, gv, pv in zip(items, g, p)},
    }
    if shuffle_draws:
        sub1b['shuffle_control'] = shuffle_control(pred_labels, gold, set(per_word),
                                                   b['rho'], shuffle_draws, seed)
    return {'subtask1a': {'macro': a['macro'], 'per_word': per_word,
                          'one_cluster_reference': float(np.mean(
                              [v['one_cluster_f1'] for v in per_word.values()]))},
            'subtask1b': sub1b, '_labels': (pred_labels, gold_labels)}


def shuffle_control(pred_labels, gold, words, observed, n=200, seed=42):
    """
    Assignment 1 control: keep each word's predicted cluster sizes, shuffle
    WHICH usage is in which cluster, recompute rho. If the system's rho is
    not above most shuffles, its 1b score comes from cluster sizes, not content.
    """
    rng = np.random.default_rng(seed)
    by_word = _by_word(pred_labels)
    gold_items = {k: v for k, v in gold['gold_jsd'].items() if k.rsplit('_', 2)[0] in words}
    null = []
    for _ in range(n):
        shuffled = {}
        for w, keys in by_word.items():
            perm = rng.permutation(len(keys))
            for k, j in zip(keys, perm):
                shuffled[k] = pred_labels[keys[j]]
        pj = _pred_jsd_from_labels(shuffled, gold['period'], words)
        null.append(compute_spearman(gold_items, pj, '')['rho'])
    null = np.array(null, float)
    return {'draws': n, 'shuffled_mean': float(null.mean()), 'shuffled_sd': float(null.std()),
            'p': float((null >= observed).mean())}


def evaluate_subtask2(pred_file, gold, words):
    with tempfile.TemporaryDirectory() as tmp:
        gp, pp = _prepare_gold(gold['g2'], pred_file, words, False, tmp)
        pred, gl = load_subtask2_labels(pp, gp)
    off = score_2(pred_file, gold['g2'], words=words)          # official numbers
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    per_word, y_all, p_all = {}, [], []
    for w, keys in sorted(_by_word(gl).items()):
        y = np.array([gl[k] for k in keys])
        p = np.array([pred[k] for k in keys])
        y_all += y.tolist()
        p_all += p.tolist()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            per_word[w] = {
                'macro_f1': off['per_word'][w],
                'positive_f1': float(f1_score(y, p, zero_division=0)),
                'accuracy': float(accuracy_score(y, p)),
                'balanced_accuracy': float(balanced_accuracy_score(y, p)),
                'gold_positive_rate': float(y.mean()), 'pred_positive_rate': float(p.mean()),
            }
    mean = lambda key: float(np.mean([v[key] for v in per_word.values()]))  # noqa: E731
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pooled = {'accuracy': float(accuracy_score(y_all, p_all)),
                  'balanced_accuracy': float(balanced_accuracy_score(y_all, p_all)),
                  'macro_f1': float(f1_score(y_all, p_all, average='macro', zero_division=0))}
    return {'macro_f1': off['macro'],
            'word_mean': {'positive_f1': mean('positive_f1'), 'accuracy': mean('accuracy'),
                          'balanced_accuracy': mean('balanced_accuracy')},
            'pooled_over_usages': pooled, 'per_word': per_word,
            '_labels': (pred, gl)}


def evaluate_graded(graded_file, gold, words):
    gold_jsd = {k: v for k, v in gold['gold_jsd'].items() if k.rsplit('_', 2)[0] in words}
    chk = check_graded(graded_file, gold_jsd)
    if chk['errors']:
        return {'errors': chk['errors'], 'warnings': chk['warnings']}
    items = sorted(gold_jsd)
    g = np.array([gold_jsd[i] for i in items])
    p = np.array([chk['scores'][i] for i in items])
    cross = np.array([is_historical(i.rsplit('_', 2)[1]) and not is_historical(i.rsplit('_', 2)[2])
                      for i in items])
    return {'rho': _safe_corr(spearmanr, g, p), 'pearson_r': _safe_corr(pearsonr, g, p),
            'rho_within_era': _safe_corr(spearmanr, g[~cross], p[~cross]),
            'rho_pred_vs_cross_era_flag': _safe_corr(spearmanr, p, cross.astype(float)),
            'n_pairs': len(items), 'warnings': chk['warnings'],
            'note': 'Local score only: the official 1b score uses the submitted cluster labels.',
            'pairs': {i: {'gold': float(gv), 'pred': float(pv)} for i, gv, pv in zip(items, g, p)}}


# ===========================================================================
# 4. Bootstrap confidence intervals over words
# ===========================================================================

def _rho_from_pairs(pairs_by_word, words):
    g = [v['gold'] for w in words for v in pairs_by_word.get(w, [])]
    p = [v['pred'] for w in words for v in pairs_by_word.get(w, [])]
    return _safe_corr(spearmanr, g, p)


def _pairs_by_word(pairs):
    out = defaultdict(list)
    for key, v in sorted(pairs.items()):
        out[key.rsplit('_', 2)[0]].append(v)
    return out


def bootstrap_ci(values_by_word=None, pairs=None, n=10000, seed=42, level=0.95):
    """
    Percentile bootstrap over WORDS: draw the words with replacement, recompute
    the macro score (or rho from all pairs of the drawn words). Words, not
    usages, are resampled because the test set is a new set of words.
    """
    rng = np.random.default_rng(seed)
    if values_by_word is not None:
        words = sorted(values_by_word)
        vals = np.array([values_by_word[w] for w in words], float)
        idx = rng.integers(0, len(words), size=(n, len(words)))
        stats = vals[idx].mean(axis=1)
    else:
        by_word = _pairs_by_word(pairs)
        words = sorted(by_word)
        idx = rng.integers(0, len(words), size=(n, len(words)))
        stats = np.array([_rho_from_pairs(by_word, [words[i] for i in row]) for row in idx])
        stats = np.nan_to_num(stats, nan=0.0)   # official rule: no ranking = rho 0
    lo, hi = np.percentile(stats, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return {'low': float(lo), 'high': float(hi), 'level': level, 'resamples': int(n),
            'unit': 'words'}


# ===========================================================================
# 5. Subgroups
# ===========================================================================

def _word_groups(gold, words):
    """word -> group label, for each grouping in the work plan."""
    freq_median = float(np.median([gold['n_usages'][w] for w in words]))
    change_vals = [gold['change'][w] for w in words if w in gold['change']]
    change_median = float(np.median(change_vals)) if change_vals else float('nan')
    groups = {
        'frequency': {w: ('high' if gold['n_usages'][w] >= freq_median else 'low') +
                      f' (median {freq_median:.0f} usages)' for w in words},
        'part_of_speech': {w: gold['pos'].get(w, 'NA') for w in words},
        'gold_change': {w: (('high' if gold['change'][w] >= change_median else 'low') +
                            f' (mean gold JSD, median {change_median:.2f})') if w in gold['change']
                        else 'n/a' for w in words},
        'time_coverage': {w: gold['coverage'].get(w, 'n/a') for w in words},
    }
    return groups


def subgroup_analysis(res, gold):
    """Rows: group type, group, n words/usages/pairs, metrics, gap to overall."""
    rows = []
    s1, s2 = res.get('subtask1'), res.get('subtask2')
    words = sorted(res['words_scored'])

    def add(gtype, group, ws, keys1=None, keys2=None, pairs=None):
        row = {'group_type': gtype, 'group': group, 'n_words': len(ws)}
        if s1:
            pl, gl = s1['_labels']
            keys = keys1 if keys1 is not None else [k for k in gl if k.split(';', 1)[0] in ws]
            if keys:
                row['n_usages'] = len(keys)
                row['1a_bcubed_f1'] = compute_bcubed({k: gl[k] for k in keys},
                                                     {k: pl[k] for k in keys}, '')['fcubed']
                row['1a_one_cluster_f1'] = compute_bcubed({k: gl[k] for k in keys},
                                                          {k: [0] for k in keys}, '')['fcubed']
            pr = pairs if pairs is not None else {
                k: v for k, v in s1['subtask1b']['pairs'].items() if k.rsplit('_', 2)[0] in ws}
            row['n_pairs'] = len(pr)
            row['1b_rho'] = _safe_corr(spearmanr, [v['gold'] for v in pr.values()],
                                       [v['pred'] for v in pr.values()]) if len(pr) >= 5 else None
        if s2:
            pl, gl = s2['_labels']
            keys = keys2 if keys2 is not None else [k for k in gl if k.split(';', 1)[0] in ws]
            if keys:
                row['2_macro_f1'] = compute_f1_score({k: gl[k] for k in keys},
                                                     {k: pl[k] for k in keys}, '')
        rows.append(row)

    add('overall', 'all words', words)
    for gtype, mapping in _word_groups(gold, words).items():
        for group in sorted(set(mapping.values())):
            add(gtype, group, [w for w in words if mapping[w] == group])

    # Time: usages of each era / period (all words that have usages there).
    all_keys1 = list(s1['_labels'][1]) if s1 else []
    all_keys2 = list(s2['_labels'][1]) if s2 else []
    periods = sorted({gold['period'][k] for k in (all_keys1 or all_keys2) if k in gold['period']},
                     key=period_sort_key)
    eras = {'historical (1880-1915)': [p for p in periods if is_historical(p)],
            'modern (2006-2023)': [p for p in periods if not is_historical(p)]}
    for gtype, groups in (('era', eras), ('period', {p: [p] for p in periods})):
        for group, plist in groups.items():
            k1 = [k for k in all_keys1 if gold['period'].get(k) in plist]
            k2 = [k for k in all_keys2 if gold['period'].get(k) in plist]
            ws = sorted({k.split(';', 1)[0] for k in (k1 or k2)})
            pairs = None
            if s1 and gtype == 'era':
                hist = group.startswith('historical')
                pairs = {k: v for k, v in s1['subtask1b']['pairs'].items()
                         if is_historical(k.rsplit('_', 2)[1]) == hist
                         and is_historical(k.rsplit('_', 2)[2]) == hist}
            elif s1:
                pairs = {}
            add(gtype, group, ws, keys1=k1 if s1 else None, keys2=k2 if s2 else None, pairs=pairs)
    if s1:
        cross = {k: v for k, v in s1['subtask1b']['pairs'].items()
                 if is_historical(k.rsplit('_', 2)[1]) and not is_historical(k.rsplit('_', 2)[2])}
        # Too few pairs for a correlation: report mean gold vs predicted JSD.
        rows.append({'group_type': 'era', 'group': 'cross-era step (1912-1915 -> 2006-2007)',
                     'n_words': len(cross), 'n_pairs': len(cross), '1b_rho': None,
                     'mean_gold_jsd': float(np.mean([v['gold'] for v in cross.values()]))
                     if cross else None,
                     'mean_pred_jsd': float(np.mean([v['pred'] for v in cross.values()]))
                     if cross else None})
    overall = rows[0]
    for r in rows:
        for m in ('1a_bcubed_f1', '2_macro_f1', '1b_rho'):
            if r.get(m) is not None and overall.get(m) is not None:
                r[f'{m}_minus_overall'] = r[m] - overall[m]
    return rows


# ===========================================================================
# Main entry point
# ===========================================================================

def evaluate_predictions(s1=None, s2=None, graded=None, name='system', out_dir=None,
                         config=None, overrides=None, subgroups=False, shuffle=True,
                         bootstrap=True, verbose=True):
    """Check, score and save. Returns the result dict (also saved as scores.json)."""
    data_dir, language, ev = _settings(config, overrides)
    gold = _gold_tables(data_dir, language)
    res = {'name': name, 'language': language,
           'official_scorer': 'ChangeIsKey/semeval2027-task3 commit 0e69238 (8 Nov 2026)',
           'files': {'s1': s1, 's2': s2, 'graded': graded}, 'checks': {}, 'scores': {}}
    words_sets = []

    for key, path, subtask, gfile in (('subtask1', s1, '1', gold['g1']),
                                      ('subtask2', s2, '2', gold['g2'])):
        if not path:
            continue
        chk = check_predictions(path, gfile, subtask)
        res['checks'][key] = chk
        if chk['errors']:
            continue
        words_sets.append(set(chk['words_scored']))
        if subtask == '1':
            res['subtask1'] = evaluate_subtask1(
                path, gold, chk['words_scored'],
                shuffle_draws=int(ev['shuffle_control_draws']) if shuffle else 0,
                seed=int(ev['bootstrap_seed']))
        else:
            res['subtask2'] = evaluate_subtask2(path, gold, chk['words_scored'])

    words = sorted(set.intersection(*words_sets)) if words_sets else sorted(gold['n_usages'])
    res['words_scored'] = words
    if graded:
        res['graded'] = evaluate_graded(graded, gold, set(words))

    # Headline table + bootstrap CIs
    n, sd, lv = int(ev['bootstrap_resamples']), int(ev['bootstrap_seed']), float(ev['ci_level'])
    sc = res['scores']
    if 'subtask1' in res:
        pw = res['subtask1']['subtask1a']['per_word']
        sc['1a_bcubed_f1'] = res['subtask1']['subtask1a']['macro']['fcubed']
        sc['1b_spearman'] = res['subtask1']['subtask1b']['rho']
        if bootstrap:
            sc['1a_bcubed_f1_ci'] = bootstrap_ci({w: v['fcubed'] for w, v in pw.items()},
                                                 n=n, seed=sd, level=lv)
            sc['1b_spearman_ci'] = bootstrap_ci(pairs=res['subtask1']['subtask1b']['pairs'],
                                                n=n, seed=sd, level=lv)
    if 'subtask2' in res:
        sc['2_macro_f1'] = res['subtask2']['macro_f1']
        if bootstrap:
            sc['2_macro_f1_ci'] = bootstrap_ci({w: v['macro_f1'] for w, v in
                                                res['subtask2']['per_word'].items()},
                                               n=n, seed=sd, level=lv)
    if graded and 'rho' in res['graded']:
        sc['graded_spearman'] = res['graded']['rho']
        if bootstrap:
            sc['graded_spearman_ci'] = bootstrap_ci(pairs=res['graded']['pairs'],
                                                    n=n, seed=sd, level=lv)

    if subgroups and ('subtask1' in res or 'subtask2' in res):
        res['subgroups'] = subgroup_analysis(res, gold)

    out_dir = Path(out_dir) if out_dir else Path(s1 or s2 or graded).resolve().parent
    _save(res, out_dir)
    if verbose:
        print_report(res, out_dir)
    return res


def _save(res, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in res.items() if k not in ('subtask1', 'subtask2')}
    for key in ('subtask1', 'subtask2'):
        if key in res:
            clean[key] = {k: v for k, v in res[key].items() if not k.startswith('_')}
    (out_dir / 'scores.json').write_text(
        json.dumps(_clean(clean), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

    rows = []
    for w in res['words_scored']:
        row = {'word': w}
        if 'subtask1' in res:
            pw = res['subtask1']['subtask1a']['per_word'].get(w, {})
            row.update({f'1a_{k}': v for k, v in pw.items()})
        if 'subtask2' in res:
            row.update({f'2_{k}': v for k, v in res['subtask2']['per_word'].get(w, {}).items()})
        rows.append(row)
    _write_csv(out_dir / 'per_word.csv', rows)
    if 'subgroups' in res:
        _write_csv(out_dir / 'subgroups.csv', res['subgroups'])


def _write_csv(path, rows):
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in r.items()})


def _fmt(x, ci=None):
    if x is None or (isinstance(x, float) and x != x):
        return 'n/a'
    s = f'{x:.3f}'
    return s + (f'  [{ci["low"]:.3f}, {ci["high"]:.3f}]' if ci else '')


def print_report(res, out_dir=None):
    line = '=' * 72
    print(line)
    print(f"Evaluation: {res['name']}   ({res['language']}, {len(res['words_scored'])} words)")
    print(line)
    for key, chk in res['checks'].items():
        status = 'OK' if not chk['errors'] and not chk['warnings'] else (
            'REJECTED' if chk['errors'] else 'OK with warnings')
        print(f'Format check {key}: {status}')
        for e in chk['errors']:
            print(f'   ERROR   {e}')
        for w in chk['warnings']:
            print(f'   WARNING {w}')
    if res.get('graded'):
        for e in res['graded'].get('errors', []):
            print(f'   ERROR   graded: {e}')
        for w in res['graded'].get('warnings', []):
            print(f'   WARNING graded: {w}')
    sc = res['scores']
    ci_txt = ' (95% CI over words)' if any(k.endswith('_ci') for k in sc) else ''
    print(f'\nOFFICIAL METRICS{ci_txt}')
    if '1a_bcubed_f1' in sc:
        s1 = res['subtask1']
        m = s1['subtask1a']['macro']
        print(f"  1a  BCubed F1        {_fmt(sc['1a_bcubed_f1'], sc.get('1a_bcubed_f1_ci'))}"
              f"   (P {m['precision']:.3f}, R {m['recall']:.3f}; one cluster = "
              f"{s1['subtask1a']['one_cluster_reference']:.3f})")
        b = s1['subtask1b']
        print(f"  1b  Spearman rho     {_fmt(sc['1b_spearman'], sc.get('1b_spearman_ci'))}"
              f"   ({b['n_pairs']} pairs)")
        if b.get('note'):
            print(f"      note: {b['note']}")
    if '2_macro_f1' in sc:
        print(f"  2   macro F1         {_fmt(sc['2_macro_f1'], sc.get('2_macro_f1_ci'))}")
    if 'graded_spearman' in sc:
        print(f"  graded change rho    {_fmt(sc['graded_spearman'], sc.get('graded_spearman_ci'))}"
              f"   (LOCAL only, not an official score)")

    print('\nSECONDARY METRICS')
    if 'subtask1' in res:
        b = res['subtask1']['subtask1b']
        print(f"  1b  Pearson r {_fmt(b['pearson_r'])} | rho within-era {_fmt(b['rho_within_era'])}"
              f" ({b['n_within_era_pairs']} pairs) | rho vs cross-era flag "
              f"{_fmt(b['rho_pred_vs_cross_era_flag'])}")
        if 'shuffle_control' in b:
            s = b['shuffle_control']
            print(f"      shuffle control: {s['shuffled_mean']:.3f} +- {s['shuffled_sd']:.3f} "
                  f"({s['draws']} draws), p = {s['p']:.3f}")
    if 'subtask2' in res:
        s2 = res['subtask2']
        wm, po = s2['word_mean'], s2['pooled_over_usages']
        print(f"  2   accuracy {wm['accuracy']:.3f} | balanced accuracy {wm['balanced_accuracy']:.3f}"
              f" | positive-class F1 (A1 metric) {wm['positive_f1']:.3f}   (means over words)")
        print(f"      pooled over usages: accuracy {po['accuracy']:.3f}, balanced accuracy "
              f"{po['balanced_accuracy']:.3f}, macro F1 {po['macro_f1']:.3f}")
    if res.get('graded') and 'rho' in res['graded']:
        g = res['graded']
        print(f"  graded: Pearson r {_fmt(g['pearson_r'])} | rho within-era "
              f"{_fmt(g['rho_within_era'])} | rho vs cross-era flag "
              f"{_fmt(g['rho_pred_vs_cross_era_flag'])}")

    print('\nPER WORD')
    hdr = f"  {'word':<11}"
    if 'subtask1' in res:
        hdr += f"{'1a F1':>7}{'1clust':>8}{'#pred':>7}{'#gold':>7}"
    if 'subtask2' in res:
        hdr += f"{'2 mF1':>8}{'gold+':>7}{'pred+':>7}"
    print(hdr)
    for w in res['words_scored']:
        row = f'  {w:<11}'
        if 'subtask1' in res:
            p = res['subtask1']['subtask1a']['per_word'][w]
            row += (f"{p['fcubed']:>7.3f}{p['one_cluster_f1']:>8.3f}"
                    f"{p['n_pred_clusters']:>7}{p['n_gold_senses']:>7}")
        if 'subtask2' in res:
            p = res['subtask2']['per_word'][w]
            row += f"{p['macro_f1']:>8.3f}{p['gold_positive_rate']:>7.2f}{p['pred_positive_rate']:>7.2f}"
        print(row)

    if res.get('subgroups'):
        print('\nSUBGROUPS (difference to the overall score in brackets)')
        print(f"  {'group':<52}{'1a F1':>15}{'1b rho':>16}{'2 mF1':>16}")
        for r in res['subgroups']:
            if r['group_type'] == 'period':
                continue
            cells = []
            for m in ('1a_bcubed_f1', '1b_rho', '2_macro_f1'):
                v = r.get(m)
                d = r.get(f'{m}_minus_overall')
                cells.append('' if v is None or v != v else
                             f'{v:.3f}' + (f' ({d:+.3f})' if d is not None and r['group_type'] != 'overall' else ''))
            label = f"{r['group_type']}: {r['group']}"[:51]
            if r.get('mean_gold_jsd') is not None:
                cells[1] = f"JSD gold {r['mean_gold_jsd']:.2f}/pred {r['mean_pred_jsd']:.2f}"
            print(f'  {label:<52}{cells[0]:>15}{cells[1]:>16}{cells[2]:>16}')
        print('  (per-period rows are in subgroups.csv)')
    if out_dir:
        print(f'\nSaved: {Path(out_dir) / "scores.json"}, per_word.csv'
              + (', subgroups.csv' if res.get('subgroups') else ''))
    print(line)


# ===========================================================================
# compare: is system A really better than system B?   (Section 10.2)
# ===========================================================================

def _per_word_metric(res, metric):
    if metric == '1a':
        return {w: v['fcubed'] for w, v in res['subtask1']['subtask1a']['per_word'].items()}
    if metric == '2':
        return {w: v['macro_f1'] for w, v in res['subtask2']['per_word'].items()}
    raise ValueError(metric)


def _pairs(res, metric):
    return res['subtask1']['subtask1b']['pairs'] if metric == '1b' else res['graded']['pairs']


def compare(res_a, res_b, metric='1a', metric_b=None, n=10000, seed=42, level=0.95):
    """
    Paired comparison over the SAME words (A minus B).
      1a, 2      : exact sign-flip permutation test on the per-word
                   differences (all 2^n_words sign patterns) + bootstrap CI
                   of the mean difference.
      1b, graded : exact per-word swap permutation test (for each pattern,
                   the chosen words take the other system's predictions, rho
                   difference recomputed) + paired bootstrap CI over words.
    Call A better only if the CI excludes 0 AND p < 0.05 (Section 10.2).
    """
    metric_b = metric_b or metric
    rng = np.random.default_rng(seed)
    if metric in ('1a', '2'):
        a, b = _per_word_metric(res_a, metric), _per_word_metric(res_b, metric_b)
        words = sorted(set(a) & set(b))
        d = np.array([a[w] - b[w] for w in words])
        observed = float(d.mean())
        signs = np.array(list(itertools.product([1, -1], repeat=len(words))))
        null = (signs * d).mean(axis=1)
        p = float((np.abs(null) >= abs(observed) - 1e-12).mean())
        idx = rng.integers(0, len(words), size=(n, len(words)))
        boot = d[idx].mean(axis=1)
        test = 'exact paired sign-flip permutation test'
    else:
        pa, pb = _pairs_by_word(_pairs(res_a, metric)), _pairs_by_word(_pairs(res_b, metric_b))
        words = sorted(set(pa) & set(pb))

        def rho_diff(swap, wsel):
            ga, gb = [], []
            for w, s in zip(wsel, swap):
                x, y = (pb[w], pa[w]) if s else (pa[w], pb[w])
                ga += x
                gb += y
            ra = _safe_corr(spearmanr, [v['gold'] for v in ga], [v['pred'] for v in ga])
            rb = _safe_corr(spearmanr, [v['gold'] for v in gb], [v['pred'] for v in gb])
            return np.nan_to_num(ra) - np.nan_to_num(rb)

        observed = float(rho_diff([0] * len(words), words))
        if metric == metric_b:
            null = np.array([rho_diff(s, words)
                             for s in itertools.product([0, 1], repeat=len(words))])
            p = float((np.abs(null) >= abs(observed) - 1e-12).mean())
            test = 'exact per-word swap permutation test'
        else:
            p, test = None, 'no permutation test across different score types (CI only)'
        idx = rng.integers(0, len(words), size=(n, len(words)))
        boot = np.array([rho_diff([0] * len(words), [words[i] for i in row]) for row in idx])
    lo, hi = np.percentile(boot, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    better = bool((lo > 0 or hi < 0) and p is not None and p < 0.05)
    return {'a': res_a['name'], 'b': res_b['name'], 'metric_a': metric, 'metric_b': metric_b,
            'n_words': len(words), 'difference_a_minus_b': observed,
            'ci': {'low': float(lo), 'high': float(hi), 'level': level},
            'p_value': p, 'test': test,
            'verdict': ('significant difference' if better else
                        'not significant: report as a difference, not an improvement')}


def summarise(results):
    """Mean +- sd of the headline scores over several runs (e.g. 5 seeds)."""
    keys = ['1a_bcubed_f1', '1b_spearman', '2_macro_f1', 'graded_spearman']
    out = {'runs': [r['name'] for r in results]}
    for k in keys:
        vals = [r['scores'][k] for r in results if r['scores'].get(k) is not None]
        if vals:
            out[k] = {'mean': float(np.mean(vals)),
                      'sd': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                      'n_runs': len(vals), 'values': vals}
    return out


# ===========================================================================
# Command line
# ===========================================================================

def main(argv=None):
    from src.config import add_cli_arguments
    ap = argparse.ArgumentParser(description='SenseShift shared evaluation script.')
    sub = ap.add_subparsers(dest='cmd', required=True)

    for name in ('score', 'check'):
        s = sub.add_parser(name)
        s.add_argument('--s1', help='Subtask 1 predictions (cluster labels), official format')
        s.add_argument('--s2', help='Subtask 2 predictions (0/1), official format')
        s.add_argument('--graded', help='graded change scores (JSONL: word, period1, period2, score)')
        add_cli_arguments(s)
        if name == 'score':
            s.add_argument('--name', default='system')
            s.add_argument('--out', help='output folder (default: folder of the prediction file)')
            s.add_argument('--subgroups', action='store_true')
            s.add_argument('--no-shuffle', action='store_true', help='skip the 1b shuffle control')
            s.add_argument('--no-bootstrap', action='store_true', help='skip confidence intervals')

    c = sub.add_parser('compare')
    c.add_argument('scores_a')
    c.add_argument('scores_b')
    c.add_argument('--metric', choices=['1a', '1b', '2', 'graded'], default='1a')
    c.add_argument('--metric-b', choices=['1a', '1b', '2', 'graded'])
    add_cli_arguments(c)

    m = sub.add_parser('summarise')
    m.add_argument('scores', nargs='+')
    m.add_argument('--out')

    a = ap.parse_args(argv)
    if a.cmd in ('score', 'check') and not (a.s1 or a.s2 or a.graded):
        ap.error('give at least one of --s1, --s2, --graded')

    if a.cmd == 'check':
        data_dir, language, _ = _settings(a.config, a.overrides)
        bad = False
        for label, path, subtask in (('subtask1', a.s1, '1'), ('subtask2', a.s2, '2')):
            if path:
                chk = check_predictions(path, Path(data_dir) / language / f'{label}.jsonl', subtask)
                print(f"{label}: {'REJECTED' if chk['errors'] else 'OK'} ({chk['n_rows'] if 'n_rows' in chk else 0} rows)")
                for e in chk['errors']:
                    print('   ERROR  ', e)
                for w in chk['warnings']:
                    print('   WARNING', w)
                bad |= bool(chk['errors'])
        if a.graded:
            gold = _gold_tables(data_dir, language)
            chk = check_graded(a.graded, gold['gold_jsd'])
            print(f"graded: {'REJECTED' if chk['errors'] else 'OK'}")
            for e in chk['errors']:
                print('   ERROR  ', e)
            for w in chk['warnings']:
                print('   WARNING', w)
            bad |= bool(chk['errors'])
        sys.exit(1 if bad else 0)

    if a.cmd == 'score':
        res = evaluate_predictions(a.s1, a.s2, a.graded, name=a.name, out_dir=a.out,
                                   config=a.config, overrides=a.overrides,
                                   subgroups=a.subgroups, shuffle=not a.no_shuffle,
                                   bootstrap=not a.no_bootstrap)
        rejected = any(c['errors'] for c in res['checks'].values()) or \
            bool(res.get('graded', {}).get('errors'))
        sys.exit(1 if rejected else 0)

    if a.cmd == 'compare':
        _, _, ev = _settings(a.config, a.overrides)
        ra = json.loads(Path(a.scores_a).read_text(encoding='utf-8'))
        rb = json.loads(Path(a.scores_b).read_text(encoding='utf-8'))
        r = compare(ra, rb, a.metric, a.metric_b, n=int(ev['bootstrap_resamples']),
                    seed=int(ev['bootstrap_seed']), level=float(ev['ci_level']))
        print(json.dumps(_clean(r), indent=2, ensure_ascii=False))

    if a.cmd == 'summarise':
        rs = [json.loads(Path(p).read_text(encoding='utf-8')) for p in a.scores]
        r = summarise(rs)
        print(json.dumps(_clean(r), indent=2, ensure_ascii=False))
        if a.out:
            Path(a.out).write_text(json.dumps(_clean(r), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
