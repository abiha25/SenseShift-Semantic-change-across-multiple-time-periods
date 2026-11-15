"""
Error analysis for SemEval-2027 Task 3 (Section 9 of the report).

Works on ANY system's prediction files (official jsonl format), so Baseline A
and Baseline B are analysed with exactly the same code.

    python -m src.error_analysis --data data --system A_combined \
        --s1 results/classical/SV_subtask1_combined.jsonl \
        --s2 results/classical/SV_subtask2_LR_rank_thr_0.5.jsonl \
        --out results/error_analysis

Outputs (CSV) in <out>/<system>_*.csv :
    s1_by_word          B3 P/R/F1 per word, number of predicted vs gold senses
    s1_by_rarity        per-usage B3 recall/precision by rarity of the gold sense
    s1_by_period        per-usage B3 F1 per period (all words pooled, and by era)
    s1b_pairs           gold vs predicted JSD and their rank error per word-period pair
    s1b_by_word         mean gold/pred JSD and within-word rank agreement
    s2_by_word          TP/FP/FN/TN, precision, recall, F1, positive rates
    s2_by_period        gold vs predicted positive rate and recall per period
    s2_examples         a fixed-seed sample of false positives and false negatives

Per-usage B3 uses the FIRST gold sense only (13.2% of usages have several
senses; this is a diagnostic, the official score in Section 8 uses the full
extended BCubed).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from data_loader import load_language
from src.scoring import score_1a, score_1b, score_2

ERA_OF = lambda p: 'historical' if int(p[:4]) < 1950 else 'modern'
RARE_SHARE = 0.10          # a gold sense with < 10% of a word's usages is "rare"


def _read(path):
    with open(path, encoding='utf-8') as fh:
        return [json.loads(l) for l in fh if l.strip()]


def attach_predictions(df, s1_file=None, s2_file=None):
    df = df.copy()
    if s1_file:
        p = {(r['word'], r['sentence_id']): r['label'] for r in _read(s1_file)}
        df['pred_cluster'] = [p[(w, s)][0] for w, s in zip(df['word'], df['sentence_id'])]
    if s2_file:
        p = {(r['word'], r['sentence_id']): r['label'] for r in _read(s2_file)}
        df['pred_s2'] = [int(p[(w, s)]) for w, s in zip(df['word'], df['sentence_id'])]
    df['era'] = df['period_label'].map(ERA_OF)
    df['gold1'] = df['senses'].apply(lambda s: int(s[0]))
    return df


# ----------------------------------------------------------------------------
# Subtask 1a
# ----------------------------------------------------------------------------
def per_usage_b3(g):
    """Per-usage B3 precision/recall inside one word (first gold sense)."""
    gold, pred = g['gold1'].values, g['pred_cluster'].values
    same_g = gold[:, None] == gold[None, :]
    same_p = pred[:, None] == pred[None, :]
    both = same_g & same_p
    prec = both.sum(1) / same_p.sum(1)          # includes the usage itself
    rec = both.sum(1) / same_g.sum(1)
    out = g[['word', 'sentence_id', 'period_label', 'era']].copy()
    out['b3_p'], out['b3_r'] = prec, rec
    out['b3_f'] = 2 * prec * rec / (prec + rec)
    share = g['gold1'].map(g['gold1'].value_counts(normalize=True))
    out['gold_share'] = share.values
    out['rarity'] = np.where(g['gold1'].values == g['gold1'].mode()[0], 'dominant',
                             np.where(share.values < RARE_SHARE, 'rare (<10%)', 'minor (>=10%)'))
    return out


def s1_tables(df, s1_file, gold_s1):
    res = score_1a(s1_file, gold_s1)
    by_word = pd.DataFrame({
        w: {'B3_P': v['precision'], 'B3_R': v['recall'], 'B3_F1': v['fcubed'],
            'n_usages': int((df.word == w).sum()),
            'gold_senses': df[df.word == w]['gold1'].nunique(),
            'pred_clusters': df[df.word == w]['pred_cluster'].nunique(),
            'dominant_gold_share': df[df.word == w]['gold1'].value_counts(normalize=True).iloc[0]}
        for w, v in res['per_word'].items()}).T
    pu = pd.concat([per_usage_b3(g) for _, g in df.groupby('word')])
    by_rarity = pu.groupby('rarity').agg(n=('b3_f', 'size'), b3_p=('b3_p', 'mean'),
                                         b3_r=('b3_r', 'mean'), b3_f=('b3_f', 'mean'))
    by_rarity['share_of_usages'] = by_rarity['n'] / by_rarity['n'].sum()
    by_period = pu.groupby('period_label').agg(n=('b3_f', 'size'), b3_p=('b3_p', 'mean'),
                                               b3_r=('b3_r', 'mean'), b3_f=('b3_f', 'mean'))
    return res, by_word, by_rarity, by_period, pu


# ----------------------------------------------------------------------------
# Subtask 1b
# ----------------------------------------------------------------------------
def s1b_tables(s1_file, gold_s1):
    r = score_1b(s1_file, gold_s1)
    rows = []
    for k in sorted(r['gold_jsd']):
        w, p1, p2 = k.rsplit('_', 2)
        rows.append({'word': w, 'period_from': p1, 'period_to': p2,
                     'cross_era': ERA_OF(p1) != ERA_OF(p2),
                     'gold_jsd': r['gold_jsd'][k], 'pred_jsd': r['pred_jsd'][k]})
    pairs = pd.DataFrame(rows)
    # ranks over all 60 pairs, as the official Spearman does
    pairs['gold_rank_pct'] = pairs['gold_jsd'].rank(pct=True)
    pairs['pred_rank_pct'] = pairs['pred_jsd'].rank(pct=True)
    pairs['rank_error'] = (pairs['pred_rank_pct'] - pairs['gold_rank_pct']).abs()

    def within(g):
        if g['pred_jsd'].nunique() < 2 or len(g) < 3:
            return np.nan
        return spearmanr(g['gold_jsd'], g['pred_jsd'])[0]

    by_word = pairs.groupby('word').agg(
        n_pairs=('gold_jsd', 'size'), gold_mean=('gold_jsd', 'mean'),
        pred_mean=('pred_jsd', 'mean'), mean_rank_error=('rank_error', 'mean'))
    by_word['within_word_rho'] = pairs.groupby('word').apply(within)
    return r, pairs, by_word


# ----------------------------------------------------------------------------
# Subtask 2
# ----------------------------------------------------------------------------
def s2_tables(df, s2_file, gold_s2, n_examples=4, seed=42):
    res = score_2(s2_file, gold_s2)
    y, p = df['target_sense'].values, df['pred_s2'].values
    rows = {}
    for w, g in df.groupby('word'):
        y_, p_ = g['target_sense'].values, g['pred_s2'].values
        tp, fp = int(((y_ == 1) & (p_ == 1)).sum()), int(((y_ == 0) & (p_ == 1)).sum())
        fn, tn = int(((y_ == 1) & (p_ == 0)).sum()), int(((y_ == 0) & (p_ == 0)).sum())
        rows[w] = {'gold_pos_rate': y_.mean(), 'pred_pos_rate': p_.mean(), 'TP': tp, 'FP': fp,
                   'FN': fn, 'TN': tn, 'precision': tp / max(tp + fp, 1),
                   'recall': tp / max(tp + fn, 1), 'F1': res['per_word'][w]}
    by_word = pd.DataFrame(rows).T
    by_period = df.groupby('period_label').apply(lambda g: pd.Series({
        'n': len(g), 'gold_pos_rate': g['target_sense'].mean(), 'pred_pos_rate': g['pred_s2'].mean(),
        'recall_of_positives': (g['pred_s2'][g['target_sense'] == 1]).mean()
        if (g['target_sense'] == 1).any() else np.nan}))
    rng = np.random.default_rng(seed)
    ex = []
    for kind, mask in [('false_positive', (y == 0) & (p == 1)), ('false_negative', (y == 1) & (p == 0))]:
        sub = df[mask]
        take = sub.sample(min(n_examples * 3, len(sub)), random_state=int(rng.integers(1e6)))
        for _, r in take.iterrows():
            a, b = int(r['start_fixed']), int(r['end_fixed'])
            ex.append({'type': kind, 'word': r['word'], 'period': r['period_label'],
                       'snippet': (r['text'][max(0, a - 70):a] + '[[' + r['text'][a:b] + ']]'
                                   + r['text'][b:b + 70]).replace('\n', ' ')})
    return res, by_word, by_period, pd.DataFrame(ex)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--system', required=True)
    ap.add_argument('--s1')
    ap.add_argument('--s2')
    ap.add_argument('--out', default='results/error_analysis')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_language(a.data, 'SV')
    df = attach_predictions(df, a.s1, a.s2)
    gold1, gold2 = f'{a.data}/SV/subtask1.jsonl', f'{a.data}/SV/subtask2.jsonl'
    if a.s1:
        res, bw, br, bp, pu = s1_tables(df, a.s1, gold1)
        bw.to_csv(out / f'{a.system}_s1_by_word.csv'); br.to_csv(out / f'{a.system}_s1_by_rarity.csv')
        bp.to_csv(out / f'{a.system}_s1_by_period.csv'); pu.to_csv(out / f'{a.system}_s1_per_usage.csv', index=False)
        r, pairs, bw1b = s1b_tables(a.s1, gold1)
        pairs.to_csv(out / f'{a.system}_s1b_pairs.csv', index=False); bw1b.to_csv(out / f'{a.system}_s1b_by_word.csv')
        print(a.system, 'B3 F1', round(res['macro']['fcubed'], 3), 'rho', r['rho'])
    if a.s2:
        res, bw, bp, ex = s2_tables(df, a.s2, gold2)
        bw.to_csv(out / f'{a.system}_s2_by_word.csv'); bp.to_csv(out / f'{a.system}_s2_by_period.csv')
        ex.to_csv(out / f'{a.system}_s2_examples.csv', index=False)
        print(a.system, 'S2 macro F1', round(res['macro'], 3) if not isinstance(res['macro'], dict) else res['macro'])


if __name__ == '__main__':
    main()
