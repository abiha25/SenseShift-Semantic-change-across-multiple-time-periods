"""
JSD + APD combination for graded change (Assignment 3, ablation E3 / RQ2).

Question: does the clustering step lose change information that the usage
vectors carry? A run of the proposed system already gives two graded change
scores for every pair of consecutive periods:

    JSD   from the predicted sense clusters (what the official 1b score uses)
    APD   average pairwise cosine distance of the vectors (no clustering)

This script adds the third variant of E3 and makes the comparison possible
with the shared evaluation script:

    both  for each pair: the mean of its JSD rank and its APD rank, where the
          ranks are percentile ranks over all consecutive period pairs of
          all words (ties get the average rank)

Usage (repo root; no GPU, no retraining - it only reads a run folder):

    python -m src.combine_change --run results/runs/<time>_proposed_ft_syn_sv_seed42
    python -m src.combine_change --run <run folder> --mode jsd     # JSD as a graded file

For every mode it writes a NEW run folder next to the source run,
    <time>_<system>_change<mode>_seed<seed>/
holding copies of the official prediction files (so 1a, 1b and 2 are
unchanged), the new SV_graded_change.jsonl, config.yaml and run_info.json
(with 'derived_from'), and the scores from src/evaluate.py. Its
'graded_spearman' is the E3 score of that mode; APD alone is the
'graded_spearman' of the source run itself. Paired tests:
    python -m src.evaluate compare <both>/scores.json <source>/scores.json --metric graded

JSD is recomputed from SV_subtask1.jsonl with the organisers' own functions
(evaluation_utils.calculate_JSD_distances), so 'jsd' mode reproduces the
official 1b Spearman exactly (checked in tests/test_combine_change.py).
All graded scores are LOCAL: the official 1b score always comes from the
submitted clusters.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation_utils import calculate_JSD_distances  # noqa: E402
from src.evaluate import evaluate_predictions, read_jsonl, write_graded  # noqa: E402

MODES = ('both', 'jsd')
RUN_NAME = re.compile(r'^\d{8}-\d{6}_(?P<system>.+)_seed(?P<seed>\d+)$')


def _key(word, p1, p2):
    return f'{word}_{p1}_{p2}'


def jsd_from_predictions(s1_file, usages_file):
    """{word_p1_p2: JSD} for every consecutive period pair, from the predicted
    cluster labels, with the official fractional counting and JSD code.
    Periods come from usages.jsonl (they are not gold labels)."""
    period = {(u['word'], u['sentence_id']): u['period_label'] for u in read_jsonl(usages_file)}
    counts = {}
    for r in read_jsonl(s1_file):
        c = counts.setdefault(r['word'], {}).setdefault(period[(r['word'], r['sentence_id'])], Counter())
        for lab in r['label']:
            c[lab] += 1.0 / len(r['label'])
    return {k: float(v) for k, v in calculate_JSD_distances(counts).items()}


def apd_from_graded(graded_file):
    return {_key(r['word'], r['period1'], r['period2']): float(r['score'])
            for r in read_jsonl(graded_file)}


def rank_mean(*scores):
    """
    Mean of percentile ranks. Each argument is {pair: score} over the SAME
    pairs; each is ranked over all pairs (average rank for ties, divided by
    the number of pairs), and the ranks are averaged per pair.
    """
    keys = sorted(scores[0])
    for s in scores[1:]:
        if sorted(s) != keys:
            missing = sorted(set(keys) ^ set(s))
            raise ValueError(f'The scores do not cover the same period pairs, e.g. {missing[:3]}')
    ranks = [rankdata([s[k] for k in keys], method='average') / len(keys) for s in scores]
    return dict(zip(keys, np.mean(ranks, axis=0).tolist()))


def combine(run_dir, mode='both', out_root=None, config=None, verbose=True):
    """Write the derived run folder for `mode` and score it. Returns (folder, result)."""
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}')
    run_dir = Path(run_dir)
    m = RUN_NAME.match(run_dir.name)
    if not m:
        raise ValueError(f'{run_dir.name} is not a run folder name (<time>_<system>_seed<seed>)')
    cfg_used = __import__('yaml').safe_load((run_dir / 'config.yaml').read_text(encoding='utf-8'))
    lang = cfg_used['data']['language']
    s1, s2 = run_dir / f'{lang}_subtask1.jsonl', run_dir / f'{lang}_subtask2.jsonl'
    graded = run_dir / f'{lang}_graded_change.jsonl'
    # Periods are read from the data of THIS checkout (the run folder stores the
    # absolute path of the machine it ran on, e.g. /content/... on Colab).
    from src.config import load_config
    cfg_now = load_config(config or REPO_ROOT / 'configs' / 'default.yaml')
    usages = Path(cfg_now['data']['data_dir']) / lang / 'usages.jsonl'
    if not usages.exists():
        raise FileNotFoundError(usages)

    jsd = jsd_from_predictions(s1, usages)
    if mode == 'jsd':
        scores = jsd
    else:
        if not graded.exists():
            raise FileNotFoundError(f'{graded} (the source run has no APD file)')
        scores = rank_mean(jsd, apd_from_graded(graded))

    system = f"{m['system']}_change{mode}"
    out_root = Path(out_root) if out_root else run_dir.parent
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    out = out_root / f"{stamp}_{system}_seed{m['seed']}"
    out.mkdir(parents=True, exist_ok=False)
    for f in (s1, s2, run_dir / 'config.yaml'):
        if f.exists():
            shutil.copy2(f, out / f.name)
    rows = [{'word': k.rsplit('_', 2)[0], 'period1': k.rsplit('_', 2)[1],
             'period2': k.rsplit('_', 2)[2], 'score': v} for k, v in sorted(scores.items())]
    new_graded = write_graded(rows, out / f'{lang}_graded_change.jsonl')
    info = json.loads((run_dir / 'run_info.json').read_text(encoding='utf-8')) \
        if (run_dir / 'run_info.json').exists() else {}
    info.update({'derived_from': run_dir.name, 'derived_at': datetime.now().isoformat(timespec='seconds'),
                 'derivation': ('graded change = mean of the percentile ranks of JSD (from the '
                                'predicted clusters) and APD' if mode == 'both' else
                                'graded change = JSD from the predicted clusters'),
                 'command': ' '.join(sys.argv), 'system': system})
    (out / 'run_info.json').write_text(json.dumps(info, indent=2, default=str), encoding='utf-8')
    res = evaluate_predictions(str(out / s1.name), str(out / s2.name), str(new_graded), name=system,
                               out_dir=out, config=config, subgroups=True, verbose=verbose)
    return out, res


def main(argv=None):
    from src.config import add_cli_arguments
    ap = argparse.ArgumentParser(description='JSD + APD combination for graded change (E3).')
    ap.add_argument('--run', nargs='+', required=True, help='source run folder(s) of the proposed system')
    ap.add_argument('--mode', nargs='+', choices=MODES, default=['both'])
    ap.add_argument('--out', help='where to write the new run folders (default: next to the source)')
    add_cli_arguments(ap)
    a = ap.parse_args(argv)
    for run in a.run:
        for mode in a.mode:
            out, res = combine(run, mode, a.out, a.config)
            print(f"{out.name}: graded rho = {res['scores'].get('graded_spearman'):.3f}")


if __name__ == '__main__':
    main()
