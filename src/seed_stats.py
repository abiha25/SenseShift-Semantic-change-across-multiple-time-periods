"""
Comparisons over several seeds (Assignment 3, Person 2; report Section 7 and 8.1).

The decision rule of the Assignment 2 protocol (Section 10.2) says: with
several seeds, a system is called better only if the 95% interval of the
paired difference excludes zero AND the exact permutation test gives p < 0.05,
"applied to the scores averaged over seeds". This file applies that rule.

How two systems A (normally the main system) and B are compared
    Seeds used    the seeds both systems were run with. A system that was run
                  once and has no randomness (majority, classical, pretrained,
                  the frozen-encoder runs; see DETERMINISTIC) is compared with
                  every seed of A.
    1a and 2      the per-word score of each system is averaged over those
                  seeds, then src.evaluate.compare runs the exact sign-flip
                  test on the 10 per-word differences (+ bootstrap CI). This is
                  the "averaged over seeds" rule.
    1b and APD    rho is a single number over 60 period pairs and cannot be
                  averaged per word, so each seed is tested on its own (exact
                  per-word swap test) and the row reports the mean difference,
                  the per-seed differences and p-values, and how many seeds
                  meet the rule. The verdict is "significant" only if EVERY
                  seed meets it.

Nothing here re-implements a metric: every test is src.evaluate.compare.

    python -m src.seed_stats                       # main system = proposed_ft_syn_sv
    python -m src.seed_stats --main proposed_ft_syn_sv --out results/summary
    python -m src.seed_stats --subgroups-only      # only subgroups_seed_mean.csv (fast)

Writes results/summary/seed_comparisons.csv and subgroups_seed_mean.csv
(both also written by python -m src.run_pipeline report).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import add_cli_arguments, load_config  # noqa: E402

METRICS = ('1a', '1b', '2', 'graded')
SCORE_KEY = {'1a': '1a_bcubed_f1', '1b': '1b_spearman', '2': '2_macro_f1', 'graded': 'graded_spearman'}


def _per_word(res, metric):
    from src.evaluate import _per_word_metric
    return _per_word_metric(res, metric)


def averaged_result(results, name):
    """A result dict whose per-word 1a and Subtask-2 scores are the means over `results`
    (one scores.json per seed). Only the fields src.evaluate.compare reads for 1a/2."""
    out = {'name': name, 'subtask1': {'subtask1a': {'per_word': {}}}, 'subtask2': {'per_word': {}}}
    for metric, path in (('1a', ('subtask1', 'subtask1a')), ('2', ('subtask2',))):
        per_seed = [_per_word(r, metric) for r in results]
        words = sorted(set.intersection(*(set(p) for p in per_seed)))
        target = out[path[0]] if len(path) == 1 else out[path[0]][path[1]]
        field = 'fcubed' if metric == '1a' else 'macro_f1'
        for w in words:
            target['per_word'][w] = {field: float(np.mean([p[w] for p in per_seed]))}
    return out


# Systems without any random choice: their single run stands for every seed.
# (The A2 re-clustering ablations of seed 42 are NOT in this list: they reuse
# the seed-42 vectors, so they are paired with seed 42 only.)
DETERMINISTIC = ('baseline_majority', 'baseline_classical', 'baseline_pretrained',
                 'proposed_frozen')


def is_deterministic(system):
    return system.startswith(DETERMINISTIC)


def paired_seeds(seeds_a, seeds_b, deterministic_b=False):
    """(seed of A, seed of B) pairs on which A and B are compared: the common
    seeds, or every seed of A when B is a deterministic single run."""
    if deterministic_b and len(seeds_b) == 1:
        b = next(iter(seeds_b))
        return [(s, b) for s in sorted(seeds_a)]
    return [(s, s) for s in sorted(set(seeds_a) & set(seeds_b))]


def compare_over_seeds(runs, main, other, metric, ev):
    """One row: main vs other on `metric`, over their paired seeds."""
    from src.evaluate import compare
    pairs = paired_seeds(runs[main], runs[other], is_deterministic(other))
    if not pairs:
        return None
    key = SCORE_KEY[metric]
    a_res = [runs[main][sa][1] for sa, _ in pairs]
    b_res = [runs[other][sb][1] for _, sb in pairs]
    if any(r['scores'].get(key) is None for r in a_res + b_res):
        return None
    kw = dict(n=int(ev['bootstrap_resamples']), seed=int(ev['bootstrap_seed']), level=float(ev['ci_level']))
    row = {'main': main, 'other': other, 'metric': metric,
           'seeds_main': ' '.join(str(sa) for sa, _ in pairs),
           'seeds_other': ' '.join(str(sb) for _, sb in pairs),
           'n_seeds': len(pairs),
           'main_mean': float(np.mean([r['scores'][key] for r in a_res])),
           'other_mean': float(np.mean([r['scores'][key] for r in b_res]))}
    per = [compare(a, b, metric, **kw) for a, b in zip(a_res, b_res)]
    row['per_seed_difference'] = ' '.join(f"{p['difference_a_minus_b']:+.3f}" for p in per)
    row['per_seed_p'] = ' '.join('-' if p['p_value'] is None else f"{p['p_value']:.3f}" for p in per)
    passed = [p['verdict'] == 'significant difference' for p in per]
    row['n_seeds_significant'] = int(sum(passed))
    if metric in ('1a', '2'):
        r = compare(averaged_result(a_res, f'{main} (seed mean)'),
                    averaged_result(b_res, f'{other} (seed mean)'), metric, **kw)
        row.update({'method': 'per-word scores averaged over seeds; exact sign-flip test',
                    'difference': r['difference_a_minus_b'], 'ci_low': r['ci']['low'],
                    'ci_high': r['ci']['high'], 'p_value': r['p_value'], 'verdict': r['verdict']})
    else:
        diffs = [p['difference_a_minus_b'] for p in per]
        row.update({'method': 'each seed tested separately; exact per-word swap test',
                    'difference': float(np.mean(diffs)), 'ci_low': None, 'ci_high': None,
                    'p_value': None,
                    'verdict': ('significant difference on every seed' if all(passed) else
                                f'significant on {sum(passed)} of {len(passed)} seeds: '
                                'report as a difference, not an improvement')})
    return row


def seed_comparison_rows(runs, main, ev, others=None):
    if main not in runs:
        return []
    rows = []
    for other in sorted(others or runs):
        if other == main or other not in runs:
            continue
        for metric in METRICS:
            r = compare_over_seeds(runs, main, other, metric, ev)
            if r:
                rows.append(r)
    return rows


SUBGROUP_COLUMNS = ('n_words', 'n_usages', '1a_bcubed_f1', '1a_one_cluster_f1', 'n_pairs', '1b_rho', '2_macro_f1')


def subgroup_seed_means(runs, systems=None):
    """Mean, min and max over seeds of each run's subgroups.csv (written by
    src.evaluate.subgroup_analysis), one row per system and subgroup."""
    import pandas as pd
    rows = []
    for system in sorted(systems or runs):
        frames = []
        for seed, (run_dir, _) in sorted(runs.get(system, {}).items()):
            f = Path(run_dir) / 'subgroups.csv'
            if f.exists():
                frames.append(pd.read_csv(f).assign(seed=seed))
        if not frames:
            continue
        g = pd.concat(frames)
        for (gt, grp), d in g.groupby(['group_type', 'group'], sort=False):
            row = {'system': system, 'group_type': gt, 'group': grp,
                   'seeds': ' '.join(str(s) for s in d['seed']), 'n_seeds': len(d)}
            for c in SUBGROUP_COLUMNS:
                v = pd.to_numeric(d[c], errors='coerce')
                row[f'{c}_mean'] = None if v.isna().all() else float(v.mean())
                if c not in ('n_words', 'n_usages', 'n_pairs', '1a_one_cluster_f1'):
                    row[f'{c}_min'] = None if v.isna().all() else float(v.min())
                    row[f'{c}_max'] = None if v.isna().all() else float(v.max())
            rows.append(row)
    return rows


def main(argv=None):
    from src.run_pipeline import _write_csv, collect_runs
    ap = argparse.ArgumentParser(description='Comparisons of the main system over several seeds.')
    ap.add_argument('--main', default='proposed_ft_syn_sv')
    ap.add_argument('--out', help='summary folder (default: results/summary)')
    ap.add_argument('--subgroups-only', action='store_true',
                    help='write only subgroups_seed_mean.csv (no tests)')
    add_cli_arguments(ap)
    a = ap.parse_args(argv)
    cfg = load_config(a.config, overrides=a.overrides)
    runs_dir = Path(cfg['output']['results_dir'])
    out = Path(a.out) if a.out else runs_dir.parent / 'summary'
    out.mkdir(parents=True, exist_ok=True)
    runs = collect_runs(runs_dir)
    _write_csv(out / 'subgroups_seed_mean.csv', subgroup_seed_means(runs))
    print(f'Saved: {out / "subgroups_seed_mean.csv"}')
    if a.subgroups_only:
        return []
    rows = seed_comparison_rows(runs, a.main, cfg['evaluation'])
    _write_csv(out / 'seed_comparisons.csv', rows)
    for r in rows:
        ci = '' if r['ci_low'] is None else f" CI=[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] p={r['p_value']:.3f}"
        print(f"{r['other']:48s} {r['metric']:6s} seeds={r['n_seeds']} diff={r['difference']:+.3f}{ci}"
              f"  per seed {r['per_seed_difference']}  -> {r['verdict']}")
    print(f'\nSaved: {out / "seed_comparisons.csv"}')
    return rows


if __name__ == '__main__':
    main()
