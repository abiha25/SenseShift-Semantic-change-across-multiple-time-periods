"""
End-to-end run script (team SenseShift, Assignment 2).

ONE entry point for the whole experiment:

    load data -> run a system -> save predictions -> run the evaluation
              -> save results with the config used -> summary over systems and seeds

It does not re-implement anything. Every system is run by its own module
(src/baselines.py, src/proposed.py), which loads the data with the shared
loader, uses the shared split, writes the official prediction files and
scores them with the shared evaluation script (src/evaluate.py). Each run
writes its own folder results/runs/<time>_<system>_seed<seed>/ with
config.yaml and run_info.json next to the predictions and scores.

Commands (repo root; on Colab prefix with !):

    python -m src.run_pipeline run --systems baselines proposed frozen --seeds 42 13
        run the chosen systems for the chosen seeds (default seed: config 'seed')
          baselines : majority, random (5 seeds), classical, pretrained frozen XLM-R
          proposed  : fine-tuned encoder (Stage A + Stage B), all folds
          frozen    : proposed pipeline with the frozen encoder
        then 'verify' and 'report' are run automatically.

    python -m src.run_pipeline verify
        re-scores the saved prediction files of every run folder with the
        shared evaluation script and checks that the numbers in its
        scores.json are reproduced (exit code 1 if not). Use it after copying
        run folders from Colab/Drive into the repository.

    python -m src.run_pipeline report [--main proposed_ft_syn_sv]
        collects all run folders into results/summary/:
          summary.csv / summary.md   one row per system: the score of each seed,
                                     mean +- sd over seeds, 95% CI over words
                                     (single seed), 1b shuffle control
          comparisons.csv            paired tests of the main system against
                                     every other system, per metric (exact
                                     permutation test + bootstrap CI over words,
                                     the decision rule of Section 10)
          seed_comparisons.csv       the same tests over several seeds
                                     (src/seed_stats.py, Assignment 3)
          subgroups_seed_mean.csv    each run's subgroups.csv, mean/min/max
                                     over seeds (src/seed_stats.py)
          runs.json                  which run folder fed which row

Settings: configs/default.yaml (evaluation.bootstrap_resamples, bootstrap_seed,
ci_level); any value can be overridden with --set key=value.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import add_cli_arguments, load_config  # noqa: E402

METRICS = [('1a', '1a_bcubed_f1', '1a BCubed F1'),
           ('1b', '1b_spearman', '1b Spearman rho'),
           ('2', '2_macro_f1', '2 macro F1'),
           ('graded', 'graded_spearman', 'APD rho (local)')]
RUN_NAME = re.compile(r'^\d{8}-\d{6}_(?P<system>.+)_seed(?P<seed>\d+)$')
# The 1b shuffle mean averages 200 Spearman values; different platforms can
# differ in the 7th decimal through floating-point summation order.
TOLERANCE = {'shuffle_mean': 1e-5}


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def run_systems(systems, seeds, config, overrides):
    """Run each system with each seed through its own module."""
    from src import baselines, proposed
    common = ['--config', config]
    for o in overrides:
        common += ['--set', o]
    for seed in seeds:
        s = common + ['--set', f'seed={seed}']
        if 'baselines' in systems:
            # random is always run with the 5 configured seeds inside src/baselines.py,
            # so the baselines are run once, with the first seed.
            if seed == seeds[0]:
                baselines.main(['--which', 'simple', 'classical', 'pretrained'] + s)
        if 'proposed' in systems:
            proposed.main(['run'] + s)
        if 'frozen' in systems:
            proposed.main(['run', '--set', 'training.enabled=false'] + s)


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def _headline(res):
    out = dict(res['scores'])
    sc = res.get('subtask1', {}).get('subtask1b', {}).get('shuffle_control') or {}
    out['shuffle_mean'], out['shuffle_p'] = sc.get('shuffled_mean'), sc.get('p')
    for key in ['1a_bcubed_f1', '1b_spearman', '2_macro_f1', 'graded_spearman']:
        ci = out.get(key + '_ci') or {}
        out[key + '_ci_low'], out[key + '_ci_high'] = ci.get('low'), ci.get('high')
    return {k: v for k, v in out.items() if not isinstance(v, dict)}


def verify_runs(runs_dir, config=None, overrides=None):
    """Re-score every run folder from its saved predictions; compare with scores.json."""
    from src.evaluate import evaluate_predictions
    failed = []
    for run in sorted(p for p in Path(runs_dir).iterdir() if (p / 'scores.json').exists()):
        saved = json.loads((run / 'scores.json').read_text(encoding='utf-8'))
        files = {'s1': run / 'SV_subtask1.jsonl', 's2': run / 'SV_subtask2.jsonl',
                 'graded': run / 'SV_graded_change.jsonl'}
        kw = {k: str(p) if p.exists() else None for k, p in files.items()}
        with tempfile.TemporaryDirectory() as tmp:
            res = evaluate_predictions(**kw, name=run.name, out_dir=tmp, config=config,
                                       overrides=overrides, verbose=False)
        old, new = _headline(saved), _headline(res)
        diffs = {}
        for k in set(old) | set(new):
            a, b = old.get(k), new.get(k)
            if (a is None) != (b is None) or (a is not None and abs(a - b) > TOLERANCE.get(k, 1e-9)):
                diffs[k] = (a, b)
        complete = all(c.get('complete', True) and not c.get('errors')
                       for c in res['checks'].values())
        ok = complete and not diffs
        if not ok:
            failed.append(run.name)
        print(f"{'OK  ' if ok else 'FAIL'} {run.name}" + (f'  {diffs}' if diffs else '')
              + ('' if complete else '  (prediction files incomplete or rejected)'))
    print(f'\n{len(failed)} run folder(s) failed.' if failed else
          '\nEvery run folder is reproduced exactly from its saved predictions.')
    return failed


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def collect_runs(runs_dir):
    """{system: {seed: (run_dir, scores.json dict)}}; the latest folder wins."""
    out = defaultdict(dict)
    for run in sorted(p for p in Path(runs_dir).iterdir() if (p / 'scores.json').exists()):
        m = RUN_NAME.match(run.name)
        if m:
            res = json.loads((run / 'scores.json').read_text(encoding='utf-8'))
            out[m['system']][int(m['seed'])] = (run, res)
    return dict(out)


def _fmt(x, nd=3):
    return '' if x is None else f'{x:.{nd}f}'


def summary_rows(runs):
    rows = []
    for system in sorted(runs):
        seeds = sorted(runs[system])
        row = {'system': system, 'seeds': ' '.join(map(str, seeds)), 'n_seeds': len(seeds)}
        for _, key, _ in METRICS:
            vals = [runs[system][s][1]['scores'].get(key) for s in seeds]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            row[f'{key}_mean'] = float(np.mean(vals))
            row[f'{key}_sd'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
            row[f'{key}_values'] = ' '.join(_fmt(v) for v in vals)
            if len(seeds) == 1:
                ci = runs[system][seeds[0]][1]['scores'].get(key + '_ci') or {}
                row[f'{key}_ci_low'], row[f'{key}_ci_high'] = ci.get('low'), ci.get('high')
        shuffles = [(_headline(runs[system][s][1]).get('shuffle_mean'),
                     _headline(runs[system][s][1]).get('shuffle_p')) for s in seeds]
        if shuffles and all(x[0] is not None for x in shuffles):
            row['1b_shuffle_mean'] = float(np.mean([x[0] for x in shuffles]))
            row['1b_shuffle_p'] = ' '.join(_fmt(x[1], 3) for x in shuffles)
        rows.append(row)
    return rows


def comparison_rows(runs, main_system, ev):
    """Paired tests (same seed where possible) of the main system against every other system."""
    from src.evaluate import compare
    if main_system not in runs:
        return []
    rows = []
    for other in sorted(runs):
        if other == main_system:
            continue
        common = sorted(set(runs[main_system]) & set(runs[other]))
        pairs = [(s, s) for s in common] or [(sorted(runs[main_system])[0], sorted(runs[other])[0])]
        for s_main, s_other in pairs:
            a, b = runs[main_system][s_main][1], runs[other][s_other][1]
            for metric, key, _ in METRICS:
                if a['scores'].get(key) is None or b['scores'].get(key) is None:
                    continue
                r = compare(a, b, metric, n=int(ev['bootstrap_resamples']),
                            seed=int(ev['bootstrap_seed']), level=float(ev['ci_level']))
                rows.append({'main': main_system, 'other': other, 'seed_main': s_main,
                             'seed_other': s_other, 'metric': metric,
                             'difference': r['difference_a_minus_b'],
                             'ci_low': r['ci']['low'], 'ci_high': r['ci']['high'],
                             'p_value': r['p_value'], 'test': r['test'], 'verdict': r['verdict']})
    return rows


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _markdown(rows):
    head = ['system', 'seeds'] + [label for _, _, label in METRICS] + ['1b shuffle']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for r in rows:
        cells = [r['system'], r['seeds']]
        for _, key, _ in METRICS:
            if f'{key}_mean' not in r:
                cells.append('–')
            elif r.get(f'{key}_sd') is not None:
                cells.append(f"{_fmt(r[f'{key}_mean'])} ± {_fmt(r[f'{key}_sd'])}")
            elif r.get(f'{key}_ci_low') is not None:
                cells.append(f"{_fmt(r[f'{key}_mean'])} [{_fmt(r[f'{key}_ci_low'])}, {_fmt(r[f'{key}_ci_high'])}]")
            else:
                cells.append(_fmt(r[f'{key}_mean']))
        cells.append(_fmt(r.get('1b_shuffle_mean')))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def report(cfg, main_system, out_dir=None):
    runs_dir = Path(cfg['output']['results_dir'])
    out_dir = Path(out_dir) if out_dir else runs_dir.parent / 'summary'
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = collect_runs(runs_dir)
    rows = summary_rows(runs)
    comps = comparison_rows(runs, main_system, cfg['evaluation'])
    from src.seed_stats import seed_comparison_rows, subgroup_seed_means   # Assignment 3: several seeds
    seed_comps = seed_comparison_rows(runs, main_system, cfg['evaluation'])
    _write_csv(out_dir / 'summary.csv', rows)
    _write_csv(out_dir / 'comparisons.csv', comps)
    _write_csv(out_dir / 'seed_comparisons.csv', seed_comps)
    _write_csv(out_dir / 'subgroups_seed_mean.csv', subgroup_seed_means(runs))
    md = _markdown(rows)
    (out_dir / 'summary.md').write_text(md, encoding='utf-8')
    (out_dir / 'runs.json').write_text(json.dumps(
        {sy: {str(s): str(v[0].name) for s, v in seeds.items()} for sy, seeds in runs.items()},
        indent=2), encoding='utf-8')
    print(md)
    for c in comps:
        print(f"{c['main']} vs {c['other']} (seed {c['seed_main']}/{c['seed_other']}) {c['metric']:6s} "
              f"diff={c['difference']:+.3f} CI=[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] "
              f"p={_fmt(c['p_value'])} -> {c['verdict']}")
    print(f'\nSaved: {out_dir}')
    return rows, comps


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description='SenseShift end-to-end run script.')
    ap.add_argument('command', choices=['run', 'verify', 'report'])
    ap.add_argument('--systems', nargs='+', choices=['baselines', 'proposed', 'frozen'],
                    default=['baselines', 'proposed'])
    ap.add_argument('--seeds', nargs='+', type=int, help='default: the config seed')
    ap.add_argument('--main', default='proposed_ft_syn_sv',
                    help='system the others are compared with in comparisons.csv')
    ap.add_argument('--out', help='summary folder (default: results/summary)')
    add_cli_arguments(ap)
    a = ap.parse_args(argv)
    cfg = load_config(a.config, overrides=a.overrides)

    if a.command == 'run':
        run_systems(a.systems, a.seeds or [cfg['seed']], a.config, a.overrides)
    if a.command in ('run', 'verify'):
        failed = verify_runs(cfg['output']['results_dir'], a.config, a.overrides)
        if a.command == 'verify':
            sys.exit(1 if failed else 0)
    report(cfg, a.main, a.out)


if __name__ == '__main__':
    main()
