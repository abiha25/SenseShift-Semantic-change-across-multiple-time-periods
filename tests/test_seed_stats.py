"""
Tests for the comparisons over several seeds (src/seed_stats.py). Run from the repo root:

    python -m pytest tests -q

They use the run folders in results/runs/ (no model, no download).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import seed_stats as ss  # noqa: E402
from src.run_pipeline import collect_runs  # noqa: E402

RUNS = ROOT / 'results' / 'runs'
EV = {'bootstrap_resamples': 200, 'bootstrap_seed': 42, 'ci_level': 0.95}   # small n: fast tests
needs_runs = pytest.mark.skipif(not any(RUNS.glob('*_proposed_ft_syn_sv_seed*')),
                                reason='no saved run folders in results/runs')


def test_paired_seeds():
    assert ss.paired_seeds({13: 0, 42: 0}, {42: 0, 87: 0}) == [(42, 42)]
    assert ss.paired_seeds({13: 0}, {42: 0, 87: 0}) == []
    # a deterministic single run is compared with every seed of the main system ...
    assert ss.paired_seeds({13: 0, 42: 0}, {42: 0}, deterministic_b=True) == [(13, 42), (42, 42)]
    # ... but a seed-42 re-clustering ablation only with seed 42
    assert ss.paired_seeds({13: 0, 42: 0}, {42: 0}, deterministic_b=False) == [(42, 42)]
    assert ss.is_deterministic('baseline_majority') and ss.is_deterministic('proposed_frozen_last')
    assert not ss.is_deterministic('baseline_random') and not ss.is_deterministic('proposed_abl_noera')
    assert not ss.is_deterministic('proposed_ft_syn_sv_fixed_absolute')


def test_averaged_result_is_the_mean_per_word():
    a = {'subtask1': {'subtask1a': {'per_word': {'x': {'fcubed': 0.2}, 'y': {'fcubed': 0.4}}}},
         'subtask2': {'per_word': {'x': {'macro_f1': 0.5}, 'y': {'macro_f1': 0.7}}}}
    b = {'subtask1': {'subtask1a': {'per_word': {'x': {'fcubed': 0.6}, 'y': {'fcubed': 0.8}}}},
         'subtask2': {'per_word': {'x': {'macro_f1': 0.1}, 'y': {'macro_f1': 0.3}}}}
    m = ss.averaged_result([a, b], 'mean')
    assert m['subtask1']['subtask1a']['per_word']['x']['fcubed'] == pytest.approx(0.4)
    assert m['subtask2']['per_word']['y']['macro_f1'] == pytest.approx(0.5)


@needs_runs
def test_same_system_gives_zero_difference():
    runs = collect_runs(RUNS)
    runs = {'A': runs['proposed_ft_syn_sv'], 'B': runs['proposed_ft_syn_sv']}
    for metric in ('1a', '2'):
        r = ss.compare_over_seeds(runs, 'A', 'B', metric, EV)
        assert r['difference'] == pytest.approx(0.0) and r['n_seeds'] == 5
        assert r['verdict'].startswith('not significant')


@needs_runs
def test_seed_mean_matches_summary():
    runs = collect_runs(RUNS)
    r = ss.compare_over_seeds(runs, 'proposed_ft_syn_sv', 'baseline_majority', '1a', EV)
    vals = [res['scores']['1a_bcubed_f1'] for _, res in runs['proposed_ft_syn_sv'].values()]
    assert r['main_mean'] == pytest.approx(np.mean(vals))
    assert r['other_mean'] == pytest.approx(0.7563, abs=1e-4)      # one cluster per word
    # macro average of the seed-averaged per-word scores = mean of the seed scores
    assert r['difference'] == pytest.approx(np.mean(vals) - r['other_mean'], abs=1e-9)


@needs_runs
def test_subgroup_seed_means_overall_matches_summary():
    runs = collect_runs(RUNS)
    rows = ss.subgroup_seed_means(runs, ['proposed_ft_syn_sv'])
    overall = [r for r in rows if r['group_type'] == 'overall'][0]
    vals = [res['scores']['1a_bcubed_f1'] for _, res in runs['proposed_ft_syn_sv'].values()]
    assert overall['n_seeds'] == 5
    assert overall['1a_bcubed_f1_mean'] == pytest.approx(np.mean(vals), abs=1e-4)   # csv has 4 decimals
