"""
Tests for the end-to-end run script (src/run_pipeline.py). Run from the repo root:

    python -m pytest tests -q

They use the run folders that are already in results/runs/, so no model is
trained and nothing is downloaded.
"""

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import run_pipeline as rp  # noqa: E402

RUNS = ROOT / 'results' / 'runs'
needs_runs = pytest.mark.skipif(not any(RUNS.glob('*_baseline_majority_seed*')),
                                reason='no saved run folders in results/runs')


def test_run_folder_names_are_parsed():
    m = rp.RUN_NAME.match('20261115-111845_proposed_ft_syn_sv_seed42')
    assert m['system'] == 'proposed_ft_syn_sv' and m['seed'] == '42'
    assert rp.RUN_NAME.match('notes') is None


@needs_runs
def test_seeds_are_grouped_per_system():
    rows = {r['system']: r for r in rp.summary_rows(rp.collect_runs(RUNS))}
    rnd = rows['baseline_random']
    assert rnd['n_seeds'] == 5 and rnd['1a_bcubed_f1_sd'] is not None
    assert round(rows['baseline_majority']['1a_bcubed_f1_mean'], 3) == 0.756


@needs_runs
def test_verify_reproduces_a_saved_run(tmp_path):
    src = sorted(RUNS.glob('*_baseline_majority_seed*'))[-1]
    shutil.copytree(src, tmp_path / src.name)
    assert rp.verify_runs(tmp_path) == []


@needs_runs
def test_verify_catches_an_edited_score(tmp_path):
    src = sorted(RUNS.glob('*_baseline_majority_seed*'))[-1]
    dst = tmp_path / src.name
    shutil.copytree(src, dst)
    text = (dst / 'scores.json').read_text(encoding='utf-8')
    (dst / 'scores.json').write_text(text.replace('"1a_bcubed_f1": 0.75', '"1a_bcubed_f1": 0.95', 1),
                                     encoding='utf-8')
    assert rp.verify_runs(tmp_path) == [src.name]
