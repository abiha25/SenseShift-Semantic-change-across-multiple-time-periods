"""
Tests for the Stage A guard and the final configuration (E8).

    python -m pytest tests -q
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.proposed import stage_a_key  # noqa: E402

RUNS = ROOT / 'results' / 'runs'


def _have_main_runs():
    try:
        from src.final_config import main_runs
        main_runs(RUNS)
        return True
    except FileNotFoundError:
        return False


needs_runs = pytest.mark.skipif(not _have_main_runs(), reason='five main seed runs not in results/runs')


def test_guard_is_off_by_default_and_keeps_the_checkpoint_key():
    cfg = load_config(ROOT / 'configs' / 'default.yaml', ['seed=42'])
    assert cfg['training']['stage_a_guard'] is False
    assert stage_a_key(cfg) == 'ff87edb4ef'          # Assignment 2 seed-42 checkpoint
    on = load_config(ROOT / 'configs' / 'default.yaml', ['seed=42', 'training.stage_a_guard=true'])
    assert stage_a_key(on) != stage_a_key(cfg)       # a guarded Stage A is cached separately


def test_final_yaml_loads_and_differs_from_default_only_where_intended():
    final = load_config(ROOT / 'configs' / 'final.yaml')
    default = load_config(ROOT / 'configs' / 'default.yaml')
    assert final['training']['stage_a_guard'] is True
    for k in ('stage_b_epochs', 'cluster_threshold', 'min_cluster_size', 'subtask2_threshold'):
        assert k in final['final']
    for section in ('data', 'split', 'preprocessing', 'model', 'clustering', 'subtask2', 'evaluation'):
        assert final[section] == default[section]
    rest = {k: v for k, v in final['training'].items() if k != 'stage_a_guard'}
    assert rest == {k: v for k, v in default['training'].items() if k != 'stage_a_guard'}


@needs_runs
def test_final_values_are_the_medians_of_the_five_seed_runs():
    from src.final_config import main_runs, medians
    values, rows = medians(main_runs(RUNS))
    assert len(rows) == 50 and {r['seed'] for r in rows} == {13, 42, 87, 1234, 2027}
    assert values['cluster_threshold'] == round(float(np.median([r['t'] for r in rows])), 4)
    final = load_config(ROOT / 'configs' / 'final.yaml')['final']
    for k, v in values.items():
        assert final[k] == v, f'configs/final.yaml is out of date for {k}: run python -m src.final_config'
