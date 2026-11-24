"""
Tests for the Assignment 3 options (Person 1). No model download needed:

    python -m pytest tests -q

  * pooling options (model.pooling)             ablation E6
  * K-means clustering (clustering.method)      ablation E4
  * JSD + APD combination (src/combine_change)  ablation E3
  * the default settings still give the Assignment 2 Stage A checkpoint key
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RUNS = ROOT / 'results' / 'runs'
PROPOSED = sorted(RUNS.glob('*_proposed_ft_syn_sv_seed42'))
needs_run = pytest.mark.skipif(not PROPOSED, reason='no saved proposed run in results/runs')


# ---------------------------------------------------------------------------
# Pooling (E6)
# ---------------------------------------------------------------------------

def _toy_batch():
    """One sentence '<s> A dum pade B </s> <pad>': the target 'dumpade' is
    characters 2-9 and is split into the sub-words 'dum' (2-5) and 'pade' (5-9)."""
    torch = pytest.importorskip('torch')
    offs = torch.tensor([[[0, 0], [0, 1], [2, 5], [5, 9], [10, 11], [0, 0], [0, 0]]])
    attn = torch.tensor([[1, 1, 1, 1, 1, 1, 0]])
    return offs, torch.tensor([[2]]), torch.tensor([[9]]), attn


def test_pooling_masks():
    from src.usage_encoder import pooling_mask
    offs, s, e, attn = _toy_batch()
    assert pooling_mask(offs, s, e, attn, 'target_mean').tolist() == [[0, 0, 1, 1, 0, 0, 0]]
    assert pooling_mask(offs, s, e, attn, 'first_subword').tolist() == [[0, 0, 1, 0, 0, 0, 0]]
    # every real token, without <s>, </s> and padding
    assert pooling_mask(offs, s, e, attn, 'sentence_mean').tolist() == [[0, 1, 1, 1, 1, 0, 0]]


def test_pooling_falls_back_when_target_is_lost():
    torch = pytest.importorskip('torch')
    from src.usage_encoder import pooling_mask
    offs, _, _, attn = _toy_batch()
    m = pooling_mask(offs, torch.tensor([[50]]), torch.tensor([[60]]), attn, 'first_subword')
    assert m.tolist() == [attn[0].bool().tolist()]


def test_unknown_pooling_is_rejected():
    from src.usage_encoder import pooling_mask
    offs, s, e, attn = _toy_batch()
    with pytest.raises(ValueError):
        pooling_mask(offs, s, e, attn, 'cls')


def test_default_settings_keep_the_stage_a_checkpoint_key():
    from src.config import load_config
    from src.proposed import stage_a_key
    cfg = load_config(ROOT / 'configs' / 'default.yaml', ['seed=42'])
    # Key of the Assignment 2 seed-42 checkpoint: adding the pooling option must
    # not invalidate the cached Stage A checkpoints of the default settings.
    assert stage_a_key(cfg) == 'ff87edb4ef'
    other = load_config(ROOT / 'configs' / 'default.yaml', ['seed=42', 'model.pooling=first_subword'])
    assert stage_a_key(other) != 'ff87edb4ef'


# ---------------------------------------------------------------------------
# K-means (E4)
# ---------------------------------------------------------------------------

def _blobs(sizes=(20, 12, 8), dim=16, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(len(sizes), dim)) * 3
    V = np.vstack([c + 0.1 * rng.normal(size=(n, dim)) for c, n in zip(centres, sizes)])
    truth = np.repeat(np.arange(len(sizes)), sizes)
    return V, truth


def test_kmeans_finds_the_number_of_senses_and_orders_by_size():
    from src.proposed import kmeans_senses
    V, truth = _blobs()
    lab, info = kmeans_senses(V, (2, 10), seed=42)
    assert info['k'] == 3
    assert np.bincount(lab).tolist() == [20, 12, 8]          # label 0 = largest cluster
    assert all(len(set(lab[truth == t])) == 1 for t in range(3))
    lab2, _ = kmeans_senses(V, (2, 10), seed=42)
    assert (lab == lab2).all()                               # seeded, reproducible


def test_kmeans_never_returns_one_cluster_and_handles_tiny_words():
    from src.proposed import kmeans_senses
    V, _ = _blobs(sizes=(30,))
    lab, info = kmeans_senses(V, (2, 10), seed=1)
    assert info['k'] >= 2 and lab.max() >= 1
    lab, info = kmeans_senses(V[:2], (2, 10))
    assert info['k'] == 1 and lab.tolist() == [0, 0]


# ---------------------------------------------------------------------------
# JSD + APD combination (E3)
# ---------------------------------------------------------------------------

def test_rank_mean_ties_and_agreement():
    from src.combine_change import rank_mean
    a = {'w_1_2': 0.1, 'w_2_3': 0.5, 'w_3_4': 0.9}
    assert rank_mean(a, a) == pytest.approx({'w_1_2': 1 / 3, 'w_2_3': 2 / 3, 'w_3_4': 1.0})
    tied = rank_mean({'x': 0.0, 'y': 0.0, 'z': 1.0}, {'x': 0.2, 'y': 0.1, 'z': 0.3})
    assert tied['x'] > tied['y'] and tied['z'] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        rank_mean(a, {'w_1_2': 1.0})


@needs_run
def test_jsd_mode_reproduces_the_official_1b_score(tmp_path):
    from src.combine_change import combine
    src = PROPOSED[-1]
    out, res = combine(src, 'jsd', out_root=tmp_path, verbose=False)
    official = json.loads((src / 'scores.json').read_text(encoding='utf-8'))['scores']['1b_spearman']
    assert res['scores']['graded_spearman'] == pytest.approx(official, abs=1e-12)
    assert res['scores']['1a_bcubed_f1'] == pytest.approx(
        json.loads((src / 'scores.json').read_text(encoding='utf-8'))['scores']['1a_bcubed_f1'])
    assert json.loads((out / 'run_info.json').read_text(encoding='utf-8'))['derived_from'] == src.name


@needs_run
def test_both_mode_covers_every_pair_and_is_verified(tmp_path):
    from src.combine_change import combine
    from src import run_pipeline as rp
    out, res = combine(PROPOSED[-1], 'both', out_root=tmp_path, verbose=False)
    rows = [json.loads(l) for l in (out / 'SV_graded_change.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 65 and all(0 < r['score'] <= 1 for r in rows)
    assert res['graded']['n_pairs'] == 60
    assert rp.RUN_NAME.match(out.name)['system'] == 'proposed_ft_syn_sv_changeboth'
    assert rp.verify_runs(tmp_path) == []
