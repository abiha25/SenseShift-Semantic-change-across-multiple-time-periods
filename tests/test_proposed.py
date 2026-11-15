"""Tests for the proposed system. No model download needed:
    python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.usage_encoder import PairSampler, mark_target, pair_auc  # noqa: E402


def test_mark_target_keeps_offsets():
    text = 'Polisen misstänker att de dumpade oljefat i grusgropen.'
    s = text.index('dumpade')
    new, s2, e2 = mark_target(text, s, s + 7)
    assert new[s2:e2] == 'dumpade'
    assert '<t> dumpade </t>' in new


def test_mark_target_crops_long_text_around_target():
    text = 'a ' * 2000 + 'målet' + ' b' * 2000
    s = text.index('målet')
    new, s2, e2 = mark_target(text, s, s + 5, max_chars=200)
    assert new[s2:e2] == 'målet'
    assert len(new) < 250


def _items():
    rows = []
    for era in (0, 1):
        for k in range(6):
            rows.append({'text': f'w1 s1 {era} {k}', 'start': 0, 'end': 2, 'word': 'w1',
                         'senses': (1,), 'era': era})
            rows.append({'text': f'w1 s2 {era} {k}', 'start': 0, 'end': 2, 'word': 'w1',
                         'senses': (2,), 'era': era})
    rows.append({'text': 'w1 multi', 'start': 0, 'end': 2, 'word': 'w1', 'senses': (1, 2), 'era': 0})
    return rows


def test_pair_sampler_labels_and_same_word():
    items = _items()
    pairs = PairSampler(items, soft_labels=True).sample(400, np.random.default_rng(0))
    assert len(pairs) == 400
    for i, j, y in pairs:
        assert i != j and items[i]['word'] == items[j]['word']
        a, b = set(items[i]['senses']), set(items[j]['senses'])
        assert y == pytest.approx(len(a & b) / len(a | b))
    pos = np.mean([y > 0 for _, _, y in pairs])
    assert 0.35 < pos < 0.65


def test_pair_sampler_is_era_aware():
    items = [it for it in _items() if len(it['senses']) == 1]
    rng = np.random.default_rng(1)
    aware = PairSampler(items, era_aware=True, soft_labels=False).sample(2000, rng)
    plain = PairSampler(items, era_aware=False, soft_labels=False).sample(2000, np.random.default_rng(1))

    def cross_pos(pairs):
        return np.mean([items[i]['era'] != items[j]['era'] for i, j, y in pairs if y == 1])

    assert cross_pos(aware) > cross_pos(plain) + 0.1


def test_pair_auc_perfect_and_degenerate():
    V = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=float)
    assert pair_auc(V, [(1,), (1,), (2,), (2,)]) == pytest.approx(1.0)
    assert np.isnan(pair_auc(V, [(1,)] * 4))


def test_cluster_and_change_scores():
    from src.proposed import cluster_senses, score_change
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=16), rng.normal(size=16)
    V = np.vstack([a + 0.05 * rng.normal(size=16) for _ in range(12)] +
                  [b + 0.05 * rng.normal(size=16) for _ in range(12)])
    lab = cluster_senses(V, 0.3, 5)
    assert lab.max() == 1 and len(set(lab[:12])) == 1 and len(set(lab[12:])) == 1
    df = pd.DataFrame({'word': 'w', 'period_label': ['1880-1883'] * 12 + ['2006-2007'] * 12})
    rows = score_change(df, {'w': lab})
    assert len(rows) == 1 and rows[0]['jsd'] == pytest.approx(1.0)
