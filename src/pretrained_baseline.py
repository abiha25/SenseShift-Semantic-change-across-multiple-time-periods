"""
Pretrained baseline (Baseline B) for SemEval-2027 Task 3.   Owner: Moteba.

NOTHING is fine-tuned. Every encoder is FROZEN; every Subtask 2 model is used
ZERO-SHOT (the only things fitted are a rank threshold / a 2-feature logistic
regression, leave-one-word-out, exactly like Baseline A).

Subtask 1 (1a + 1b)
    encoder : XL-LEXEME  (frozen, WiC-trained SBERT-style bi-encoder, Cassotti et al. 2023)
              XLM-R base layer 8, mean over the target sub-words (frozen, reference)
    input   : the sentence with the target span marked (repaired offsets,
              start_fixed/end_fixed); optional spelling normalisation (EDA F4)
    cluster : average-linkage agglomerative on cosine distance, cut at t,
              clusters < m merged into nearest large cluster, (t, m) chosen
              leave-one-word-out (same code as Baseline A, so results compare)
              + K-means with silhouette-chosen K (Giulianelli et al. 2020)

Subtask 2
    S2-XL  : cosine(usage vector, definition vector) from frozen XL-LEXEME.
             The definition is encoded as a pseudo-usage "<word>: <definition>"
             with the span on <word>.
    S2-LLM : zero-shot prompting of an instruction-tuned LLM (optional, GPU):
             P("Ja") vs P("Nej") for "does this usage express this sense?"
    Each score is converted to a within-word percentile rank (as in Baseline A,
    EDA F7), then a rank threshold is chosen on the nine training words
    (macro per-word F1) and applied to the held-out word.

Run (Colab, GPU recommended, ~10-20 min):
    !pip install -r requirements.txt sentence-transformers transformers torch
    !pip install git+https://github.com/pierluigic/xl-lexeme.git
    !python -m src.pretrained_baseline --data data --out results/pretrained
    # optional LLM prompting:
    !python -m src.pretrained_baseline --data data --out results/pretrained --llm Qwen/Qwen2.5-7B-Instruct

NOTE: written against the public XL-LEXEME / transformers APIs but not executed
by the author's environment (no model download there). If the first run fails,
send the traceback.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

from data_loader import load_language, leave_one_word_out, write_subtask1, write_subtask2
from src import classical_baseline as cb
from src.features import SPELLING_RULES
from src.scoring import score_1a, score_1b, score_2

SEED = 42
THR_GRID = np.round(np.arange(0.05, 1.0, 0.05), 2)


# ----------------------------------------------------------------------------
# Encoders (all frozen)
# ----------------------------------------------------------------------------
def _norm_chunk(s: str) -> str:
    for pat, rep in SPELLING_RULES:
        s = pat.sub(rep, s.lower())
    return s


def prepare_inputs(df, spelling_norm: bool):
    """(text, start, end) with the repaired target span; optional spelling
    normalisation applied to the left/right context and to the target
    separately, so the span offsets stay correct."""
    out = []
    for t, s, e in zip(df['text'], df['start_fixed'], df['end_fixed']):
        s, e = int(s), int(e)
        if not spelling_norm:
            out.append((t, s, e))
            continue
        L, T, R = _norm_chunk(t[:s]), _norm_chunk(t[s:e]), _norm_chunk(t[e:])
        out.append((L + T + R, len(L), len(L) + len(T)))
    return out


def encode_xllexeme(items, batch_size=32):
    from WordTransformer import WordTransformer, InputExample   # pip: xl-lexeme repo
    model = WordTransformer('pierluigi/xl-lexeme')
    ex = [InputExample(texts=t, positions=[s, e]) for t, s, e in items]
    return np.asarray(model.encode(ex, batch_size=batch_size))


def encode_xlmr_layer(items, layer=8, batch_size=32, name='xlm-roberta-base'):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name, output_hidden_states=True).eval()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(dev)
    vecs = []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        enc = tok([t for t, _, _ in chunk], return_offsets_mapping=True, return_tensors='pt',
                  padding=True, truncation=True, max_length=256)
        offs = enc.pop('offset_mapping')
        with torch.no_grad():
            h = model(**{k: v.to(dev) for k, v in enc.items()}).hidden_states[layer].cpu()
        for j, (_, s, e) in enumerate(chunk):
            m = (offs[j, :, 0] < e) & (offs[j, :, 1] > s) & (offs[j, :, 1] > offs[j, :, 0])
            vecs.append((h[j][m] if m.any() else h[j][enc['attention_mask'][j].bool()]).mean(0).numpy())
    return np.vstack(vecs)


def encode_definitions(df, enc_fn, spelling_norm):
    """One 'word: definition' pseudo-usage per word; span on the word."""
    items, words = [], []
    for w, g in df.groupby('word'):
        d = g['definition'].iloc[0]
        items.append((f'{w}: {d}', 0, len(w)))
        words.append(w)
    return dict(zip(words, enc_fn(items)))


# ----------------------------------------------------------------------------
# Subtask 1: clustering frozen vectors
# ----------------------------------------------------------------------------
def DenseWrap(X):
    """L2-normalise dense vectors and wrap them as a scipy CSR matrix, the type
    the Baseline A clustering helpers (cosine_distance_matrix etc.) expect."""
    import scipy.sparse as sp
    return sp.csr_matrix(X / np.linalg.norm(X, axis=1, keepdims=True))


def cluster_all_words(df, V, thresholds=cb.THRESHOLDS, min_sizes=cb.MIN_SIZES):
    """{(t, m): {word: labels}} on dense vectors V (rows aligned with df)."""
    grid = {}
    for w, g in df.groupby('word'):
        Xw = DenseWrap(V[g.index.values])
        for key, lab in cb.agglomerative_labels(Xw, thresholds, min_sizes).items():
            grid.setdefault(key, {})[w] = lab
    return grid


def run_subtask1(df, V, tag, gold1, out_dir):
    """LOWO selection with all three objectives + K-means; returns summary dict."""
    grid = cluster_all_words(df, V)
    scorer = cb.FastScorer(df, gold1)
    # build the same {config: labels} structure lowo_select expects
    cfgs = list(grid.keys())
    gs = pd.DataFrame([{'config': c, **{f'f1_{w}': scorer.f1(w, grid[c][w]) for w in scorer.words},
                        'f1_macro': np.mean([scorer.f1(w, grid[c][w]) for w in scorer.words]),
                        'rho_all': scorer.rho(grid[c])} for c in cfgs])
    one = {w: scorer.f1(w, np.zeros(len(g), int)) for w, g in df.groupby('word')}
    summary = {}
    for obj in ['1a', 'combined', '1b']:
        sel = cb.lowo_select(df, grid, gs, scorer, objective=obj, single=one)
        path = Path(out_dir) / f'SV_subtask1_{tag}_{obj}.jsonl'
        res = cb.score_labels(df, sel['labels'], gold1, keep_file=path)
        diag = cb.era_diagnostics(res)
        perm = cb.permutation_control(scorer, sel['labels'])
        summary[f'{tag}_{obj}'] = {'precision': res['macro']['precision'], 'recall': res['macro']['recall'],
                                   'f1': res['macro']['fcubed'], 'rho': res['rho'], **diag,
                                   'shuffled_mean': perm['shuffled_mean'], 'shuffled_sd': perm['shuffled_sd'],
                                   'p': perm['p'],
                                   'per_word_f1': res['per_word_f1'],
                                   'chosen': {w: str(c) for w, c in sel['chosen'].items()}}
    # K-means + silhouette (Giulianelli et al. 2020): needs no tuning
    km = {w: cb.kmeans_silhouette_labels(DenseWrap(V[g.index.values])) for w, g in df.groupby('word')}
    path = Path(out_dir) / f'SV_subtask1_{tag}_kmeans.jsonl'
    res = cb.score_labels(df, km, gold1, keep_file=path)
    perm = cb.permutation_control(scorer, km)
    summary[f'{tag}_kmeans'] = {'precision': res['macro']['precision'], 'recall': res['macro']['recall'],
                                'f1': res['macro']['fcubed'], 'rho': res['rho'], **cb.era_diagnostics(res),
                                'shuffled_mean': perm['shuffled_mean'], 'shuffled_sd': perm['shuffled_sd'],
                                'p': perm['p'], 'per_word_f1': res['per_word_f1']}
    return summary


# ----------------------------------------------------------------------------
# Subtask 2: score -> within-word rank -> LOWO threshold
# ----------------------------------------------------------------------------
def to_rank(df, score):
    return pd.Series(score, index=df.index).groupby(df['word']).rank(pct=True).values


def macro_f1(y, p, words):
    return float(np.mean([f1_score(y[words == w], p[words == w], zero_division=0) for w in np.unique(words)]))


def lowo_rank_threshold(df, rank):
    """Predict 1 if within-word percentile rank >= q ; q chosen on 9 words."""
    y, w = df['target_sense'].values, df['word'].values
    pred, chosen = np.zeros(len(df), int), {}
    for train_words, test_word in leave_one_word_out(df):
        tr, te = np.isin(w, train_words), w == test_word
        best = max(THR_GRID, key=lambda q: macro_f1(y[tr], (rank[tr] >= q).astype(int), w[tr]))
        pred[te] = (rank[te] >= best).astype(int)
        chosen[test_word] = float(best)
    return pred, chosen


def lowo_lr(df, F):
    """Logistic regression on the rank features + rank-threshold (class_weight balanced)."""
    y, w = df['target_sense'].values, df['word'].values
    pred, prob = np.zeros(len(df), int), np.zeros(len(df))
    for train_words, test_word in leave_one_word_out(df):
        tr, te = np.isin(w, train_words), w == test_word
        clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEED).fit(F[tr], y[tr])
        prob[te] = clf.predict_proba(F[te])[:, 1]
        pred[te] = (prob[te] >= 0.5).astype(int)
    return pred, prob


def run_subtask2(df, scores: dict, gold2, out_dir):
    """scores: {name: raw score array}. Writes predictions, returns summary."""
    y, w = df['target_sense'].values, df['word'].values
    summary = {}
    for name, s in scores.items():
        r = to_rank(df, s)
        pred, chosen = lowo_rank_threshold(df, r)
        path = Path(out_dir) / f'SV_subtask2_{name}_rankthr.jsonl'
        write_subtask2(df.assign(_p=pred), '_p', path)
        res = score_2(path, gold2)
        auc = {x: roc_auc_score(y[w == x], s[w == x]) for x in np.unique(w) if len(set(y[w == x])) > 1}
        summary[f'{name}_rankthr'] = {'macro_f1': res['macro'], 'per_word_f1': res['per_word'],
                                      'mean_auc': float(np.mean(list(auc.values()))), 'auc': auc, 'thr': chosen}
    if len(scores) >= 2:                                   # combination of all scores
        F = np.column_stack([to_rank(df, s) for s in scores.values()])
        pred, prob = lowo_lr(df, F)
        path = Path(out_dir) / 'SV_subtask2_combo_LR.jsonl'
        write_subtask2(df.assign(_p=pred), '_p', path)
        res = score_2(path, gold2)
        summary['combo_LR'] = {'macro_f1': res['macro'], 'per_word_f1': res['per_word']}
    return summary


# ----------------------------------------------------------------------------
# Optional: zero-shot LLM prompting for Subtask 2
# ----------------------------------------------------------------------------
PROMPT = ("Mening: {sent}\n\nOrdet \"{target}\" i meningen ovan. Definition av en betydelse: \"{definition}\"\n\n"
          "Används ordet \"{target}\" i just den betydelse som definitionen beskriver? Svara endast Ja eller Nej.\nSvar:")


def llm_scores(df, model_name, batch_size=8):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto').eval()
    yes, no = tok(' Ja', add_special_tokens=False).input_ids[-1], tok(' Nej', add_special_tokens=False).input_ids[-1]
    prompts = [PROMPT.format(sent=r.text, target=r.target_fixed, definition=r.definition)
               for r in df.itertuples()]
    out = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors='pt', padding=True).to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :].float()
        out.extend((logits[:, yes] - logits[:, no]).cpu().numpy())
    return np.asarray(out)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--out', default='results/pretrained')
    ap.add_argument('--llm', default=None, help='HF id of an instruction-tuned LLM (optional)')
    ap.add_argument('--skip-xlmr', action='store_true')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_language(a.data, 'SV').reset_index(drop=True)
    gold1, gold2 = f'{a.data}/SV/subtask1.jsonl', f'{a.data}/SV/subtask2.jsonl'
    summary, s2_scores = {}, {}

    # ---- XL-LEXEME (main pretrained system), raw and spelling-normalised
    for norm in [True, False]:
        tag = 'xllex' + ('' if norm else '_rawspelling')
        items = prepare_inputs(df, norm)
        V = encode_xllexeme(items)
        np.save(out / f'{tag}_vectors.npy', V)
        summary.update(run_subtask1(df, V, tag, gold1, out))
        if norm:
            D = encode_definitions(df, lambda it: encode_xllexeme(it), norm)
            Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
            s2_scores['xllex_cos'] = np.array([Vn[i] @ (D[w] / np.linalg.norm(D[w]))
                                               for i, w in enumerate(df['word'])])

    # ---- XLM-R layer 8 (frozen reference; EDA F8 predicted form/era grouping)
    if not a.skip_xlmr:
        items = prepare_inputs(df, True)
        Vx = encode_xlmr_layer(items, layer=8)
        np.save(out / 'xlmr8_vectors.npy', Vx)
        summary.update(run_subtask1(df, Vx, 'xlmr8', gold1, out))

    # ---- LLM prompting (optional)
    if a.llm:
        s2_scores['llm_yes_minus_no'] = llm_scores(df, a.llm)
        np.save(out / 'llm_scores.npy', s2_scores['llm_yes_minus_no'])

    summary['subtask2'] = run_subtask2(df, s2_scores, gold2, out)

    with open(out / 'summary.json', 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=float)
    print(json.dumps({k: {m: v[m] for m in ('f1', 'rho', 'p') if m in v}
                      for k, v in summary.items() if k != 'subtask2'}, indent=1, default=float))
    print({k: round(v['macro_f1'], 3) for k, v in summary['subtask2'].items()})


if __name__ == '__main__':
    main()
