"""
Fine-tuned usage encoder (XL-LEXEME style) for SenseShift.

What this file contains
    UsageEncoder        XLM-R bi-encoder. The target word is marked with <t> ... </t>
                        and one vector per usage is the mean of the hidden states of
                        the target's sub-words.
    load_synthetic()    the organisers' synthetic English set (training only)
    swedish_items()     the Swedish usages of the TRAINING words of one fold
    PairSampler         same-sense / different-sense usage pairs within one word,
                        with era-aware sampling and soft labels (sense overlap)
    soft_contrastive_loss()
    train_pairs()       training loop with warm-up, AMP, gradient clipping and
                        early stopping on a validation score
    pair_auc()          validation score: how well cosine distance separates
                        same-sense from different-sense pairs of the validation word

Training objective (Section 1 of the report)
    For a pair of usages (u, v) of the same word with sense sets S_u, S_v:
        y = |S_u & S_v| / |S_u | S_v|          (1 = same sense, 0 = no shared sense;
                                                 in between for multi-sense usages)
        d = 1 - cos(f(u), f(v))
        L = y * d^2 + (1 - y) * max(0, m - d)^2  (contrastive loss, margin m)
    Same-sense usages are pulled together, different-sense usages are pushed at
    least m apart. With hard labels (soft_labels=false) y is 1 if the usages share
    a sense, else 0, which is the loss XL-LEXEME was trained with.

Era-aware sampling (EDA F4/F8: frozen vectors group usages by era and form)
    A positive partner is drawn from the OTHER era when possible and a negative
    partner from the SAME era when possible, so era (spelling, style) is never
    the easy cue for "same sense".
"""

from __future__ import annotations

import copy
import glob
import json
import math
import time
from pathlib import Path

import numpy as np

MARK_L, MARK_R = '<t>', '</t>'


# ===========================================================================
# Input preparation
# ===========================================================================

def mark_target(text, start, end, markers=True, max_chars=1000):
    """
    Returns (text, start, end) with the target wrapped in <t> ... </t> and the
    offsets moved so that text[start:end] is still the target. Very long texts
    are cropped around the target first, so it is never cut off by truncation.
    """
    start, end = int(start), int(end)
    if len(text) > max_chars:
        left = max(0, start - max_chars // 2)
        right = min(len(text), left + max_chars)
        left = max(0, right - max_chars)
        text, start, end = text[left:right], start - left, end - left
    if not markers:
        return text, start, end
    pre, tgt, post = text[:start], text[start:end], text[end:]
    new = f'{pre}{MARK_L} {tgt} {MARK_R}{post}'
    s = len(pre) + len(MARK_L) + 1
    return new, s, s + len(tgt)


# ===========================================================================
# Model
# ===========================================================================

class UsageEncoder:
    """
    Wraps a Hugging Face encoder. Not an nn.Module itself, so that saving and
    loading go through the standard save_pretrained()/from_pretrained().

        enc = UsageEncoder('xlm-roberta-base', layer=-1, markers=True)
        V = enc.encode([(text, start, end), ...])          # numpy (n, hidden)
    """

    def __init__(self, name_or_path, layer=-1, markers=True, max_length=256,
                 freeze_embeddings=True, dropout=None, device=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.name = str(name_or_path)
        self.layer = int(layer)
        self.markers = bool(markers)
        self.max_length = int(max_length)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        kw = {} if dropout is None else {'hidden_dropout_prob': float(dropout),
                                         'attention_probs_dropout_prob': float(dropout)}
        self.model = AutoModel.from_pretrained(self.name, **kw)
        if self.markers and MARK_L not in self.tokenizer.get_vocab():
            self._add_markers()
        if freeze_embeddings:
            for p in self.model.get_input_embeddings().parameters():
                p.requires_grad = False
        self.model.to(self.device)

    def _add_markers(self):
        """Add <t>, </t> as special tokens. Each new embedding starts as the mean
        of the embeddings of the pieces the old tokenizer split it into, so it is
        a sensible input even when the embedding matrix stays frozen."""
        import torch
        pieces = {m: self.tokenizer(m, add_special_tokens=False)['input_ids'] for m in (MARK_L, MARK_R)}
        self.tokenizer.add_special_tokens({'additional_special_tokens': [MARK_L, MARK_R]})
        self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)
        emb = self.model.get_input_embeddings().weight
        with torch.no_grad():
            for m, ids in pieces.items():
                emb[self.tokenizer.convert_tokens_to_ids(m)] = emb[ids].mean(0)

    # -----------------------------------------------------------------------
    def parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def forward(self, items):
        """items: list of (text, start, end) in ORIGINAL offsets. Returns a
        torch tensor (n, hidden) with gradients (call inside no_grad to encode)."""
        import torch
        marked = [mark_target(t, s, e, self.markers) for t, s, e in items]
        enc = self.tokenizer([t for t, _, _ in marked], return_offsets_mapping=True,
                             return_tensors='pt', padding=True, truncation=True,
                             max_length=self.max_length)
        offs = enc.pop('offset_mapping')
        enc = {k: v.to(self.device) for k, v in enc.items()}
        use_amp = self.device == 'cuda'
        with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
            if self.layer == -1:
                h = self.model(**enc).last_hidden_state
            else:
                h = self.model(**enc, output_hidden_states=True).hidden_states[self.layer]
        h = h.float()
        starts = torch.tensor([s for _, s, _ in marked]).unsqueeze(1)
        ends = torch.tensor([e for _, _, e in marked]).unsqueeze(1)
        mask = (offs[:, :, 0] < ends) & (offs[:, :, 1] > starts) & (offs[:, :, 1] > offs[:, :, 0])
        empty = ~mask.any(1)                          # target lost (should not happen)
        if empty.any():
            mask[empty] = enc['attention_mask'][empty].bool().cpu()
        mask = mask.to(h.device).unsqueeze(-1).float()
        return (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)

    def encode(self, items, batch_size=32):
        import torch
        self.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(items), batch_size):
                out.append(self.forward(items[i:i + batch_size]).cpu().numpy())
        return np.vstack(out).astype(np.float32) if out else np.zeros((0, self.model.config.hidden_size))

    # -----------------------------------------------------------------------
    def state_dict_cpu(self):
        return {k: v.detach().to('cpu', copy=True) for k, v in self.model.state_dict().items()}

    def load_state_dict(self, state):
        self.model.load_state_dict(state)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        (path / 'usage_encoder.json').write_text(json.dumps(
            {'layer': self.layer, 'markers': self.markers, 'max_length': self.max_length,
             'base': self.name}), encoding='utf-8')
        return path


# ===========================================================================
# Training data
# ===========================================================================

def _norm(s):
    return s.lower().replace('-', ' ').replace('_', ' ').strip()


def load_synthetic(root, max_per_sense=20, seed=42, holdout_fraction=0.03, max_words=None):
    """
    Read data/external/synthetic_en/generated_sentences/*/*.jsonl.

    Filters lines whose marked span is not the target word (first three
    letters differ from the lemma, e.g. 'asked' for 'question'), keeps at most
    `max_per_sense` sentences per sense (seeded sample), and drops words with a
    single sense (they give no negative pairs).

    Returns (train_items, holdout_items, stats); each item is a dict with
    text, start, end, word, senses (tuple), era.  Era = before/after 1900.
    """
    rng = np.random.default_rng(seed)
    files = sorted(glob.glob(str(Path(root) / 'generated_sentences' / '*' / '*.jsonl')))
    if not files:
        raise FileNotFoundError(
            f'No synthetic data under {root}/generated_sentences. See data/external/README.md.')
    by_word, n_lines, n_bad = {}, 0, 0
    for f in files:
        with open(f, encoding='utf-8') as fh:
            for line in fh:
                r = json.loads(line)
                n_lines += 1
                span = r['text'][r['start']:r['end']]
                if not span or _norm(span)[:3] != _norm(r['lemma'])[:3]:
                    n_bad += 1
                    continue
                by_word.setdefault(r['lemma'], {}).setdefault(r['sense_id'], []).append(r)
    words = sorted(w for w, s in by_word.items() if len(s) >= 2)
    rng.shuffle(words)
    if max_words:
        words = words[:max_words]
    n_hold = max(1, int(round(holdout_fraction * len(words))))
    hold, train = set(words[:n_hold]), words[n_hold:]
    out = {'train': [], 'holdout': []}
    for w in sorted(words):
        part = 'holdout' if w in hold else 'train'
        for sid in sorted(by_word[w]):
            rows = by_word[w][sid]
            if len(rows) > max_per_sense:
                rows = [rows[i] for i in sorted(rng.choice(len(rows), max_per_sense, replace=False))]
            for r in rows:
                out[part].append({'text': r['text'], 'start': int(r['start']), 'end': int(r['end']),
                                  'word': w, 'senses': (sid,), 'era': int(r['year'] >= 1900)})
    stats = {'lines': n_lines, 'span_filtered': n_bad, 'words_used': len(train),
             'words_holdout': len(hold), 'items_train': len(out['train']),
             'items_holdout': len(out['holdout'])}
    return out['train'], out['holdout'], stats


def swedish_items(df, words, exclude_conflicts=True):
    """Usages of `words` (the TRAINING words of one fold) as training items.
    Era = historical (before 1950) vs modern. Sentences whose duplicate copies
    received different gold senses are left out (label noise)."""
    sub = df[df['word'].isin(words)]
    if exclude_conflicts and 'dup_label_conflict' in sub:
        sub = sub[~sub['dup_label_conflict']]
    return [{'text': t, 'start': int(s), 'end': int(e), 'word': w, 'senses': tuple(sorted(ss)),
             'era': int(y >= 1950)}
            for t, s, e, w, ss, y in zip(sub['model_text'], sub['model_start'], sub['model_end'],
                                         sub['word'], sub['senses'], sub['year'])]


class PairSampler:
    """
    Draws usage pairs within one word.

        sampler = PairSampler(items, positive_fraction=0.5, era_aware=True, soft_labels=True)
        pairs = sampler.sample(n, rng)      # list of (i, j, label)

    Words are drawn uniformly (so large words do not dominate), the anchor
    uniformly within the word.
    """

    def __init__(self, items, positive_fraction=0.5, era_aware=True, p_cross_era_positive=0.7,
                 p_same_era_negative=0.7, soft_labels=True):
        self.items = items
        self.pos_frac = float(positive_fraction)
        self.era_aware = bool(era_aware)
        self.p_cross = float(p_cross_era_positive)
        self.p_same = float(p_same_era_negative)
        self.soft = bool(soft_labels)
        self.words = {}
        for i, it in enumerate(items):
            self.words.setdefault(it['word'], []).append(i)
        self.word_list = sorted(self.words)
        self._cache = {}

    def _word_arrays(self, w):
        if w not in self._cache:
            idx = np.array(self.words[w])
            senses = sorted({s for i in idx for s in self.items[i]['senses']})
            col = {s: k for k, s in enumerate(senses)}
            M = np.zeros((len(idx), len(senses)), dtype=np.float32)
            for r, i in enumerate(idx):
                for s in self.items[i]['senses']:
                    M[r, col[s]] = 1.0
            era = np.array([self.items[i]['era'] for i in idx])
            text = np.array([hash(self.items[i]['text']) for i in idx])
            self._cache[w] = (idx, M, era, text)
        return self._cache[w]

    def _pick(self, rng, cand, prefer):
        if self.era_aware and prefer.any() and rng.random() < (self.p_cross if self._want_pos else self.p_same):
            cand = cand[prefer]
        return int(rng.choice(cand))

    def sample(self, n, rng):
        pairs = []
        tries = 0
        while len(pairs) < n and tries < 20 * n:
            tries += 1
            w = self.word_list[int(rng.integers(len(self.word_list)))]
            idx, M, era, text = self._word_arrays(w)
            if len(idx) < 2:
                continue
            a = int(rng.integers(len(idx)))
            inter = M @ M[a]
            union = M.sum(1) + M[a].sum() - inter
            jac = inter / np.maximum(union, 1e-9)
            ok = (np.arange(len(idx)) != a) & (text != text[a])
            pos, neg = np.where(ok & (jac > 0))[0], np.where(ok & (jac == 0))[0]
            self._want_pos = rng.random() < self.pos_frac
            if self._want_pos and len(pos) == 0:
                self._want_pos = False
            if not self._want_pos and len(neg) == 0:
                if len(pos) == 0:
                    continue
                self._want_pos = True
            if self._want_pos:
                b = self._pick(rng, pos, era[pos] != era[a])
            else:
                b = self._pick(rng, neg, era[neg] == era[a])
            label = float(jac[b]) if self.soft else float(jac[b] > 0)
            pairs.append((int(idx[a]), int(idx[b]), label))
        return pairs


# ===========================================================================
# Loss, validation score, training loop
# ===========================================================================

def soft_contrastive_loss(u, v, y, margin=0.5):
    import torch
    d = 1.0 - torch.nn.functional.cosine_similarity(u, v)
    return (y * d.pow(2) + (1.0 - y) * torch.clamp(margin - d, min=0.0).pow(2)).mean()


def pair_auc(V, sense_sets):
    """ROC AUC of cosine similarity for 'shares a sense' over all usage pairs
    of ONE word (vectors V, rows aligned with sense_sets). NaN if one class is empty."""
    from sklearn.metrics import roc_auc_score
    X = V / np.linalg.norm(V, axis=1, keepdims=True)
    S = X @ X.T
    iu = np.triu_indices(len(X), 1)
    same = np.array([[bool(set(a) & set(b)) for b in sense_sets] for a in sense_sets])[iu]
    if same.all() or not same.any():
        return float('nan')
    return float(roc_auc_score(same, S[iu]))


def train_pairs(encoder, items, sample_epoch, epochs, lr, batch_size=16, warmup_ratio=0.1,
                weight_decay=0.01, max_grad_norm=1.0, margin=0.5, eval_fn=None, patience=2,
                eval_before=True, seed=42, log=print, tag='', keep_best=True):
    """
    sample_epoch(epoch) -> list of (i, j, label) pairs for that epoch
    (fresh pairs every epoch). eval_fn(encoder) -> higher-is-better score,
    computed after every epoch (and before training if eval_before).
    keep_best=True: keeps the best weights (in CPU memory) and restores them at
    the end (early stopping). keep_best=False: only monitors, keeps the final weights.
    Returns a history list.
    """
    import torch
    from transformers import get_linear_schedule_with_warmup
    g = torch.Generator().manual_seed(int(seed))
    first = sample_epoch(0)
    steps_per_epoch = math.ceil(len(first) / batch_size)
    total = max(1, steps_per_epoch * epochs)
    opt = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    sched = get_linear_schedule_with_warmup(opt, int(warmup_ratio * total), total)
    scaler = torch.amp.GradScaler('cuda', enabled=encoder.device == 'cuda')

    history, best, best_state, bad = [], -float('inf'), None, 0
    if eval_fn is not None and eval_before:
        best = eval_fn(encoder)
        best_state = encoder.state_dict_cpu() if keep_best else None
        history.append({'epoch': 0, 'val': best, 'loss': None})
        log(f'{tag} epoch 0 (before training) val={best:.4f}')
    for ep in range(1, epochs + 1):
        pairs = first if ep == 1 else sample_epoch(ep - 1)
        order = torch.randperm(len(pairs), generator=g).tolist()
        encoder.train()
        t0, losses = time.time(), []
        for k in range(0, len(order), batch_size):
            batch = [pairs[i] for i in order[k:k + batch_size]]
            texts = [(items[a]['text'], items[a]['start'], items[a]['end']) for a, _, _ in batch] + \
                    [(items[b]['text'], items[b]['start'], items[b]['end']) for _, b, _ in batch]
            y = torch.tensor([l for _, _, l in batch], dtype=torch.float32, device=encoder.device)
            emb = encoder.forward(texts)
            loss = soft_contrastive_loss(emb[:len(batch)], emb[len(batch):], y, margin)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_grad_norm)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(loss))
        rec = {'epoch': ep, 'loss': float(np.mean(losses)), 'pairs': len(pairs),
               'seconds': round(time.time() - t0, 1)}
        if eval_fn is not None:
            rec['val'] = eval_fn(encoder)
        history.append(rec)
        log(f"{tag} epoch {ep} loss={rec['loss']:.4f} "
            + (f"val={rec['val']:.4f} " if 'val' in rec else '') + f"({rec['seconds']}s)")
        if eval_fn is None or not keep_best:
            continue
        if rec['val'] > best + 1e-4:
            best, best_state, bad = rec['val'], encoder.state_dict_cpu(), 0
        else:
            bad += 1
            if bad >= patience:
                log(f'{tag} early stop (no improvement for {patience} epochs)')
                break
    if keep_best and best_state is not None:
        encoder.load_state_dict(best_state)
    for h in history:
        h['best'] = h.get('val') == best
    return history
