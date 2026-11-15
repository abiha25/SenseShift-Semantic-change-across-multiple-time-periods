"""
Feature engineering for the classical baseline (Baseline A).

Every feature here is motivated by an EDA finding (Section 4, F1-F8):

  normalise()          F4  pre-1906 spelling (hv-, -fv-, -dt, af, qv-) is a
                           near-perfect era signal, so we rewrite it to modern
                           spelling before building any feature.
  context_tokens()     F3  historical sentences are longer, so we use a fixed
                           window of k words on each side of the target.
                       F5/F8  the inflected form of the target mostly encodes
                           form and era, so the target itself is left out.
  tfidf_matrix()       F4/F5  character n-grams survive spelling and
                           inflection variants that word features split.
  definition_features()       Subtask 2: how close a usage is to the given
                           sense definition (cosine and stem overlap).

Features we deliberately do NOT use, because the EDA showed they encode
WHEN a sentence was written rather than WHAT the word means:
period label, year, era, sentence length (F3), archaic-spelling rate (F4).

Import from a notebook (repo root on sys.path):
    from src.features import FeatureConfig, word_matrix, definition_features
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l2_normalize

# ---------------------------------------------------------------------------
# 1. Spelling normalisation (EDA F4)
# ---------------------------------------------------------------------------

# The same five markers that eda.py counts as pre-1906 spelling, written as
# (pattern, replacement) so that old and modern spellings become one token.
SPELLING_RULES = [
    (re.compile(r'\bhv'), 'v'),     # hvad -> vad, hvita -> vita
    (re.compile(r'fv'), 'v'),       # öfver -> över, blifva -> bliva
    (re.compile(r'dt\b'), 'tt'),    # godt -> gott, rödt -> rött
    (re.compile(r'\baf\b'), 'av'),  # af -> av
    (re.compile(r'\bqv'), 'kv'),    # qvinna -> kvinna
]

# A token counts as a word if it contains at least one letter. This drops
# punctuation ("," "—") and pure numbers, which carry no sense information.
_WORD_RE = re.compile(r'[a-zåäöéü]', re.IGNORECASE)


def normalise(text: str, spelling: bool = True) -> str:
    """Lower-case, and (optionally) rewrite pre-1906 spelling to modern."""
    text = text.lower()
    if spelling:
        for pattern, repl in SPELLING_RULES:
            text = pattern.sub(repl, text)
    return text


# ---------------------------------------------------------------------------
# 2. Context window around the target (EDA F3, F5, F8)
# ---------------------------------------------------------------------------

def context_tokens(text: str, start: int, end: int, window: int | None,
                   spelling: bool = True, keep_target: bool = False) -> list[str]:
    """
    The words around the target, without the target itself.

    window = k   -> up to k words on each side
    window = None -> the whole sentence
    keep_target  -> only for the ablation that tests the form feature (F5)
    """
    left = [t for t in normalise(text[:start], spelling).split() if _WORD_RE.search(t)]
    right = [t for t in normalise(text[end:], spelling).split() if _WORD_RE.search(t)]
    if window is not None:
        left, right = left[-window:], right[:window]
    middle = [normalise(text[start:end], spelling)] if keep_target else []
    return left + middle + right


def contexts(df: pd.DataFrame, window: int | None, spelling: bool = True,
             keep_target: bool = False) -> list[str]:
    """One context string per row; always uses the repaired offsets."""
    return [' '.join(context_tokens(t, int(s), int(e), window, spelling, keep_target))
            for t, s, e in zip(df['text'], df['start_fixed'], df['end_fixed'])]


# ---------------------------------------------------------------------------
# 3. TF-IDF matrices, fitted per target word
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureConfig:
    """
    One feature setting for Subtask 1.

    kind     'char'  character 3-5-grams inside word boundaries
             'word'  word unigrams
             'both'  both blocks side by side, each L2-normalised and
                     weighted 1/sqrt(2) so neither dominates the cosine
    window   words on each side of the target (None = whole sentence)
    spelling apply the F4 normalisation
    keep_target  include the target form (ablation for F5 only)
    """
    kind: str = 'both'
    window: int | None = 10
    spelling: bool = True
    keep_target: bool = False

    def name(self) -> str:
        w = 'all' if self.window is None else f'pm{self.window}'
        extra = ('' if self.spelling else '-nospell') + ('+form' if self.keep_target else '')
        return f'{self.kind}-{w}{extra}'


def _vectorizer(kind: str) -> TfidfVectorizer:
    # min_df=2: a feature must occur in at least two usages of the word,
    # otherwise it cannot make two usages similar and only adds noise.
    # sublinear_tf: a repeated word counts log-scaled, not linearly.
    if kind == 'char':
        return TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2,
                               sublinear_tf=True)
    if kind == 'word':
        return TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\b\w\w+\b',
                               min_df=2, sublinear_tf=True)
    raise ValueError(kind)


def tfidf_matrix(docs: list[str], kind: str) -> sp.csr_matrix:
    """Rows are L2-normalised, so a dot product is a cosine similarity."""
    if kind == 'both':
        blocks = [tfidf_matrix(docs, 'char'), tfidf_matrix(docs, 'word')]
        return sp.hstack([b / np.sqrt(2) for b in blocks]).tocsr()
    return l2_normalize(_vectorizer(kind).fit_transform(docs)).tocsr()


def word_matrix(df_word: pd.DataFrame, cfg: FeatureConfig) -> sp.csr_matrix:
    """
    Feature matrix for the usages of ONE target word.

    It is fitted on that word's (unlabelled) usages only. This is allowed in
    the evaluation phase too, because the test usages are given to us; no
    gold label is used here.
    """
    docs = contexts(df_word, cfg.window, cfg.spelling, cfg.keep_target)
    return tfidf_matrix(docs, cfg.kind)


# ---------------------------------------------------------------------------
# 4. Subtask 2: usage-definition similarity features
# ---------------------------------------------------------------------------

# Short list of Swedish function words (modern spelling; old spellings are
# normalised to these first). Used only for the overlap features, where a
# shared "och" or "som" says nothing about the sense.
SWEDISH_STOPWORDS = set("""
alla allt att av blev bli blir de dem den denna deras dess det detta dig din
ditt du där då efter ej eller en er ert ett från för ha hade han hans har hen
henne hennes hon honom hur här i icke ingen inom inte jag ju kan kunde man
med mellan men mig min mina mitt mot mycket ni nu när någon något några och
om oss på samma sedan sig sin sina sitt själv ska skall skulle som så sådan
till under upp ut utan vad var vara varit varje vars vem vi vid vilka vilken
vilket vår våra vårt än är åt över ock också även
""".split())

_STEMMER = SnowballStemmer('swedish')


def content_stems(text: str) -> set[str]:
    """Stems of the content words (no stopwords, at least 3 letters)."""
    toks = re.findall(r'[a-zåäöéü]+', text.lower())
    return {_STEMMER.stem(t) for t in toks if len(t) >= 3 and t not in SWEDISH_STOPWORDS}


DEF_FEATURES = ['cos_char', 'cos_word', 'stem_overlap', 'stem_jaccard']


def definition_features(df: pd.DataFrame, window: int | None = 10) -> pd.DataFrame:
    """
    Four similarity scores between each usage context and its word's
    sense definition, plus a within-word percentile rank of each.

    Why the rank versions: the raw scale differs a lot between words (a long
    definition shares more n-grams with everything). The rank says "this
    usage is among the X% most definition-like usages OF THIS WORD", which
    transfers across words in leave-one-word-out validation.
    """
    parts = []
    for word, g in df.groupby('word', sort=True):
        ctx = contexts(g, window, spelling=True)
        definition = normalise(g['definition'].iloc[0])
        out = pd.DataFrame(index=g.index)
        for kind, col in [('char', 'cos_char'), ('word', 'cos_word')]:
            # Fit the vocabulary on the usages AND the definition, so the
            # definition's words are in the vocabulary.
            vec = _vectorizer(kind).set_params(min_df=1)
            X = l2_normalize(vec.fit_transform(ctx + [definition]))
            out[col] = (X[:-1] @ X[-1].T).toarray().ravel()
        d = content_stems(definition)
        stems = [content_stems(c) for c in ctx]
        out['stem_overlap'] = [len(s & d) for s in stems]
        out['stem_jaccard'] = [len(s & d) / max(len(s | d), 1) for s in stems]
        for col in DEF_FEATURES:
            out[col + '_rank'] = out[col].rank(pct=True, method='average')
        parts.append(out)
    return pd.concat(parts).loc[df.index]


# ---------------------------------------------------------------------------
# 5. Era-proxy features: ONLY for the shortcut test (EDA F2, F3, F4)
# ---------------------------------------------------------------------------

# Same five pre-1906 markers that eda.py counts (F4).
_ARCHAIC_RE = re.compile(r'^hv|fv|dt$|^af$|^qv')

ERA_PROXIES = ['n_tokens', 'archaic_rate', 'is_modern']


def era_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features that say WHEN a sentence was written, not what the word means:
    sentence length (F3), share of old-spelling tokens (F4) and the era
    itself (F2). They are never part of the baseline; they are added in one
    ablation to test whether a classifier would exploit the time shortcut.
    """
    toks = df['text'].str.lower().str.split()
    words = toks.apply(lambda ts: [t for t in ts if _WORD_RE.search(t)])
    return pd.DataFrame({
        'n_tokens': toks.str.len(),
        'archaic_rate': words.apply(lambda ws: sum(bool(_ARCHAIC_RE.search(w)) for w in ws) / max(len(ws), 1)),
        'is_modern': (df['period_index'] >= 5).astype(int),
    }, index=df.index)
