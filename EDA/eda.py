"""
EDA figures for SemEval-2027 Task 3 (Swedish dev data).

    from eda import add_features, compute_embeddings, make_eda_pdf, eda_numbers
    df = add_features(df)                       # df from data_loader.load_language
    emb = compute_embeddings(df)                # optional, needs a GPU runtime
    make_eda_pdf(df, 'EDA.pdf', emb=emb, png_dir='eda_png')
    eda_numbers(df, emb)                        # numbers for the write-up

Every figure has a title and axis labels and no caption, as the brief asks.
"""

import os
import re
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from data_loader import ordered_periods, period_start_year, period_end_year

# ---------------------------------------------------------------------------
# Style: one validated categorical palette, one blue sequential ramp,
# quiet axes. Colours are assigned in fixed order, never cycled.
# ---------------------------------------------------------------------------

CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
OTHER = '#c3c2b7'
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
SEQ = LinearSegmentedColormap.from_list(
    'blue', ['#f3f8fe', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])
ERA_COLORS = {'historical (1880-1915)': CAT[0], 'modern (2006-2023)': CAT[1]}

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
    'axes.titlesize': 11, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.grid': False, 'grid.color': GRID,
    'grid.linewidth': 0.6, 'legend.frameon': False, 'legend.fontsize': 8,
    'font.size': 9, 'figure.titlesize': 13, 'figure.titleweight': 'bold',
})


def _short(period):
    """'1880-1883' -> '1880-83' for compact tick labels."""
    a, b = str(period).split('-')
    return f'{a}-{b[-2:]}'


def _ygrid(ax):
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Derived features (shared by figures and numbers)
# ---------------------------------------------------------------------------

ARCHAIC_PATTERNS = [
    r'^hv',        # hvad, hvita  -> vad, vita (spelling reform 1906)
    r'fv',         # öfver, blifva -> över, bliva
    r'dt$',        # godt, rödt   -> gott, rött
    r'^af$',       # af -> av
    r'^qv',        # qvinna -> kvinna
]
ARCHAIC_RE = re.compile('|'.join(ARCHAIC_PATTERNS))
OCR_RE = re.compile(r'(?=.*[A-Za-zÅÄÖåäö])(?=.*\d)|[<>|\\{}~^„]')


def add_features(df):
    """Add the per-usage features the EDA uses. Returns a copy."""
    df = df.copy()
    tokens = df['text'].str.split()
    df['n_tokens'] = tokens.str.len()
    df['n_chars'] = df['text'].str.len()
    df['era'] = df['period_label'].map(
        lambda p: 'historical (1880-1915)' if period_start_year(p) < 1950 else 'modern (2006-2023)')

    def rate(toks, regex):
        words = [t.lower() for t in toks if re.search(r'[A-Za-zÅÄÖåäö0-9]', t)]
        return sum(bool(regex.search(t)) for t in words) / max(len(words), 1)

    df['archaic_rate'] = tokens.apply(lambda t: rate(t, ARCHAIC_RE))
    df['ocr_noise_rate'] = tokens.apply(lambda t: rate(t, OCR_RE))
    df['form'] = df['target_fixed'].str.lower()
    df['rel_position'] = df['start_fixed'] / df['n_chars'].clip(lower=1)
    return df


# ---------------------------------------------------------------------------
# Figure 1: coverage
# ---------------------------------------------------------------------------

def fig_coverage(df):
    periods = ordered_periods(df)
    words = sorted(df['word'].unique())
    counts = (df.pivot_table(index='word', columns='period_label', values='sentence_id',
                             aggfunc='count', fill_value=0)
                .reindex(index=words, columns=periods, fill_value=0))
    fig, ax = plt.subplots(figsize=(10, 5.2))
    masked = np.ma.masked_equal(counts.values, 0)
    cmap = SEQ.copy()
    cmap.set_bad('#f0efec')
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=30, aspect='auto')
    for i in range(len(words)):
        for j in range(len(periods)):
            n = counts.values[i, j]
            if n == 0:
                ax.text(j, i, '–', ha='center', va='center', color=MUTED, fontsize=8)
            else:
                color = 'white' if n >= 18 else INK
                weight = 'bold' if n < 10 else 'normal'
                ax.text(j, i, f'{n}{"*" if n < 10 else ""}', ha='center', va='center',
                        color=color, fontsize=8, fontweight=weight)
    ax.set_xticks(range(len(periods)), [_short(p) for p in periods], rotation=0)
    ax.set_yticks(range(len(words)), words)
    ax.axvline(4.5, color=INK2, lw=1.2, ls='--')
    ax.set_xlabel('Period label (dashed line = 90-year gap between 1915 and 2006)')
    ax.set_ylabel('Target word')
    ax.set_title('Usages per word and period (* = fewer than 10, pair not scored in Subtask 1b; – = no usages)')
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label('Number of usages')
    cb.outline.set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2: year vs period label on the real timeline
# ---------------------------------------------------------------------------

def fig_year_vs_period(df):
    periods = ordered_periods(df)
    per_year = df.groupby('year').size()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True,
                                 gridspec_kw={'width_ratios': [38, 20], 'wspace': 0.05})
    for ax, lo, hi in [(a1, 1878, 1916), (a2, 2004.5, 2024.8)]:
        for k, p in enumerate(periods):
            s, e = period_start_year(p), period_end_year(p)
            if lo <= s <= hi:
                ax.axvspan(s - 0.5, e + 0.5, color='#f0efec', zorder=0)
                ax.text((s + e) / 2, per_year.max() * 1.08, _short(p), ha='center',
                        fontsize=7, color=INK2)
        yrs = per_year[(per_year.index >= lo) & (per_year.index <= hi)]
        ax.bar(yrs.index, yrs.values, width=0.7, color=CAT[0], zorder=2)
        ax.set_xlim(lo, hi)
        _ygrid(ax)
    a1.set_ylim(0, per_year.max() * 1.18)
    a2.spines['left'].set_visible(False)
    a2.tick_params(left=False)
    a1.set_ylabel('Number of usages')
    a1.set_xlabel('Publication year (1880–1915)')
    a2.set_xlabel('Publication year (2006–2023)')
    for ax, x in [(a1, 1), (a2, 0)]:
        ax.plot([x, x], [0, 1], transform=ax.transAxes, color=INK2, lw=1, ls=(0, (3, 3)),
                clip_on=False)
    fig.suptitle('Usage years on the real timeline (shaded = period window; axis broken over 1916–2005)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3: sentence length by period
# ---------------------------------------------------------------------------

def fig_sentence_length(df):
    periods = ordered_periods(df)
    data = [df.loc[df['period_label'] == p, 'n_tokens'].values for p in periods]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    bp = ax.boxplot(data, positions=range(len(periods)), widths=0.55, patch_artist=True,
                    showfliers=True, medianprops=dict(color=INK, lw=1.5),
                    flierprops=dict(marker='o', markersize=3, markerfacecolor=MUTED,
                                    markeredgecolor='none'),
                    whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
    for k, box in enumerate(bp['boxes']):
        era = 'historical (1880-1915)' if period_start_year(periods[k]) < 1950 else 'modern (2006-2023)'
        box.set_facecolor(ERA_COLORS[era])
        box.set_alpha(0.85)
        box.set_edgecolor('white')
    ax.axhline(16, color=MUTED, lw=1, ls=':')
    ax.text(len(periods) - 0.5, 16.5, 'minimum = 16 tokens in every period', ha='right',
            va='bottom', fontsize=7, color=INK2)
    ax.set_xticks(range(len(periods)), [_short(p) for p in periods])
    ax.set_xlabel('Period label')
    ax.set_ylabel('Sentence length (whitespace tokens)')
    ax.set_title('Sentence length by period')
    handles = [matplotlib.patches.Patch(color=c, label=l) for l, c in ERA_COLORS.items()]
    ax.legend(handles=handles, loc='upper right')
    _ygrid(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4: archaic spelling and OCR noise by period
# ---------------------------------------------------------------------------

def fig_spelling_noise(df):
    """Pre-1906 spelling markers per period. (An OCR-noise proxy was tested
    and dropped: it mostly counted digits inside modern tokens.)"""
    periods = ordered_periods(df)
    g = df.groupby('period_label')['archaic_rate'].mean().reindex(periods) * 100
    any_ = (df.assign(a=df['archaic_rate'] > 0).groupby('period_label')['a'].mean()
              .reindex(periods) * 100)
    x = np.arange(len(periods))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    colors = [CAT[0] if period_start_year(p) < 1950 else CAT[1] for p in periods]
    a1.bar(x, g.values, 0.7, color=colors)
    a1.set_ylabel('Pre-1906 spelling tokens (% of tokens, mean)')
    a1.set_title('Share of tokens with old spelling')
    a2.bar(x, any_.values, 0.7, color=colors)
    a2.set_ylabel('Usages with at least one old-spelling token (%)')
    a2.set_title('Share of usages containing old spelling')
    for a in (a1, a2):
        a.set_xticks(x, [_short(p) for p in periods], rotation=45)
        a.set_xlabel('Period label')
        _ygrid(a)
    handles = [matplotlib.patches.Patch(color=c, label=l) for l, c in ERA_COLORS.items()]
    a2.legend(handles=handles, loc='upper right')
    fig.suptitle('Pre-1906 spelling markers by period (hv-, -fv-, -dt, af, qv-)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 5: inflected forms of the target word by period
# ---------------------------------------------------------------------------

def fig_forms(df, words=None, top=4):
    periods = ordered_periods(df)
    if words is None:
        full = df.groupby('word')['period_label'].nunique()
        words = sorted(full[full == len(periods)].index)[:5]
    fig, axes = plt.subplots(1, len(words), figsize=(14, 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, w in zip(axes, words):
        sub = df[df['word'] == w]
        top_forms = [f for f, _ in Counter(sub['form']).most_common(top)]
        tab = (sub.assign(f=sub['form'].where(sub['form'].isin(top_forms), 'other'))
                  .pivot_table(index='period_label', columns='f', values='sentence_id',
                               aggfunc='count', fill_value=0)
                  .reindex(index=periods, fill_value=0))
        tab = tab.div(tab.sum(axis=1).replace(0, np.nan), axis=0) * 100
        cols = top_forms + (['other'] if 'other' in tab.columns else [])
        bottom = np.zeros(len(periods))
        for k, f in enumerate(cols):
            vals = tab[f].fillna(0).values
            ax.bar(range(len(periods)), vals, bottom=bottom, width=0.75,
                   color=CAT[k] if f != 'other' else OTHER, edgecolor='white', linewidth=1, label=f)
            bottom += vals
        ax.set_title(w)
        ax.set_xticks(range(len(periods)), [p[:4] for p in periods], rotation=90)
        ax.set_xlabel('Period (start year)')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.32), ncol=3, fontsize=7)
    axes[0].set_ylabel('Share of usages (%)')
    fig.suptitle('Inflected form of the target word by period (words present in all 10 periods)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 6: gold sense distribution per period (what Subtask 1b measures)
# ---------------------------------------------------------------------------

def sense_distribution(df):
    """Fractional sense counts per word and period, exactly like the scorer."""
    rows = []
    for _, r in df.iterrows():
        for s in r['senses']:
            rows.append((r['word'], r['period_label'], s, 1.0 / len(r['senses'])))
    return pd.DataFrame(rows, columns=['word', 'period_label', 'sense', 'weight'])


def fig_sense_distribution(df, max_senses=7):
    periods = ordered_periods(df)
    dist = sense_distribution(df)
    words = sorted(df['word'].unique())
    ncol = 5
    nrow = int(np.ceil(len(words) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 3.6 * nrow), sharey=True)
    axes = axes.ravel()
    for ax, w in zip(axes, words):
        sub = dist[dist['word'] == w]
        order = sub.groupby('sense')['weight'].sum().sort_values(ascending=False).index.tolist()
        keep = order[:max_senses]
        sub = sub.assign(s=sub['sense'].where(sub['sense'].isin(keep), 'other'))
        tab = (sub.pivot_table(index='period_label', columns='s', values='weight', aggfunc='sum',
                               fill_value=0).reindex(index=periods))
        tab = tab.div(tab.sum(axis=1), axis=0) * 100
        bottom = np.zeros(len(periods))
        cols = keep + (['other'] if 'other' in tab.columns else [])
        for k, s in enumerate(cols):
            vals = tab[s].fillna(0).values if s in tab.columns else np.zeros(len(periods))
            ax.bar(range(len(periods)), vals, bottom=bottom, width=0.8,
                   color=CAT[k] if s != 'other' else OTHER, edgecolor='white', linewidth=0.8,
                   label=f'sense {s}' if s != 'other' else 'other senses')
            bottom += vals
        ax.set_title(f'{w} ({len(order)} senses)')
        ax.set_xticks(range(len(periods)), [p[:4] for p in periods], rotation=90, fontsize=7)
        ax.set_xlabel('Period (start year)')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize=6.5,
                  handlelength=1, columnspacing=0.8)
    for ax in axes[len(words):]:
        ax.set_visible(False)
    for ax in axes[::ncol]:
        ax.set_ylabel('Share of usages (%)')
    fig.suptitle('Gold sense distribution per period (fractional counts, as used for the Subtask 1b JSD)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 7: Subtask 2 target-sense share per period (imbalance)
# ---------------------------------------------------------------------------

def fig_target_share(df):
    periods = ordered_periods(df)
    words = sorted(df['word'].unique())
    pos = df.pivot_table(index='word', columns='period_label', values='target_sense', aggfunc='sum')
    n = df.pivot_table(index='word', columns='period_label', values='target_sense', aggfunc='count')
    pos, n = pos.reindex(index=words, columns=periods), n.reindex(index=words, columns=periods)
    share = pos / n * 100
    fig, ax = plt.subplots(figsize=(10, 5.2))
    cmap = SEQ.copy()
    cmap.set_bad('#f0efec')
    im = ax.imshow(np.ma.masked_invalid(share.values), cmap=cmap, vmin=0, vmax=100, aspect='auto')
    for i in range(len(words)):
        for j in range(len(periods)):
            if np.isnan(share.values[i, j]):
                ax.text(j, i, '–', ha='center', va='center', color=MUTED, fontsize=8)
            else:
                v = share.values[i, j]
                ax.text(j, i, f'{int(pos.values[i, j])}/{int(n.values[i, j])}', ha='center',
                        va='center', fontsize=7, color='white' if v >= 55 else INK,
                        fontweight='bold' if pos.values[i, j] == 0 else 'normal')
    ax.set_xticks(range(len(periods)), [_short(p) for p in periods])
    ax.set_yticks(range(len(words)), words)
    ax.axvline(4.5, color=INK2, lw=1.2, ls='--')
    ax.set_xlabel('Period label')
    ax.set_ylabel('Target word')
    ax.set_title('Share of usages expressing the Subtask 2 target sense (cell text = positives / usages)')
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label('Target-sense share (%)')
    cb.outline.set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 8: embedding projection (period vs sense)
# ---------------------------------------------------------------------------

def compute_embeddings(df, model_name='xlm-roberta-base', layer=8, batch_size=32,
                       cache_path=None, device=None):
    """
    One vector per usage: the mean of the sub-word vectors covering the
    target word (start_fixed:end_fixed) at one hidden layer. Layer 8 follows
    Periti & Tahmasebi (2024), who found middle layers work best.
    The model is frozen (no fine-tuning). Results are cached to cache_path.
    """
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)

    import torch
    from transformers import AutoModel, AutoTokenizer

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device).eval()

    texts = df['text'].tolist()
    spans = list(zip(df['start_fixed'].astype(int), df['end_fixed'].astype(int)))
    out = np.zeros((len(df), model.config.hidden_size), dtype=np.float32)

    for b in range(0, len(df), batch_size):
        enc = tok(texts[b:b + batch_size], return_tensors='pt', padding=True, truncation=True,
                  max_length=512, return_offsets_mapping=True)
        offsets = enc.pop('offset_mapping')
        with torch.no_grad():
            hidden = model(**{k: v.to(device) for k, v in enc.items()}).hidden_states[layer]
        for i in range(hidden.shape[0]):
            s, e = spans[b + i]
            off = offsets[i]
            mask = (off[:, 0] < e) & (off[:, 1] > s) & (off[:, 1] > off[:, 0])
            if mask.sum() == 0:            # target cut off by truncation: fall back to <s>
                mask[0] = True
            out[b + i] = hidden[i][mask.to(hidden.device)].mean(0).cpu().numpy()
        print(f'\rembedded {min(b + batch_size, len(df))}/{len(df)}', end='')
    print()
    if cache_path:
        np.save(cache_path, out)
    return out


def _pca2(x):
    x = x - x.mean(0)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def fig_embeddings(df, emb, words=None):
    periods = ordered_periods(df)
    if words is None:
        full = df.groupby('word')['period_label'].nunique()
        words = sorted(full[full == len(periods)].index)[:5]
    fig, axes = plt.subplots(2, len(words), figsize=(15, 8))
    axes = np.atleast_2d(axes)
    for j, w in enumerate(words):
        idx = np.where(df['word'].values == w)[0]
        xy = _pca2(emb[idx])
        sub = df.iloc[idx]
        ax = axes[0, j]
        for era, c in ERA_COLORS.items():
            m = (sub['era'] == era).values
            ax.scatter(xy[m, 0], xy[m, 1], s=14, color=c, edgecolor='white', linewidth=0.4,
                       label=era)
        ax.set_title(f'{w}: coloured by era')
        first = sub['senses'].apply(lambda s: s[0]).values
        order = [s for s, _ in Counter(first).most_common()]
        keep = order[:5]
        ax2 = axes[1, j]
        for k, s in enumerate(keep + ['other']):
            m = np.isin(first, keep, invert=True) if s == 'other' else (first == s)
            if m.sum():
                ax2.scatter(xy[m, 0], xy[m, 1], s=14, color=CAT[k] if s != 'other' else OTHER,
                            edgecolor='white', linewidth=0.4,
                            label=f'sense {s}' if s != 'other' else 'other senses')
        ax2.set_title(f'{w}: coloured by gold sense')
        ax2.legend(fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3,
                   markerscale=1.2, handletextpad=0.2, columnspacing=0.8)
        for a in (ax, ax2):
            a.set_xlabel('PC 1')
            a.set_ylabel('PC 2')
            a.tick_params(labelsize=6)
    axes[0, 0].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2)
    fig.suptitle('XLM-R (layer 8, frozen) target-word embeddings, PCA per word: era vs gold sense',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


def separation_scores(df, emb, words=None):
    """
    Silhouette (cosine) of the embeddings grouped by era and by first gold
    sense, per word. Higher = the grouping is more visible in the embedding.
    If era > sense, the representation encodes time more than meaning.
    """
    from sklearn.metrics import silhouette_score
    rows = []
    for w in (words or sorted(df['word'].unique())):
        idx = np.where(df['word'].values == w)[0]
        sub = df.iloc[idx]
        first = sub['senses'].apply(lambda s: s[0]).values
        row = {'word': w}
        if sub['era'].nunique() > 1:
            row['silhouette_era'] = silhouette_score(emb[idx], sub['era'].values, metric='cosine')
        if len(set(first)) > 1:
            row['silhouette_sense'] = silhouette_score(emb[idx], first, metric='cosine')
        forms = sub['form'].values
        if 1 < len(set(forms)) < len(forms):
            row['silhouette_form'] = silhouette_score(emb[idx], forms, metric='cosine')
        rows.append(row)
    return pd.DataFrame(rows).set_index('word').round(3)


# ---------------------------------------------------------------------------
# PDF and numbers
# ---------------------------------------------------------------------------

FIGURES = [
    ('01_coverage', fig_coverage),
    ('02_year_vs_period', fig_year_vs_period),
    ('03_sentence_length', fig_sentence_length),
    ('04_spelling_ocr', fig_spelling_noise),
    ('05_forms', fig_forms),
    ('06_sense_distribution', fig_sense_distribution),
    ('07_target_share', fig_target_share),
]


def make_eda_pdf(df, out_pdf, emb=None, png_dir=None, show=True):
    """Draw every figure, save them into one PDF (and optional PNGs for Overleaf)."""
    if png_dir:
        os.makedirs(png_dir, exist_ok=True)
    figs = list(FIGURES)
    if emb is not None:
        figs.append(('08_embeddings', lambda d: fig_embeddings(d, emb)))
    with PdfPages(out_pdf) as pdf:
        for name, fn in figs:
            fig = fn(df)
            pdf.savefig(fig, bbox_inches='tight', pad_inches=0.25)
            if png_dir:
                fig.savefig(os.path.join(png_dir, name + '.png'), dpi=200, bbox_inches='tight', pad_inches=0.2)
            if show:
                plt.show()
            plt.close(fig)
    print(f'saved {len(figs)} figures to {out_pdf}')


def eda_numbers(df, emb=None):
    """Print the numbers behind each figure, for writing the observations."""
    periods = ordered_periods(df)
    print('== Sentence length by era (tokens)')
    print(df.groupby('era')['n_tokens'].describe().round(1).to_string())
    print('\n== Spelling / OCR rates by era (% of tokens, mean)')
    print((df.groupby('era')[['archaic_rate', 'ocr_noise_rate']].mean() * 100).round(2).to_string())
    print('\n== Usages with at least one archaic / OCR-like token, by era')
    print(df.assign(any_arch=df['archaic_rate'] > 0, any_ocr=df['ocr_noise_rate'] > 0)
            .groupby('era')[['any_arch', 'any_ocr']].mean().mul(100).round(1).to_string())
    print('\n== Distinct inflected forms per word, by era')
    print(df.groupby(['word', 'era'])['form'].nunique().unstack().to_string())
    print('\n== Senses present per word: historical only / modern only / both')
    for w, sub in df.groupby('word'):
        h = {s for lst in sub[sub['era'].str.startswith('hist')]['senses'] for s in lst}
        m = {s for lst in sub[sub['era'].str.startswith('mod')]['senses'] for s in lst}
        if h and m:
            print(f'  {w:10s} only historical {sorted(h - m)}  only modern {sorted(m - h)}  both {sorted(h & m)}')
    print('\n== Relative position of the target in the sentence (0 = start, 1 = end)')
    print(df.groupby('era')['rel_position'].describe().round(2).to_string())
    if emb is not None:
        print('\n== Embedding separation (cosine silhouette): era vs first gold sense vs word form')
        print(separation_scores(df, emb).to_string())
