"""
Figures for the paper (Assignment 3, Person 2; report Sections 7 and 8.1).

Reads ONLY results/summary/ (written by python -m src.run_pipeline report), so
every plotted number is a number in the summary tables:

    summary.csv            scores per system: per-seed values, mean, sd, 95% CI
    seed_comparisons.csv   paired differences to the main system (src/seed_stats.py)

    python -m src.make_figures                  # writes results/figures/*.pdf (+ .png)
    python -m src.make_figures --out figures    # another folder

Figures
    fig_main_results       every comparison system and the main system on the four
                           scores: one run = point with its 95% interval over words;
                           several seeds = mean +- sd with every seed shown
    fig_seed_variation     the five seeds of the main system on every score
                           (seed 87, whose Stage A collapsed, is marked)
    fig_tradeoff_1a_1b     1a BCubed F1 against 1b rho for every run of every
                           system (one cluster per word = 0.756)
    fig_ablation_effects   difference to the main system on the same seeds, per
                           ablation: 1a and 2 with the 95% interval of the
                           seed-averaged paired difference; 1b and APD with the
                           per-seed differences
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MAIN = 'proposed_ft_syn_sv'
METRICS = [('1a_bcubed_f1', '1a BCubed F1', '1a'), ('1b_spearman', r'1b Spearman $\rho$', '1b'),
           ('2_macro_f1', '2 macro F1', '2'), ('graded_spearman', r'APD $\rho$ (local)', 'graded')]
ONE_CLUSTER = 0.756   # one cluster per word (Section 4.1)
COMPARISON_ORDER = [
    ('baseline_majority', 'Majority'),
    ('baseline_random', 'Random'),
    ('baseline_classical', 'Classical'),
    ('baseline_pretrained_frozen_xlmr', 'Pretrained (frozen XLM-R)'),
    ('proposed_frozen', 'Proposed pipeline, frozen L8'),
    ('proposed_frozen_last', 'Proposed pipeline, frozen last layer'),
    (MAIN, 'Proposed (fine-tuned)'),
]
ABLATION_LABELS = {
    'proposed_frozen_last': 'E2 frozen encoder (last layer)',
    'proposed_ft_syn_sv_changejsd': 'E3 JSD only (= official 1b)',
    'proposed_ft_syn_sv_changeboth': 'E3 JSD + APD (rank mean)',
    'proposed_ft_syn_sv_kmeans_absolute': 'E4 K-means (silhouette)',
    'proposed_ft_syn_sv_nine_words_absolute': 'E4 clustering chosen on 9 words',
    'proposed_ft_syn_sv_fixed_absolute': 'E4 fixed cut t = 0.25',
    'proposed_ft_syn_sv_validation_bcubed_absolute': 'E4 cut by validation BCubed',
    'proposed_abl_nospell': 'E5 no spelling normalisation',
    'proposed_abl_layer4': 'E6 layer 4',
    'proposed_abl_layer8': 'E6 layer 8',
    'proposed_abl_poolfirst': 'E6 first-sub-word pooling',
    'proposed_abl_poolsent': 'E6 sentence-mean pooling',
    'proposed_abl_nostagea': 'E7 no Stage A (synthetic)',
    'proposed_abl_nostageb': 'E7 no Stage B (Swedish)',
    'proposed_abl_noera': 'E7 no era-aware sampling',
    'proposed_abl_hardlabels': 'E7 hard pair labels',
    'proposed_abl_nomarkers': 'E7 no target markers',
}
# E3 variants copy the prediction files of their source run: only the graded score differs.
GRADED_ONLY = ('proposed_ft_syn_sv_changejsd', 'proposed_ft_syn_sv_changeboth')
# E4 variants re-cluster the same vectors with the same Subtask 2 rule: only 1a and 1b can change.
CLUSTERING_ONLY = ('proposed_ft_syn_sv_kmeans_absolute', 'proposed_ft_syn_sv_nine_words_absolute',
                   'proposed_ft_syn_sv_fixed_absolute', 'proposed_ft_syn_sv_validation_bcubed_absolute')


def _shown(system, metric):
    if system in GRADED_ONLY:
        return metric == 'graded'
    if system in CLUSTERING_ONLY:
        return metric in ('1a', '1b')
    return True


def _f(x):
    try:
        return None if x in (None, '') else float(x)
    except ValueError:
        return None


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def _values(row, key):
    raw = row.get(f'{key}_values') or ''
    return [float(v) for v in raw.split()] if raw.strip() else []


def _seeds(row):
    return [int(s) for s in (row.get('seeds') or '').split()]


def _style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9,
                         'legend.fontsize': 8, 'figure.dpi': 150, 'savefig.bbox': 'tight',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.6})
    return plt


def _save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'{name}.{ext}')
    print('wrote', out / f'{name}.pdf')


def fig_main_results(summary, out):
    plt = _style()
    rows = {r['system']: r for r in summary}
    systems = [(s, lab) for s, lab in COMPARISON_ORDER if s in rows]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.3), sharey=True)
    ys = list(range(len(systems)))[::-1]
    for ax, (key, title, _) in zip(axes, METRICS):
        for y, (s, _) in zip(ys, systems):
            r = rows[s]
            mean = _f(r.get(f'{key}_mean'))
            if mean is None:
                continue
            main = s == MAIN
            colour = 'C3' if main else 'C0'
            vals = _values(r, key)
            if len(vals) > 1:
                sd = _f(r.get(f'{key}_sd')) or 0.0
                ax.errorbar(mean, y, xerr=sd, fmt='D', color=colour, ms=5, capsize=3, lw=1.2)
                for seed, v in zip(_seeds(r), vals):
                    ax.plot(v, y + 0.22, 'o', ms=3.2, mfc='none' if seed == 87 else colour,
                            mec=colour, alpha=0.8)
            else:
                lo, hi = _f(r.get(f'{key}_ci_low')), _f(r.get(f'{key}_ci_high'))
                if lo is not None and hi is not None:
                    ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]], fmt='o', color=colour,
                                ms=4.5, capsize=3, lw=1.0)
                else:
                    ax.plot(mean, y, 'o', color=colour, ms=4.5)
        if key == '1a_bcubed_f1':
            ax.axvline(ONE_CLUSTER, color='0.4', ls='--', lw=0.8)
        if key in ('1b_spearman', 'graded_spearman'):
            ax.axvline(0, color='0.4', lw=0.6)
        ax.set_title(title)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([lab for _, lab in systems])
    fig.text(0.5, -0.04, 'One run: point and 95% interval over words. Several seeds: mean ± sd '
             '(diamond) and each seed (small circles; open circle = seed 87). Dashed: one cluster '
             'per word (0.756).', ha='center', fontsize=7.5)
    fig.tight_layout()
    _save(fig, out, 'fig_main_results')
    plt.close(fig)


def fig_seed_variation(summary, out):
    plt = _style()
    rows = {r['system']: r for r in summary}
    if MAIN not in rows:
        return
    r = rows[MAIN]
    seeds = _seeds(r)
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    markers = ['o', 's', '^', 'v', 'P', 'X', 'D']
    for i, seed in enumerate(seeds):
        xs, ys = [], []
        for j, (key, _, _) in enumerate(METRICS):
            vals = _values(r, key)
            if i < len(vals):
                xs.append(j + (i - (len(seeds) - 1) / 2) * 0.08)
                ys.append(vals[i])
        ax.plot(xs, ys, markers[i % len(markers)], ms=6,
                mfc='none' if seed == 87 else f'C{i}', mec=f'C{i}',
                label=f'seed {seed}' + (' (Stage A collapsed)' if seed == 87 else ''))
    for j, (key, _, _) in enumerate(METRICS):
        m, sd = _f(r.get(f'{key}_mean')), _f(r.get(f'{key}_sd'))
        if m is not None:
            ax.hlines(m, j - 0.25, j + 0.25, color='k', lw=1.2)
            if sd:
                ax.add_patch(plt.Rectangle((j - 0.25, m - sd), 0.5, 2 * sd, color='0.85', zorder=0))
    ax.axhline(0, color='0.4', lw=0.6)
    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([t for _, t, _ in METRICS])
    ax.set_ylabel('score')
    ax.set_title('Main system: score of every seed (line = mean, grey = ± sd)')
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
    _save(fig, out, 'fig_seed_variation')
    plt.close(fig)


def fig_tradeoff(summary, out):
    plt = _style()
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    groups = [('comparison', [s for s, _ in COMPARISON_ORDER if s != MAIN], 'C0', 'o'),
              ('ablation', list(ABLATION_LABELS), 'C2', '^'),
              ('main', [MAIN], 'C3', 'D')]
    rows = {r['system']: r for r in summary}
    for name, systems, colour, marker in groups:
        first = True
        for s in systems:
            if s not in rows or s in GRADED_ONLY or (name == 'ablation' and s == 'proposed_frozen_last'):
                continue
            a, b = _values(rows[s], '1a_bcubed_f1'), _values(rows[s], '1b_spearman')
            for seed, x, y in zip(_seeds(rows[s]), a, b):
                ax.plot(x, y, marker, color=colour, ms=5 if name != 'main' else 6,
                        mfc='none' if (s == MAIN and seed == 87) else colour, alpha=0.85,
                        label={'comparison': 'comparison systems', 'ablation': 'ablations',
                               'main': 'main system (5 seeds)'}[name] if first else None)
                first = False
    ax.axvline(ONE_CLUSTER, color='0.4', ls='--', lw=0.8)
    ax.text(ONE_CLUSTER, ax.get_ylim()[1], ' one cluster\n per word', va='top', fontsize=7.5, color='0.3')
    ax.axhline(0, color='0.4', lw=0.6)
    ax.set_xlabel('1a BCubed F1')
    ax.set_ylabel(r'1b Spearman $\rho$')
    ax.set_title('Sense clustering (1a) against change ranking (1b), one point per run')
    ax.legend(loc='lower left', frameon=False)
    _save(fig, out, 'fig_tradeoff_1a_1b')
    plt.close(fig)


def fig_ablation_effects(seed_comps, out):
    plt = _style()
    rows = [r for r in seed_comps if r['main'] == MAIN and r['other'] in ABLATION_LABELS]
    systems = [s for s in ABLATION_LABELS if any(r['other'] == s for r in rows)]
    if not systems:
        print('no ablation comparisons yet: fig_ablation_effects skipped')
        return
    fig, axes = plt.subplots(1, 4, figsize=(11, 0.42 * len(systems) + 1.3), sharey=True)
    ys = list(range(len(systems)))[::-1]
    for ax, (_, title, metric) in zip(axes, METRICS):
        for y, s in zip(ys, systems):
            r = next((r for r in rows if r['other'] == s and r['metric'] == metric), None)
            if r is None or not _shown(s, metric):
                continue
            # the figure shows ablation minus main (positive = the ablation scores higher)
            diff = -float(r['difference'])
            lo, hi = _f(r.get('ci_low')), _f(r.get('ci_high'))
            sig = r['verdict'].startswith('significant difference')
            colour = 'C3' if sig else 'C0'
            if lo is not None and hi is not None:
                ax.errorbar(diff, y, xerr=[[diff - (-hi)], [(-lo) - diff]], fmt='o', color=colour,
                            ms=4.5, capsize=3, lw=1.0, mfc=colour if sig else 'white')
            else:
                ax.plot(diff, y, 'o', color=colour, ms=4.5, mfc=colour if sig else 'white')
            for d in (r.get('per_seed_difference') or '').split():
                ax.plot(-float(d), y + 0.2, '|', color='0.35', ms=6)
        ax.axvline(0, color='0.3', lw=0.8)
        ax.set_title(title)
        ax.set_xlabel('ablation − main system')
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([ABLATION_LABELS[s] for s in systems])
    fig.text(0.5, -0.02, 'Same seeds as the ablation. Circle: mean difference (1a, 2: with the 95% '
             'interval of the seed-averaged paired difference); ticks: each seed. Filled red: meets '
             'the decision rule (interval excludes 0 and p < 0.05). Empty cell: the ablation cannot '
             'change that score (E3: only APD; E4: only 1a and 1b).', ha='center', fontsize=7.5)
    fig.tight_layout()
    _save(fig, out, 'fig_ablation_effects')
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Paper figures from results/summary/.')
    ap.add_argument('--summary', default=str(REPO_ROOT / 'results' / 'summary'))
    ap.add_argument('--out', default=str(REPO_ROOT / 'results' / 'figures'))
    a = ap.parse_args(argv)
    summary_dir, out = Path(a.summary), Path(a.out)
    summary = read_csv(summary_dir / 'summary.csv')
    fig_main_results(summary, out)
    fig_seed_variation(summary, out)
    fig_tradeoff(summary, out)
    sc = summary_dir / 'seed_comparisons.csv'
    if sc.exists():
        fig_ablation_effects(read_csv(sc), out)
    else:
        print('results/summary/seed_comparisons.csv missing: run python -m src.run_pipeline report')


if __name__ == '__main__':
    main()
