# semeval2027-task3
Semantic change across multiple time period
FOR DATA LOADING:
## Data and EDA (Person 2)

1. Download `dev.zip` from Codabench and unzip it so you have `<DATA_DIR>/SV/usages.jsonl` etc.
   The data is not stored in this repo.
2. Load it with the shared loader:

```python
from data_loader import load_language, check_integrity, leave_one_word_out, write_subtask1, write_subtask2
df = load_language(DATA_DIR, 'SV')     # one row per usage, gold labels included
check_integrity(df)                    # prints all data checks
```

- Use `start_fixed` / `end_fixed` / `target_fixed` (not `start` / `end`) to extract the target word:
  five skär offsets in the gold data are wrong and are repaired here.
- Tune with `leave_one_word_out(df)` (no separate training set exists).
- Write predictions with `write_subtask1` / `write_subtask2` (official format).

3. EDA: `eda.py` builds all figures (`results/eda/EDA.pdf`); the notebook is in `notebooks/`.
   Key finding for the baselines: frozen XLM-R vectors group usages by word form and era
   more than by sense (Section 4, F8).

## Features and Classical Baseline (Person 3)

Baseline A: interpretable features + clustering (Subtask 1) and logistic regression (Subtask 2).
Report: Section 5 (Feature Engineering) and Section 6A (Classical Baseline).

### Files

| File | What it does |
|---|---|
| `src/features.py` | Spelling normalisation (pre-1906 → modern), ±k-word context windows (target removed), TF-IDF matrices, Subtask 2 usage–definition similarity features, era proxies (shortcut test only) |
| `src/classical_baseline.py` | Agglomerative clustering, leave-one-word-out (LOWO) selection, 1b diagnostics (within-era ρ, shuffle control, era tracking), Subtask 2 logistic regression, TF-IDF + LR, JSD export |
| `notebooks/classical_baselineA.ipynb` | Runs everything step by step (Colab, about 8–10 min) |
| `run_person3.py` | Same steps as the notebook, from the command line (about 2–3 min) |
| `results/classical/` | Predictions (official format), all tables (CSV), figure, `summary.json` |

### How to run

```bash
pip install -r requirements.txt
python src/sanity_check.py        # scoring setup still OK (one cluster = 0.756)
python run_person3.py             # or run notebooks/classical_baselineA.ipynb in Colab
```

In Colab, set `REPO` in Cell 1 to the repo folder on your Drive.

### Method in short

- **Features:** context words within ±k of the target (k = 3, 5, 10 or the whole sentence), target removed, old spelling normalised; word and/or character 3–5-gram TF-IDF, fitted per word on unlabelled usages. Period, year, length and archaic spelling are **never** features (time shortcut; EDA F2–F4).
- **Subtask 1:** average-linkage agglomerative clustering on cosine distance, cut at threshold *t*; clusters smaller than *m* merge into the nearest large cluster (can return one sense). The official scorer computes 1b (JSD) from the same labels.
- **Subtask 2:** four context–definition similarity scores (char/word cosine, stem overlap, Jaccard) as within-word percentile ranks → logistic regression; threshold tuned on the training words.
- **Validation:** LOWO everywhere. Settings and thresholds are chosen on 9 words and applied to the 10th; the 10 held-out predictions are scored once with `src/scoring.py`. Seed 42.

### Results (Swedish dev, LOWO)

| | Baseline A | Trivial baseline |
|---|---|---|
| 1a BCubed F1 | 0.697 | 0.756 (one cluster per word) |
| 1b Spearman ρ | 0.001 | undefined |
| 2 macro F1 | 0.472 | 0.472 (predict all 1) |

Key findings:
- TF-IDF context vectors carry too little sense information to beat one cluster per word.
- In-sample tuning reaches ρ = 0.35 on 1b but about 0.00 under LOWO. K-means' ρ = 0.24 matches randomly shuffled labels with the same cluster sizes, so it comes from cluster granularity, not content.
- TF-IDF + LR (0.244) learns which target word a sentence belongs to, not the sense; within a single word the contexts do carry sense information (AUC 0.725).

### Use in other baselines

```python
from src import classical_baseline as cb
scorer = cb.FastScorer(df, 'data/SV/subtask1.jsonl')   # fast in-memory official scoring for tuning
cb.permutation_control(scorer, labels_by_word)          # shuffle control: report next to any 1b rho
```

For error analysis: `results/classical/subtask1b_jsd_per_pair.csv` (gold vs predicted JSD per word-period pair) and `subtask1_sense_shares_per_period.csv`.

### Change to a shared file

`src/scoring.py`: scoring a subset of words (`words=` / `--only-predicted-words`) now filters the predictions as well as gold. Before, the official loader rejected the other words' predictions. Full-file scores are unchanged.
