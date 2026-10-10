# semeval2027-task3
Semantic change across multiple time period
FOR DATA LOADING:
## Data and EDA 

1. Download `dev.zip` from Codabench and unzip it so you have `<DATA_DIR>/SV/usages.jsonl` etc.
   The dev data (data/SV) is included in this repo for convenience; the official source is Codabench.
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

3. EDA files are in the `EDA/` folder:
   - `EDA/eda.py`: builds all EDA figures (needs `data_loader.py` importable, see below)
   - `EDA/eda_8_features_groups.pdf`: the final EDA figures (8 plots)
   - `EDA/person2_data_eda.ipynb`: the notebook that produced every number and figure

   Key finding for the baselines: frozen XLM-R vectors group usages by word form and era
   more than by sense (Section 4, F8).

To use `eda.py`, Python must be able to find both files. In Colab, after cloning the repo:

```python
import sys
sys.path.insert(0, '/content/<repo-folder>')        # folder containing data_loader.py
sys.path.insert(0, '/content/<repo-folder>/EDA')    # folder containing eda.py
from eda import add_features, make_eda_pdf, eda_numbers
```

## Features and Classical Baseline 

Baseline A: interpretable features + clustering (Subtask 1) and logistic regression (Subtask 2).
Report: Section 5 (Feature Engineering) and Section 6.1 (Classical Baseline).

### Files

| File | What it does |
|---|---|
| `src/features.py` | Spelling normalisation (pre-1906 → modern), ±k-word context windows (target removed), TF-IDF matrices, Subtask 2 usage–definition similarity features, era proxies (shortcut test only) |
| `src/classical_baseline.py` | Agglomerative clustering, leave-one-word-out (LOWO) selection, 1b diagnostics (within-era ρ, shuffle control, era tracking), Subtask 2 logistic regression, TF-IDF + LR, JSD export |
| `notebooks/classical_baselineA.ipynb` | Runs everything step by step (Colab, about 8–10 min) |
| `results/classical/` | Predictions (official format), all tables (CSV), figure, `summary.json` |

### How to run

Open `notebooks/classical_baselineA.ipynb` in Colab, set `REPO` in Cell 1 to the repo folder on your Drive, and run all cells. Results are written to `results/classical/`.

To check the scoring setup first:

```bash
pip install -r requirements.txt
python src/sanity_check.py        # one cluster per word should give 0.756
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

## Pretrained Baseline, Results and Error Analysis (Person 4: Moteba)

Baseline B: frozen multilingual encoders with the same clustering, LOWO protocol and official scorer as Baseline A.
Report: Section 6.2 (Baseline B), Section 8 (Results), Section 9 (Error Analysis), Sections 10–11.

### Files

| File | What it does |
|---|---|
| `src/pretrained_baseline.py` | Baseline B end to end: extracts frozen XL-LEXEME and XLM-R (layer 8) usage vectors, clusters them (agglomerative with per-encoder distance-percentile thresholds, and K-means + silhouette), LOWO selection, shuffle control, Subtask 2 zero-shot scoring, writes official-format predictions. |
| `src/error_analysis.py` | Reads any system's official prediction files and writes the error-analysis tables (by word, sense rarity, period/era, period pair; Subtask 2 false positives/negatives) |
| `results/pretrained/` | Baseline B predictions (official format), `summary.json` (final threshold grid), `summary_v1_original_grid.json` (first run with the Baseline A grid, kept to document why the grid was changed). The `*_vectors.npy` embedding files are not stored; regenerate them with the command in step 3 |
| `results/error_analysis/` | CSV tables used in Section 9 of the report (`A_*` = Baseline A, `B_*` = Baseline B) |

### How to run (reproduction order)

Run everything from the repo root.

```bash
pip install -r requirements.txt            # includes torch and transformers

# 1. Check the scoring setup (trivial baselines: one cluster = 0.756, all-1 = 0.472)
python src/sanity_check.py

# 2. Baseline A: run notebooks/classical_baselineA.ipynb (writes results/classical/)

# 3. Baseline B: extracts the embeddings and runs clustering + scoring (GPU recommended, ~10-20 min)
python -m src.pretrained_baseline --data data --out results/pretrained

#    Clustering only (needs the *_vectors.npy files written by the command above):
python -m src.pretrained_baseline --data data --out results/pretrained --reuse-vectors results/pretrained

# 4. Error analysis (one call per system)
python -m src.error_analysis --data data --system B_xllex \
    --s1 results/pretrained/SV_subtask1_xllex_combined.jsonl \
    --s2 results/pretrained/SV_subtask2_xllex_cos_rankthr.jsonl --out results/error_analysis
python -m src.error_analysis --data data --system B_xllex_kmeans \
    --s1 results/pretrained/SV_subtask1_xllex_kmeans.jsonl --out results/error_analysis
python -m src.error_analysis --data data --system A_combined \
    --s1 results/classical/SV_subtask1_combined.jsonl \
    --s2 results/classical/SV_subtask2_LR_rank_thr_0.5.jsonl --out results/error_analysis
python -m src.error_analysis --data data --system A_kmeans \
    --s1 results/classical/SV_subtask1_kmeans.jsonl --out results/error_analysis
python -m src.error_analysis --data data --system A_tfidf \
    --s2 results/classical/SV_subtask2_tfidf_LR.jsonl --out results/error_analysis
```

Seed 42 everywhere. The embeddings were extracted once on Google Colab (T4 GPU) with PyTorch 2.11.0 and Transformers 5.18.0.

### Method in short

- **Encoders (frozen, nothing fine-tuned):** XL-LEXEME and XLM-R base (layer 8). The whole sentence is encoded with the target span marked (repaired offsets `start_fixed` / `end_fixed`), pre-1906 spelling is normalised, the sub-word vectors of the target are averaged, and sentences are truncated at 256 sub-words.
- **Subtask 1:** average-linkage agglomerative clustering on cosine distance. Thresholds are percentiles (5th … 90th) of the pooled within-word distances of each encoder, because the Baseline A grid (0.80–0.99) puts every word into one cluster for dense vectors. Also K-means with K in 2..10 chosen by silhouette. LOWO selection with three objectives (1a, 1b, combined).
- **Subtask 2:** zero-shot. The definition is encoded as the pseudo-usage "word: definition", each usage is scored by cosine similarity, the score is converted to a within-word percentile rank, and a rank threshold is fitted on nine words (LOWO).
- **Reported with every 1b ρ:** ρ on the 56 within-era pairs and the shuffle control (200 random reassignments that keep cluster sizes).

### Results (Swedish dev, LOWO)

| | Trivial baseline | Baseline A (selected) | Baseline B (XL-LEXEME) |
|---|---|---|---|
| 1a BCubed F1 | 0.756 (one cluster per word) | 0.697 | 0.760 |
| 1b Spearman ρ | undefined | 0.001 (K-means 0.242, shuffle control 0.168) | 0.492 (shuffle control 0.387 ± 0.042) |
| 2 macro F1 | 0.472 (predict all 1) | 0.472 | 0.468 |

Key findings:
- No classical system beats the trivial references. The only result above its control is the Baseline B ranking on Subtask 1b.
- Subtask 1a scores follow the share of the dominant sense; rare and minor senses are hidden by the macro average.
- XL-LEXEME splits *motiv* by sense but *herre* by era, and BCubed F1 cannot tell the two apart.
- Subtask 2 needs per-word calibration: a shared rank threshold gives about the same positive rate on every word.

### Notes and limitations

- XL-LEXEME was loaded with plain `transformers` because the official wrapper package did not work with the installed versions, so the pooling may differ slightly from the official one.
- No LLM prompting or cross-encoder was run for Subtask 2 (future work, Section 11 of the report).
- Only ten labelled Swedish words exist, so differences of a few hundredths are within noise.
