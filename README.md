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
   - `EDA/data_eda.ipynb`: the notebook that produced every number and figure

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
| 2 F1 (A1: old positive-class metric) | 0.472 | 0.472 (predict all 1) |

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

## Pretrained Baseline, Results and Error Analysis

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

# 1. Check the scoring setup (trivial baselines: one cluster = 0.756, all-1 = 0.236 with the 8 Nov 2026 official scorer)
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
| 2 F1 (A1: old positive-class metric) | 0.472 (predict all 1) | 0.472 | 0.468 |

Key findings:
- No classical system beats the trivial references. The only result above its control is the Baseline B ranking on Subtask 1b.
- Subtask 1a scores follow the share of the dominant sense; rare and minor senses are hidden by the macro average.
- XL-LEXEME splits *motiv* by sense but *herre* by era, and BCubed F1 cannot tell the two apart.
- Subtask 2 needs per-word calibration: a shared rank threshold gives about the same positive rate on every word.

### Notes and limitations

- XL-LEXEME was loaded with plain `transformers` because the official wrapper package did not work with the installed versions, so the pooling may differ slightly from the official one.
- No LLM prompting or cross-encoder was run for Subtask 2 (future work, Section 11 of the report).
- Only ten labelled Swedish words exist, so differences of a few hundredths are within noise.

---

## Assignment 2:

Everyone loads data, splits, settings and seeds through these files, so all
systems (baselines and the proposed method) use identical inputs.

| File | What it gives you |
|---|---|
| `data_loader.py` (bottom section) | `load_and_prepare()`: A1 loader + model inputs (`model_text`, `model_start`, `model_end`, `model_target`) + duplicate flags; `group_usages()`: `{word: {period: usages}}` in time order. Old A1 functions are unchanged. |
| `src/splits.py` → `data/splits/SV_splits.json` | Nested leave-one-word-out: per fold 1 test word, 1 validation word (early stopping / checkpoint choice), 8 training words. No word in two parts. Verified against a data fingerprint. |
| `configs/default.yaml` + `src/config.py` | Every setting in one file; `--set key=value` overrides; `save_run_config()` writes `config.yaml` + `run_info.json` (versions, GPU, git commit, data hash) next to each result. |
| `src/reproducibility.py` | `set_seed()` (Python, NumPy, PyTorch, deterministic GPU), DataLoader seeding, `save_checkpoint` / `load_checkpoint`, `CheckpointKeeper` (best-on-validation + last, early stopping). |
| `tests/test_data_pipeline.py` | 23 checks: counts, A1 compatibility, offsets, splits, config, seeds, checkpoints. |

```python
from data_loader import load_and_prepare, group_usages
from src.config import load_config
from src.splits import load_splits, fold_frames
from src.reproducibility import set_seed

cfg = load_config('configs/default.yaml')              # or overrides=['training.epochs=1']
set_seed(cfg['seed'], cfg['training']['deterministic'])
df = load_and_prepare(cfg['data']['data_dir'], cfg['data']['language'], cfg['preprocessing'])
splits = load_splits(cfg['split']['file'], df, cfg['data']['data_dir'])
for fold in splits['folds']:
    train, val, test = fold_frames(df, fold)   # fine-tune on train, select on val, predict test
```

Rules:
- Models read `model_text` / `model_start` / `model_end` (repaired offsets + preprocessing), never `start` / `end`.
- Never tune or pick checkpoints on the test word. Write the 10 test-word predictions into one file and score it once with `src/scoring.py`.
- Never cap the usages you predict for (`max_usages_per_period` is for training only): the scorer needs every `sentence_id`.
- Split, preprocessing and evaluation are FIXED. Tell the team before changing them.

```bash
pip install -r requirements.txt
python -m src.splits            # (re)checks / writes data/splits/SV_splits.json
python -m pytest tests -q       # all checks should pass
python src/sanity_check.py      # one cluster per word = 0.756, all-1 = 0.236 (new official metric)
```

### Evaluation script - use it for EVERY result in the report

The official scorer was updated by the organisers on 8 Nov 2026 (commit `0e69238`) and
`evaluation.py` / `evaluation_utils.py` in this repo are now that version:

- **Subtask 2 is now macro F1 over both classes per word** (then averaged over words).
  Predicting all 1 now scores 0.236 (it was 0.472). Assignment 1 Subtask 2 numbers
  were computed with the old metric; `src/evaluate.py` still prints the old one as
  "positive-class F1 (A1 metric)" so both can be compared.
- Subtask 1a: precision and recall are now named correctly by the official code, so the
  old swap in `src/scoring.py` was removed. F1 is unchanged (one cluster = 0.756).
- Subtask 1b: a constant prediction (e.g. one cluster per word) now gets rho = 0.

```bash
# score a run (writes scores.json, per_word.csv [, subgroups.csv] next to the predictions)
python -m src.evaluate score --name <system> \
    --s1 results/runs/<run>/SV_subtask1.jsonl \
    --s2 results/runs/<run>/SV_subtask2.jsonl \
    [--graded results/runs/<run>/SV_graded_change.jsonl] [--subgroups]

python -m src.evaluate check --s1 FILE --s2 FILE                 # format check only
python -m src.evaluate compare A/scores.json B/scores.json --metric 1a   # 1a | 1b | 2 | graded
python -m src.evaluate summarise seed13/scores.json seed42/scores.json ...  # mean +- sd
```

```python
from src.evaluate import evaluate_predictions, write_graded
res = evaluate_predictions(s1=..., s2=..., graded=..., name='proposed', out_dir=run_dir)
```

It prints and saves: a format check (errors = the official scorer would reject the file;
warnings for missing or extra words), the three official metrics with 95% bootstrap
confidence intervals over words, secondary metrics (BCubed P/R, cluster counts, Pearson r,
within-era rho, cross-era tracking, 1b shuffle control; accuracy, balanced accuracy and
positive-class F1 for Subtask 2), per-word scores, and with `--subgroups` the scores by
era/period, word frequency, part of speech, amount of gold change and time coverage.

- **Graded change (APD etc.)**: save one score per scored consecutive period pair with
  `write_graded(rows, path)` (rows: `word`, `period1`, `period2`, `score`). It is scored
  against gold JSD on the same 60 pairs. This is a LOCAL score: the official 1b score is
  always computed from the submitted cluster labels.
- **compare** implements Section 10.2: exact paired sign-flip test (1a, 2) or per-word swap
  test (1b, graded) plus a bootstrap CI of the difference. Call a system better only if
  the CI excludes 0 and p < 0.05.
- Do not change `src/evaluate.py`, `src/scoring.py` or the official files without telling
  the team.

### Comparison baselines

`src/baselines.py` runs the three comparison systems on the shared split and scores each with
`src/evaluate.py`. Colab: `notebooks/baselines.ipynb` (T4 GPU, about 10 minutes).

```bash
python -m src.baselines --which simple classical    # majority, random (5 seeds), classical
python -m src.baselines --which pretrained          # frozen XLM-R + APD (GPU recommended)
python -m src.baselines --table                     # results/baselines/summary.csv and .md
```

| System | Subtask 1 (1a, 1b) | Graded change | Subtask 2 |
|---|---|---|---|
| majority | one cluster per word | (constant: rho = 0) | majority 0/1 label of the 9 non-test words |
| random (5 seeds) | random clusters, K = median senses of the 9 non-test words | random | random at their positive rate |
| classical | A1 Baseline A: TF-IDF context + average-linkage clustering | - | LR on usage-definition similarity ranks |
| pretrained | frozen XLM-R layer 8, target mean, agglomerative clustering | APD | usage-definition cosine rank |

- For each fold the test word is predicted once; every setting, threshold or label is chosen on the
  other nine words (training + validation word). The baselines have no training loop, so nothing is
  early-stopped.
- Subtask 2 thresholds are tuned on the new official metric (macro F1), not the A1 metric.
- XLM-R vectors are cached in `cache/` (not committed).


## Proposed System (Assignment 2)

Fine-tuned XLM-R usage encoder (XL-LEXEME style) + agglomerative sense clustering + JSD/APD change scores.
Report: Sections 1-3. Runs on the shared split and is scored by `src/evaluate.py` like every baseline.

| File | What it does |
|---|---|
| `src/usage_encoder.py` | `UsageEncoder` (XLM-R, target marked `<t> ... </t>`, mean of the target's sub-word vectors), synthetic + Swedish training items, era-aware `PairSampler` with soft labels (sense overlap), contrastive loss, training loop with early stopping |
| `src/proposed.py` | `extract_embeddings()`, `cluster_senses()`, `score_change()`, Subtask 2, `predict()`; the fold loop (`run`) and `recluster` (new clustering settings on saved vectors) |
| `notebooks/proposed_model.ipynb` | Colab notebook: set-up, quick check, full run, frozen ablation, clustering ablations, results table |
| `tests/test_proposed.py` | Tests that need no model download |
| `data/external/README.md` | Where to put the organisers' synthetic English set (training only) |

Training: **Stage A** synthetic English (once per seed, shared by all folds, no Swedish data) ->
**Stage B** the 8 training words of each fold, early stopping on the validation word (pair AUC).
The test word is never used for training or for any choice. Settings: `configs/default.yaml`
(`model`, `training`, `clustering`, `subtask2`).

```bash
python -m src.proposed run                                     # full system (GPU)
python -m src.proposed run --set training.enabled=false        # frozen encoder, same pipeline
python -m src.proposed recluster --run results/runs/<dir> --set clustering.threshold_selection=fixed
```
