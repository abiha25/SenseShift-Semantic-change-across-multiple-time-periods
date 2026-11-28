# Deviations from the Assignment 2 protocol

Every change to the experimental protocol of Assignment 2 (report Sections 1-3 and 7-10) is
recorded here, with the reason and its effect on the results. Additions that do not change any
existing result (new options, new scripts) are listed separately at the end.
Everyone adds their own entries; the paper summarises this file in Section 6.6.

## Changes to the protocol

### D1. Compute platform: Kaggle instead of Google Colab (Person 1, 24 Nov 2026)
- **What:** seeds 87, 1234 and 2027 of the main system and the frozen last-layer run (E2) ran on
  Kaggle (`notebooks/a3_person1_main_runs_kaggle.ipynb`). Seeds 42 and 13 (Assignment 2) ran on Colab.
- **Why:** no Colab GPU could be assigned on 24 Nov 2026.
- **Effect:** same GPU type (Tesla T4) and the same code. Library versions differ: Colab used
  PyTorch 2.14.1 and Transformers 5.22.1, Kaggle PyTorch 2.11.0 and Transformers 5.20.2. Both are
  recorded in every `run_info.json`. Small numerical differences between platforms are possible and
  cannot be separated from seed variation with five seeds. The Stage A training data was identical
  on both platforms (672,420 lines, 35,692 removed, 2,193 training words; see `training_history.json`).

### D2. Synthetic set stored as `sentences.bin` on Kaggle (Person 1, 24 Nov 2026)
- **What:** `sentences.zip` was uploaded to a private Kaggle dataset under the name `sentences.bin`
  and unpacked by the notebook.
- **Why:** Kaggle unpacks uploaded zip files and rejects four file names that contain an apostrophe
  (for example `devil's bird.jsonl`).
- **Effect:** none. The file content is unchanged and all 2,347 word files were read. The code reads
  the word from the `lemma` field, not from the file name. `metadata.zip` is not read by any code
  and is not needed.

### D3. Runs started from a git clone (Person 1, 24 Nov 2026)
- **What:** the Assignment 3 runs were started from a fresh clone of the repository at commit
  `6278e43`, with no local changes (`uncommitted_changes: false` in `run_info.json`).
- **Why:** the Assignment 2 runs copied the code from a Drive zip, so their `run_info.json` has
  `"git": null`. Section 10.3 of Assignment 2 requires the commit of every reported number.
- **Effect:** the commit is known for every Assignment 3 run. Seeds 42 and 13 keep `git: null`;
  their code is the Assignment 2 code that `run_pipeline verify` still reproduces exactly.

### D4. Name of the frozen last-layer run (E2) (Person 1, 24 Nov 2026)
- **What:** E2 runs as `proposed_frozen_last` (`--name frozen_last --set training.enabled=false
  --set model.layer=-1`).
- **Why:** every frozen run is otherwise named `proposed_frozen`, and the summary keeps only the
  latest run of a name, so the Assignment 2 layer-8 run would have disappeared from the tables.
- **Effect:** both frozen runs appear in `results/summary/` (layer 8: `proposed_frozen`; last
  layer: `proposed_frozen_last`).

### D5. Seed 87: Stage A collapsed; kept in the results (Person 1, 24 Nov 2026)
- **What happened:** in the run of seed 87, Stage A lowered the pair AUC on the held-out synthetic
  words from 0.608 to 0.511 (chance level). The other four seeds raised it to 0.735-0.759. Its final
  Stage A loss was 0.073 (others 0.059-0.061), its Stage B losses stayed near 0.064 in every fold
  (others fall to about 0.025), and its Subtask 2 threshold reached 1.0 in five folds. The vectors of
  this run carry almost no sense information: 1b rho -0.238, APD rho 0.055.
- **Decision:** seed 87 stays in all five-seed results exactly as it ran. It is not replaced by
  another seed and not rerun, because choosing seeds after seeing their results would bias the
  mean. The collapse is reported as a finding about training stability.
- **Not done:** a rerun of the seed-87 Stage A (to tell a reproducible instability from a
  non-deterministic GPU effect) and a Stage A run with a sixth seed were prepared
  (`notebooks/a3_person1_seed87_diagnosis_kaggle.ipynb`) but could not run before the deadline,
  because no Kaggle GPU became free. The cause of the collapse is therefore open.
- **Effect:** the standard deviations over seeds include this run. Under Assignment 2 the
  collapse was not caught because Stage A keeps its final weights by design (it only monitors
  the held-out AUC), while Stage B keeps its best epoch.

### D6. Stage A guard for the final configuration (Person 1, 24 Nov 2026)
- **What:** new key `training.stage_a_guard` (default `false`). With `true`, Stage A keeps its
  pre-training weights if training does not raise the pair AUC on the held-out synthetic words.
  It is switched on only in `configs/final.yaml`.
- **Why:** the official submission trains one model, so a collapse like seed 87 would decide the
  whole submission. The guard uses only held-out synthetic English words, never Swedish data.
- **Effect:** no reported result uses the guard (all runs have `stage_a_guard: false`). The default
  keeps the Stage A checkpoint key unchanged (pinned in `tests/test_final_config.py`).

### D7. Final configuration (E8) includes seed 87 (Person 1, 24 Nov 2026)
- **What:** `configs/final.yaml` fixes the Stage B epochs, t, m and tau to their medians over all
  50 folds of the five seeds (`python -m src.final_config`): 1 epoch, t = 0.2117, m = 30,
  tau = 0.7855. Per-fold values: `results/summary/final_config.json`.
- **Why:** the planned rule uses all five seeds; dropping one after seeing it would be a choice
  based on results.
- **Effect:** without seed 87 the medians are the same except tau (0.7814 instead of 0.7855).

### D8. GPU memory logged only on Kaggle (Person 1, 24 Nov 2026)
- **What:** peak GPU memory comes from `nvidia-smi`, logged every 5 s during the Kaggle runs
  (`results/compute/`). It was not logged for the Assignment 2 Colab runs.
- **Effect:** the compute table reports peak memory for E2 and seeds 87, 1234 and 2027 only
  (1.7 GiB frozen; 6.6-8.2 GiB fine-tuning).

## Additions (no existing result changes)

- **K-means clustering** (`clustering.method=kmeans`, K = 2-10 chosen by silhouette on the word's
  own vectors) for E4.
- **Pooling options** (`model.pooling` = `target_mean` | `first_subword` | `sentence_mean`) for E6.
  `sentence_mean` averages every real token, including the `<t>` `</t>` markers. A non-default
  pooling retrains Stage A.
- **JSD + APD combination** (`src/combine_change.py`) for E3: mean of the two percentile ranks per
  period pair, ranks taken over all 65 consecutive pairs (not only the 60 the scorer keeps).
- **Final configuration** script (`src/final_config.py`) and compute record (`results/compute/`).
- **Comparisons and subgroups over seeds** (`src/seed_stats.py`, Person 2): `results/summary/seed_comparisons.csv` (see D11) and `subgroups_seed_mean.csv` (mean, min and max over seeds of each run's `subgroups.csv`). **Figures** (`src/make_figures.py`, reads only `results/summary/`) in `results/figures/`.

## Further entries

<!-- Person 2, 3, 4: add your deviations below in the same format (D9, D10, ...). -->

### D9. Training ablations (E5-E7) on seeds 1234 and 2027 instead of 42 and 13 (Person 2, 28 Nov 2026)
- **What:** the ten training ablations run with seeds 1234 and 2027
  (`notebooks/Ablations_seed1234_kaggle.ipynb`, `notebooks/Ablations_seed2027_kaggle.ipynb`).
  The work plan named seeds 42 and 13.
- **Why:** an ablation is compared with the main run of the same seed. Seeds 42 and 13 ran on
  Colab (PyTorch 2.14.1, Transformers 5.22.1); the ablations can only run on Kaggle (no Colab GPU,
  see D1). Seeds 1234 and 2027 ran on Kaggle (PyTorch 2.11.0, Transformers 5.20.2), so with them
  the ablation and its reference differ only in the ablated setting, not in the platform. The
  choice was made before any ablation was run, from the platform alone; seed 87 was not
  considered because the plan uses two of the healthy reference runs.
- **Effect:** seeds 42 and 13 have no training ablations. Cell 7 of each notebook checks that
  Kaggle reproduces the reference run's Stage A (the `abl_nospell` run has the same Stage A key).
- **Result of the check:** not bit-identical on either seed. Held-out synthetic pair AUC before
  Stage A 0.6036 in all runs; after Stage A, seed 1234: 0.7469 (main run) vs 0.7487
  (`abl_nospell`), difference 0.0018; seed 2027: 0.7358 vs 0.7350, difference 0.0007. All runs
  used the same environment (Python 3.13.16, PyTorch 2.11.0+cu130,
  Transformers 5.20.2, Tesla T4), and the code between commit `6278e43` (main runs) and
  `30965fb` (ablations) leaves the default settings unchanged (the new options are off by
  default; the default pooling mask is the same code moved into `pooling_mask`). The
  difference is therefore GPU run-to-run non-determinism that `training.deterministic: true`
  does not remove. It is about a tenth of the spread of this AUC over seeds (0.735-0.759);
  ablation differences of this size are not attributed to the ablated setting.

### D10. Two seeds for the training ablations, five for the others (Person 2, 28 Nov 2026)
- **What:** E5-E7 use two seeds; E3 and E4 (no training: they reuse saved predictions or
  vectors) use all five seeds. The Assignment 2 protocol planned five seeds for every ablation.
- **Why:** five seeds for E5-E7 would be 50 runs of about 40 minutes; the time before the
  deadline allowed 20 (decided in the work plan of 21 Nov 2026).
- **Effect:** the training ablations are compared on two seeds only; Section 8.1 treats
  them as indicative, and a difference counts only under the decision rule.

### D11. Decision rule over several seeds (Person 2, 28 Nov 2026)
- **What:** `src/seed_stats.py` applies the rule of Assignment 2 Section 10.2 ("applied to the
  scores averaged over seeds"). For 1a and Subtask 2 the per-word scores are averaged over the
  paired seeds and the exact sign-flip test is run on the 10 per-word differences. For 1b and
  APD, rho is one number over 60 period pairs and has no per-word score to average, so each seed
  is tested on its own (exact per-word swap test) and a difference counts only if every seed
  meets the rule. Systems without any random choice (majority, classical, pretrained, the
  frozen-encoder runs) are compared with every seed of the main system.
- **Why:** Assignment 2 did not say how to average rho over seeds.
- **Effect:** `results/summary/seed_comparisons.csv`; the single-seed tests in
  `comparisons.csv` are unchanged.
