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

## Further entries

<!-- Person 2, 3, 4: add your deviations below in the same format (D9, D10, ...). -->
