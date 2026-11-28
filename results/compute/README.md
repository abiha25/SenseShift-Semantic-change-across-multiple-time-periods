# Compute cost of the proposed system

`compute.csv` lists, for every run of the proposed system and the frozen-encoder runs:
platform, GPU, Python / PyTorch / Transformers versions (from `run_info.json`), Stage A and
total minutes (from `selection.json`), and the peak GPU memory.

Peak GPU memory was logged with `nvidia-smi` every 5 seconds during the Kaggle runs
(`gpu_logs/<run>.csv`, columns: timestamp, MiB used on GPU 0). It was not logged for the
two Colab runs of Assignment 2 (seeds 42 and 13).

## Ablation runs E5-E7 (Assignment 3, Person 2)

The 20 rows `*_proposed_abl_*` were added from each run's `run_info.json` and `selection.json`
(`timing`) and the `nvidia-smi` logs in `ablations_seed1234/` and `ablations_seed2027/`
(`<ablation>_gpu.csv`: timestamp, MiB used on the run's own GPU; `<ablation>.log`: the run's output).
Both notebooks ran two queues at once, one per T4 GPU of a Kaggle "GPU T4 x2" session.
Stage A minutes: `abl_nostagea` only loads the base model; `abl_nostageb` reuses the Stage A
checkpoint of `abl_nospell` (same Stage A key) and trains nothing, so it only encodes.
