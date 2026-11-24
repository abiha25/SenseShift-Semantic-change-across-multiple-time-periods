# Compute cost of the proposed system

`compute.csv` lists, for every run of the proposed system and the frozen-encoder runs:
platform, GPU, Python / PyTorch / Transformers versions (from `run_info.json`), Stage A and
total minutes (from `selection.json`), and the peak GPU memory.

Peak GPU memory was logged with `nvidia-smi` every 5 seconds during the Kaggle runs
(`gpu_logs/<run>.csv`, columns: timestamp, MiB used on GPU 0). It was not logged for the
two Colab runs of Assignment 2 (seeds 42 and 13).
