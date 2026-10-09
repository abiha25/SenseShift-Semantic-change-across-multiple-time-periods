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
