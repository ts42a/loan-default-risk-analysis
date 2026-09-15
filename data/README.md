# Dataset

The dataset is **not committed** to this repository: `data.csv` is ~317 MB, which is
above GitHub's 100 MB per-file limit.

## Getting the data

Place the file provided with the CSCI316 project brief here:

```
data/data.csv
```

Every script resolves that path relative to the repository root, so nothing else
needs to be configured.

## Expected shape

Verified against the file used for the reported results:

| Property | Value |
|---|---|
| Rows | 855,969 |
| Columns | 73 |
| Target column | `default_ind` |
| Class balance | 809,502 non-default / 46,467 default (**5.43% default rate**) |
| Missing values in target | 0 |
| Duplicate loan `id` values | 0 |
| Encoding / line endings | UTF-8, CRLF, no embedded newlines inside quoted fields |

`src/stage1_eda.py` re-checks the row count after parsing and aborts if Spark's
parsed row count disagrees with the raw line count, which catches the silent
mis-parse that occurs when a CSV extract *does* contain newlines inside the
free-text `desc` field.

## Feature documentation

The data dictionary for all 73 columns is in
[`docs/Project_Specification.pdf`](../docs/Project_Specification.pdf) (Appendix).

## Post-origination columns (target leakage)

These columns describe what happened *after* the loan was funded, so they cannot be
used to predict default at application time. Stage 1 flags them and Stage 2 never
loads them as predictors:

```
out_prncp, out_prncp_inv, total_pymnt, total_pymnt_inv, total_rec_prncp,
total_rec_int, total_rec_late_fee, recoveries, collection_recovery_fee,
last_pymnt_d, last_pymnt_amnt, next_pymnt_d, last_credit_pull_d
```
