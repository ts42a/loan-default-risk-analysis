# Loan Default Risk Analysis

CSCI316 Big Data Mining Techniques and Implementation — Autumn 2026 group project.

End-to-end workflow that predicts `default_ind` (1 = borrower defaulted, 0 = did not) on a Lending Club loan extract. The assignment requires two stages:

1. **Stage 1** — exploratory analysis with Apache Spark only (no pandas / scikit-learn).
2. **Stage 2** — two independent modelling processes on the same origination-safe features:
   - Process One: three Spark MLlib classifiers
   - Process Two: three TensorFlow / Keras networks

Every figure and table used in the slides is produced by the three scripts under `src/`.

## Repository layout

```
loan-default-risk-analysis/
├── src/
│   ├── stage1_eda.py            # Stage 1 — Spark EDA + cleaning
│   ├── stage2_spark.py          # Stage 2 Process One — Spark MLlib
│   ├── stage2_tensorflow.py     # Stage 2 Process Two — TensorFlow/Keras
│   └── run_paths.py             # timestamped runs, runs.log, FINAL pin
├── notebooks/                   # thin wrappers around the src/ scripts
├── data/                        # place data.csv here (not committed)
├── results/
│   ├── stage1/<stamp>_full/     # plus final/ for slides
│   ├── spark/<stamp>_full/
│   └── tensorflow/<stamp>_full/
├── slides/                      # presentation PDF / PPTX
├── docs/                        # assignment PDFs
├── requirements.txt
└── .gitignore
```

`data/data.csv` is ~317 MB, which is over GitHub's 100 MB file limit, so it is gitignored. See [`data/README.md`](data/README.md).

## Problem

The target is loan default at origination time. Columns that only exist after the loan is funded (`total_pymnt`, `recoveries`, `last_pymnt_amnt`, outstanding principal, …) are **never** used as predictors. Using them would leak the answer.

The class is heavily imbalanced (~5.4% default). Accuracy is reported, but ranking uses **PR-AUC** and default-class recall / precision / F1 at a threshold chosen on the validation fold.

## Setup

Python **3.11 or 3.12** is required. PySpark 4 and TensorFlow 2.19 do not publish wheels for 3.14 (the Windows `py` launcher default on this machine).

Stage 1 and Stage 2 Process One also need a **JDK 11 or 17** on `PATH` (Temurin: https://adoptium.net/).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place the dataset at `data/data.csv`.

## Run

Scripts resolve the repository root from `__file__`, so they can be launched from any working directory.

```powershell
python src/stage1_eda.py          # once — writes results/stage1/data/data_clean.csv
python src/stage2_spark.py        # samples that file via STAGE2_SAMPLE
python src/stage2_tensorflow.py
```

Useful environment switches:

| Variable | Default | Meaning |
|---|---|---|
| `STAGE1_SAMPLE` | `1.0` | Fraction of rows after the parser check, e.g. `0.1` |
| `STAGE2_TUNE` | `1` | Set `0` to skip Spark CrossValidator (much faster) |
| `STAGE2_SAMPLE` | `1.0` | Fraction of rows for a Stage 2 smoke test, e.g. `0.1` |
| `STAGE2_EPOCHS` | `60` | Keras max epochs (early stopping usually cuts this short) |
| `STAGE2_BATCH_SIZE` | `1024` | Keras batch size |
| `STAGE2_DRIVER_MEMORY` | `8g` | Spark driver heap for Process One |
| `STAGE_PIN_FINAL` | auto | `1` force-pin this run as `final/`; `0` skip pinning |
| `STAGE_USE_FINAL` | off | `1` makes Stage 2 read the pinned full Stage 1/Spark run |

Example of a quick Spark development run:

```powershell
$env:STAGE2_TUNE = "0"
$env:STAGE2_SAMPLE = "0.05"
python src/stage2_spark.py
```

## Pipeline

**Stage 1 (`src/stage1_eda.py`)** — Spark DataFrame / RDD APIs only:

- parser sanity check (parsed rows vs raw CSV lines)
- duplicates, missing values, summary statistics
- class balance
- type casting, leakage / free-text / sparse-column drop
- numeric distributions, correlations, categorical default rates
- 7 most and 7 least relevant attributes with written justification
- export `results/stage1/data/data_clean.csv` (stable file; Stage 2 reuses it)

**Stage 2 Process One (`src/stage2_spark.py`)** — `pyspark.ml`:

- Logistic Regression, Random Forest, Gradient-Boosted Trees
- stratified 64 / 16 / 20 split, class weights, train-only imputation and encoding
- optional CrossValidator on PR-AUC
- threshold tuned on validation F1 of the default class, then evaluated once on test
- ROC / PR curves, confusion matrices, feature importance

**Stage 2 Process Two (`src/stage2_tensorflow.py`)** — TensorFlow / Keras:

- LogisticNN (single sigmoid unit), MLP (64/32 + dropout), DeepMLP (128/64/32 + batch norm)
- same feature contract and split fractions as Process One
- class weights, output-bias initialisation, early stopping on validation PR-AUC
- same threshold protocol and metrics; comparison chart if Spark results already exist

## Team

Fill in before Moodle submission (first slide of the presentation must match this table).

| Name | Student number | Contribution |
|---|---|---|
| | | |

## Licence

Coursework for CSCI316. Dataset is provided with the project brief and is not redistributed here.
