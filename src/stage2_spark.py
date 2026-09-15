#!/usr/bin/env python3
"""CSCI316 Stage 2, Process One — Spark MLlib classifiers for loan default.

Three classifiers from ``pyspark.ml`` are trained on the origination-safe feature
set exported by Stage 1:

1. Logistic Regression  — linear baseline on standardised features
2. Random Forest        — bagged trees, captures non-linear interactions
3. Gradient-Boosted Trees — sequential boosting, usually the strongest on tabular credit data

Design points that matter for a 5% positive class:

* the split is **stratified** 64/16/20, so every fold keeps the same default rate;
* missing values are imputed with a median learned on the **training fold only**,
  with a companion missing-value indicator, so no test information leaks back;
* the decision threshold is chosen on the **validation** fold (max F1 on the
  default class) and only then applied to the untouched test fold;
* PR-AUC is reported alongside ROC-AUC because accuracy is meaningless here.

Run:

    python src/stage2_spark.py            # full run with cross-validation
    STAGE2_TUNE=0 python src/stage2_spark.py    # skip CrossValidator (much faster)

Inputs   : latest results/stage1/<stamp>/data/data_clean.csv
Outputs  : results/spark/<YYYYMMDD_HHMMSS>/ plus results/spark/runs.log
"""

from __future__ import annotations

import csv
import functools
import os
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyspark import StorageLevel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import GBTClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import Imputer, OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

import run_paths

# NumPy 2 renamed trapz; keep working on either major version.
trapezoid = getattr(np, "trapezoid", None) or np.trapz

RANDOM_STATE = 192
LABEL_COL = "default_ind"
SPLIT = (0.64, 0.16, 0.20)

# Must match the columns exported by src/stage1_eda.py.
BASE_NUMERIC = [
    "loan_amnt", "term", "int_rate", "installment", "annual_inc", "dti",
    "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc", "collections_12_mths_ex_med", "acc_now_delinq",
    "tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim",
    "credit_history_years", "grade_ord", "emp_length_years",
]
ENGINEERED_NUMERIC = [
    "loan_to_income", "installment_to_income", "revol_util_x_dti",
    "open_to_total_acc", "bal_to_limit", "int_rate_x_term",
]
CATEGORICAL = [
    "sub_grade", "home_ownership", "purpose", "verification_status",
    "addr_state", "application_type",
]
# The credit-bureau block is absent for older loans as a group; one shared flag
# records that fact so the models can distinguish "zero" from "not reported".
MISSING_FLAG_SOURCE = "tot_cur_bal"

ALL_NUMERIC = BASE_NUMERIC + ENGINEERED_NUMERIC

# Environment switches (documented in README.md).
ENABLE_TUNE = os.environ.get("STAGE2_TUNE", "1") != "0"
SAMPLE_FRAC = float(os.environ.get("STAGE2_SAMPLE", "1.0"))
CV_METRIC = os.environ.get("STAGE2_CV_METRIC", "areaUnderPR")
THRESHOLD_STEP = float(os.environ.get("STAGE2_THRESHOLD_STEP", "0.005"))

COLOURS = {"LogisticRegression": "#3498db", "RandomForest": "#27ae60", "GBT": "#9b59b6"}


def resolve_root() -> Path:
    try:
        root = Path(__file__).resolve().parents[1]
    except NameError:
        root = Path.cwd()
    if run_paths.clean_loan_path(root) is not None:
        return root
    for candidate in (Path.cwd(), Path.cwd().parent, Path("/content/loan-default-risk-analysis")):
        if run_paths.clean_loan_path(candidate) is not None:
            return candidate.resolve()
    return root


ROOT = resolve_root()
DATA_PATH = run_paths.clean_loan_path(ROOT) or (ROOT / "results" / "stage1" / "data" / run_paths.CLEAN_CSV)
OUT_DIR = ROOT / "results" / "spark"


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
def find_jdk() -> Path | None:
    """Locate a JDK 11/17. Prefers a portable copy under the repo `.jdk/` folder."""
    bases = [
        ROOT / ".jdk",
        Path(r"C:\Program Files\Eclipse Adoptium"),
        Path(r"C:\Program Files\Java"),
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Eclipse Adoptium",
    ]
    for base in bases:
        if not base.is_dir():
            continue
        for jdk in sorted(base.glob("jdk*"), reverse=True):
            if (jdk / "bin" / "java.exe").is_file():
                return jdk
    return None


def start_spark() -> SparkSession:
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    jdk = find_jdk()
    if jdk is not None:
        os.environ["JAVA_HOME"] = str(jdk)
        os.environ["PATH"] = str(jdk / "bin") + os.pathsep + os.environ.get("PATH", "")
    elif not shutil.which("java"):
        raise RuntimeError(
            "Java was not found. Install Temurin JDK 17 from https://adoptium.net/ "
            "or place a portable JDK under .jdk/ in this repository, then re-run."
        )
    if "_" in socket.gethostname():
        os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

    java_banner = subprocess.check_output(
        ["java", "-version"], stderr=subprocess.STDOUT, text=True
    ).splitlines()[0]
    spark_tmp = ROOT / ".spark-tmp"
    spark_tmp.mkdir(exist_ok=True)
    spark = (
        SparkSession.builder.appName("CSCI316_Stage2_SparkMLlib")
        .master("local[*]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.memory", os.environ.get("STAGE2_DRIVER_MEMORY", "8g"))
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.local.dir", str(spark_tmp))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    print(f"Java: {java_banner}")
    print(f"Spark {spark.version}  |  root: {ROOT}")
    return spark


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path.relative_to(ROOT)}")


def print_table(rows: list[dict], columns: list[str], *, pct_cols: tuple = ()) -> None:
    view = []
    for row in rows:
        rendered = {}
        for column in columns:
            value = row.get(column)
            if column in pct_cols and value is not None:
                rendered[column] = f"{float(value) * 100:.2f}%"
            elif isinstance(value, float):
                rendered[column] = f"{value:.4f}"
            elif isinstance(value, int):
                rendered[column] = f"{value:,}"
            else:
                rendered[column] = "" if value is None else str(value)
        view.append(rendered)
    if not view:
        return
    widths = [max(len(c), max(len(r[c]) for r in view)) for c in columns]
    line = "  ".join(f"{{:<{w}}}" for w in widths)
    print(line.format(*columns))
    print(line.format(*["-" * w for w in widths]))
    for row in view:
        print(line.format(*[row[c] for c in columns]))


def save_fig(fig, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def load_model_frame(spark: SparkSession) -> DataFrame:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"{DATA_PATH} is missing. Run `python src/stage1_eda.py` first — Stage 2 consumes "
            "the cleaned dataset that Stage 1 exports."
        )
    df = spark.read.csv(str(DATA_PATH), header=True, inferSchema=True)
    missing = [c for c in ALL_NUMERIC + CATEGORICAL + [LABEL_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"data_clean.csv is missing expected columns: {missing}")

    for column in ALL_NUMERIC + [LABEL_COL]:
        df = df.withColumn(column, F.col(column).try_cast("double"))
    for column in CATEGORICAL:
        df = df.withColumn(
            column,
            F.when(F.col(column).isNull() | (F.trim(F.col(column)) == ""), F.lit("UNKNOWN"))
            .otherwise(F.col(column).cast("string")),
        )
    df = df.withColumn(
        "bureau_data_missing",
        F.when(F.col(MISSING_FLAG_SOURCE).isNull(), F.lit(1.0)).otherwise(F.lit(0.0)),
    )
    df = df.dropna(subset=[LABEL_COL])
    return df.select(*(ALL_NUMERIC + ["bureau_data_missing"] + CATEGORICAL + [LABEL_COL]))


def stratified_split(
    df: DataFrame, fracs: tuple[float, float, float], seed: int
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Split each class separately so the default rate is preserved everywhere."""
    per_split: list[list[DataFrame]] = [[], [], []]
    for label in (0.0, 1.0):
        subset = df.filter(F.col(LABEL_COL) == label)
        for index, piece in enumerate(subset.randomSplit(list(fracs), seed=seed)):
            per_split[index].append(piece)
    train, validation, test = (
        functools.reduce(DataFrame.unionByName, pieces) for pieces in per_split
    )
    return train, validation, test


def add_class_weights(df: DataFrame) -> tuple[DataFrame, float, int, int]:
    """Balanced weights: every default counts as (n_negative / n_positive) rows."""
    counts = {int(r[LABEL_COL]): int(r["count"]) for r in df.groupBy(LABEL_COL).count().collect()}
    negatives, positives = counts.get(0, 0), counts.get(1, 0)
    if positives == 0:
        raise ValueError("The training data contains no defaults.")
    ratio = negatives / positives
    weighted = df.withColumn(
        "weight", F.when(F.col(LABEL_COL) == 1.0, F.lit(ratio)).otherwise(F.lit(1.0))
    )
    return weighted, positives / (negatives + positives), negatives, positives


def build_preprocessor() -> Pipeline:
    """Impute -> index -> one-hot -> assemble. Fitted on the training fold only."""
    imputed = [f"{c}_imp" for c in ALL_NUMERIC]
    stages: list = [
        Imputer(inputCols=ALL_NUMERIC, outputCols=imputed, strategy="median")
    ]
    stages += [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in CATEGORICAL
    ]
    # handleInvalid="keep" on both stages means a category that only appears in
    # validation/test is encoded as an all-zero block instead of aborting the run.
    stages.append(
        OneHotEncoder(
            inputCols=[f"{c}_idx" for c in CATEGORICAL],
            outputCols=[f"{c}_ohe" for c in CATEGORICAL],
            dropLast=True,
            handleInvalid="keep",
        )
    )
    stages.append(
        VectorAssembler(
            inputCols=imputed + ["bureau_data_missing"] + [f"{c}_ohe" for c in CATEGORICAL],
            outputCol="features",
            handleInvalid="keep",
        )
    )
    return Pipeline(stages=stages)


def feature_names(model: PipelineModel) -> list[str]:
    """Expand the assembled vector back into readable column names."""
    labels = {stage.getOutputCol(): list(stage.labels) for stage in model.stages if hasattr(stage, "labels")}
    assembler = next(s for s in model.stages if isinstance(s, VectorAssembler))
    names: list[str] = []
    for column in assembler.getInputCols():
        if column.endswith("_ohe"):
            source = column[: -len("_ohe")]
            categories = labels.get(f"{source}_idx", [])
            # With handleInvalid="keep" the encoder appends an extra "unseen"
            # category and dropLast removes that one, so every known label keeps
            # a column of its own.
            names += [f"{source}={value}" for value in categories]
        elif column.endswith("_imp"):
            names.append(column[: -len("_imp")])
        else:
            names.append(column)
    return names


# --------------------------------------------------------------------------- #
# Metrics (numpy only — this process deliberately avoids scikit-learn)
# --------------------------------------------------------------------------- #
def collect_scores(predictions: DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Pull (label, P(default)) pairs to the driver for metric computation."""
    rows = predictions.select(
        F.col(LABEL_COL).cast("int").alias("y"),
        vector_to_array(F.col("probability"))[1].alias("p"),
    ).collect()
    y = np.fromiter((r["y"] for r in rows), dtype=np.int8, count=len(rows))
    p = np.fromiter((r["p"] for r in rows), dtype=np.float64, count=len(rows))
    return y, p


def roc_curve(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-scores)
    labels = y[order].astype(np.float64)
    positives, negatives = labels.sum(), len(labels) - labels.sum()
    if positives == 0 or negatives == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5
    tpr = np.concatenate([[0.0], np.cumsum(labels) / positives])
    fpr = np.concatenate([[0.0], np.cumsum(1.0 - labels) / negatives])
    return fpr, tpr, float(trapezoid(tpr, fpr))


def pr_curve(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Precision/recall plus average precision (the step-wise PR-AUC)."""
    order = np.argsort(-scores)
    labels = y[order].astype(np.float64)
    positives = labels.sum()
    if positives == 0:
        return np.array([0.0]), np.array([0.0]), 0.0
    tp = np.cumsum(labels)
    precision = tp / np.arange(1, len(labels) + 1)
    recall = tp / positives
    average_precision = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))
    return recall, precision, average_precision


def counts_at(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, int]:
    predicted = scores >= threshold
    return {
        "tp": int(np.sum(predicted & (y == 1))),
        "fp": int(np.sum(predicted & (y == 0))),
        "tn": int(np.sum(~predicted & (y == 0))),
        "fn": int(np.sum(~predicted & (y == 1))),
    }


def metrics_at(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    cm = counts_at(y, scores, threshold)
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / len(y),
        "precision_default": precision,
        "recall_default": recall,
        "f1_default": f1,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        **cm,
    }


def best_threshold(y: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Threshold maximising F1 on the default class, searched on validation data."""
    grid = np.arange(0.05, 0.951, THRESHOLD_STEP)
    best, best_f1 = 0.5, -1.0
    for threshold in grid:
        score = metrics_at(y, scores, float(threshold))["f1_default"]
        if score > best_f1:
            best, best_f1 = float(threshold), score
    return best, best_f1


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def plot_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], scores: dict[str, float],
                *, xlabel: str, ylabel: str, title: str, filename: str, baseline: float | None) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (x, y) in curves.items():
        ax.plot(x, y, label=f"{name} ({scores[name]:.4f})", color=COLOURS.get(name), linewidth=2)
    if baseline is None:
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="random (0.5000)")
    else:
        ax.axhline(baseline, color="black", linestyle="--", linewidth=1, label=f"random ({baseline:.4f})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, filename)


def plot_confusion(cm: dict[str, int], name: str, threshold: float, filename: str) -> None:
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], dtype=float)
    shares = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    image = ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1], ["predicted\nno default", "predicted\ndefault"])
    ax.set_yticks([0, 1], ["actual\nno default", "actual\ndefault"])
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{int(matrix[i, j]):,}\n{shares[i, j]:.1%} of row",
                ha="center", va="center", fontsize=10,
                color="white" if shares[i, j] > 0.5 else "black",
            )
    ax.set_title(f"{name} — test set at threshold {threshold:.3f}")
    fig.colorbar(image, ax=ax, fraction=0.046, label="share of actual class")
    fig.tight_layout()
    save_fig(fig, filename)


def plot_importance(names: list[str], values: np.ndarray, name: str, filename: str, top_n: int = 20) -> list[dict]:
    order = np.argsort(-np.abs(values))[:top_n]
    rows = [{"feature": names[i], "importance": float(values[i])} for i in order]
    fig, ax = plt.subplots(figsize=(8, max(4.0, 0.32 * len(rows))))
    ax.barh([r["feature"] for r in rows][::-1], [r["importance"] for r in rows][::-1], color=COLOURS.get(name, "#27ae60"))
    ax.set_xlabel("importance")
    ax.set_title(f"Top {len(rows)} features — {name}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, filename)
    return rows


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def fit_model(estimator, train: DataFrame, grid, name: str):
    if not ENABLE_TUNE or not grid:
        print(f"  Fitting {name} with fixed hyper-parameters...")
        return estimator.fit(train), None
    # The imputer/encoder/scaler are fitted once on the whole training fold rather
    # than inside each CV fold. That is the usual trade-off for tractable runtimes;
    # it can flatter the CV score slightly, but the test fold stays untouched, so
    # the reported metrics are unaffected.
    folds = 3
    print(f"  Cross-validating {name}: {len(grid)} candidates x {folds} folds, metric={CV_METRIC}")
    cv = CrossValidator(
        estimator=estimator,
        estimatorParamMaps=grid,
        evaluator=BinaryClassificationEvaluator(
            labelCol=LABEL_COL, rawPredictionCol="rawPrediction", metricName=CV_METRIC
        ),
        numFolds=folds,
        seed=RANDOM_STATE,
        parallelism=2,
        collectSubModels=False,
    )
    model = cv.fit(train)
    best = model.bestModel
    chosen = {
        param.name: value
        for param, value in grid[int(np.argmax(model.avgMetrics))].items()
    }
    print(f"  Best {CV_METRIC}={max(model.avgMetrics):.4f} with {chosen}")
    return best, chosen


def main() -> None:
    global OUT_DIR, DATA_PATH
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    found = run_paths.clean_loan_path(ROOT)
    if found is None:
        raise FileNotFoundError(
            "No Stage 1 cleaned dataset found. Run `python src/stage1_eda.py` first."
        )
    DATA_PATH = found
    note = f"tune={int(ENABLE_TUNE)} sample={SAMPLE_FRAC}"
    run_dir = run_paths.start_run(ROOT, "spark", sample=SAMPLE_FRAC, note=note)
    OUT_DIR = run_dir
    plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "figure.facecolor": "white"})

    status = "failed"
    try:
        _run_spark()
        status = "ok"
    finally:
        run_paths.finish_run(ROOT, "spark", run_dir, status=status, sample=SAMPLE_FRAC)


def _run_spark() -> None:

    spark = start_spark()
    section("2.1 Configuration")
    print(f"Started         : {datetime.now():%Y-%m-%d %H:%M}")
    print(f"Input           : {DATA_PATH.relative_to(ROOT)}")
    print(f"Output          : {OUT_DIR.relative_to(ROOT)}/")
    print(f"Cross-validation: {'on' if ENABLE_TUNE else 'off (STAGE2_TUNE=0)'}  |  metric: {CV_METRIC}")
    print(f"Threshold step  : {THRESHOLD_STEP}  |  seed: {RANDOM_STATE}")

    # ----------------------------------------------------------------- 2.2 ---
    section("2.2 Load features")
    model_df = load_model_frame(spark)
    if SAMPLE_FRAC < 1.0:
        model_df = model_df.sample(withReplacement=False, fraction=SAMPLE_FRAC, seed=RANDOM_STATE)
        print(f"Development sample: STAGE2_SAMPLE={SAMPLE_FRAC}")
    model_df = model_df.persist(StorageLevel.MEMORY_AND_DISK)
    n_total = model_df.count()
    print(f"Rows: {n_total:,}")
    print(f"Numeric features ({len(ALL_NUMERIC)} + 1 missing flag): {', '.join(ALL_NUMERIC)}")
    print(f"Categorical features ({len(CATEGORICAL)}): {', '.join(CATEGORICAL)}")
    print("No post-origination payment column is loaded, so target leakage is structurally impossible.")

    # ----------------------------------------------------------------- 2.3 ---
    section("2.3 Stratified split and class weighting")
    train_df, val_df, test_df = stratified_split(model_df, SPLIT, RANDOM_STATE)
    train_df, train_rate, negatives, positives = add_class_weights(train_df)
    train_df = train_df.persist(StorageLevel.MEMORY_AND_DISK)
    val_df = val_df.persist(StorageLevel.MEMORY_AND_DISK)
    test_df = test_df.persist(StorageLevel.MEMORY_AND_DISK)
    n_train, n_val, n_test = train_df.count(), val_df.count(), test_df.count()
    split_rows = []
    for name, frame, count in (("train", train_df, n_train), ("validation", val_df, n_val), ("test", test_df, n_test)):
        rate = float(frame.select(F.avg(F.col(LABEL_COL))).collect()[0][0])
        split_rows.append({"split": name, "rows": count, "default_rate": rate})
    print_table(split_rows, ["split", "rows", "default_rate"], pct_cols=("default_rate",))
    print(f"Class weight applied to defaults: {negatives / positives:.2f} (={negatives:,}/{positives:,})")
    write_csv(OUT_DIR / "split_summary.csv", split_rows, ["split", "rows", "default_rate"])

    # ----------------------------------------------------------------- 2.4 ---
    section("2.4 Feature pipeline")
    preprocessor = build_preprocessor().fit(train_df)
    train_feat = preprocessor.transform(train_df).persist(StorageLevel.MEMORY_AND_DISK)
    val_feat = preprocessor.transform(val_df).persist(StorageLevel.MEMORY_AND_DISK)
    test_feat = preprocessor.transform(test_df).persist(StorageLevel.MEMORY_AND_DISK)
    names = feature_names(preprocessor)
    print(f"Assembled vector width: {len(names)} (median imputation and one-hot encoding fitted on train only)")

    # withMean=False keeps the one-hot block sparse; centring would densify a
    # ~120-column vector across every row for no benefit to the linear model.
    scaler = StandardScaler(
        inputCol="features", outputCol="scaled_features", withMean=False, withStd=True
    ).fit(train_feat)
    train_scaled = scaler.transform(train_feat).persist(StorageLevel.MEMORY_AND_DISK)
    val_scaled = scaler.transform(val_feat)
    test_scaled = scaler.transform(test_feat)
    train_scaled.count()

    results: list[dict] = []
    roc_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    roc_scores: dict[str, float] = {}
    prc_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    prc_scores: dict[str, float] = {}
    chosen_params: list[dict] = []

    def evaluate(name: str, val_predictions: DataFrame, test_predictions: DataFrame) -> dict:
        y_val, p_val = collect_scores(val_predictions)
        threshold, val_f1 = best_threshold(y_val, p_val)
        y_test, p_test = collect_scores(test_predictions)

        fpr, tpr, auc = roc_curve(y_test, p_test)
        recall, precision, average_precision = pr_curve(y_test, p_test)
        roc_curves[name], roc_scores[name] = (fpr, tpr), auc
        prc_curves[name], prc_scores[name] = (recall, precision), average_precision

        default = metrics_at(y_test, p_test, 0.5)
        tuned = metrics_at(y_test, p_test, threshold)
        row = {
            "model": name,
            "auc_roc": auc,
            "auc_pr": average_precision,
            "val_f1_default": val_f1,
            "tuned_threshold": threshold,
            "train_rows": n_train,
            "test_rows": n_test,
            **{f"{k}_at_0.5": v for k, v in default.items() if k != "threshold"},
            **{f"{k}_tuned": v for k, v in tuned.items() if k != "threshold"},
        }
        print(
            f"  {name}: ROC-AUC={auc:.4f}  PR-AUC={average_precision:.4f}\n"
            f"    at 0.500 -> recall={default['recall_default']:.4f} precision={default['precision_default']:.4f} "
            f"F1={default['f1_default']:.4f} accuracy={default['accuracy']:.4f}\n"
            f"    at {threshold:.3f} -> recall={tuned['recall_default']:.4f} precision={tuned['precision_default']:.4f} "
            f"F1={tuned['f1_default']:.4f} accuracy={tuned['accuracy']:.4f}"
        )
        plot_confusion(
            {k: tuned[k] for k in ("tp", "fp", "tn", "fn")},
            name, threshold, f"confusion_matrix_{name.lower()}.png",
        )
        results.append(row)
        return row

    # -------------------------------------------------- model 1: logistic ---
    section("2.5 Model 1 — Logistic Regression")
    logistic = LogisticRegression(
        featuresCol="scaled_features", labelCol=LABEL_COL, weightCol="weight",
        maxIter=100, regParam=0.01,
    )
    logistic_grid = (
        ParamGridBuilder()
        .addGrid(logistic.regParam, [0.001, 0.01, 0.1])
        .addGrid(logistic.elasticNetParam, [0.0, 0.5])
        .build()
    )
    logistic_model, logistic_choice = fit_model(logistic, train_scaled, logistic_grid, "LogisticRegression")
    chosen_params.append({"model": "LogisticRegression", "parameters": str(logistic_choice)})
    evaluate("LogisticRegression", logistic_model.transform(val_scaled), logistic_model.transform(test_scaled))

    coefficients = np.array(logistic_model.coefficients.toArray())
    coefficient_rows = plot_importance(
        names, coefficients, "LogisticRegression", "coefficients_logisticregression.png"
    )
    write_csv(OUT_DIR / "coefficients_logisticregression.csv", coefficient_rows, ["feature", "importance"])

    # ---------------------------------------------- model 2: random forest ---
    section("2.6 Model 2 — Random Forest")
    forest = RandomForestClassifier(
        featuresCol="features", labelCol=LABEL_COL, weightCol="weight",
        numTrees=150, maxDepth=10, seed=RANDOM_STATE, subsamplingRate=0.8,
    )
    forest_grid = (
        ParamGridBuilder()
        .addGrid(forest.numTrees, [100, 150])
        .addGrid(forest.maxDepth, [8, 12])
        .build()
    )
    forest_model, forest_choice = fit_model(forest, train_feat, forest_grid, "RandomForest")
    chosen_params.append({"model": "RandomForest", "parameters": str(forest_choice)})
    evaluate("RandomForest", forest_model.transform(val_feat), forest_model.transform(test_feat))
    forest_rows = plot_importance(
        names, np.array(forest_model.featureImportances.toArray()), "RandomForest",
        "feature_importance_randomforest.png",
    )
    write_csv(OUT_DIR / "feature_importance_randomforest.csv", forest_rows, ["feature", "importance"])

    # ------------------------------------------------------- model 3: GBT ---
    section("2.7 Model 3 — Gradient-Boosted Trees")
    boosted = GBTClassifier(
        featuresCol="features", labelCol=LABEL_COL, weightCol="weight",
        maxIter=80, maxDepth=5, stepSize=0.1, subsamplingRate=0.8, seed=RANDOM_STATE,
    )
    boosted_grid = (
        ParamGridBuilder()
        .addGrid(boosted.maxDepth, [4, 6])
        .addGrid(boosted.maxIter, [60, 100])
        .build()
    )
    boosted_model, boosted_choice = fit_model(boosted, train_feat, boosted_grid, "GBT")
    chosen_params.append({"model": "GBT", "parameters": str(boosted_choice)})
    evaluate("GBT", boosted_model.transform(val_feat), boosted_model.transform(test_feat))
    boosted_rows = plot_importance(
        names, np.array(boosted_model.featureImportances.toArray()), "GBT",
        "feature_importance_gbt.png",
    )
    write_csv(OUT_DIR / "feature_importance_gbt.csv", boosted_rows, ["feature", "importance"])

    # ----------------------------------------------------------------- 2.8 ---
    section("2.8 Comparison and export")
    test_rate = next(r["default_rate"] for r in split_rows if r["split"] == "test")
    plot_curves(
        roc_curves, roc_scores,
        xlabel="false positive rate", ylabel="true positive rate",
        title="ROC curves on the held-out test set — Spark MLlib",
        filename="roc_curves.png", baseline=None,
    )
    plot_curves(
        prc_curves, prc_scores,
        xlabel="recall (default class)", ylabel="precision (default class)",
        title="Precision-recall curves on the held-out test set — Spark MLlib",
        filename="pr_curves.png", baseline=test_rate,
    )

    metric_columns = [
        "model", "auc_roc", "auc_pr", "accuracy_at_0.5", "precision_default_at_0.5",
        "recall_default_at_0.5", "f1_default_at_0.5", "tuned_threshold", "accuracy_tuned",
        "precision_default_tuned", "recall_default_tuned", "f1_default_tuned",
        "balanced_accuracy_tuned", "tp_tuned", "fp_tuned", "tn_tuned", "fn_tuned",
        "train_rows", "test_rows",
    ]
    write_csv(OUT_DIR / "model_metrics.csv", results, metric_columns)
    write_csv(
        OUT_DIR / "confusion_matrices.csv",
        [
            {
                "model": r["model"], "threshold": r["tuned_threshold"],
                "tp": r["tp_tuned"], "fp": r["fp_tuned"], "tn": r["tn_tuned"], "fn": r["fn_tuned"],
            }
            for r in results
        ],
        ["model", "threshold", "tp", "fp", "tn", "fn"],
    )
    write_csv(OUT_DIR / "chosen_hyperparameters.csv", chosen_params, ["model", "parameters"])

    print("\nRanked by PR-AUC (the metric that matters at a 5% default rate):")
    print_table(
        sorted(results, key=lambda r: -r["auc_pr"]),
        ["model", "auc_roc", "auc_pr", "accuracy_at_0.5", "recall_default_at_0.5"],
    )
    print("\nAt the validation-tuned threshold:")
    print_table(
        results,
        ["model", "tuned_threshold", "precision_default_tuned", "recall_default_tuned",
         "f1_default_tuned", "accuracy_tuned"],
    )

    champion = max(results, key=lambda r: r["auc_pr"])
    baseline_accuracy = 1 - test_rate
    summary = [
        "CSCI316 Stage 2, Process One — Spark MLlib",
        f"Run {datetime.now():%Y-%m-%d %H:%M}",
        f"Rows train/validation/test: {n_train:,} / {n_val:,} / {n_test:,} (stratified {SPLIT})",
        f"Test default rate: {test_rate:.2%}  |  majority-class accuracy: {baseline_accuracy:.2%}",
        f"Features: {len(names)} assembled columns from {len(ALL_NUMERIC)} numeric + "
        f"{len(CATEGORICAL)} categorical inputs",
        f"Cross-validation: {'enabled' if ENABLE_TUNE else 'disabled'} ({CV_METRIC})",
        "",
        "Test-set results:",
    ]
    for row in results:
        summary.append(
            f"  {row['model']}: ROC-AUC={row['auc_roc']:.4f}  PR-AUC={row['auc_pr']:.4f}  "
            f"F1(default)@{row['tuned_threshold']:.3f}={row['f1_default_tuned']:.4f}  "
            f"recall={row['recall_default_tuned']:.4f}  precision={row['precision_default_tuned']:.4f}"
        )
    summary += [
        "",
        f"Best PR-AUC: {champion['model']} ({champion['auc_pr']:.4f}), "
        f"{champion['auc_pr'] / test_rate:.2f}x the random baseline of {test_rate:.4f}.",
        f"At its tuned threshold it finds {champion['recall_default_tuned']:.1%} of defaults "
        f"({champion['tp_tuned']:,} of {champion['tp_tuned'] + champion['fn_tuned']:,}) while "
        f"flagging {champion['fp_tuned']:,} good loans.",
        f"Accuracy at 0.5 ({champion['accuracy_at_0.5']:.2%}) must be read against the "
        f"{baseline_accuracy:.2%} a constant 'no default' rule already achieves.",
        "No post-origination payment feature was used as a predictor.",
    ]
    (OUT_DIR / "spark_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"\nSaved: {(OUT_DIR / 'spark_summary.txt').relative_to(ROOT)}")
    for line in summary[-5:]:
        print(line)

    for frame in (train_scaled, train_feat, val_feat, test_feat, train_df, val_df, test_df, model_df):
        frame.unpersist()
    try:
        spark.stop()
    except OSError as exc:
        print(f"Note: Spark shutdown reported {exc}; all outputs were already written.")
    section("Stage 2 (Spark MLlib) complete")


if __name__ == "__main__":
    main()
