#!/usr/bin/env python3
"""CSCI316 Stage 2, Process Two — TensorFlow / Keras classifiers for loan default.

Three feed-forward networks of increasing capacity are trained on exactly the same
origination-safe features and the same stratified split as the Spark process, so
the two pipelines are directly comparable:

1. LogisticNN — a single sigmoid unit, i.e. logistic regression expressed in Keras.
   Included as the linear reference point for the deeper networks.
2. MLP        — 64/32 hidden units with dropout; enough capacity for interactions.
3. DeepMLP    — 128/64/32 with batch normalisation and stronger dropout.

Imbalance handling mirrors Process One: balanced class weights during training,
early stopping on validation PR-AUC, and a decision threshold selected on the
validation fold before a single evaluation on the untouched test fold.

Run:

    python src/stage2_tensorflow.py

Inputs   : latest results/stage1/<stamp>/data/data_clean.csv
Outputs  : results/tensorflow/<YYYYMMDD_HHMMSS>/ plus results/tensorflow/runs.log
"""

from __future__ import annotations

import csv
import os
import random
import sys
from datetime import datetime
from pathlib import Path

# TensorFlow's native DLL must load before NumPy/MKL on Windows, otherwise
# `_pywrap_tensorflow_internal` fails with "DLL initialization routine failed".
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow import keras

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import run_paths

RANDOM_STATE = 192
LABEL_COL = "default_ind"
SPLIT = (0.64, 0.16, 0.20)
BATCH_SIZE = int(os.environ.get("STAGE2_BATCH_SIZE", "1024"))
MAX_EPOCHS = int(os.environ.get("STAGE2_EPOCHS", "60"))
THRESHOLD_STEP = float(os.environ.get("STAGE2_THRESHOLD_STEP", "0.005"))
SAMPLE_FRAC = float(os.environ.get("STAGE2_SAMPLE", "1.0"))
if not 0.0 < SAMPLE_FRAC <= 1.0:
    raise ValueError("STAGE2_SAMPLE must be in (0, 1]")

# Identical to the feature contract in src/stage2_spark.py.
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
MISSING_FLAG_SOURCE = "tot_cur_bal"
ALL_NUMERIC = BASE_NUMERIC + ENGINEERED_NUMERIC

COLOURS = {"LogisticNN": "#3498db", "MLP": "#27ae60", "DeepMLP": "#9b59b6"}


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
OUT_DIR = ROOT / "results" / "tensorflow"
SPARK_METRICS = run_paths.spark_metrics_path(ROOT) or (ROOT / "results" / "spark" / "model_metrics.csv")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
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
            elif isinstance(value, (int, np.integer)):
                rendered[column] = f"{int(value):,}"
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# --------------------------------------------------------------------------- #
# Metrics (shared definitions with src/stage2_spark.py)
# --------------------------------------------------------------------------- #
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
    grid = np.arange(0.05, 0.951, THRESHOLD_STEP)
    best, best_f1 = 0.5, -1.0
    for threshold in grid:
        score = metrics_at(y, scores, float(threshold))["f1_default"]
        if score > best_f1:
            best, best_f1 = float(threshold), score
    return best, best_f1


def roc_points(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores)
    labels = y[order].astype(np.float64)
    positives, negatives = labels.sum(), len(labels) - labels.sum()
    tpr = np.concatenate([[0.0], np.cumsum(labels) / positives])
    fpr = np.concatenate([[0.0], np.cumsum(1.0 - labels) / negatives])
    return fpr, tpr


def pr_points(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores)
    labels = y[order].astype(np.float64)
    tp = np.cumsum(labels)
    return tp / labels.sum(), tp / np.arange(1, len(labels) + 1)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_frame() -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"{DATA_PATH} is missing. Run `python src/stage1_eda.py` first — Stage 2 consumes "
            "the cleaned dataset that Stage 1 exports."
        )
    usecols = ALL_NUMERIC + CATEGORICAL + [LABEL_COL]
    available = set(pd.read_csv(DATA_PATH, nrows=0).columns)
    missing = [c for c in usecols if c not in available]
    if missing:
        raise ValueError(f"data_clean.csv is missing expected columns: {missing}")
    df = pd.read_csv(DATA_PATH, usecols=usecols, low_memory=False)
    for column in ALL_NUMERIC + [LABEL_COL]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in CATEGORICAL:
        df[column] = df[column].fillna("UNKNOWN").astype(str)
    # Same indicator as Process One: distinguishes "no bureau record" from a real zero.
    df["bureau_data_missing"] = df[MISSING_FLAG_SOURCE].isna().astype(float)
    return df.dropna(subset=[LABEL_COL]).reset_index(drop=True)


def stratified_indices(labels: np.ndarray, fracs: tuple[float, float, float], seed: int):
    """Shuffle within each class, then cut — preserves the default rate per split."""
    rng = np.random.default_rng(seed)
    train, validation, test = [], [], []
    for label in (0, 1):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        first = int(round(fracs[0] * len(idx)))
        second = first + int(round(fracs[1] * len(idx)))
        train.append(idx[:first])
        validation.append(idx[first:second])
        test.append(idx[second:])
    out = []
    for pieces in (train, validation, test):
        combined = np.concatenate(pieces)
        rng.shuffle(combined)
        out.append(combined)
    return out


def build_preprocessor() -> ColumnTransformer:
    """Median-impute + standardise numerics, one-hot the categoricals.

    Wrapped in a scikit-learn pipeline so it can be fitted on the training rows
    only; validation and test rows are merely transformed.
    """
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, ALL_NUMERIC),
            ("flag", "passthrough", ["bureau_data_missing"]),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=50),
                CATEGORICAL,
            ),
        ],
        remainder="drop",
    )


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def build_logistic(n_features: int, bias: float) -> keras.Model:
    """Logistic regression as a network: one sigmoid unit, no hidden layer."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(n_features,)),
            keras.layers.Dense(
                1, activation="sigmoid",
                kernel_regularizer=keras.regularizers.l2(1e-4),
                bias_initializer=keras.initializers.Constant(bias),
            ),
        ],
        name="LogisticNN",
    )
    return model


def build_mlp(n_features: int, bias: float) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(n_features,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation="sigmoid", bias_initializer=keras.initializers.Constant(bias)),
        ],
        name="MLP",
    )
    return model


def build_deep_mlp(n_features: int, bias: float) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(n_features,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation="sigmoid", bias_initializer=keras.initializers.Constant(bias)),
        ],
        name="DeepMLP",
    )
    return model


def compile_model(model: keras.Model, learning_rate: float) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="pr_auc", curve="PR"),
            keras.metrics.AUC(name="roc_auc"),
        ],
    )
    return model


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def plot_history(histories: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for name, history in histories.items():
        colour = COLOURS.get(name)
        epochs = range(1, len(history["loss"]) + 1)
        axes[0].plot(epochs, history["loss"], color=colour, linewidth=1.6, label=f"{name} train")
        axes[0].plot(epochs, history["val_loss"], color=colour, linewidth=1.6, linestyle="--", label=f"{name} val")
        axes[1].plot(epochs, history["val_pr_auc"], color=colour, linewidth=1.8, label=name)
    axes[0].set_title("Weighted binary cross-entropy")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend(fontsize=8)
    axes[1].set_title("Validation PR-AUC (early-stopping criterion)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("PR-AUC")
    axes[1].legend(fontsize=9)
    for axis in axes:
        axis.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "training_history.png")


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


def plot_confusion_grid(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, len(rows), figsize=(5.2 * len(rows), 4.4))
    axes = np.atleast_1d(axes)
    for axis, row in zip(axes, rows):
        matrix = np.array([[row["tn_tuned"], row["fp_tuned"]], [row["fn_tuned"], row["tp_tuned"]]], dtype=float)
        shares = matrix / matrix.sum(axis=1, keepdims=True)
        image = axis.imshow(shares, cmap="Blues", vmin=0, vmax=1)
        axis.set_xticks([0, 1], ["pred.\nno default", "pred.\ndefault"])
        axis.set_yticks([0, 1], ["actual\nno default", "actual\ndefault"])
        for i in range(2):
            for j in range(2):
                axis.text(
                    j, i, f"{int(matrix[i, j]):,}\n{shares[i, j]:.1%}",
                    ha="center", va="center", fontsize=9,
                    color="white" if shares[i, j] > 0.5 else "black",
                )
        axis.set_title(f"{row['model']} @ {row['tuned_threshold']:.3f}")
    fig.colorbar(image, ax=list(axes), fraction=0.025, label="share of actual class")
    fig.suptitle("Test-set confusion matrices — TensorFlow/Keras", y=1.02)
    save_fig(fig, "confusion_matrices.png")


def plot_process_comparison(keras_rows: list[dict]) -> None:
    """Side-by-side PR-AUC for both Stage 2 processes, when Spark has already run."""
    if SPARK_METRICS is None or not SPARK_METRICS.is_file():
        print("No Spark model_metrics.csv found — skipping the cross-process chart.")
        return
    spark = pd.read_csv(SPARK_METRICS)
    names = list(spark["model"]) + [r["model"] for r in keras_rows]
    values = list(spark["auc_pr"]) + [r["auc_pr"] for r in keras_rows]
    groups = ["Spark MLlib"] * len(spark) + ["TensorFlow"] * len(keras_rows)
    palette = {"Spark MLlib": "#2c3e50", "TensorFlow": "#e67e22"}

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(names, values, color=[palette[g] for g in groups])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colour) for colour in palette.values()]
    ax.legend(handles, palette.keys(), fontsize=9)
    ax.set_ylabel("test PR-AUC")
    ax.set_title("Process One vs Process Two — PR-AUC on the same test split")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    save_fig(fig, "process_comparison.png")

    combined = [{"process": g, "model": n, "auc_pr": v} for g, n, v in zip(groups, names, values)]
    write_csv(OUT_DIR / "process_comparison.csv", combined, ["process", "model", "auc_pr"])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    global OUT_DIR, DATA_PATH, SPARK_METRICS
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    found = run_paths.clean_loan_path(ROOT)
    if found is None:
        raise FileNotFoundError(
            "No Stage 1 cleaned dataset found. Run `python src/stage1_eda.py` first."
        )
    DATA_PATH = found
    SPARK_METRICS = run_paths.spark_metrics_path(ROOT) or (ROOT / "results" / "spark" / "model_metrics.csv")
    note = f"epochs={MAX_EPOCHS} sample={SAMPLE_FRAC}"
    run_dir = run_paths.start_run(ROOT, "tensorflow", sample=SAMPLE_FRAC, note=note)
    OUT_DIR = run_dir
    plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "figure.facecolor": "white"})
    seed_everything(RANDOM_STATE)

    status = "failed"
    try:
        _run_tensorflow()
        status = "ok"
    finally:
        run_paths.finish_run(ROOT, "tensorflow", run_dir, status=status, sample=SAMPLE_FRAC)


def _run_tensorflow() -> None:

    section("2.1 Configuration")
    print(f"Started      : {datetime.now():%Y-%m-%d %H:%M}")
    print(f"TensorFlow   : {tf.__version__}  |  Keras: {keras.__version__}")
    devices = [d.device_type for d in tf.config.list_physical_devices()]
    print(f"Devices      : {', '.join(sorted(set(devices)))}")
    print(f"Input        : {DATA_PATH.relative_to(ROOT)}")
    print(f"Output       : {OUT_DIR.relative_to(ROOT)}/")
    print(f"Batch size   : {BATCH_SIZE}  |  max epochs: {MAX_EPOCHS}  |  seed: {RANDOM_STATE}")

    # ----------------------------------------------------------------- 2.2 ---
    section("2.2 Load features")
    df = load_frame()
    if SAMPLE_FRAC < 1.0:
        df = df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"Development sample: STAGE2_SAMPLE={SAMPLE_FRAC}")
    labels = df[LABEL_COL].to_numpy(dtype=np.int8)
    print(f"Rows: {len(df):,}  |  numeric: {len(ALL_NUMERIC)} + 1 flag  |  categorical: {len(CATEGORICAL)}")
    print(f"Overall default rate: {labels.mean():.2%}")
    print("The feature list is identical to src/stage2_spark.py, so the two processes are comparable.")

    # ----------------------------------------------------------------- 2.3 ---
    section("2.3 Stratified split")
    train_idx, val_idx, test_idx = stratified_indices(labels, SPLIT, RANDOM_STATE)
    split_rows = [
        {"split": name, "rows": len(idx), "default_rate": float(labels[idx].mean())}
        for name, idx in (("train", train_idx), ("validation", val_idx), ("test", test_idx))
    ]
    print_table(split_rows, ["split", "rows", "default_rate"], pct_cols=("default_rate",))
    write_csv(OUT_DIR / "split_summary.csv", split_rows, ["split", "rows", "default_rate"])

    y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]

    # ----------------------------------------------------------------- 2.4 ---
    section("2.4 Preprocessing")
    preprocessor = build_preprocessor()
    x_train = preprocessor.fit_transform(df.iloc[train_idx]).astype(np.float32)
    x_val = preprocessor.transform(df.iloc[val_idx]).astype(np.float32)
    x_test = preprocessor.transform(df.iloc[test_idx]).astype(np.float32)
    n_features = x_train.shape[1]
    print(f"Design matrix: {x_train.shape[0]:,} x {n_features} (fitted on the training rows only)")
    print("Rare categories (fewer than 50 training rows) are folded into an 'infrequent' level.")

    negatives, positives = int((y_train == 0).sum()), int((y_train == 1).sum())
    class_weight = {0: 1.0, 1: negatives / positives}
    # Starting the output bias at the log-odds of the base rate stops the first
    # epochs being wasted learning that defaults are rare.
    initial_bias = float(np.log(positives / negatives))
    print(f"Class weight for defaults: {class_weight[1]:.2f}  |  output bias init: {initial_bias:.4f}")

    # ----------------------------------------------------------------- 2.5 ---
    section("2.5 Train the three networks")
    builders = [
        ("LogisticNN", build_logistic, 1e-2),
        ("MLP", build_mlp, 1e-3),
        ("DeepMLP", build_deep_mlp, 1e-3),
    ]
    results: list[dict] = []
    histories: dict[str, dict] = {}
    roc_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    roc_scores: dict[str, float] = {}
    prc_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    prc_scores: dict[str, float] = {}

    for name, builder, learning_rate in builders:
        print(f"\n--- {name} ---")
        seed_everything(RANDOM_STATE)
        model = compile_model(builder(n_features, initial_bias), learning_rate)
        model.summary(print_fn=print, line_length=78)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_pr_auc", mode="max", patience=8,
                restore_best_weights=True, verbose=0,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_pr_auc", mode="max", factor=0.5, patience=4, min_lr=1e-5, verbose=0
            ),
        ]
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=2,
        )
        histories[name] = {k: [float(v) for v in values] for k, values in history.history.items()}
        epochs_run = len(history.history["loss"])
        best_epoch = int(np.argmax(history.history["val_pr_auc"])) + 1
        print(f"Stopped after {epochs_run} epochs; best validation PR-AUC at epoch {best_epoch}.")

        p_val = model.predict(x_val, batch_size=4096, verbose=0).ravel()
        p_test = model.predict(x_test, batch_size=4096, verbose=0).ravel()
        threshold, val_f1 = best_threshold(y_val, p_val)

        auc = float(roc_auc_score(y_test, p_test))
        average_precision = float(average_precision_score(y_test, p_test))
        roc_curves[name], roc_scores[name] = roc_points(y_test, p_test), auc
        prc_curves[name], prc_scores[name] = pr_points(y_test, p_test), average_precision

        default = metrics_at(y_test, p_test, 0.5)
        tuned = metrics_at(y_test, p_test, threshold)
        results.append(
            {
                "model": name,
                "auc_roc": auc,
                "auc_pr": average_precision,
                "val_f1_default": val_f1,
                "tuned_threshold": threshold,
                "epochs_run": epochs_run,
                "best_epoch": best_epoch,
                "parameters": int(model.count_params()),
                "train_rows": len(y_train),
                "test_rows": len(y_test),
                **{f"{k}_at_0.5": v for k, v in default.items() if k != "threshold"},
                **{f"{k}_tuned": v for k, v in tuned.items() if k != "threshold"},
            }
        )
        print(
            f"  {name}: ROC-AUC={auc:.4f}  PR-AUC={average_precision:.4f}\n"
            f"    at 0.500 -> recall={default['recall_default']:.4f} precision={default['precision_default']:.4f} "
            f"F1={default['f1_default']:.4f}\n"
            f"    at {threshold:.3f} -> recall={tuned['recall_default']:.4f} precision={tuned['precision_default']:.4f} "
            f"F1={tuned['f1_default']:.4f}"
        )

    # ----------------------------------------------------------------- 2.6 ---
    section("2.6 Results and export")
    test_rate = float(y_test.mean())
    plot_history(histories)
    plot_curves(
        roc_curves, roc_scores,
        xlabel="false positive rate", ylabel="true positive rate",
        title="ROC curves on the held-out test set — TensorFlow/Keras",
        filename="roc_curves.png", baseline=None,
    )
    plot_curves(
        prc_curves, prc_scores,
        xlabel="recall (default class)", ylabel="precision (default class)",
        title="Precision-recall curves on the held-out test set — TensorFlow/Keras",
        filename="pr_curves.png", baseline=test_rate,
    )
    plot_confusion_grid(results)

    metric_columns = [
        "model", "parameters", "epochs_run", "best_epoch", "auc_roc", "auc_pr",
        "accuracy_at_0.5", "precision_default_at_0.5", "recall_default_at_0.5", "f1_default_at_0.5",
        "tuned_threshold", "accuracy_tuned", "precision_default_tuned", "recall_default_tuned",
        "f1_default_tuned", "balanced_accuracy_tuned", "tp_tuned", "fp_tuned", "tn_tuned", "fn_tuned",
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
    history_rows = [
        {"model": name, "epoch": epoch + 1, **{k: values[epoch] for k, values in history.items()}}
        for name, history in histories.items()
        for epoch in range(len(history["loss"]))
    ]
    write_csv(
        OUT_DIR / "training_history.csv",
        history_rows,
        ["model", "epoch", "loss", "val_loss", "pr_auc", "val_pr_auc", "roc_auc", "val_roc_auc"],
    )

    print("\nRanked by PR-AUC:")
    print_table(
        sorted(results, key=lambda r: -r["auc_pr"]),
        ["model", "parameters", "epochs_run", "auc_roc", "auc_pr", "accuracy_at_0.5"],
    )
    print("\nAt the validation-tuned threshold:")
    print_table(
        results,
        ["model", "tuned_threshold", "precision_default_tuned", "recall_default_tuned",
         "f1_default_tuned", "accuracy_tuned"],
    )

    plot_process_comparison(results)

    champion = max(results, key=lambda r: r["auc_pr"])
    summary = [
        "CSCI316 Stage 2, Process Two — TensorFlow / Keras",
        f"Run {datetime.now():%Y-%m-%d %H:%M} with TensorFlow {tf.__version__}",
        f"Rows train/validation/test: {len(y_train):,} / {len(y_val):,} / {len(y_test):,} "
        f"(stratified {SPLIT}, seed {RANDOM_STATE})",
        f"Design matrix width: {n_features}  |  test default rate: {test_rate:.2%}",
        f"Imbalance handling: class weight {class_weight[1]:.2f} on defaults, output bias "
        f"initialised to {initial_bias:.4f}",
        "Early stopping on validation PR-AUC (patience 8), threshold tuned on validation F1.",
        "",
        "Test-set results:",
    ]
    for row in results:
        summary.append(
            f"  {row['model']} ({row['parameters']:,} params, {row['epochs_run']} epochs): "
            f"ROC-AUC={row['auc_roc']:.4f}  PR-AUC={row['auc_pr']:.4f}  "
            f"F1(default)@{row['tuned_threshold']:.3f}={row['f1_default_tuned']:.4f}  "
            f"recall={row['recall_default_tuned']:.4f}  precision={row['precision_default_tuned']:.4f}"
        )
    linear = next((r for r in results if r["model"] == "LogisticNN"), None)
    summary += [
        "",
        f"Best PR-AUC: {champion['model']} ({champion['auc_pr']:.4f}) against a random baseline "
        f"of {test_rate:.4f}.",
    ]
    if linear is not None and champion["model"] != "LogisticNN":
        gain = champion["auc_pr"] - linear["auc_pr"]
        summary.append(
            f"Depth is worth {gain:+.4f} PR-AUC over the single-unit LogisticNN "
            f"({linear['auc_pr']:.4f}), which quantifies the non-linear signal in the features."
        )
    if SPARK_METRICS is not None and SPARK_METRICS.is_file():
        spark = pd.read_csv(SPARK_METRICS)
        best_spark = spark.loc[spark["auc_pr"].idxmax()]
        summary.append(
            f"Process One's best model ({best_spark['model']}) reached PR-AUC "
            f"{best_spark['auc_pr']:.4f} on the same split, a difference of "
            f"{champion['auc_pr'] - float(best_spark['auc_pr']):+.4f}."
        )
    summary.append("No post-origination payment feature was used as a predictor.")
    (OUT_DIR / "tensorflow_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"\nSaved: {(OUT_DIR / 'tensorflow_summary.txt').relative_to(ROOT)}")
    for line in summary[-4:]:
        print(line)

    section("Stage 2 (TensorFlow/Keras) complete")


if __name__ == "__main__":
    main()
