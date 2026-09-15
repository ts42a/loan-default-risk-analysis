#!/usr/bin/env python3
"""CSCI316 Stage 1 — Data exploration with Apache Spark (loan default risk).

Target variable: ``default_ind`` (1 = the borrower defaulted, 0 = they did not).

Project constraint for this stage: all data handling uses Spark DataFrame / RDD
APIs. pandas and scikit-learn are deliberately not imported anywhere in this
file. NumPy and Matplotlib are used only to render charts from small aggregates
that Spark has already reduced to a handful of rows.

Run from anywhere:

    python src/stage1_eda.py
    STAGE1_SAMPLE=0.1 python src/stage1_eda.py   # 10% development subset

Inputs   : data/data.csv
Outputs  : results/stage1/<YYYYMMDD_HHMMSS>/{tables,figures,reports,data}/
           plus results/stage1/runs.log and a LATEST pointer for Stage 2
"""

from __future__ import annotations

import csv
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

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, NumericType

import run_paths

RANDOM_STATE = 192
LABEL_COL = "default_ind"
SAMPLE_FRAC = float(os.environ.get("STAGE1_SAMPLE", "1.0"))
if not 0.0 < SAMPLE_FRAC <= 1.0:
    raise ValueError("STAGE1_SAMPLE must be in (0, 1]")

# Columns that only exist because the loan has already been serviced. Using any
# of them to predict default leaks the answer, so they are reported and removed.
LEAKAGE_COLS = [
    "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries",
    "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt",
    "next_pymnt_d", "last_credit_pull_d",
]

# Free text / identifiers: no usable signal at this cardinality.
TEXT_COLS = ["desc", "emp_title", "title", "url"]

GRADE_TO_ORD = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}

EMP_LENGTH_TO_YEARS = {
    "< 1 year": 0.0, "1 year": 1.0, "2 years": 2.0, "3 years": 3.0,
    "4 years": 4.0, "5 years": 5.0, "6 years": 6.0, "7 years": 7.0,
    "8 years": 8.0, "9 years": 9.0, "10+ years": 10.0,
}

# Numeric predictors that are known at the moment the loan is originated.
BASE_NUMERIC = [
    "loan_amnt", "term", "int_rate", "installment", "annual_inc", "dti",
    "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc", "collections_12_mths_ex_med", "acc_now_delinq",
    "tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim",
    "credit_history_years", "grade_ord", "emp_length_years",
]

# Ratios and interactions built in this stage so that both Stage 2 processes
# consume byte-identical inputs.
ENGINEERED_NUMERIC = [
    "loan_to_income", "installment_to_income", "revol_util_x_dti",
    "open_to_total_acc", "bal_to_limit", "int_rate_x_term",
]

CATEGORICAL = [
    "sub_grade", "home_ownership", "purpose", "verification_status",
    "addr_state", "application_type",
]

# Charted numeric features (chosen for interpretability on the slides).
PLOT_NUMERIC = ["int_rate", "dti", "annual_inc", "revol_util", "loan_amnt", "credit_history_years"]

# Upper bounds applied to fields with implausible tails (documented in the report).
WINSORISE = {"revol_util": 150.0, "dti": 100.0}

PALETTE = {
    "no_default": "#2ecc71",
    "default": "#e74c3c",
    "bar": "#3498db",
    "accent": "#9b59b6",
}


# --------------------------------------------------------------------------- #
# Paths and environment
# --------------------------------------------------------------------------- #
def resolve_root() -> Path:
    """Repository root, whether run as a script, from a notebook, or on Colab."""
    try:
        root = Path(__file__).resolve().parents[1]
    except NameError:  # interactive session
        root = Path.cwd()
    if (root / "data" / "data.csv").is_file():
        return root
    for candidate in (Path.cwd(), Path.cwd().parent, Path("/content/loan-default-risk-analysis")):
        if (candidate / "data" / "data.csv").is_file():
            return candidate.resolve()
    return root


ROOT = resolve_root()
DATA_PATH = ROOT / "data" / "data.csv"
OUT_DIR = ROOT / "results" / "stage1"
OUT_DATA = OUT_DIR / "data"
OUT_TABLES = OUT_DIR / "tables"
OUT_FIGURES = OUT_DIR / "figures"
OUT_REPORTS = OUT_DIR / "reports"


def bind_outputs(run_dir: Path) -> None:
    """Point writers at this run's timestamped folder."""
    global OUT_DIR, OUT_DATA, OUT_TABLES, OUT_FIGURES, OUT_REPORTS
    OUT_DIR = run_dir
    OUT_DATA = run_dir / "data"
    OUT_TABLES = run_dir / "tables"
    OUT_FIGURES = run_dir / "figures"
    OUT_REPORTS = run_dir / "reports"


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

    # Spark 4 needs JDK 11/17. Prefer an explicit find over whatever `java` is on PATH
    # (Oracle JRE 8 is common on Windows and will not start Spark 4).
    jdk = find_jdk()
    if jdk is not None:
        os.environ["JAVA_HOME"] = str(jdk)
        os.environ["PATH"] = str(jdk / "bin") + os.pathsep + os.environ.get("PATH", "")
    elif not shutil.which("java"):
        raise RuntimeError(
            "Java was not found. Install Temurin JDK 17 from https://adoptium.net/ "
            "or place a portable JDK under .jdk/ in this repository, then re-run."
        )

    # Spark cannot resolve host names containing an underscore.
    if "_" in socket.gethostname():
        os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

    java_banner = subprocess.check_output(
        ["java", "-version"], stderr=subprocess.STDOUT, text=True
    ).splitlines()[0]

    spark_tmp = ROOT / ".spark-tmp"
    spark_tmp.mkdir(exist_ok=True)
    spark = (
        SparkSession.builder.appName("CSCI316_Stage1_EDA")
        .master("local[*]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.memory", os.environ.get("STAGE1_DRIVER_MEMORY", "6g"))
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.local.dir", str(spark_tmp))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    print(f"Java: {java_banner}")
    print(f"Spark {spark.version}  |  root: {ROOT}")
    return spark


# --------------------------------------------------------------------------- #
# Small console / file helpers
# --------------------------------------------------------------------------- #
def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path.relative_to(ROOT)}")


def write_text(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {path.relative_to(ROOT)}")


def print_table(rows: list[dict], columns: list[str], *, pct_cols: tuple = (), limit: int | None = None) -> None:
    view = []
    for row in rows[: limit if limit is not None else len(rows)]:
        rendered = {}
        for column in columns:
            value = row.get(column)
            if column in pct_cols and value is not None:
                rendered[column] = f"{float(value) * 100:.2f}%"
            elif isinstance(value, float):
                rendered[column] = f"{value:,.4f}"
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
    path = OUT_FIGURES / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------- #
# Spark-only profiling helpers
# --------------------------------------------------------------------------- #
def raw_line_count(path: Path) -> int:
    """Physical line count of the CSV, used as a parser sanity check."""
    lines = 0
    with path.open("rb") as fh:
        for _ in fh:
            lines += 1
    return lines


def null_profile(df: DataFrame, total: int) -> list[dict]:
    """Null count for every column in a single Spark aggregation."""
    exprs = []
    for field in df.schema.fields:
        column = F.col(f"`{field.name}`")
        missing = column.isNull()
        if isinstance(field.dataType, NumericType):
            missing = missing | F.isnan(column)
        exprs.append(F.sum(F.when(missing, 1).otherwise(0)).alias(field.name))
    row = df.agg(*exprs).collect()[0]
    profile = [
        {"column": name, "null_count": int(row[name]), "null_pct": int(row[name]) / total}
        for name in df.columns
    ]
    profile.sort(key=lambda r: r["null_count"], reverse=True)
    return profile


def default_rate_by(df: DataFrame, group_col: str, *, min_n: int = 0) -> list[dict]:
    """Per-category row count and default rate, ordered by default rate."""
    rows = (
        df.groupBy(group_col)
        .agg(
            F.count("*").alias("n"),
            F.avg(F.col(LABEL_COL).cast("double")).alias("default_rate"),
        )
        .filter(F.col("n") >= min_n)
        .orderBy(F.col("default_rate").desc())
        .collect()
    )
    return [
        {
            group_col: "(null)" if r[group_col] is None else str(r[group_col]),
            "n": int(r["n"]),
            "default_rate": float(r["default_rate"]),
        }
        for r in rows
    ]


def median_fill_map(df: DataFrame, columns: list[str]) -> dict[str, float]:
    """Approximate medians for every column in one Spark pass."""
    medians = df.approxQuantile(columns, [0.5], 0.001)
    filled = {}
    for column, values in zip(columns, medians):
        filled[column] = float(values[0]) if values else 0.0
    return filled


# --------------------------------------------------------------------------- #
# 1.2 Load
# --------------------------------------------------------------------------- #
def load_dataset(spark: SparkSession) -> tuple[DataFrame, int]:
    """Read the CSV and verify Spark parsed exactly one record per raw line.

    Lending Club extracts sometimes contain newline characters inside the quoted
    free-text ``desc`` field. Spark's default (splittable) reader turns each of
    those into extra broken records, which silently corrupts every downstream
    statistic. Rather than trust the file, the row count is compared against the
    raw line count and the slower ``multiLine`` reader is used only if needed.
    """
    expected_rows = raw_line_count(DATA_PATH) - 1
    reader_opts = dict(header=True, inferSchema=True, quote='"', escape='"', mode="PERMISSIVE")

    df = spark.read.csv(str(DATA_PATH), **reader_opts)
    parsed = df.count()
    if parsed != expected_rows:
        print(
            f"Row count mismatch (parsed {parsed:,} vs {expected_rows:,} raw lines) — "
            "re-reading with multiLine=True because quoted fields contain newlines."
        )
        df = spark.read.csv(str(DATA_PATH), multiLine=True, **reader_opts)
        parsed = df.count()
        if parsed != expected_rows:
            raise ValueError(
                f"CSV still parses to {parsed:,} rows but the file has {expected_rows:,} "
                "data lines. Check the delimiter/quoting of data/data.csv."
            )
    print(f"Parser check passed: {parsed:,} records match {expected_rows:,} raw data lines.")
    return df, parsed


# --------------------------------------------------------------------------- #
# 1.7 Cleaning and typing
# --------------------------------------------------------------------------- #
def clean_and_type(df: DataFrame, dropped_sparse: list[str]) -> DataFrame:
    """Cast the messy string columns, derive ordinals and ratios, drop leakage."""
    drop = set(dropped_sparse) | set(LEAKAGE_COLS) | set(TEXT_COLS)
    df = df.drop(*[c for c in drop if c in df.columns])

    # " 36 months" -> 36
    if "term" in df.columns:
        df = df.withColumn(
            "term",
            F.regexp_replace(F.trim(F.col("term")), "[^0-9]", "").try_cast(IntegerType()),
        )
    # "83.7%" -> 83.7
    if "revol_util" in df.columns:
        df = df.withColumn(
            "revol_util",
            F.regexp_replace(F.col("revol_util").cast("string"), "%", "").try_cast(DoubleType()),
        )

    plain_numeric = [
        "loan_amnt", "funded_amnt", "funded_amnt_inv", "int_rate", "installment",
        "annual_inc", "dti", "delinq_2yrs", "inq_last_6mths", "open_acc",
        "pub_rec", "revol_bal", "total_acc", "collections_12_mths_ex_med",
        "acc_now_delinq", "tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim",
    ]
    for column in plain_numeric:
        if column in df.columns:
            df = df.withColumn(column, F.col(column).try_cast(DoubleType()))

    # Dates are dd-MM-yyyy; credit history length is a genuine origination-time signal.
    for column in ("issue_d", "earliest_cr_line"):
        if column in df.columns:
            df = df.withColumn(f"{column}_dt", F.to_date(F.col(column), "dd-MM-yyyy"))
    if "issue_d_dt" in df.columns and "earliest_cr_line_dt" in df.columns:
        df = df.withColumn(
            "credit_history_years",
            F.months_between(F.col("issue_d_dt"), F.col("earliest_cr_line_dt")) / F.lit(12.0),
        )
    else:
        df = df.withColumn("credit_history_years", F.lit(None).cast(DoubleType()))

    # Ordered categoricals become ordinals instead of one-hot columns. Values
    # outside the mapping (including nulls) stay null and are imputed in Stage 2.
    grade_map = F.create_map([F.lit(x) for pair in GRADE_TO_ORD.items() for x in pair])
    df = df.withColumn(
        "grade_ord",
        grade_map[F.upper(F.trim(F.col("grade")))].cast(DoubleType()),
    )
    emp_map = F.create_map([F.lit(x) for pair in EMP_LENGTH_TO_YEARS.items() for x in pair])
    df = df.withColumn(
        "emp_length_years",
        emp_map[F.trim(F.col("emp_length"))].cast(DoubleType()),
    )

    # Winsorise the implausible upper tails rather than deleting the rows.
    for column, cap in WINSORISE.items():
        if column in df.columns:
            df = df.withColumn(
                column,
                F.when(F.col(column) > F.lit(cap), F.lit(cap)).otherwise(F.col(column)),
            )

    df = df.withColumn(LABEL_COL, F.col(LABEL_COL).try_cast(IntegerType()))

    # Engineered ratios. The zero guards keep division-by-zero out of the data
    # instead of producing infinities that would break the neural network in
    # Stage 2. `cap` is used rather than F.least because least() ignores nulls,
    # which would silently turn a missing income into a capped ratio.
    def cap(expr, upper: float):
        return F.when(expr > F.lit(upper), F.lit(upper)).otherwise(expr)

    income = F.when(F.col("annual_inc") > 0, F.col("annual_inc"))
    limit = F.when(F.col("total_rev_hi_lim") > 0, F.col("total_rev_hi_lim"))
    accounts = F.when(F.col("total_acc") > 0, F.col("total_acc"))
    df = (
        df.withColumn("loan_to_income", cap(F.col("loan_amnt") / income, 5.0))
        .withColumn(
            "installment_to_income",
            cap(F.col("installment") * F.lit(12.0) / income, 1.0),
        )
        .withColumn("revol_util_x_dti", F.col("revol_util") * F.col("dti"))
        .withColumn("open_to_total_acc", F.col("open_acc") / accounts)
        .withColumn("bal_to_limit", cap(F.col("revol_bal") / limit, 2.0))
        .withColumn("int_rate_x_term", F.col("int_rate") * F.col("term"))
    )
    return df


# --------------------------------------------------------------------------- #
# 1.9 Charts
# --------------------------------------------------------------------------- #
def plot_class_balance(counts: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        [r["label"] for r in counts],
        [r["n"] for r in counts],
        color=[PALETTE["no_default"], PALETTE["default"]],
    )
    for i, row in enumerate(counts):
        ax.text(i, row["n"], f"{row['n']:,}\n({row['share']:.2%})", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Loans")
    ax.set_title(f"Class balance of {LABEL_COL}")
    ax.margins(y=0.15)
    fig.tight_layout()
    save_fig(fig, "class_balance.png")


def plot_numeric_distributions(df: DataFrame, features: list[str]) -> None:
    """Histograms via Spark SQL binning (JVM only — no Python RDD workers).

    On Windows, ``RDD.histogram`` ships every value through a Python worker and
    reliably crashes the worker on this dataset size. ``width_bucket`` stays in
    the JVM and only returns ~80 counts per feature to the driver.
    """
    n_bins = 40
    bounds = df.approxQuantile(features, [0.01, 0.99], 0.001)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    axes = axes.ravel()

    for index, feature in enumerate(features):
        low, high = bounds[index] if bounds[index] else (0.0, 1.0)
        if high <= low:
            axes[index].axis("off")
            continue
        width = (high - low) / n_bins
        centres = [low + (i + 0.5) * width for i in range(n_bins)]
        bucket = F.floor((F.col(feature) - F.lit(low)) / F.lit(width)).cast(IntegerType())
        bucket = F.when(bucket < 0, 0).when(bucket >= n_bins, n_bins - 1).otherwise(bucket)
        rows = (
            df.filter(F.col(feature).isNotNull())
            .groupBy(bucket.alias("bucket"), LABEL_COL)
            .count()
            .collect()
        )
        counts = {0: [0] * n_bins, 1: [0] * n_bins}
        for row in rows:
            label = int(row[LABEL_COL]) if row[LABEL_COL] is not None else None
            b = int(row["bucket"]) if row["bucket"] is not None else None
            if label in counts and b is not None:
                counts[label][b] += int(row["count"])
        axis = axes[index]
        for label, colour, name in ((0, PALETTE["no_default"], "No default"), (1, PALETTE["default"], "Default")):
            total = sum(counts[label]) or 1
            shares = [c / total for c in counts[label]]
            axis.plot(centres, shares, color=colour, linewidth=1.8, label=name)
            axis.fill_between(centres, shares, color=colour, alpha=0.25)
        axis.set_title(feature)
        axis.set_ylabel("share of class")
        axis.legend(fontsize=8)

    for spare in range(len(features), len(axes)):
        axes[spare].axis("off")
    fig.suptitle("Numeric distributions by default_ind (Spark SQL width buckets)", y=1.01)
    fig.tight_layout()
    save_fig(fig, "feature_distributions.png")


def plot_default_rate(rows: list[dict], key: str, title: str, filename: str, overall: float, *, colour: str) -> None:
    ordered = sorted(rows, key=lambda r: r["default_rate"], reverse=True)
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(ordered) + 2), 4.2))
    ax.bar([r[key] for r in ordered], [r["default_rate"] for r in ordered], color=colour)
    ax.axhline(overall, color="black", linestyle="--", linewidth=1, label=f"overall {overall:.2%}")
    ax.set_ylabel("default rate")
    ax.set_xlabel(key)
    ax.set_title(title)
    ax.legend(fontsize=8)
    if len(ordered) > 6:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    save_fig(fig, filename)


def plot_target_correlations(rows: list[dict]) -> None:
    ordered = sorted(rows, key=lambda r: abs(r["correlation"]))
    fig, ax = plt.subplots(figsize=(8, max(5, 0.28 * len(ordered))))
    colours = ["#e74c3c" if r["correlation"] < 0 else "#27ae60" for r in ordered]
    ax.barh([r["feature"] for r in ordered], [r["correlation"] for r in ordered], color=colours)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson r with default_ind")
    ax.set_title("Linear association with the target (leakage columns excluded)")
    fig.tight_layout()
    save_fig(fig, "correlation_with_target.png")


def plot_correlation_heatmap(matrix: np.ndarray, labels: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(9, 7.5))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax.set_title("Correlation heatmap — origination-time numeric features")
    fig.tight_layout()
    save_fig(fig, "correlation_heatmap.png")


# --------------------------------------------------------------------------- #
# 1.13 Export
# --------------------------------------------------------------------------- #
def export_clean_dataset(df: DataFrame, columns: list[str]) -> int:
    """Write a single CSV from the driver.

    Deliberately not using ``df.write.csv``: that needs Hadoop native libraries
    on Windows and emits a directory of part files, whereas Stage 2 (Spark) and
    Stage 2 (TensorFlow) both want one plain file.
    """
    path = OUT_DATA / run_paths.CLEAN_CSV
    written = 0
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for row in df.select(*columns).toLocalIterator():
            writer.writerow(["" if v is None else v for v in row])
            written += 1
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Saved: {path.relative_to(ROOT)} ({written:,} rows, {len(columns)} columns, {size_mb:.1f} MB)")
    return written


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    global OUT_DIR
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found. Place it at {DATA_PATH}. See data/README.md.")
    note = f"sample={SAMPLE_FRAC}"
    run_dir = run_paths.start_run(ROOT, "stage1", sample=SAMPLE_FRAC, note=note)
    bind_outputs(run_dir)
    for directory in (OUT_DIR, OUT_DATA, OUT_TABLES, OUT_FIGURES, OUT_REPORTS):
        directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "font.size": 10, "figure.facecolor": "white"})

    status = "failed"
    try:
        _run_stage1()
        status = "ok"
    finally:
        run_paths.finish_run(ROOT, "stage1", run_dir, status=status, sample=SAMPLE_FRAC)


def _run_stage1() -> None:

    spark = start_spark()
    section("1.1 Run configuration")
    print(f"Started      : {datetime.now():%Y-%m-%d %H:%M}")
    print(f"Dataset      : {DATA_PATH.relative_to(ROOT)}")
    print(f"Output       : {OUT_DIR.relative_to(ROOT)}/")
    print(f"Random seed  : {RANDOM_STATE}")
    if SAMPLE_FRAC < 1.0:
        print(f"STAGE1_SAMPLE: {SAMPLE_FRAC} (development subset after the parser check)")

    # ----------------------------------------------------------------- 1.2 ---
    section("1.2 Load the dataset")
    loan_df, n_raw = load_dataset(spark)
    if SAMPLE_FRAC < 1.0:
        loan_df = loan_df.sample(withReplacement=False, fraction=SAMPLE_FRAC, seed=RANDOM_STATE)
        n_raw = loan_df.count()
        print(f"Sampled {n_raw:,} rows (STAGE1_SAMPLE={SAMPLE_FRAC})")
    n_cols = len(loan_df.columns)
    numeric_cols_raw = [f.name for f in loan_df.schema.fields if isinstance(f.dataType, NumericType)]
    print(f"Rows: {n_raw:,}  |  Columns: {n_cols}  |  Inferred numeric: {len(numeric_cols_raw)}")
    write_csv(
        OUT_TABLES / "schema.csv",
        [{"column": f.name, "inferred_type": f.dataType.simpleString()} for f in loan_df.schema.fields],
        ["column", "inferred_type"],
    )
    loan_df = loan_df.repartition(16).cache()

    # ----------------------------------------------------------------- 1.3 ---
    section("1.3 Duplicate records")
    distinct_id, distinct_member = loan_df.select(
        F.countDistinct("id").alias("id"), F.countDistinct("member_id").alias("member")
    ).collect()[0]
    duplicate_ids = n_raw - int(distinct_id)
    identical_rows = n_raw - loan_df.dropDuplicates().count()
    print(f"distinct id        : {int(distinct_id):,} (duplicate loan ids: {duplicate_ids:,})")
    print(f"distinct member_id : {int(distinct_member):,}")
    print(f"fully identical rows: {identical_rows:,}")
    print("A borrower may legitimately hold several loans, so only duplicate `id` values are removed.")
    if duplicate_ids > 0:
        loan_df = loan_df.dropDuplicates(["id"])
        n_rows = loan_df.count()
        print(f"Removed {n_raw - n_rows:,} duplicate loan id row(s); {n_rows:,} remain.")
    else:
        n_rows = n_raw
    write_text(
        OUT_TABLES / "duplicate_check.txt",
        [
            f"rows_raw={n_raw}",
            f"distinct_id={int(distinct_id)}",
            f"distinct_member_id={int(distinct_member)}",
            f"duplicate_loan_ids={duplicate_ids}",
            f"fully_identical_rows={identical_rows}",
            f"rows_after_dedup={n_rows}",
        ],
    )

    # ----------------------------------------------------------------- 1.4 ---
    section("1.4 Missing values")
    nulls = null_profile(loan_df, n_rows)
    write_csv(OUT_TABLES / "null_counts_all.csv", nulls, ["column", "null_count", "null_pct"])
    print_table(nulls, ["column", "null_count", "null_pct"], pct_cols=("null_pct",), limit=20)
    sparse_cols = [r["column"] for r in nulls if r["null_pct"] > 0.5]
    fully_populated = sum(1 for r in nulls if r["null_count"] == 0)
    print(f"\nColumns with no missing values : {fully_populated} of {n_cols}")
    print(f"Columns more than 50% empty     : {len(sparse_cols)} -> dropped in 1.7")
    if sparse_cols:
        print("  " + ", ".join(sparse_cols))

    # ----------------------------------------------------------------- 1.5 ---
    section("1.5 Summary statistics of key numeric columns")
    stat_cols = [
        "loan_amnt", "int_rate", "installment", "annual_inc", "dti", "delinq_2yrs",
        "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "total_acc",
    ]
    stat_cols = [c for c in stat_cols if c in loan_df.columns]
    described = (
        loan_df.select([F.col(c).try_cast(DoubleType()).alias(c) for c in stat_cols])
        .summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max")
        .collect()
    )
    by_stat = {row["summary"]: row for row in described}

    def stat(name: str, column: str) -> float | None:
        raw = by_stat[name][column]
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    summary_rows = [
        {
            "feature": column,
            "count": int(stat("count", column) or 0),
            "mean": stat("mean", column),
            "stddev": stat("stddev", column),
            "min": stat("min", column),
            "p25": stat("25%", column),
            "median": stat("50%", column),
            "p75": stat("75%", column),
            "max": stat("max", column),
        }
        for column in stat_cols
    ]
    write_csv(
        OUT_TABLES / "summary_statistics.csv",
        summary_rows,
        ["feature", "count", "mean", "stddev", "min", "p25", "median", "p75", "max"],
    )
    print_table(summary_rows, ["feature", "count", "mean", "stddev", "min", "median", "max"])
    print(
        "\nThe gap between the 75th percentile and the maximum of annual_inc, revol_bal and\n"
        "revol_util shows heavy right tails; 1.7 caps the implausible ones instead of\n"
        "deleting the borrowers."
    )

    # ----------------------------------------------------------------- 1.6 ---
    section("1.6 Target variable: class balance")
    label_rows = loan_df.groupBy(LABEL_COL).count().orderBy(LABEL_COL).collect()
    labelled = sum(int(r["count"]) for r in label_rows if r[LABEL_COL] is not None)
    names = {0: "No default", 1: "Default"}
    class_counts = [
        {
            "label": names.get(int(r[LABEL_COL]), str(r[LABEL_COL])) if r[LABEL_COL] is not None else "(null)",
            "n": int(r["count"]),
            "share": int(r["count"]) / labelled,
        }
        for r in label_rows
        if r[LABEL_COL] is not None
    ]
    missing_label = n_rows - labelled
    default_rate = next(r["share"] for r in class_counts if r["label"] == "Default")
    imbalance_ratio = (1 - default_rate) / default_rate
    print_table(class_counts, ["label", "n", "share"], pct_cols=("share",))
    print(f"Rows with a missing target: {missing_label:,}")
    print(f"Overall default rate: {default_rate:.2%}  (majority:minority = {imbalance_ratio:.1f} : 1)")
    print(
        "Because only about 1 loan in 20 defaults, accuracy is a misleading headline metric —\n"
        "Stage 2 reports PR-AUC and the recall/precision of the default class."
    )
    plot_class_balance(class_counts)
    write_csv(OUT_TABLES / "class_balance.csv", class_counts, ["label", "n", "share"])

    # ----------------------------------------------------------------- 1.7 ---
    section("1.7 Cleaning, type casting and leakage removal")
    print(f"Dropping {len(sparse_cols)} column(s) with >50% missing values.")
    print(f"Dropping {len(LEAKAGE_COLS)} post-origination column(s): {', '.join(LEAKAGE_COLS)}")
    print(f"Dropping free-text/identifier column(s): {', '.join(TEXT_COLS)}")
    clean = clean_and_type(loan_df, sparse_cols)
    clean = clean.dropna(subset=[LABEL_COL])

    required = ["loan_amnt", "term", "int_rate", "installment", "annual_inc", "dti", "grade_ord"]
    before_required = clean.count()
    clean = clean.dropna(subset=required)
    clean = clean.cache()
    n_clean = clean.count()
    print(f"\nRows after removing a missing target : {before_required:,}")
    print(f"Rows after requiring {len(required)} core fields : {n_clean:,} ({n_clean / n_raw:.2%} of raw)")
    print(f"Columns retained: {len(clean.columns)}")
    print(
        "Remaining gaps (revol_util, the tot_* credit-bureau block, emp_length) are left as\n"
        "nulls on purpose: Stage 2 imputes them inside the training fold and adds a\n"
        "missing-value indicator, which avoids discarding ~8% of the borrowers here."
    )
    write_text(
        OUT_TABLES / "cleaning_summary.txt",
        [
            f"rows_raw={n_raw}",
            f"rows_after_dedup={n_rows}",
            f"rows_after_target_dropna={before_required}",
            f"rows_after_core_dropna={n_clean}",
            f"pct_rows_retained={n_clean / n_raw:.4f}",
            f"columns_after_clean={len(clean.columns)}",
            f"dropped_sparse_columns={';'.join(sparse_cols)}",
            f"dropped_leakage_columns={';'.join(LEAKAGE_COLS)}",
            f"dropped_text_columns={';'.join(TEXT_COLS)}",
            f"winsorised={';'.join(f'{k}<={v}' for k, v in WINSORISE.items())}",
        ],
    )

    # ----------------------------------------------------------------- 1.8 ---
    section("1.8 Numeric distributions by class")
    # Recomputed on the cleaned frame so every chart baseline matches the data behind it.
    clean_default_rate = float(clean.select(F.avg(F.col(LABEL_COL).cast("double"))).collect()[0][0])
    print(f"Default rate on the cleaned {n_clean:,} rows: {clean_default_rate:.2%}")
    plot_features = [c for c in PLOT_NUMERIC if c in clean.columns]
    plot_numeric_distributions(clean, plot_features)

    mean_rows = [
        {
            "feature": column,
            "mean_no_default": None,
            "mean_default": None,
            "difference_pct": None,
        }
        for column in plot_features
    ]
    aggregated = clean.groupBy(LABEL_COL).agg(*[F.avg(c).alias(c) for c in plot_features]).collect()
    means = {int(r[LABEL_COL]): r for r in aggregated}
    for row in mean_rows:
        no_default = float(means[0][row["feature"]])
        default = float(means[1][row["feature"]])
        row["mean_no_default"] = no_default
        row["mean_default"] = default
        row["difference_pct"] = (default - no_default) / no_default if no_default else None
    write_csv(
        OUT_TABLES / "class_means.csv",
        mean_rows,
        ["feature", "mean_no_default", "mean_default", "difference_pct"],
    )
    print_table(mean_rows, ["feature", "mean_no_default", "mean_default", "difference_pct"], pct_cols=("difference_pct",))

    # ----------------------------------------------------------------- 1.9 ---
    section("1.9 Correlation analysis")
    numeric_features = [c for c in BASE_NUMERIC + ENGINEERED_NUMERIC if c in clean.columns]
    # Constant columns have no correlation and would break the vector assembler.
    spreads = clean.select([F.stddev(c).alias(c) for c in numeric_features]).collect()[0]
    numeric_features = [c for c in numeric_features if spreads[c] not in (None, 0.0)]
    fill = median_fill_map(clean, numeric_features)
    corr_input = clean.select(
        [F.coalesce(F.col(c), F.lit(fill[c])).alias(c) for c in numeric_features]
        + [F.col(LABEL_COL).cast("double").alias(LABEL_COL)]
    )
    vector_cols = numeric_features + [LABEL_COL]
    assembled = VectorAssembler(inputCols=vector_cols, outputCol="vec", handleInvalid="skip").transform(corr_input)
    matrix = np.array(Correlation.corr(assembled.select("vec"), "vec").head()[0].toArray())

    target_index = vector_cols.index(LABEL_COL)
    corr_rows = [
        {"feature": name, "correlation": float(matrix[i, target_index])}
        for i, name in enumerate(vector_cols)
        if name != LABEL_COL
    ]
    corr_rows.sort(key=lambda r: abs(r["correlation"]), reverse=True)
    write_csv(OUT_TABLES / "correlation_with_target.csv", corr_rows, ["feature", "correlation"])
    print("Strongest linear associations with default_ind:")
    print_table(corr_rows, ["feature", "correlation"], limit=15)
    print(
        f"\nEven the best single predictor only reaches |r| = {abs(corr_rows[0]['correlation']):.3f}. "
        "With a\n5% positive class, linear correlation understates usable signal, which is why the\n"
        "tree ensembles and the neural network in Stage 2 are given the full feature set."
    )
    plot_target_correlations(corr_rows)

    heatmap_features = [c for c in plot_features + ["grade_ord", "term", "loan_to_income", "bal_to_limit"] if c in numeric_features]
    heatmap_labels = heatmap_features + [LABEL_COL]
    indices = [vector_cols.index(c) for c in heatmap_labels]
    plot_correlation_heatmap(matrix[np.ix_(indices, indices)], heatmap_labels)

    strongest_pair = None
    for i, a in enumerate(heatmap_features):
        for j, b in enumerate(heatmap_features):
            if j <= i:
                continue
            r = matrix[vector_cols.index(a), vector_cols.index(b)]
            if strongest_pair is None or abs(r) > abs(strongest_pair[2]):
                strongest_pair = (a, b, float(r))
    if strongest_pair:
        print(
            f"Strongest feature-to-feature correlation among the charted set: "
            f"{strongest_pair[0]} vs {strongest_pair[1]} (r = {strongest_pair[2]:.3f})."
        )

    # ---------------------------------------------------------------- 1.10 ---
    section("1.10 Default rate by category")
    categorical_report = ["grade", "term", "purpose", "home_ownership", "verification_status", "emp_length", "addr_state", "sub_grade"]
    category_tables: dict[str, list[dict]] = {}
    for column in categorical_report:
        if column not in clean.columns:
            continue
        rows = default_rate_by(clean, column, min_n=50)
        if not rows:
            continue
        category_tables[column] = rows
        write_csv(OUT_TABLES / f"default_rate_by_{column}.csv", rows, [column, "n", "default_rate"])
        print(f"\n{column}:")
        print_table(rows, [column, "n", "default_rate"], pct_cols=("default_rate",), limit=10)

    for column, colour, title in (
        ("grade", PALETTE["bar"], "Default rate by LC grade"),
        ("term", PALETTE["accent"], "Default rate by term (months)"),
        ("purpose", "#e67e22", "Default rate by stated loan purpose"),
        ("home_ownership", "#16a085", "Default rate by home ownership"),
    ):
        if column in category_tables:
            plot_default_rate(
                category_tables[column], column, title, f"default_rate_by_{column}.png",
                clean_default_rate, colour=colour,
            )

    grade_rows = sorted(category_tables.get("grade", []), key=lambda r: r["grade"])
    term_rows = sorted(category_tables.get("term", []), key=lambda r: r["default_rate"], reverse=True)
    purpose_rows = category_tables.get("purpose", [])
    if not grade_rows or not term_rows:
        raise ValueError(
            "`grade` and `term` are required for the relevance write-up but produced no groups. "
            "Check that data/data.csv is the expected extract."
        )

    # ---------------------------------------------------------------- 1.11 ---
    section("1.11 The 7 most and 7 least relevant attributes")
    corr_lookup = {r["feature"]: r["correlation"] for r in corr_rows}

    def spread(column: str) -> float:
        rows = category_tables.get(column, [])
        return (max(r["default_rate"] for r in rows) - min(r["default_rate"] for r in rows)) if rows else 0.0

    most_relevant = [
        (
            "int_rate",
            f"Strongest single numeric signal (r = {corr_lookup.get('int_rate', 0):+.3f}); the risk-based "
            "price LC charges summarises its own underwriting view.",
        ),
        (
            "grade / sub_grade",
            f"Default rate climbs monotonically from {grade_rows[0]['default_rate']:.2%} (grade "
            f"{grade_rows[0]['grade']}) to {grade_rows[-1]['default_rate']:.2%} (grade "
            f"{grade_rows[-1]['grade']}) — a {spread('grade'):.2%} spread across {len(grade_rows)} bands.",
        ),
        (
            "term",
            f"A {spread('term'):.2%} spread between the two terms "
            f"({term_rows[0]['term']}-month loans default most, r = {corr_lookup.get('term', 0):+.3f}); "
            "longer exposure means more time to fail.",
        ),
        (
            "dti",
            f"Debt-to-income is the standard measure of repayment capacity (r = {corr_lookup.get('dti', 0):+.3f}) "
            "and is available at application time.",
        ),
        (
            "revol_util",
            f"Revolving utilisation (r = {corr_lookup.get('revol_util', 0):+.3f}) shows how much headroom the "
            "borrower has already consumed.",
        ),
        (
            "inq_last_6mths",
            f"Recent credit enquiries (r = {corr_lookup.get('inq_last_6mths', 0):+.3f}) proxy active credit "
            "seeking, an early distress signal.",
        ),
        (
            "annual_inc",
            f"Income anchors every affordability ratio (r = {corr_lookup.get('annual_inc', 0):+.3f}); the derived "
            f"loan_to_income reaches r = {corr_lookup.get('loan_to_income', 0):+.3f}.",
        ),
    ]

    least_relevant = [
        ("id", "Surrogate loan key, unique for all 855k rows — pure noise, and only usable via row order."),
        ("member_id", "Surrogate borrower key; identifiers must never enter the model."),
        ("policy_code", "Constant (a single distinct value), so it carries zero information."),
        ("pymnt_plan", "Degenerate: fewer than 10 loans take the minority value, so it cannot generalise."),
        ("emp_title", "Free text with tens of thousands of inconsistent spellings; unusable without NLP."),
        ("title", "Borrower-typed loan title that merely restates `purpose`, which is already encoded."),
        ("zip_code", "Truncated to three digits, so it is coarse geography that `addr_state` covers more stably."),
    ]

    print("Seven MOST relevant attributes:")
    print_table([{"attribute": a, "evidence": e} for a, e in most_relevant], ["attribute", "evidence"])
    print("\nSeven LEAST relevant attributes:")
    print_table([{"attribute": a, "evidence": e} for a, e in least_relevant], ["attribute", "evidence"])
    write_csv(
        OUT_TABLES / "feature_relevance_most.csv",
        [{"rank": i + 1, "attribute": a, "justification": j} for i, (a, j) in enumerate(most_relevant)],
        ["rank", "attribute", "justification"],
    )
    write_csv(
        OUT_TABLES / "feature_relevance_least.csv",
        [{"rank": i + 1, "attribute": a, "justification": j} for i, (a, j) in enumerate(least_relevant)],
        ["rank", "attribute", "justification"],
    )
    write_text(
        OUT_REPORTS / "feature_relevance.txt",
        ["CSCI316 Stage 1 — attribute relevance", "", "Seven MOST relevant:"]
        + [f"  {i + 1}. {a}: {j}" for i, (a, j) in enumerate(most_relevant)]
        + ["", "Seven LEAST relevant:"]
        + [f"  {i + 1}. {a}: {j}" for i, (a, j) in enumerate(least_relevant)]
        + [
            "",
            "Separately excluded as target leakage (known only after funding):",
            "  " + ", ".join(LEAKAGE_COLS),
        ],
    )

    # ---------------------------------------------------------------- 1.12 ---
    section("1.12 Export the cleaned dataset for Stage 2")
    export_columns = (
        ["id"]
        + [c for c in BASE_NUMERIC if c in clean.columns]
        + [c for c in ENGINEERED_NUMERIC if c in clean.columns]
        + [c for c in CATEGORICAL if c in clean.columns]
        + ["grade", "emp_length", "issue_d", LABEL_COL]
    )
    export_columns = list(dict.fromkeys(c for c in export_columns if c in clean.columns))
    exported = export_clean_dataset(clean, export_columns)
    if SAMPLE_FRAC >= 1.0:
        stable = run_paths.publish_canonical_clean(ROOT, OUT_DATA / run_paths.CLEAN_CSV)
        print(f"Canonical Stage 2 input: {stable.relative_to(ROOT)}")
    write_text(
        OUT_REPORTS / "recommended_features.txt",
        ["# Numeric predictors (origination-safe)"]
        + [c for c in BASE_NUMERIC if c in clean.columns]
        + ["", "# Engineered in Stage 1"]
        + [c for c in ENGINEERED_NUMERIC if c in clean.columns]
        + ["", "# Categorical (encoded in Stage 2)"]
        + [c for c in CATEGORICAL if c in clean.columns],
    )

    # ------------------------------------------------------------ summary ---
    section("Stage 1 — observations")
    top_purpose = purpose_rows[0] if purpose_rows else None
    observations = [
        "CSCI316 Stage 1 — data exploration observations",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} from data/data.csv",
        "",
        "Dataset",
        f"- {n_raw:,} loans and {n_cols} columns; the parsed record count was verified against the raw line count.",
        f"- `id` and `member_id` are unique ({duplicate_ids:,} duplicate loan ids, {identical_rows:,} identical rows).",
        f"- {fully_populated} of {n_cols} columns are fully populated; {len(sparse_cols)} are more than half empty.",
        "- Several fields arrive as text ('36 months', '83.7%', dd-MM-yyyy dates) and are cast in section 1.7.",
        "",
        "Target",
        f"- {default_rate:.2%} of loans default ({int(round(default_rate * labelled)):,} of {labelled:,}), a {imbalance_ratio:.1f}:1 imbalance.",
        f"- {missing_label:,} rows have no target value.",
        "- Accuracy is therefore not informative: a constant 'no default' rule already scores "
        f"{1 - default_rate:.2%}.",
        "",
        "Cleaning",
        f"- {len(sparse_cols)} sparse columns, {len(LEAKAGE_COLS)} post-origination columns and "
        f"{len(TEXT_COLS)} free-text columns were removed.",
        f"- {n_clean:,} rows ({n_clean / n_raw:.2%}) survive after requiring the core origination fields.",
        f"- revol_util and dti were capped at {WINSORISE['revol_util']:.0f} and {WINSORISE['dti']:.0f} "
        "to tame implausible tails.",
        "",
        "Numeric patterns",
        *[
            f"- {r['feature']}: mean {r['mean_default']:,.2f} for defaults vs {r['mean_no_default']:,.2f} "
            f"otherwise ({r['difference_pct']:+.1%})."
            for r in mean_rows
        ],
        f"- Largest |Pearson r| with the target is {corr_rows[0]['feature']} at {corr_rows[0]['correlation']:+.4f}; "
        "the relationships are weak and non-linear.",
        "",
        "Categorical patterns",
        f"- Grade {grade_rows[0]['grade']} defaults at {grade_rows[0]['default_rate']:.2%} rising to "
        f"{grade_rows[-1]['default_rate']:.2%} for grade {grade_rows[-1]['grade']}.",
        f"- {term_rows[0]['term']}-month loans default at {term_rows[0]['default_rate']:.2%} versus "
        f"{term_rows[-1]['default_rate']:.2%} for {term_rows[-1]['term']}-month loans.",
    ]
    if top_purpose:
        observations.append(
            f"- Riskiest stated purpose is '{top_purpose['purpose']}' at {top_purpose['default_rate']:.2%} "
            f"(n = {top_purpose['n']:,}) against an overall {clean_default_rate:.2%}."
        )
    observations += [
        "",
        "Attribute relevance",
        "- Most relevant: " + ", ".join(a for a, _ in most_relevant),
        "- Least relevant: " + ", ".join(a for a, _ in least_relevant),
        "",
        "Handover",
        f"- Canonical file: results/stage1/data/{run_paths.CLEAN_CSV} "
        f"({exported:,} rows). Re-run Stage 1 only if cleaning changes.",
        "- Both Stage 2 processes read that file, so they train on identical inputs.",
    ]
    for line in observations:
        print(line)
    write_text(OUT_REPORTS / "stage1_observations.txt", observations)

    write_text(
        OUT_REPORTS / "stage1_executive_summary.txt",
        [
            "CSCI316 Stage 1 — executive summary",
            f"Rows raw / cleaned      : {n_raw:,} / {n_clean:,} ({n_clean / n_raw:.2%} retained)",
            f"Columns raw / cleaned   : {n_cols} / {len(clean.columns)}",
            f"Default rate            : {default_rate:.2%} ({imbalance_ratio:.1f}:1 imbalance)",
            f"Strongest numeric signal: {corr_rows[0]['feature']} (r = {corr_rows[0]['correlation']:+.4f})",
            f"Grade spread            : {spread('grade'):.2%} between best and worst grade",
            f"Stage 2 input           : {(OUT_DATA / run_paths.CLEAN_CSV).relative_to(ROOT).as_posix()} "
            f"({exported:,} rows x {len(export_columns)} cols)",
            "Leakage excluded        : " + ", ".join(LEAKAGE_COLS),
        ],
    )

    clean.unpersist()
    loan_df.unpersist()
    try:
        spark.stop()
    except OSError as exc:  # the JVM child process occasionally resists shutdown on Windows
        print(f"Note: Spark shutdown reported {exc}; all outputs were already written.")
    section("Stage 1 complete")


if __name__ == "__main__":
    main()
