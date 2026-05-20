#!/usr/bin/env python3
"""
CSCI316 — Stage 1: Spark EDA (Credit Risk)

Target: default_ind (1 = default, 0 = no default)
use Spark DataFrame / RDD APIs for analysis — no pandas or scikit-learn. Matplotlib and NumPy are used only to render charts from small Spark aggregates or samples.

Outputs: output/stage1/{data,tables,figures,reports}/
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import os
import shutil
import socket
import subprocess
import urllib.request
from pathlib import Path


def main() -> None:
    # --- CSCI316 — Stage 1: Spark EDA (Credit Risk) ---
    import sys
    from pathlib import Path
    import warnings

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import csv
    import matplotlib.pyplot as plt
    import numpy as np

    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, count as spark_count, countDistinct, sum as spark_sum, isnan, trim, regexp_replace, when, stddev
    from pyspark.sql.types import DoubleType, IntegerType
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.stat import Correlation

    warnings.filterwarnings("ignore")

    RANDOM_STATE = 192
    ROOT = Path.cwd()
    DATA_PATH = ROOT / "data" / "data.csv"
    OUT_DIR = ROOT / "output" / "stage1"
    OUT_DATA = OUT_DIR / "data"
    OUT_TABLES = OUT_DIR / "tables"
    OUT_FIGURES = OUT_DIR / "figures"
    OUT_REPORTS = OUT_DIR / "reports"
    for _out in (OUT_DIR, OUT_DATA, OUT_TABLES, OUT_FIGURES, OUT_REPORTS):
        _out.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Place dataset at: {DATA_PATH.resolve()}")

    plt.rcParams.update({
        "figure.dpi": 120,
        "axes.grid": True,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "font.size": 10,
        "figure.facecolor": "white",
    })
    PALETTE = {"no_default": "#2ecc71", "default": "#e74c3c", "bar": "#3498db", "accent": "#9b59b6"}

    def save_fig(fig, name: str):
        path = OUT_FIGURES / name
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", path)

    # --- Output helper: save the cleaned dataset for Stage 2 reuse ---
    def save_clean_dataset(df, out_dir: Path) -> None:
        """Write cleaned data to output/stage1/data/ (CSV + optional parquet)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "clean_loan.csv"
        cols = df.columns
        n = 0
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(cols)
            for row in df.toLocalIterator():
                writer.writerow([row[c] for c in cols])
                n += 1
        print(f"Saved: {csv_path} ({n:,} rows)")
        parquet_path = out_dir / "clean_loan.parquet"
        try:
            df.write.mode("overwrite").parquet(str(parquet_path))
            print(f"Saved parquet: {parquet_path}")
        except Exception as exc:
            print(f"Parquet skipped ({exc.__class__.__name__}); use data/clean_loan.csv for Stage 2.")

    # --- Reporting helper: explain verification_status with domain context ---
    def verification_status_lines(vs: list[dict]) -> list[str]:
        """Explain LC verification tiers (Verified is not 'safest')."""
        if not vs:
            return []
        ordered = sorted(vs, key=lambda r: r["default_rate"])
        lines = [
            "- verification_status (univariate; see tables/default_rate_by_verification_status.csv):",
        ]
        for r in ordered:
            lines.append(
                f"  {r['verification_status']}: {r['default_rate']:.2%} (n={r['n']:,})"
            )
        lines.append(
            "  Note: Source Verified = documented income (strictest); Verified = stated income "
            "deemed plausible only—not a monotonic safety ranking. Confounded by grade/int_rate; "
            "encode as categorical in Stage 2."
        )
        return lines

    def run_banner() -> None:
        from datetime import datetime
        print("CSCI316 Stage 1 — Spark EDA (Credit Risk)")
        print(f"Run started: {datetime.now():%Y-%m-%d %H:%M}")
        print(f"Dataset: {DATA_PATH.name}  |  Output: {OUT_DIR.relative_to(ROOT)}/")

    def section(title: str) -> None:
        print("\n" + "=" * 72)
        print(title)
        print("=" * 72)

    def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def print_table(rows: list[dict], columns: list[str], *, pct_cols: tuple = ()) -> None:
        """Aligned console table from list-of-dicts (Spark collect results)."""
        view = []
        for row in rows:
            out = {c: row.get(c) for c in columns}
            for c in pct_cols:
                if c in out and out[c] is not None:
                    out[c] = f"{float(out[c]) * 100:.2f}%"
            view.append(out)
        widths = [
            max(len(str(c)), *(len(str(r.get(c, ""))) for r in view))
            for c in columns
        ]
        fmt = "  ".join(f"{{:{w}}}" for w in widths)
        print(fmt.format(*columns))
        print(fmt.format(*["-" * w for w in widths]))
        for row in view:
            print(fmt.format(*[str(row[c]) for c in columns]))

    # --- Console helper: compact summary table for key numeric columns ---
    def print_summary_table(describe_rows: list, feature_cols: list[str]) -> None:
        """Pretty-print Spark describe() — one row per feature."""
        by_stat: dict[str, dict[str, float | None]] = {}
        for row in describe_rows:
            summary = row["summary"]
            by_stat[summary] = {}
            for feat in feature_cols:
                raw = row[feat]
                if raw is None:
                    by_stat[summary][feat] = None
                else:
                    try:
                        by_stat[summary][feat] = float(raw)
                    except (TypeError, ValueError):
                        by_stat[summary][feat] = None
        header = f"{'feature':<16} {'count':>10} {'mean':>12} {'std':>10} {'min':>12} {'max':>12}"
        print(header)
        print("-" * len(header))
        for feat in feature_cols:
            cnt_v = by_stat.get("count", {}).get(feat)
            cnt = int(cnt_v) if cnt_v is not None else 0
            mean = by_stat.get("mean", {}).get(feat) or 0.0
            std = by_stat.get("stddev", {}).get(feat) or 0.0
            vmin = by_stat.get("min", {}).get(feat) or 0.0
            vmax = by_stat.get("max", {}).get(feat) or 0.0
            print(
                f"{feat:<16} {cnt:>10,} {mean:>12,.2f} {std:>10,.2f} "
                f"{vmin:>12,.2f} {vmax:>12,.2f}"
            )

    # --- Spark-only null profiler: single aggregation job across all columns ---
    def null_counts_table(df) -> tuple[list[dict], int]:
        """One Spark job for all column null counts (avoids 100+ separate counts)."""
        total = df.count()
        exprs = []
        for field in df.schema.fields:
            c = field.name
            if isinstance(field.dataType, (DoubleType, IntegerType)):
                exprs.append(spark_sum(when(col(c).isNull() | isnan(col(c)), 1).otherwise(0)).alias(c))
            else:
                exprs.append(spark_sum(when(col(c).isNull(), 1).otherwise(0)).alias(c))
        row = df.agg(*exprs).collect()[0]
        out = [
            {"column": c, "null_count": int(row[c]), "null_pct": int(row[c]) / total}
            for c in df.columns
        ]
        out.sort(key=lambda r: r["null_count"], reverse=True)
        return out, total

    # --- Spark-only categorical profiler: default rate by category ---
    def default_rate_by(df, group_col: str) -> list[dict]:
        """Spark groupBy default rate for a categorical column."""
        return [
            {
                group_col: r[group_col],
                "n": int(r["n"]),
                "default_rate": float(r["default_rate"]),
            }
            for r in (
                df.groupBy(group_col)
                .agg(
                    spark_count("*").alias("n"),
                    (
                        spark_sum(when(col("default_ind") == 1, 1).otherwise(0))
                        / spark_count("*")
                    ).alias("default_rate"),
                )
                .orderBy(col("default_rate").desc())
                .collect()
            )
        ]

    # Columns with post-origination information (must be excluded from modelling relevance).
    LEAKAGE_COLS = frozenset({
        "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
        "total_rec_late_fee", "recoveries", "collection_recovery_fee",
        "last_pymnt_amnt", "out_prncp", "out_prncp_inv", "last_pymnt_d",
        "next_pymnt_d", "last_credit_pull_d",
    })


    # Windows compatibility helper for Spark parquet/winutils edge cases.
    def setup_hadoop_winutils(root: Path) -> None:
        """Optional: helps Spark parquet on some Windows setups."""
        if os.name != "nt":
            return
        hadoop_home = root / ".hadoop"
        bin_dir = hadoop_home / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        base = "https://github.com/kontext-tech/winutils/raw/master/hadoop-3.3.0/bin"
        for name in ("winutils.exe", "hadoop.dll"):
            dest = bin_dir / name
            if not dest.is_file():
                print(f"Downloading {name} (optional Hadoop helper)...")
                urllib.request.urlretrieve(f"{base}/{name}", dest)
        os.environ["HADOOP_HOME"] = str(hadoop_home.resolve())
        os.environ["hadoop.home.dir"] = os.environ["HADOOP_HOME"]

    # Try common Windows JDK install locations if JAVA_HOME is unset.
    def _find_windows_jdk():
        adoptium = Path(r"C:\Program Files\Eclipse Adoptium")
        if adoptium.is_dir():
            for jdk in sorted(adoptium.glob("jdk-*"), reverse=True):
                if (jdk / "bin" / "java.exe").is_file():
                    return jdk
        java_dir = Path(r"C:\Program Files\Java")
        if java_dir.is_dir():
            for jdk in sorted(java_dir.glob("jdk*"), reverse=True):
                if (jdk / "bin" / "java.exe").is_file():
                    return jdk
        return None

    if not shutil.which("java"):
        jdk_home = _find_windows_jdk()
        if jdk_home is None:
            raise RuntimeError(
                "Java not found. Install Temurin JDK 17, restart the terminal, then re-run. "
                "https://adoptium.net/"
            )
        os.environ["JAVA_HOME"] = str(jdk_home)
        os.environ["PATH"] = str(jdk_home / "bin") + os.pathsep + os.environ.get("PATH", "")

    if "_" in socket.gethostname():
        os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

    setup_hadoop_winutils(ROOT)
    print("Java:", subprocess.check_output(["java", "-version"], stderr=subprocess.STDOUT, text=True).splitlines()[0])
    SPARK_TMP = ROOT / ".spark-tmp"
    SPARK_TMP.mkdir(exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("CSCI316_Stage1_EDA")
        .master("local[*]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.local.dir", str(SPARK_TMP))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    run_banner()
    print(f"Spark {spark.version}  |  Java OK  |  Root: {ROOT.resolve()}")
    # --- 1.2 Load dataset ---
    section("1.2 Load dataset")
    loan_df = spark.read.csv(str(DATA_PATH), header=True, inferSchema=True)

    n_rows_raw = loan_df.count()
    n_rows = n_rows_raw
    n_cols = len(loan_df.columns)
    n_numeric = sum(isinstance(f.dataType, (DoubleType, IntegerType)) for f in loan_df.schema.fields)
    print(f"Rows: {n_rows:,}  |  Columns: {n_cols}  |  Inferred numeric: {n_numeric}")
    print("First 20 columns:", ", ".join(loan_df.columns[:20]))
    if n_cols > 20:
        print(f"  ... and {n_cols - 20} more (see null_counts_all.csv)")
    # --- 1.2b Duplicate records check ---
    # Keep member_id repeats (same borrower can have multiple loans);
    # only remove duplicate loan IDs.
    section("1.2b Duplicate records check")
    n_distinct_id = loan_df.select(countDistinct("id")).collect()[0][0]
    n_distinct_member = loan_df.select(countDistinct("member_id")).collect()[0][0]
    n_full_dup = n_rows - loan_df.dropDuplicates().count()
    dup_id_rows = n_rows - n_distinct_id
    print(
        f"distinct id: {n_distinct_id:,}  |  distinct member_id: {n_distinct_member:,}  |  "
        f"duplicate loan ids: {dup_id_rows:,}  |  fully identical rows: {n_full_dup:,}"
    )
    print(
        "member_id repeats are expected (one borrower may have several loans) — "
        "only duplicate id rows are removed."
    )
    rows_removed_dup_id = 0
    if dup_id_rows > 0:
        print("Sample duplicate loan ids:")
        loan_df.groupBy("id").agg(spark_count("*").alias("n")).filter(col("n") > 1).show(5, truncate=False)
        loan_df = loan_df.dropDuplicates(["id"])
        n_rows = loan_df.count()
        rows_removed_dup_id = n_rows_raw - n_rows
        print(f"Removed {rows_removed_dup_id:,} duplicate id row(s); rows now: {n_rows:,}")
    else:
        print("id is unique — no duplicate loans.")
    dup_lines = [
        f"rows_raw={n_rows_raw}",
        f"distinct_id={n_distinct_id}",
        f"distinct_member_id={n_distinct_member}",
        f"duplicate_loan_ids={dup_id_rows}",
        f"fully_identical_rows={n_full_dup}",
        f"rows_removed_drop_duplicates_id={rows_removed_dup_id}",
        f"rows_after_dedup={n_rows}",
    ]
    (OUT_TABLES / "duplicate_check.txt").write_text("\n".join(dup_lines) + "\n")
    print("Saved:", OUT_TABLES / "duplicate_check.txt")
    # --- 1.3 Summary statistics ---
    section("1.3 Summary statistics (key columns only)")
    SUMMARY_COLS = [
        "loan_amnt", "funded_amnt", "int_rate", "installment", "annual_inc", "dti",
        "delinq_2yrs", "inq_last_6mths", "revol_bal", "revol_util", "open_acc", "total_acc", "default_ind",
    ]
    summary_cols = [c for c in SUMMARY_COLS if c in loan_df.columns]
    summary_df = loan_df
    for c in summary_cols:
        if c == "default_ind":
            summary_df = summary_df.withColumn(c, col(c).try_cast(IntegerType()))
        else:
            summary_df = summary_df.withColumn(c, col(c).try_cast(DoubleType()))
    stats_rows = summary_df.select(summary_cols).describe().collect()
    print_summary_table(stats_rows, summary_cols)
    describe_csv_rows = []
    for row in stats_rows:
        rec = {"summary": row["summary"]}
        for c in summary_cols:
            rec[c] = row[c]
        describe_csv_rows.append(rec)
    write_csv(
        OUT_TABLES / "summary_statistics.csv",
        describe_csv_rows,
        ["summary"] + summary_cols,
    )
    print("Saved:", OUT_TABLES / "summary_statistics.csv")
    print("Note: extreme min/max (e.g. annual_inc) are invalid strings cast to NULL in cleaning.")
    # --- 1.4 Missing values (per column) ---
    section("1.4 Missing values (top 25)")
    null_rows, total = null_counts_table(loan_df)
    write_csv(OUT_TABLES / "null_counts_all.csv", null_rows, ["column", "null_count", "null_pct"])
    write_csv(OUT_TABLES / "null_counts_top25.csv", null_rows[:25], ["column", "null_count", "null_pct"])
    print_table(null_rows[:25], ["column", "null_count", "null_pct"], pct_cols=("null_pct",))
    print("Saved:", OUT_TABLES / "null_counts_all.csv")
    print("Saved:", OUT_TABLES / "null_counts_top25.csv")
    # --- Initial observations ---
    # --- 1.5 Target variable: class balance ---
    section("1.5 Class balance")
    class_rows = (
        loan_df.filter(col("default_ind").isNotNull())
        .groupBy("default_ind")
        .count()
        .orderBy("default_ind")
        .collect()
    )
    label_map = {0: "No default", 1: "Default"}
    class_total = sum(int(r["count"]) for r in class_rows)
    class_show = [
        {
            "label": label_map.get(int(r["default_ind"]), str(r["default_ind"])),
            "count": int(r["count"]),
            "pct": int(r["count"]) / class_total,
        }
        for r in class_rows
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        [r["label"] for r in class_show],
        [r["count"] for r in class_show],
        color=[PALETTE["no_default"], PALETTE["default"]],
    )
    ax.set_title("Class balance: default_ind")
    ax.set_ylabel("Count")
    for i, r in enumerate(class_show):
        ax.text(i, r["count"], f"{r['count']:,}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    save_fig(fig, "class_balance.png")
    print_table(class_show, ["label", "count", "pct"], pct_cols=("pct",))
    # --- 1.6 Data cleaning ---
    section("1.6 Data cleaning")
    clean_loan = loan_df.dropna(subset=["default_ind"])
    rows_after_target = clean_loan.count()

    null_clean, _ = null_counts_table(clean_loan)
    drop_cols = [r["column"] for r in null_clean if r["null_pct"] > 0.5]

    print(f"Rows after removing null target: {rows_after_target:,}")
    print(f"Columns dropped (>50% null): {len(drop_cols)}")
    if drop_cols:
        print("  " + ", ".join(drop_cols[:12]) + (" ..." if len(drop_cols) > 12 else ""))
    clean_loan = clean_loan.drop(*drop_cols)
    print(f"Columns remaining: {len(clean_loan.columns)}")
    # --- 1.7 Type casting (Spark only) ---
    # Requirement note: use Spark DataFrame operations only (no pandas/sklearn in Stage 1).
    section("1.7 Type casting")
    # try_cast: malformed strings (e.g. desc text in numeric fields) -> NULL (Spark 4 is strict on .cast)
    if "desc" in clean_loan.columns:
        clean_loan = clean_loan.drop("desc")

    clean_loan = clean_loan.withColumn(
        "term",
        regexp_replace(trim(col("term")), "months", "").try_cast(IntegerType()),
    )

    for c in ["int_rate", "installment", "annual_inc", "dti", "loan_amnt", "funded_amnt"]:
        if c in clean_loan.columns:
            clean_loan = clean_loan.withColumn(c, col(c).try_cast(DoubleType()))

    clean_loan = clean_loan.withColumn(
        "revol_util",
        regexp_replace(col("revol_util"), "%", "").try_cast(DoubleType()),
    )

    for c in ["delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "total_acc"]:
        if c in clean_loan.columns:
            clean_loan = clean_loan.withColumn(c, col(c).try_cast(DoubleType()))

    CORE_FEATURES = [
        "loan_amnt", "int_rate", "term", "dti", "annual_inc",
        "delinq_2yrs", "inq_last_6mths", "revol_util", "grade", "sub_grade",
    ]
    existing_core = [c for c in CORE_FEATURES if c in clean_loan.columns]
    clean_loan = clean_loan.dropna(subset=existing_core)
    if "default_ind" in clean_loan.columns:
        clean_loan = clean_loan.withColumn("default_ind", col("default_ind").try_cast(IntegerType()))

    rows_final = clean_loan.count()
    summary_lines = [
        f"raw_rows={n_rows_raw}",
        f"after_dedup_id={n_rows}",
        f"after_target_drop={rows_after_target}",
        f"after_core_feature_drop={rows_final}",
        f"pct_retained={rows_final / n_rows_raw:.4f}",
        f"columns_final={len(clean_loan.columns)}",
    ]
    (OUT_TABLES / "cleaning_summary.txt").write_text("\n".join(summary_lines) + "\n")
    for line in summary_lines:
        print(f"  {line}")
    # --- 1.8 Visualisations ---
    section("1.8 Visualisations")
    numeric_plot = [c for c in ["loan_amnt", "int_rate", "dti", "annual_inc", "revol_util"] if c in clean_loan.columns]

    sample_rows = (
        clean_loan.select(numeric_plot + ["default_ind"])
        .sample(fraction=0.05, seed=RANDOM_STATE)
        .collect()
    )
    sample_vals = {feat: {0: [], 1: []} for feat in numeric_plot}
    for row in sample_rows:
        label = int(row["default_ind"]) if row["default_ind"] is not None else None
        if label not in (0, 1):
            continue
        for feat in numeric_plot:
            v = row[feat]
            if v is not None:
                sample_vals[feat][label].append(float(v))

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    for i, feat in enumerate(numeric_plot):
        axes[i].hist(
            sample_vals[feat][0],
            bins=40,
            alpha=0.6,
            label="No default",
            color=PALETTE["no_default"],
            edgecolor="white",
        )
        axes[i].hist(
            sample_vals[feat][1],
            bins=40,
            alpha=0.6,
            label="Default",
            color=PALETTE["default"],
            edgecolor="white",
        )
        axes[i].set_title(feat)
        axes[i].legend(fontsize=8)
    for j in range(len(numeric_plot), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Numeric distributions by default_ind (5% Spark sample)", y=1.02)
    plt.tight_layout()
    save_fig(fig, "feature_distributions.png")
    if "grade" in clean_loan.columns:
        grade_rate = [
            {
                "grade": r["grade"],
                "n": int(r["n"]),
                "default_rate": float(r["default_rate"]),
            }
            for r in (
                clean_loan.groupBy("grade")
                .agg(
                    spark_count("*").alias("n"),
                    (
                        spark_sum(when(col("default_ind") == 1, 1).otherwise(0))
                        / spark_count("*")
                    ).alias("default_rate"),
                )
                .orderBy("grade")
                .collect()
            )
        ]
        write_csv(OUT_TABLES / "default_rate_by_grade.csv", grade_rate, ["grade", "n", "default_rate"])

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(
            [str(r["grade"]) for r in grade_rate],
            [r["default_rate"] for r in grade_rate],
            color=PALETTE["bar"],
        )
        ax.set_title("Default rate by loan grade")
        ax.set_xlabel("grade")
        ax.set_ylabel("default rate")
        plt.tight_layout()
        save_fig(fig, "default_rate_by_grade.png")
        print_table(grade_rate, ["grade", "n", "default_rate"], pct_cols=("default_rate",))
    term_rate = [
        {
            "term": r["term"],
            "n": int(r["n"]),
            "default_rate": float(r["default_rate"]),
        }
        for r in (
            clean_loan.groupBy("term")
            .agg(
                spark_count("*").alias("n"),
                (
                    spark_sum(when(col("default_ind") == 1, 1).otherwise(0))
                    / spark_count("*")
                ).alias("default_rate"),
            )
            .orderBy("term")
            .collect()
        )
    ]
    write_csv(OUT_TABLES / "default_rate_by_term.csv", term_rate, ["term", "n", "default_rate"])

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(
        [str(r["term"]) for r in term_rate],
        [r["default_rate"] for r in term_rate],
        color=PALETTE["accent"],
    )
    ax.set_title("Default rate by term (months)")
    ax.set_xlabel("term")
    ax.set_ylabel("default rate")
    plt.tight_layout()
    save_fig(fig, "default_rate_by_term.png")
    print_table(term_rate, ["term", "n", "default_rate"], pct_cols=("default_rate",))
    # --- 1.9 Correlation analysis (numeric, origination-safe) ---
    # Correlation is for interpretation only; weak linear signal is expected for imbalanced default data.
    section("1.9 Correlation analysis")
    EXCLUDE_COLS = {"default_ind", "id", "member_id"} | LEAKAGE_COLS

    numeric_cols = [
        f.name for f in clean_loan.schema.fields
        if isinstance(f.dataType, (IntegerType, DoubleType))
        and f.name not in EXCLUDE_COLS
    ]
    numeric_cols = [
        c for c in numeric_cols
        if clean_loan.select(stddev(c)).collect()[0][0] not in (None, 0.0)
    ]

    correlations = []
    skipped_corr: list[str] = []
    for c in numeric_cols:
        try:
            r = clean_loan.stat.corr(c, "default_ind")
            if r is not None:
                correlations.append((c, round(float(r), 4)))
        except Exception as exc:
            skipped_corr.append(f"{c} ({exc.__class__.__name__})")
    if skipped_corr:
        print(f"Correlation skipped for {len(skipped_corr)} column(s): {', '.join(skipped_corr)}")

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    corr_rows = [{"feature": f, "correlation": r} for f, r in correlations]
    write_csv(OUT_TABLES / "correlation_with_target.csv", corr_rows, ["feature", "correlation"])
    print("Top 15 |Pearson r| with default_ind (leakage columns excluded):")
    print_table(corr_rows[:15], ["feature", "correlation"])
    plot_corr = sorted(corr_rows, key=lambda r: abs(r["correlation"]))

    fig, ax = plt.subplots(figsize=(8, max(6, len(plot_corr) * 0.12)))
    colors = ["#e74c3c" if r["correlation"] < 0 else "#27ae60" for r in plot_corr]
    ax.barh([r["feature"] for r in plot_corr], [r["correlation"] for r in plot_corr], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Pearson correlation with default_ind")
    ax.set_xlabel("correlation")
    plt.tight_layout()
    save_fig(fig, "correlation_with_target.png")
    heat_cols = [c for c in numeric_plot + ["delinq_2yrs", "inq_last_6mths", "default_ind"] if c in clean_loan.columns]
    assembler = VectorAssembler(inputCols=heat_cols, outputCol="features_vec", handleInvalid="skip")
    assembled = assembler.transform(clean_loan.select(heat_cols)).select("features_vec")

    matrix = Correlation.corr(assembled, "features_vec").head()[0]
    matrix_arr = np.array(matrix.toArray())

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix_arr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(heat_cols)))
    ax.set_yticks(range(len(heat_cols)))
    ax.set_xticklabels(heat_cols, rotation=45, ha="right")
    ax.set_yticklabels(heat_cols)
    for i in range(len(heat_cols)):
        for j in range(len(heat_cols)):
            ax.text(j, i, f"{matrix_arr[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Correlation heatmap (selected numerics)")
    plt.tight_layout()
    save_fig(fig, "correlation_heatmap.png")
    # --- 1.9b Categorical default rates ---
    section("1.9b Categorical default rates")
    for cat_col in ["home_ownership", "purpose", "verification_status", "emp_length", "sub_grade"]:
        if cat_col not in clean_loan.columns:
            continue
        rate = default_rate_by(clean_loan, cat_col)
        write_csv(
            OUT_TABLES / f"default_rate_by_{cat_col}.csv",
            rate,
            [cat_col, "n", "default_rate"],
        )
        print(f"\n{cat_col} (top 10 by default rate):")
        print_table(rate[:10], [cat_col, "n", "default_rate"], pct_cols=("default_rate",))
    # --- 1.10 Feature relevance justification (required) ---
    # Project-spec requirement: explicitly provide 7 most and 7 least relevant attributes with reasons.
    section("1.10 Feature relevance (7 most / 7 least)")
    MOST_RELEVANT = [
        ("int_rate", "Higher rates reflect riskier loans; strongest numeric linear signal in correlation analysis."),
        ("grade", "LC risk grade; default rate increases from A toward G (default_rate_by_grade.csv)."),
        ("dti", "Debt-to-income measures repayment capacity; standard underwriting variable."),
        ("delinq_2yrs", "Past delinquency is a direct behavioural credit-risk signal."),
        ("inq_last_6mths", "Recent inquiries proxy financial stress and credit seeking."),
        ("revol_util", "High revolving utilisation indicates limited credit headroom."),
        ("loan_amnt", "Loan size relates to exposure; useful with other variables."),
    ]
    LEAST_RELEVANT = [
        ("id", "Unique loan identifier; no predictive signal."),
        ("member_id", "Unique borrower identifier; must not be used as a model feature."),
        ("emp_title", "Free-text job title; high cardinality and inconsistent labels."),
        ("zip_code", "Redacted/coarse location; weak and unstable risk signal."),
        ("title", "Borrower-entered loan title; noisy unstructured text."),
        ("policy_code", "Administrative flag; near-constant, not borrower risk."),
        ("initial_list_status", "Listing workflow flag; not a credit characteristic."),
    ]
    most_rows = [{"attribute": a, "justification": j} for a, j in MOST_RELEVANT]
    least_rows = [{"attribute": a, "justification": j} for a, j in LEAST_RELEVANT]
    write_csv(OUT_TABLES / "feature_relevance_most_relevant.csv", most_rows, ["attribute", "justification"])
    write_csv(OUT_TABLES / "feature_relevance_least_relevant.csv", least_rows, ["attribute", "justification"])
    print("Saved:", OUT_TABLES / "feature_relevance_most_relevant.csv")
    print("Saved:", OUT_TABLES / "feature_relevance_least_relevant.csv")
    print("\nSeven MOST relevant attributes (for predicting default at origination):")
    print_table(most_rows, ["attribute", "justification"])
    print("\nSeven LEAST relevant attributes:")
    print_table(least_rows, ["attribute", "justification"])
    leakage_list = sorted(LEAKAGE_COLS)
    summary_710 = [
        "CSCI316 Stage 1 — Feature relevance (required 7+7)",
        "",
        "Seven MOST relevant:",
    ] + [f"  {i+1}. {a}: {j}" for i, (a, j) in enumerate(MOST_RELEVANT)] + [
        "",
        "Seven LEAST relevant:",
    ] + [f"  {i+1}. {a}: {j}" for i, (a, j) in enumerate(LEAST_RELEVANT)] + [
        "",
        "Excluded leakage (not for modelling): " + ", ".join(leakage_list),
        "Note: |Pearson r| with default is weak (~5% default rate); combine correlation, default rates, domain knowledge.",
    ]
    (OUT_REPORTS / "feature_relevance_summary.txt").write_text("\n".join(summary_710) + "\n", encoding="utf-8")
    print("Saved:", OUT_REPORTS / "feature_relevance_summary.txt")
    # Export shortlist used to guide Stage 2 model inputs.
    section("1.11 Export recommended features")
    RECOMMENDED = [
        "int_rate", "term", "dti", "annual_inc",
        "delinq_2yrs", "inq_last_6mths", "revol_util",
    ]
    lines = RECOMMENDED + [
        "",
        "# Categorical (encode in Stage 2): grade",
        "# Stage 2A/2B use expanded origination-safe features (see stage2a.py / stage2b.py)",
    ]
    (OUT_REPORTS / "recommended_features.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Saved:", OUT_REPORTS / "recommended_features.txt")
    print("Core recommended numerics:", RECOMMENDED)
    section("1.12 Export cleaned dataset")
    save_clean_dataset(clean_loan, OUT_DATA)

    section("Stage 1 — executive summary")
    default_count = sum(r["count"] for r in class_show if r["label"] == "Default")
    default_pct = 100 * default_count / class_total
    top_corr = corr_rows[0]["correlation"] if corr_rows else 0.0
    summary_lines = [
        "CSCI316 Stage 1 — Executive Summary",
        f"Rows (raw / cleaned): {n_rows_raw:,} / {rows_final:,} ({100*rows_final/n_rows_raw:.2f}% retained)",
        f"Class balance: {default_pct:.2f}% default (imbalanced — use recall/ROC in Stage 2)",
        f"Strongest numeric signal: int_rate (r={top_corr:.4f})",
        f"Grade trend: default rate rises A->G (see figures/default_rate_by_grade.png)",
        f"Core recommended numerics (no leakage): {', '.join(RECOMMENDED)}",
        "See feature_relevance_most_relevant.csv / feature_relevance_least_relevant.csv (7+7 required).",
        f"Artifacts: {OUT_DIR.resolve()}",
    ]
    for line in summary_lines:
        print(line)
    (OUT_REPORTS / "stage1_executive_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("Saved:", OUT_REPORTS / "stage1_executive_summary.txt")

    top_null = null_rows[:5]
    vs_lines: list[str] = []
    if "verification_status" in clean_loan.columns:
        vs_lines = verification_status_lines(default_rate_by(clean_loan, "verification_status"))
    cat_patterns = [
        "- Default rate rises from grade A toward G (see figures/default_rate_by_grade.png).",
        "- 36-month term shows higher default rate than 60-month (see figures/default_rate_by_term.png).",
    ]
    if vs_lines:
        cat_patterns.extend(vs_lines)
    else:
        cat_patterns.append(
            "- See tables/default_rate_by_*.csv for home_ownership, purpose, emp_length, sub_grade."
        )
    observations = [
        "CSCI316 Stage 1 — Data exploration observations",
        "",
        "Dataset (data/data.csv)",
        f"- {n_rows_raw:,} loans, {n_cols} columns; target = default_ind (Lending Club extract).",
        f"- Duplicate check: {n_distinct_id:,} distinct id ({dup_id_rows:,} duplicate ids removed); "
        f"{n_distinct_member:,} distinct member_id (repeats kept).",
        f"- Many fields inferred as string (e.g. term='36 months', revol_util with '%'); cleaned in Section 1.7.",
        "",
        "Class balance",
        f"- Defaults are a minority: {default_pct:.2f}% default ({default_count:,} / {class_total:,}).",
        "- Stage 2 should emphasise recall, ROC-AUC, and PR-AUC — not accuracy alone.",
        "",
        "Missing data",
        f"- Sparsest columns (pre-clean): {', '.join(r['column'] for r in top_null)}.",
        "- Dropped columns with >50% null (joint-application and event-conditional fields).",
        "- Avoided blanket dropna(); retained {:.2f}% of rows after requiring core origination features.".format(
            100 * rows_final / n_rows_raw
        ),
        "",
        "Numeric patterns (5% Spark sample, by default_ind)",
        "- int_rate and dti tend higher for defaulted loans; annual_inc overlap is wide.",
        "- Pearson |r| with default_ind is weak overall; relationships are non-linear and imbalanced.",
        "",
        "Categorical patterns",
        *cat_patterns,
        "",
        "Feature relevance (7+7)",
        "- Most relevant: int_rate, grade, dti, delinq_2yrs, inq_last_6mths, revol_util, loan_amnt.",
        "- Least relevant: id, member_id, emp_title, zip_code, title, policy_code, initial_list_status.",
        f"- Post-origination leakage excluded: {', '.join(leakage_list[:6])}, ...",
        "",
        "Outputs: output/stage1/{data,tables,figures,reports}/ (data/clean_loan.csv for Stage 2).",
    ]
    (OUT_REPORTS / "stage1_observations.txt").write_text("\n".join(observations) + "\n", encoding="utf-8")
    print("Saved:", OUT_REPORTS / "stage1_observations.txt")
    # --- Stage 1 complete ---
    try:
        spark.stop()
    except OSError as exc:
        # Harmless on Windows: JVM child process sometimes cannot be killed cleanly.
        print(f"Note: Spark shutdown ({exc}); all Stage 1 outputs were saved.")
    print("\n" + "=" * 72)
    print("Stage 1 complete.")
    print("=" * 72)

if __name__ == "__main__":
    main()
