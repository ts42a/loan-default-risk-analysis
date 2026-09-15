"""Timestamped result folders under results/{stage1,spark,tensorflow}/.

Each script call creates ``results/<stage>/<YYYYMMDD_HHMMSS>_<tag>/`` and
appends to that stage's ``runs.log``.

* ``LATEST.txt`` always points at the newest run (smoke or full) so Stage 2 can
  chain immediately after Stage 1.
* ``FINAL.txt`` plus a copied ``final/`` directory are updated only for a
  successful **full** run (sample = 1.0), or when ``STAGE_PIN_FINAL=1``.
  Slides should insert figures from ``results/<stage>/final/``.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path


STAGES = ("stage1", "spark", "tensorflow")
SKIP_WHEN_PUBLISHING = {"data"}
CLEAN_CSV = "data_clean.csv"


def stamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sample_tag(sample: float) -> str:
    if sample >= 1.0:
        return "full"
    percent = max(1, int(round(sample * 100)))
    return f"s{percent:02d}"


def should_pin_final(sample: float) -> bool:
    flag = os.environ.get("STAGE_PIN_FINAL")
    if flag == "0":
        return False
    if flag == "1":
        return True
    return sample >= 1.0


def start_run(root: Path, stage: str, *, sample: float = 1.0, note: str = "") -> Path:
    """Create a tagged run folder, write LATEST, append to runs.log."""
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {STAGES}")
    stamp = f"{stamp_now()}_{sample_tag(sample)}"
    run_dir = root / "results" / stage / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = root / "results" / stage
    (stage_dir / "LATEST.txt").write_text(stamp + "\n", encoding="utf-8")
    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    extra = f"  {note}" if note else ""
    with (stage_dir / "runs.log").open("a", encoding="utf-8") as fh:
        fh.write(f"{started}  started   {stamp}{extra}\n")
    (run_dir / "run.log").write_text(
        f"started : {started}\nstamp   : {stamp}\nsample  : {sample}\nnote    : {note}\n",
        encoding="utf-8",
    )
    return run_dir


def publish_final(run_dir: Path) -> Path:
    """Copy slide artefacts into results/<stage>/final/ (skips bulky data/)."""
    dest = run_dir.parent / "final"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for item in run_dir.iterdir():
        if item.name in SKIP_WHEN_PUBLISHING:
            continue
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    (run_dir.parent / "FINAL.txt").write_text(run_dir.name + "\n", encoding="utf-8")
    (dest / "PINNED_FROM.txt").write_text(run_dir.name + "\n", encoding="utf-8")
    return dest


def finish_run(
    root: Path,
    stage: str,
    run_dir: Path,
    *,
    status: str = "ok",
    sample: float = 1.0,
) -> None:
    stamp = run_dir.name
    finished = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pinned = ""
    if status == "ok" and should_pin_final(sample):
        dest = publish_final(run_dir)
        pinned = "  pinned=FINAL"
        print(f"Pinned slide artefacts: {dest.relative_to(root).as_posix()}/  (from {stamp})")
    line = f"{finished}  finished  {stamp}  {status}{pinned}\n"
    with (root / "results" / stage / "runs.log").open("a", encoding="utf-8") as fh:
        fh.write(line)
    log = run_dir / "run.log"
    existing = log.read_text(encoding="utf-8") if log.is_file() else ""
    log.write_text(
        existing + f"finished: {finished}\nstatus  : {status}\npinned  : {bool(pinned)}\n",
        encoding="utf-8",
    )


def _run_from_pointer(stage_dir: Path, name: str) -> Path | None:
    """Read LATEST.txt / FINAL.txt (plain LATEST/FINAL still accepted)."""
    for filename in (f"{name}.txt", name):
        pointer = stage_dir / filename
        if not pointer.is_file():
            continue
        candidate = stage_dir / pointer.read_text(encoding="utf-8").strip()
        if candidate.is_dir():
            return candidate
    return None


def latest_run(root: Path, stage: str) -> Path | None:
    stage_dir = root / "results" / stage
    found = _run_from_pointer(stage_dir, "LATEST")
    if found is not None:
        return found
    stamped = sorted(
        (p for p in stage_dir.iterdir() if p.is_dir() and p.name[:8].isdigit()),
        reverse=True,
    )
    return stamped[0] if stamped else None


def final_run(root: Path, stage: str) -> Path | None:
    return _run_from_pointer(root / "results" / stage, "FINAL")


def canonical_clean_csv(root: Path) -> Path:
    """Stable handover file. Stage 1 writes this once; Stage 2 always reads it."""
    return root / "results" / "stage1" / "data" / CLEAN_CSV


def publish_canonical_clean(root: Path, source: Path) -> Path:
    dest = canonical_clean_csv(root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != dest.resolve():
        shutil.copy2(source, dest)
    return dest


def _stage1_csv(run: Path | None) -> Path | None:
    if run is None:
        return None
    nested = run / "data" / CLEAN_CSV
    return nested if nested.is_file() else None


def clean_loan_path(root: Path) -> Path | None:
    """Stage 2 always prefers the stable full ``data_clean.csv``.

    Stage 1 is run once on the whole extract. Later 10% / 30% experiments only
    sample that file via ``STAGE2_SAMPLE``.
    """
    canonical = canonical_clean_csv(root)
    if canonical.is_file():
        return canonical
    found = _stage1_csv(latest_run(root, "stage1"))
    if found is not None:
        return found
    found = _stage1_csv(final_run(root, "stage1"))
    if found is not None:
        return found
    return None


def _metrics_csv(run: Path | None) -> Path | None:
    if run is None:
        return None
    nested = run / "model_metrics.csv"
    return nested if nested.is_file() else None


def spark_metrics_path(root: Path) -> Path | None:
    prefer_final = os.environ.get("STAGE_USE_FINAL") == "1"
    if prefer_final:
        found = _metrics_csv(final_run(root, "spark"))
        if found is not None:
            return found
    found = _metrics_csv(latest_run(root, "spark"))
    if found is not None:
        return found
    found = _metrics_csv(final_run(root, "spark"))
    if found is not None:
        return found
    legacy = root / "results" / "spark" / "model_metrics.csv"
    return legacy if legacy.is_file() else None
