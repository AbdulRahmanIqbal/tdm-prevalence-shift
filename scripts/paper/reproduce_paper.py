"""Rebuild every table and figure of the paper from the released outputs and compare them with the files in paper/.

Needs: the result folders under results/ (unpack results/*.zip first, or pass --unzip) and data/processed/ (built
with `s1bench download` and `s1bench prepare`, see REPRODUCE.md section 1; several scripts read the test labels).
Runs on a CPU. The EM simulation with distorted posteriors takes about an hour; --skip-slow leaves it out and keeps
the two simulation tables as they are.

The comparison covers exactly the tables and figures that the paper includes (read from paper/main.tex and
paper/sections/*.tex). Tables must be byte-identical; for PDF figures the creation date is ignored. The script exits
with status 1 if any file differs or any script fails.

Run: python scripts/paper/reproduce_paper.py [--unzip] [--skip-slow]
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"

# (script, arguments); order only matters for readability, every script reads results/ and data/ only
STEPS = [
    ("make_tables.py", []),
    ("exp2_descriptives.py", []),
    ("make_figures.py", []),
    ("make_pipeline_figure.py", []),
    ("budget_tables.py", []),
    ("fig_budget.py", []),
    ("fig_calibration.py", []),
    ("priorsens_table.py", []),
    ("pa_population.py", []),
    ("hwu64_overlap_sensitivity.py", []),
    ("implicit_prior_table.py", []),
    ("review3_checks.py", []),
    ("deberta_seed_shift.py", []),
    ("em_true_prior.py", []),
    ("content_free_calibration.py", []),
    ("router_calibration.py", []),
    ("prompt_precision_checks.py", []),
    ("fig_revision.py", []),
]
SLOW = [("em_candidate_simulation.py", ["--miscalibration"])]
RESULT_DIRS = ["final_local", "final_train", "final_jev", "final_analysis", "final_budget", "final_budget_analysis",
               "final_qlora", "final_contentfree", "final_router", "final_promptsens", "final_bf16",
               "final_realistic_analysis", "final_analysis_calibrob"]


def used_files() -> tuple[list[str], list[str]]:
    tex = "".join(p.read_text(encoding="utf-8") for p in [PAPER / "main.tex", *sorted((PAPER / "sections").glob("*.tex"))])
    tables = sorted(set(re.findall(r"tables/(tab_[A-Za-z0-9_]+)", tex)))
    figures = sorted(set(re.findall(r"figures/([A-Za-z0-9_]+)\.pdf", tex)))
    return [f"{t}.tex" for t in tables], [f"{f}.pdf" for f in figures]


def pdf_body(path: Path) -> bytes:
    return re.sub(rb"/CreationDate \([^)]*\)", b"", path.read_bytes())


def unzip_results() -> None:
    for z in sorted((ROOT / "results").glob("*.zip")):
        with zipfile.ZipFile(z) as f:
            f.extractall(ROOT / "results")
        print(f"unpacked {z.name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unzip", action="store_true", help="unpack results/*.zip into results/ first")
    ap.add_argument("--skip-slow", action="store_true", help="leave out the one-hour EM simulation")
    args = ap.parse_args()

    if args.unzip:
        unzip_results()
    missing = [d for d in RESULT_DIRS if not (ROOT / "results" / d).is_dir()]
    if missing:
        print("missing result folders (unpack results/*.zip or use --unzip):", ", ".join(missing))
        return 1
    if not (ROOT / "data" / "processed").is_dir():
        print("missing data/processed: run `s1bench download` and `s1bench prepare` first (REPRODUCE.md section 1)")
        return 1

    tables, figures = used_files()
    expected = Path(tempfile.mkdtemp(prefix="paper_expected_"))
    for name in tables:
        shutil.copy2(PAPER / "tables" / name, expected / name)
    for name in figures:
        shutil.copy2(PAPER / "figures" / name, expected / name)
    print(f"paper uses {len(tables)} tables and {len(figures)} figures; reference copies in {expected}")

    failed = []
    for script, extra in STEPS + ([] if args.skip_slow else SLOW):
        t0 = time.time()
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "paper" / script), *extra], cwd=ROOT,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        status = "ok" if r.returncode == 0 else "FAILED"
        print(f"  {status:6s} {script} {' '.join(extra)} ({time.time() - t0:.0f} s)")
        if r.returncode != 0:
            failed.append(script)
            print(r.stdout[-1500:] + r.stderr[-1500:])

    differ = [n for n in tables if (PAPER / "tables" / n).read_bytes() != (expected / n).read_bytes()]
    differ += [n for n in figures if pdf_body(PAPER / "figures" / n) != pdf_body(expected / n)]
    print(f"\ntables identical: {len(tables) - sum(n in differ for n in tables)}/{len(tables)}; "
          f"figures identical (creation date ignored): {len(figures) - sum(n in differ for n in figures)}/{len(figures)}")
    if differ:
        print("differ:", ", ".join(differ), f"(reference copies kept in {expected})")
    if failed:
        print("failed scripts:", ", ".join(failed))
    return 1 if (differ or failed) else 0


if __name__ == "__main__":
    sys.exit(main())
