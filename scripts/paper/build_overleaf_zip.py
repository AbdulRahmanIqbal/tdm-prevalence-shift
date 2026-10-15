"""Build paper_overleaf.zip (the files Overleaf and arXiv need), leaving out internal notes and previews.

arXiv publishes every uploaded source file, so only files needed to compile the paper go into the zip.
Run: .venv/Scripts/python scripts/paper/build_overleaf_zip.py
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
SKIP_NAMES = {"README.md", "arxiv_abstract.txt"}
SKIP_SUFFIXES = {".png", ".log", ".aux", ".out", ".blg", ".bbl", ".synctex.gz"}


def main() -> None:
    files = [f for f in sorted(PAPER.rglob("*")) if f.is_file() and f.name not in SKIP_NAMES
             and f.suffix not in SKIP_SUFFIXES and "__pycache__" not in f.parts]
    out = ROOT / "paper_overleaf.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(PAPER).as_posix())
    print(f"{out.name}: {len(files)} files")
    for f in files:
        print("  ", f.relative_to(PAPER).as_posix())


if __name__ == "__main__":
    main()
