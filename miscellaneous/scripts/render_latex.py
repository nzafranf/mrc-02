"""Render a LaTeX file to PDF locally.

  python miscellaneous/scripts/render_latex.py report/latex/main.tex                 # -> report/latex/main.pdf
  python miscellaneous/scripts/render_latex.py report/latex/main.tex report/main.pdf
  python miscellaneous/scripts/render_latex.py report/latex/main.tex out.pdf --engine xelatex --clean

Needs a TeX distribution on PATH (MiKTeX or TeX Live). Uses latexmk when it is available (it reruns LaTeX and
BibTeX/Biber as often as needed); otherwise runs engine -> bibtex/biber -> engine -> engine itself.
LaTeX runs inside the .tex file's folder, so relative paths in the document (\\graphicspath, \\input, \\bibliography)
resolve from there. Auxiliary files go to <tex folder>/build/ (ignored by git); only the PDF is copied out.
On failure the first errors from the log are printed with their line numbers.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ENGINES = {"pdflatex": "-pdf", "xelatex": "-pdfxe", "lualatex": "-pdflua"}


def run(cmd, cwd, env=None):
    r = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, encoding="utf-8", errors="replace")
    return r.returncode, r.stdout


def show_errors(log: Path, limit=5):
    if not log.exists():
        print("  no log file was written")
        return
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    hits = [i for i, l in enumerate(lines) if l.startswith("!") or re.match(r"^\S+\.tex:\d+: ", l)]
    for i in hits[:limit]:
        print("  " + "\n  ".join(lines[i:i + 4]))
        print("  ---")
    if not hits:
        print("  no '!' error lines in the log; see", log)
    for l in lines:  # missing files and references are the usual causes
        if "Warning: Citation" in l or "Warning: Reference" in l or "not found" in l:
            print("  warning:", l.strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tex", type=Path)
    ap.add_argument("output", type=Path, nargs="?", help="PDF path (default: next to the .tex file)")
    ap.add_argument("--engine", choices=list(ENGINES), default="pdflatex")
    ap.add_argument("--clean", action="store_true", help="delete the build folder first")
    ap.add_argument("--no-latexmk", action="store_true", help="use the manual engine/bibtex sequence")
    a = ap.parse_args()

    tex = a.tex.resolve()
    if not tex.exists():
        sys.exit(f"not found: {tex}")
    out = (a.output or tex.with_suffix(".pdf")).resolve()
    src, build = tex.parent, tex.parent / "build"
    if a.clean and build.exists():  # empty it rather than delete it: Windows refuses to remove an open folder
        for f in build.iterdir():
            shutil.rmtree(f, ignore_errors=True) if f.is_dir() else f.unlink(missing_ok=True)
    build.mkdir(exist_ok=True)
    if not shutil.which(a.engine):
        sys.exit(f"{a.engine} is not on PATH. Install MiKTeX (https://miktex.org) or TeX Live, then reopen the terminal.")

    flags = ["-interaction=nonstopmode", "-halt-on-error", "-file-line-error"]
    pdf, logfile = build / f"{tex.stem}.pdf", build / f"{tex.stem}.log"
    for stale in (pdf, logfile):  # a failed run must not report an old log or copy an old PDF
        stale.unlink(missing_ok=True)
    # BibTeX runs inside build/, so put the document folder first on its search path; otherwise a same-named
    # .bib shipped with the TeX distribution (MiKTeX has a references.bib) wins over the project's file.
    env = dict(os.environ, BIBINPUTS=f"{src}{os.pathsep}", BSTINPUTS=f"{src}{os.pathsep}")
    latexmk = bool(shutil.which("latexmk")) and not a.no_latexmk
    if latexmk:
        # MiKTeX's latexmk is a Perl script: without Perl on PATH (plain cmd or PowerShell) it cannot start
        ok, msg = run(["latexmk", "-v"], src)
        if ok != 0:
            print("latexmk cannot run here (" + (msg.strip().splitlines() or ["?"])[-1].strip()
                  + "); using pdflatex + bibtex instead. Install Perl (e.g. Strawberry Perl) to use latexmk.")
            latexmk = False
    if latexmk:
        tool = "latexmk"
        code, log = run(["latexmk", ENGINES[a.engine], *flags, f"-outdir={build}", tex.name], src, env)
    else:
        tool = a.engine
        engine = [a.engine, *flags, f"-output-directory={build}", tex.name]
        code, log = run(engine, src, env)
        if code == 0:
            stem = tex.stem
            if (build / f"{stem}.bcf").exists():
                tool, (bcode, log2) = "biber", run(["biber", "--input-directory", str(src), stem], build)
            elif "\\bibdata" in (build / f"{stem}.aux").read_text(encoding="utf-8", errors="replace"):
                tool, (bcode, log2) = "bibtex", run(["bibtex", stem], build, env)
            else:
                bcode, log2 = 0, ""
            log += log2
            code = bcode if bcode >= 2 else 0  # exit 1 = warnings only (e.g. a missing entry), keep going
            for _ in range(2):
                if code == 0:
                    tool = a.engine
                    code, log2 = run(engine, src, env)
                    log += log2
    if code != 0 or not pdf.exists():
        print(f"FAILED ({tool}, exit {code}). Errors:")
        if logfile.exists() and tool != "bibtex":
            show_errors(logfile)
        print(f"  last lines from {tool}:\n    " + "\n    ".join(log.strip().splitlines()[-12:]))
        sys.exit(1)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pdf, out)
    warn = [l for l in (build / f"{tex.stem}.log").read_text(encoding="utf-8", errors="replace").splitlines()
            if "undefined" in l and "Warning" in l]
    print(f"wrote {out}" + (f" ({len(warn)} undefined reference/citation warnings)" if warn else ""))


if __name__ == "__main__":
    main()
