# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This file covers `exercises/globalOpt/`. The repo-level `../../CLAUDE.md` also applies, but most of it (CasADi/ccopt/CAMINO/nosnoc) does not matter here.

## What this is

A pen-and-paper tutorial sheet on global optimization (McCormick / RLT relaxations, size reduction of conic constraints), not a coding exercise. Unlike the other exercises, there is no Python scaffold.

- `TutorialGO.tex` is the source of truth: an `exam`-class sheet with five questions.
  1. Smallest RLT relaxation of `x^T Q_i x + b^T x + a <= 0` for two 4x4 Hessians (block-diagonal `Q_1` vs. tridiagonal `Q_2`), and how to improve the RLT of `Q_2`.
  2. Find the single bilinear term hidden in a product of two affine functions on `[-1,1]^4`, and write its McCormick relaxation.
  3. RLT of the quartic `-x1^2 x2^2 + 2 x1 x2^3 - x2^4` on `[-1,1]^2`: bound-factor products for the lifted monomials `X_..`, then an alternative relaxation via `f = -(x1 x2 - x2^2)^2`, and a comparison.
  4. KKT conditions of `min/max x^T Q x s.t. x^T x = 1` (eigenvalue problem), solved for a given 4x4 `Q`.
  5. Meta-question: how far a coding agent gets on questions 1–4.
- `TutorialGO.pdf` is the compiled sheet (untracked; regenerate rather than edit).
- `description.md` defines the workflow for students: answer on paper → `/init` → `/plan` → optionally write Python to check the underestimators (or consider Lean) → compare against the student's own answers → **the deliverable is a short LaTeX document comparing the two**. It also suggests photographing a handwritten solution and asking the agent to transcribe it to LaTeX.

## Commands

```bash
latexmk -pdf TutorialGO.tex      # or: pdflatex TutorialGO.tex (pdflatex and latexmk are installed)
latexmk -c                       # clean aux files
```

For numerical checks (eigenvalues, sampling a relaxation against the original function, etc.) the repo `venv/` currently has no `python` binary. Use the conda env instead:

```bash
conda run -n optimization python q1_rlt.py   # likewise q2_mccormick.py, q3_quartic.py, q4_eig.py
cd lean && PATH=$HOME/.elan/bin:$PATH lake build
```

The `optimization` env has numpy, scipy, sympy (pip-installed for this exercise), matplotlib, casadi 3.8.1 and unopy. No LP/SDP modelling package (cvxpy etc.) is installed.

- `uno_lp.py` wraps Uno: `solve_lp` (cross-checked against HiGHS via `linprog`), `solve_casadi` (a general NLP with exact derivatives from CasADi) and `solve_qcqp_sphere`.
- SDP cones are imposed as "all principal minors >= 0" and solved with Uno. At rank-one optima this only reaches about 1e-5 feasibility.
- Uno 0.4.x fails when a problem has no constraints. `solve_casadi` adds a free dummy row to work around this.
- `plotstyle.py` holds the shared figure style. Figures go to `figs/`.
- Lean 4 + Mathlib (elan in `~/.elan`) lives in `lean/`. Its `.lake/` is about 8 GB and git-ignored.

## Working conventions

- **Don't modify `TutorialGO.tex**` when answering. It is the instructor's problem sheet. 
- Write answers, the comparison, and any helper scripts as new files in this directory. Create/update answers.tex for math description. Create/update python scripts to show the McCormick outer approximations. Use unopy (UNO) to solve relaxations.
- Name the bound-factor products and lifted variables exactly as the sheet does (`X_{12}`, `X_{122}`, `X_{1122}`, ...), so answers can be compared line by line with the students' work.
- Check every claimed relaxation numerically: a valid relaxation must underestimate (or, for constraints, contain the feasible set of) the original on the whole box. A short sampling or vertex-enumeration script is enough and is what `description.md` asks for.
- The LaTeX write-up should reuse the preamble macros from `TutorialGO.tex` (`\R`, `\st`, `\trps`, ...) so notation stays consistent.
