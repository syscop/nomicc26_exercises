# Plan: solve TutorialGO (global optimization tutorial)

## Context
`exercises/globalOpt/description.md` asks for a coding agent to solve the four questions in
`TutorialGO.tex` (RLT/McCormick relaxations, eigenvalue QCQP), with formal verification considered.
The user's decisions:
- Write the agent's answers only. **No comparison section** with the user's paper answers.
- For Q1, present **both** readings of "smallest RLT" (PSD-cone size reduction, and fewest lifted
  terms) and compare them numerically.
- Verify with sympy and sampling, **and** try Lean (needs installing first).

Rules from `exercises/globalOpt/CLAUDE.md`:
- Put all files in `exercises/globalOpt/`. Never edit `TutorialGO.tex`.
- `answers.tex` holds the maths, and Python scripts show the McCormick outer approximations.
- Solve relaxations with `unopy` (Uno).
- Keep the sheet's notation (`X_{12}`, `X_{1122}`, …) and reuse its preamble macros.

Environment:
- Python: the conda env `optimization` (numpy, scipy, sympy, matplotlib, casadi 3.8.1, unopy). The
  repo `venv/` has no python binary.
- Uno: copy the model setup in `exercises/ballthrow/ball_throw.py:116-190` (`unopy.Model`,
  `set_objective`, `set_constraints` with a triplet Jacobian, `UnoSolver`, preset `filtersqp`).
  The relaxations are small LPs, so pass them as linear callbacks. Cross-check each LP bound with
  `scipy.optimize.linprog` (HiGHS).

## Files to create (all in `exercises/globalOpt/`)
- `uno_lp.py` is a shared helper with two functions:
  - `solve_lp(c, A, lb_g, ub_g, lb_x, ub_x)` wraps a dense LP as Uno callbacks, solves it, and
    cross-checks the result with linprog.
  - `solve_nlp(...)` handles Q4.
- `q1_rlt.py`, `q2_mccormick.py`, `q3_quartic.py`, `q4_eig.py`. Each prints its results and
  PASS/FAIL validity checks, and saves PDF figures to `figs/`.
- `answers.tex` uses the same `exam`-style preamble and macros as `TutorialGO.tex`, has one section
  per question, includes `figs/*.pdf` and ends with a short section on verification and Lean.
- `lean/` is a Lake project with Mathlib, holding `GlobalOpt/Basic.lean`.

## Solution approach per question (numbers already checked with numpy)

### Q1: smallest RLT for Q1 and Q2
- Both matrices are indefinite: eig(Q1) = {−2.24, 1.38, 2.24, 3.62} and eig(Q2) = {−2.25, 1.09, 2.13, 4.03}.
- **Reading A (cone size).**
  - Q1 is block diagonal. Its (3,4) block is positive definite, so it is kept convex. Only the (1,2)
    block is lifted: a 3×3 PSD cone [1 xᵀ; x X] replaces the 5×5 one.
  - Q2 is tridiagonal with chordal cliques {1,2}, {2,3}, {3,4}, so the 5×5 cone becomes three 3×3 cones.
- **Reading B (fewest lifted terms).**
  - Q1: −2x₁² − 2x₁x₂ + 2x₂² = 2(x₂ − x₁/2)² − (5/2)x₁², which needs one lifted variable X₁₁ and a secant.
  - Q2: d* = 21/8 is the smallest d with Q₂ + d e₁e₁ᵀ ⪰ 0 (Schur complement). Then
    xᵀQ₂x = xᵀ(Q₂ + d*e₁e₁ᵀ)x − (21/8)x₁², again with a single X₁₁.
- **Numerical comparison.** Take b = 0, a = 0 and x ∈ [−1,1]⁴. For each relaxation (full McCormick
  RLT, reading A without SDP, reading B), compute the lower bound on xᵀQᵢx with Uno and compare it
  with the true minimum from vertex enumeration and Uno multistart.
  - SDP cones need an SDP solver, which isn't installed. So the PSD reduction in reading A is argued
    on paper only, and its LP/McCormick part is computed.
  - Also compare the convex quadratic part of reading B by solving the convex QCQP relaxation with Uno.

### Q2: the single bilinear term
- u = 1 − x₁ + 2x₂ − 2x₄ ∈ [−4, 6] and v = 4x₂ − 5x₃ + x₄ + 2 ∈ [−8, 12]. The constraint is w ≤ 2 with w = uv.
- McCormick:
  - Underestimators, which are the ones that matter for "≤": w ≥ −8u − 4v − 32 and w ≥ 12u + 6v − 72.
  - Overestimators, listed for completeness: w ≤ 12u − 4v + 48 and w ≤ −8u + 6v + 48.
- Script:
  - Sample x ∈ [−1,1]⁴ and check that every feasible x satisfies the relaxation.
  - Plot the uv surface and the McCormick planes on [−4,6]×[−8,12].
  - Plot the true region uv ≤ 2 against the relaxed region in the (u,v) box.

### Q3: RLT of the quartic, two relaxations
- Bounds: X₁₁, X₂₂ ∈ [0,1], and X₁₂, X₁₂₂, X₂₁₁ ∈ [−1,1]. Valid inequalities come from expanding all
  bound-factor products (1±x₁)ᵃ(1±x₂)ᵇ ≥ 0 of degree ≤ 4 with sympy and mapping the monomials to X names.
- Identity: f = −(x₁x₂ − x₂²)², checked with sympy.
- Alternative relaxation: w = x₁x₂ − x₂² ∈ [−2, 1/4]. Since −w² is concave, the secant gives
  f ≥ (7/4)w − 1/2. Then w is relaxed with McCormick on X₁₂ and the tangents/secant of X₂₂.
- Comparison:
  - LP lower bounds from Uno on the full box and on a grid of sub-boxes. Both relaxations may reach
    −4 on the full box, so the sub-boxes are what separate them.
  - Heat maps of the gap f − underestimator on [−1,1]².
  - The conclusion on which relaxation is tighter is drawn from these numbers.

### Q4: the eigenvalue QCQP
- KKT: Qx = λx and xᵀx = 1, so f = λ. min = λ_min ≈ −0.5343 and max = λ_max ≈ 6.1503.
- These KKT points are global because the feasible set is compact and every KKT point is an eigenpair.
  Equivalently, the S-lemma gives zero duality gap.
- Uno multistart for min and max, compared with `numpy.linalg.eigh`.

## Lean (needs a separate install; I'll confirm before running it)
- Install `elan` with the official script to `~/.elan`, run `lake new globalopt_lean math` in
  `exercises/globalOpt/lean/`, then `lake exe cache get` (several GB, and 290 GB is free).
- Statements over ℝ with rational data:
  1. The Q1 split identity (`ring`).
  2. Q₂ + (21/8)e₁e₁ᵀ is PSD, shown by its rational LDLᵀ sum of squares (`nlinarith`/`positivity`).
  3. The Q2 McCormick underestimators on the box (`nlinarith` with the products (u+4)(v+8) ≥ 0, …).
  4. The Q3 identity (`ring`), one expanded bound-factor product (`ring`), and the secant
     −w² ≥ (7/4)w − 1/2 on [−2, 1/4] (`nlinarith`).
  5. Q4 in general form: if Qx = λx and xᵀx = 1, then xᵀQx = λ. The irrational eigenvalues stay numerical.
- `answers.tex` reports which statements Lean proved and what Lean was not used for.

## Verification
- `conda run -n optimization python q1_rlt.py` (and likewise for q2–q4) runs and prints PASS for every
  validity check. Uno and linprog bounds agree to 1e-6.
- `cd lean && lake build` succeeds with no `sorry`.
- `latexmk -pdf answers.tex` compiles cleanly with all figures included.
