# Global optimization tutorial: by hand, by agent, and checked

This session is an old-fashioned tutorial sheet on McCormick and RLT relaxations,
[`TutorialGO.pdf`](TutorialGO.pdf) ([LaTeX source](TutorialGO.tex)). It is solved twice: on
paper by you, and by a coding agent following the workflow in [`description.md`](description.md).

The agent's answers are in [`answers.pdf`](answers.pdf) ([source](answers.tex)). Every claim
in them is backed by a Python script, which solves the relaxations with
[Uno](https://github.com/cvanaret/Uno), or by a Lean 4 proof. This README explains:
- how to rebuild those answers yourself;
- how to check them independently;
- how they were produced.

---

## 1. What needs to be implemented, and how

The sheet has four mathematical questions, plus a fifth that asks how far an AI tool gets.
To answer them *computationally*, you need four pieces.

1. **A small solver layer.** Write a wrapper that passes problems to Uno (`unopy`).
   - *LPs*: give a dense `A`, the bounds and the cost to `unopy.Model` as callbacks. Cross-check
     every LP with `scipy.optimize.linprog` (HiGHS), so that a solver bug cannot hide.
   - *NLPs*: build `f` and `g` in CasADi and let it produce the gradient, the Jacobian and the
     Hessian of the Lagrangian. Use Uno's `MULTIPLIER_POSITIVE` convention.
   - The callback pattern is in [`../ballthrow/ball_throw.py`](../ballthrow/ball_throw.py).
   - Our version is [`uno_lp.py`](uno_lp.py).
2. **One script per question**, each doing the following:
   - *Build the relaxation.* Introduce lifted variables (`X_12 ≈ x1·x2`, …) and generate the
     McCormick or bound-factor inequalities. For the degree-3 and degree-4 products in Q3, use
     sympy (`expand`, then map each monomial to its `X` name).
   - *Compute the true optimum independently.* Use face or vertex enumeration, a dense grid, or
     Uno multistart.
   - *Assert validity.* Every relaxation bound must be ≤ the true minimum. Every relaxation must
     contain all sampled feasible points. Print `PASS`/`FAIL`.
   - *Compare relaxations.* Use bounds on the full box, bounds on sub-boxes, and pointwise
     underestimators (solve the relaxation with `x` fixed).
3. **The write-up**, a short LaTeX file with figures (matplotlib → PDF).
4. **Optional: Lean 4 + Mathlib proofs.** State each identity or inequality over ℝ with rational
   data, then close the proofs:
   - identities with `ring`;
   - box inequalities with `nlinarith [hints]`, where the hints are the bound-factor products;
   - PSD claims with an explicit rational LDLᵀ sum of squares, computed with sympy;
   - numeric facts with `norm_num`.

**Tips that save time.**
- No SDP solver is installed. To get an SDP bound, impose `[1 xᵀ; x X] ⪰ 0` as
  *all principal minors ≥ 0* and solve with Uno. The objective is linear and the set convex,
  so a local solution is global. Expect only about 1e-5 accuracy at rank-one optima. If the
  `filtersqp` preset stalls, use the `ipopt` preset.
- **Uno 0.4.x fails on problems with zero constraints** (`ALGORITHMIC_ERROR`, or a segfault
  with logging on). Add one free row, `-inf ≤ x0 ≤ inf`.
- Compare objective values on the sphere through the Rayleigh quotient `xᵀQx / xᵀx`. Uno's
  `‖x‖ = 1` is only satisfied to about 1e-7.

---

## 2. Setup (fresh conda environment)

```bash
conda env create -f environment.yml     # Python 3.12 + pinned pip packages; env name "globalopt"
                                        # (tested from scratch on Linux x86_64, 2026-10-01)
conda activate globalopt
```

[`environment.yml`](environment.yml) pins the versions that produced `answers.pdf`: numpy 2.4.3,
scipy 1.17.1, sympy 1.14.0, matplotlib 3.10.9, casadi 3.8.1 and unopy 0.4.10. unopy has wheels for
Linux (glibc ≥ 2.27), macOS ≥ 15 and Windows.

- **LaTeX** (only to rebuild `answers.pdf`): any TeX Live with `latexmk`.
- **Lean** (optional):
  ```bash
  curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --no-modify-path
  export PATH=$HOME/.elan/bin:$PATH
  cd lean && lake exe cache get          # downloads prebuilt Mathlib into lean/.lake
  ```

**Time and disk budget**

| step | time | disk |
|---|---|---|
| `conda env create` | ≈ 1 min (measured on Linux) | ≈ 0.9 GB |
| all Python scripts | ≈ 40 s (Q1 ≈ 13 s, Q3 ≈ 18 s, the rest a few seconds) | – |
| `latexmk -pdf answers.tex` | seconds | – |
| `lake exe cache get` (first time) | several minutes, network bound | **≈ 8 GB** |
| `lake build` | ≈ 45 s | – |

---

## 3. Validate the answers

Run everything from this directory. Each script prints its checks and a final summary line, and
exits non-zero if any check fails. Figures are rewritten to `figs/`.

| Q | Run | What it checks | Expected last line | Lean theorems (`lean/Globalopt/Basic.lean`) |
|---|---|---|---|---|
| 1 | `python q1_rlt.py` | true minima by face enumeration; six relaxations (McCormick LP, 5×5 cone, clique cones, block cone, convex split, αBB) are valid lower bounds; clique cones = full cone; one lifted `X_11` is exact | `38/38 checks passed` | `q1_split`, `q2_split`, `q2_shift_smallest`, `q1_lower`/`q1_attained`, `q2_lower`/`q2_attained` |
| 1 | `python check_eig_q1.py` | eigenvalues of Q₁ (= those of its two blocks); `Q₁ + 5/2 e₁e₁ᵀ` is PSD and singular | `all checks passed` | – |
| 2 | `python q2_mccormick.py` | `(…)(…) = u·v`; interval bounds of u, v; McCormick valid on 10⁶ samples; both relaxations contain all feasible samples; LP bounds below the true minimum | `7/7 checks passed` | `uv_bounds`, `mccormick` |
| 3 | `python q3_quartic.py` | `f = -(x1x2 - x2²)²`; the sheet's expansion; bounds of both RLTs valid on the full box, 16 sub-boxes, pointwise and for tilted objectives | `10/10 checks passed` | `f_eq`, `sheet_bound_factor`, `w_range`, `secant`, `f_lower`/`f_attained` |
| 4 | `python q4_eig.py` | characteristic polynomial roots = numpy eigenvalues; Uno multistart reaches λ_min / λ_max; every solution is a unit eigenvector | `8/8 checks passed` | `kkt_value`, `Q4_charpoly`, `lam_min_ge`/`lam_min_le`, `lam_max_le`/`lam_max_ge` |
| all | `cd lean && lake build` | all theorems above, no `sorry` | `Build completed successfully` | – |
| – | `latexmk -pdf answers.tex` | the write-up compiles with all figures | – | – |

Without activating the env, `conda run -n globalopt python q1_rlt.py` works too.

### Expected numbers (compare with your own implementation)

- **Q1.** Lower bounds on `xᵀQx + bᵀx` over [-1,1]⁴:

  | | true min | McCormick, all terms | SDP (5×5 = cliques) | convex split, one `X_11` | αBB |
  |---|---|---|---|---|---|
  | Q₁, b = 0 | −2.5 | −6 | −2.5 | −2.5 | −8.944 |
  | Q₂, b = 0 | −2.625 | −8 | −2.625 | −2.625 | −8.988 |
  | Q₁, b = (1,−1,1,−1) | −3.275 | −6 | −3.275 | −3.275 | −9.855 |
  | Q₂, b = (1,−1,1,−1) | −4.375 | −10 | −4.375 | −4.375 | −10.330 |

  The smallest shifts are d* = 5/2 (Q₁) and d* = 21/8 (Q₂). The kernel vector of
  Q₂ + 21/8 e₁e₁ᵀ is (−8, 5, 2, 1).
- **Q2.** u ∈ [−4, 6] and v ∈ [−8, 12]. The expansion has 8 quadratic monomials.
  - Share of [−1,1]⁴ accepted: feasible 0.537, single-term McCormick 0.908, term-wise 0.867.
  - Minimum of the product over the box: true −18, single-term LP −48, term-wise LP −36.
- **Q3.**
  - The bound-factor products alone imply X₁₁, X₂₂ ≥ −1/3, so X ≥ 0 must be added for even powers.
  - w = x₁x₂ − x₂² ∈ [−2, 1/4] exactly; interval arithmetic gives [−2, 1].
  - Both relaxations give −4 on the full box and are exact on all 16 sub-boxes. RLT-B with
    interval w-bounds has gaps up to 0.3125.
  - Mean pointwise gap: RLT-A 2.2275, RLT-B 2.4021. RLT-A is tighter at 420 of 441 grid points
    and never looser.
- **Q4.**
  - det(Q − μI) = μ⁴ − 12μ³ + 42μ² − 32μ − 31.
  - λ = −0.534306, 2.322889, 4.061156, 6.150262.
  - 40/40 Uno starts reach the global min or max.

Small last-digit differences across platforms are normal. The `PASS` checks use tolerances
between 1e-6 and 1e-4.

**Numerically verified only (not proved):** LP/SDP bound values and comparisons, the chordal
(PSD-completion) theorem (cited), and the claim that RLT-A dominates RLT-B pointwise (checked
on a 21×21 grid).

---

## 4. How this was produced with a coding agent

These steps follow [`description.md`](description.md), using Claude Code.

1. **`/init`** wrote [`CLAUDE.md`](CLAUDE.md). The instructor then edited it to require
   `answers.tex`, Python scripts that show the McCormick outer approximations, and Uno for the
   relaxations.
2. **`/plan`.** The agent first discussed an approach per question, with key numbers checked in
   numpy, and then asked three questions:
   - present both readings of "smallest RLT" in Q1 (cone size, and number of lifted terms);
   - verify with sympy, and also try Lean;
   - skip the human-versus-agent comparison section.
   The plan is in [`plan-261001-solution01.md`](plan-261001-solution01.md).
3. **Auto mode** carried out the plan. Issues hit along the way:
   - `sympy` was missing from the conda env;
   - `lake new lean` fails, because `lean` is a reserved package name;
   - Uno fails on problems without constraints;
   - SDP-by-minors needed an interior-point fallback and a 1e-4 PSD tolerance.
4. **Findings the agent did not expect, and reported:**
   - In Q2 the "lazy" single bilinear term is valid but *weaker* than term-wise McCormick, because
     u and v share x₂ and x₄.
   - In Q3 box bounds cannot separate the two relaxations, since every minimiser is a vertex, so
     pointwise underestimators were used instead.
5. This README was planned in a second `/plan` session. The setup above was tested in a freshly
   created conda env.

---

## 5. Files

| file | purpose |
|---|---|
| [`TutorialGO.tex`](TutorialGO.tex) / [`.pdf`](TutorialGO.pdf) | the tutorial sheet |
| [`description.md`](description.md) | the exercise workflow (paper → agent → comparison) |
| [`answers.tex`](answers.tex) / [`.pdf`](answers.pdf) | answers to Q1–Q4, verification and Lean assessment |
| [`q1_rlt.py`](q1_rlt.py), [`q2_mccormick.py`](q2_mccormick.py), [`q3_quartic.py`](q3_quartic.py), [`q4_eig.py`](q4_eig.py) | one script per question: relaxations, Uno solves, checks, figures |
| [`check_eig_q1.py`](check_eig_q1.py) | quick numpy check of the eigenvalues of Q₁ |
| [`uno_lp.py`](uno_lp.py) | Uno wrappers: `solve_lp` (HiGHS cross-check), `solve_casadi`, `solve_qcqp_sphere` |
| [`plotstyle.py`](plotstyle.py) | shared matplotlib style |
| [`figs/`](figs/) | generated figures used by `answers.tex` |
| [`lean/`](lean/) | Lean 4 + Mathlib project; proofs in [`Globalopt/Basic.lean`](lean/Globalopt/Basic.lean) |
| [`environment.yml`](environment.yml) | pinned conda/pip environment |
| [`CLAUDE.md`](CLAUDE.md) | instructions for the coding agent |
