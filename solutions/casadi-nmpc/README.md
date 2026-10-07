# Learning CasADi: nonlinear MPC of a cart-pole (and connecting to Uno)

[CasADi](https://web.casadi.org) ([GitHub](https://github.com/casadi/casadi)) is a symbolic framework for nonlinear optimization and optimal control. You write a model as symbolic expressions. CasADi then:
- builds exact derivatives by **algorithmic differentiation** (AD),
- integrates ODEs and DAEs, and
- passes the resulting NLP, QP or MINLP to a numerical solver (Ipopt, Uno, Bonmin, HiGHS, …) through one interface, `nlpsol`/`qpsol`.

Python, MATLAB/Octave and C++ front ends are available. This directory uses Python.

| File | What it is |
|---|---|
| `01_casadi_basics.py` | A 10-minute tour: SX/MX symbols, `Function`, AD, `nlpsol` (Ipopt and Uno), `Opti`, integrators |
| `nmpc_cartpole.md` / `nmpc_cartpole.tex` (→ PDF) | The math: model, OCP, multiple shooting, receding horizon, solvers |
| `nmpc_cartpole.py` | The NMPC closed loop (command-line flags for every experiment below) |
| `make_figures.py` → `figures/` | Regenerates the multiple-shooting and NMPC-principle illustrations used in the notes |
| `02_uno_direct.py` | Stretch goal: CasADi ↔ Uno, via the `uno` plugin **and** via `unopy` callbacks |

---

## 1. Installation

### What you need

| Package | Needed for | How |
|---|---|---|
| Python ≥ 3.9 | everything | conda or system Python |
| `casadi` | everything | `pip install casadi` (tested: 3.8.1) |
| `numpy` | arrays | `pip install numpy` |
| `matplotlib` (+ `pillow`) | `--plot`, `--animate` | `pip install matplotlib pillow` |
| `unopy` | `02_uno_direct.py` only | `pip install unopy` (tested: 0.4.10, Uno 2.7.4) |
| LaTeX | building the PDF only | `pdflatex nmpc_cartpole.tex` (run twice) |

**You do not need to install any solvers separately.** The `casadi` pip wheel is self-contained and bundles:
- NLP: Ipopt (with MUMPS), **Uno** (in recent releases; 3.8.1 tested), Fatrop, `sqpmethod`, blockSQP
- MINLP: Bonmin (with Cbc)
- QP/LP: HiGHS, OSQP, qpOASES, DAQP, ProxQP, HPIPM, `qrqp`
- Integrators: CVODES/IDAS (SUNDIALS) and fixed-step RK/collocation

**Optional extras:**
- **HSL linear solvers** (MA27/MA57/MA97) make Ipopt faster and more robust on large problems. Get them from https://licences.stfc.ac.uk/product/coin-hsl, put `libhsl.so` on `LD_LIBRARY_PATH`, and pass `{"ipopt.linear_solver": "ma57"}`.
- **Gurobi / CPLEX / Knitro / SNOPT** are commercial. CasADi loads them at run time if they are installed. Gurobi needs `GUROBI_VERSION` set, which explains the harmless "Failed to load Gurobi adaptor" warning you may see.

### Option A: use the existing conda env (recommended on this machine)

```bash
conda activate optimization        # Python 3.11, casadi 3.8.1, unopy 0.4.10, numpy, matplotlib
```

### Option B: a fresh environment

```bash
python -m venv casadi-env && source casadi-env/bin/activate
pip install casadi numpy matplotlib pillow unopy
```

### Check the install

```bash
python -c "import casadi as ca; print(ca.__version__, ca.has_nlpsol('ipopt'), ca.has_nlpsol('uno'), ca.has_conic('highs'))"
# 3.8.1 True True True
```

---

## 2. Running

```bash
cd exercises/casadi-nmpc
python 01_casadi_basics.py
python nmpc_cartpole.py --plot --animate          # writes cartpole_closed_loop.png, cartpole.gif
python nmpc_cartpole.py --solver uno --uno-preset filtersqp
python 02_uno_direct.py [--preset ipopt] [--verbose]
python nmpc_cartpole.py -h                        # all options
```

`nmpc_cartpole.py` prints a summary like this:

```
solver            : ipopt
successful solves : 100 / 100
iterations        : total 581, first 60, mean after first 5.3
solve time        : total 1.49 s, max 226.4 ms, mean 14.9 ms
max |F|           : 20.00 N   (bound 20.0)
max |p|           : 1.325 m   (bound 1.5)
|theta - pi|, last 1 s: max 1.31e-04 rad  ->  UPRIGHT
```

---

## 3. Practical guide: a reading-and-doing exercise (≈ 3 hours)

**Step 1. CasADi concepts (45 min).**
- Read Sections 1–4 of the [CasADi user guide](https://web.casadi.org/docs/): symbols, `Function`, AD, and `nlpsol`. Then read Section 9 on `Opti`.
- Run `01_casadi_basics.py` section by section.

Answer these questions:
- Why does `ca.hessian` return two outputs?
- What is the difference between the `SX` and `MX` printouts?
- In section 4, why does the multiplier `lam_g` come out positive?

**Step 2. The math (30 min).** Read `nmpc_cartpole.md` (or build the PDF). For each of these items, find the line in `nmpc_cartpole.py` that implements it:
- equation (1), the ODE,
- the RK4 map $\Phi$, (4a)–(4b),
- each line (5a)–(5e) of the NLP, and
- the shift warm start.

Section 5 of the notes gives the answer for everything except the warm start, as one self-contained listing with every line tagged by its equation number. Try the exercise before you look.

**Step 3. Run the baseline.** Run `python nmpc_cartpole.py --plot --animate`, then look at the PNG and the GIF. Identify the three phases of the motion:
1. the push left,
2. the swing-up,
3. the catch.

**Step 4. Modify and observe (60 min).** Run each experiment below and compare it with the baseline. The observations in the right-hand column come from our runs.

| Experiment | Command | What you should see |
|---|---|---|
| a. Horizon | `--N 25`, `--N 10`, `--N 8`, `--N 5` | N ≥ 8 still swings up. At N = 5 the controller is too myopic: about half the solves fail, the controller applies their infeasible last iterates, the cart even leaves the track (|p| ≈ 2.2), and the pole never stays up. |
| b. Terminal cost | `--N 10 --no-terminal-cost`, `--N 8 --no-terminal-cost` | With a short horizon, dropping $V_f$ hurts. At N = 10 the cart rides along the track limit and the pole jitters by about 0.2 rad. At N = 8 the swing-up fails. The terminal cost stands in for the missing tail of the horizon. |
| c. Actuator limit | `--F-max 8`, `--F-max 4 --T-sim 10`, `--F-max 3 --T-sim 10` | Weaker motors need more swings and longer saturation. At 3 N the swing-up fails within the horizon. |
| d. Model mismatch | `--mismatch` | The plant pole is 30% heavier, yet the pendulum is still caught (|p| rises to ≈ 1.38). This shows the robustness that feedback provides. |
| d′. Measurement noise | `--noise 0.01`, `0.05`, `0.1`, `0.2` (add `--plot` to see true vs. measured states; `--seed` changes the noise draw) | σ = 0.01: still upright to within 0.02 rad. σ = 0.05–0.1: balanced but jittering by 0.1–0.2 rad, with 7–14 instead of 5 iterations per step. σ = 0.2: the pendulum is lost. Combine with `--mismatch` or with `--solver uno` to compare. |
| e. Warm start | `--no-warmstart` | Iterations rise from about 5 to about 67 per step, **and the pendulum never swings up**: cold starts converge to the local minimizer "stay hanging". See Section 8 of the notes. |

**Step 5. Edit the code (30 min).** Choose one or more of these changes:
- Replace RK4 with an exact integrator, as in `Phi = ca.integrator("Phi", "cvodes", ...)`, and compare the speed.
- Change $Q$ so that the cart must return to $p=0.5$ instead of $0$.
- Add a rate penalty $\sum_k (u_{k+1}-u_k)^2$ and watch the force profile smooth out.
- Add the path constraint "the pole tip must stay above $y=-0.3$" (the tip is at $y = -l\cos\theta$).

**Step 6. Compare solvers (15 min).** Fill in this table on your own machine:

```bash
for s in "ipopt" "uno --uno-preset ipopt" "uno --uno-preset filtersqp" "sqpmethod"; do
    python nmpc_cartpole.py --solver $s | grep -E "solver|iterations|solve time"; done
```

| solver | first solve | mean iters, warm | our result (total iters) |
|---|---|---|---|
| Ipopt | 60 | 5.3 | 581 |
| Uno / ipopt | 64 | 5.4 | 596 |
| Uno / filtersqp | 21 | 1.4 | 155 |
| sqpmethod (qrqp, eigen-clip) | 59 | 1.5 | 204 |

The lesson: an active-set SQP re-uses the previous active set and needs only about 1–2 iterations per MPC step, while an interior point method needs about 5. Wall-clock time also depends on the cost of each iteration, so measure it.

**Further reading:**
- CasADi examples: `docs/examples/python/direct_multiple_shooting.py`, `race_car.py` and `vdp_*.py` in the [CasADi repo](https://github.com/casadi/casadi/tree/main/docs/examples/python).
- J. Andersson et al., "CasADi: a software framework for nonlinear optimization and optimal control", *Math. Prog. Comp.* 11 (2019).
- J. Rawlings, D. Mayne, M. Diehl, *Model Predictive Control: Theory, Computation, and Design*, 2nd ed., Ch. 8 (free PDF from the authors' site).
- S. Gros, M. Zanon, R. Quirynen, A. Bemporad, M. Diehl, "From linear to nonlinear MPC: bridging the gap via the real-time iteration", *Int. J. Control* 93 (2020).

---

## 4. Stretch goal: CasADi ↔ Uno

> **Note:** the Uno you want is the optimization solver **https://github.com/cvanaret/Uno** (C. Vanaret & S. Leyffer). `github.com/unoplatform/uno` is an unrelated .NET UI framework.

Uno is a modular NLP solver. A *preset* combines building blocks (constraint relaxation, globalization strategy and mechanism, subproblem solver) into a known method. For example, `ipopt` gives an interior point method with a filter line search, and `filtersqp` gives trust-region SQP with a filter.

**Route A: the built-in CasADi plugin (easiest).** Recent CasADi releases (3.8.1 tested) ship an `uno` nlpsol plugin, so no extra install is needed:

```python
S = ca.nlpsol("S", "uno", nlp, {"uno.preset": "filtersqp"})        # low level
opti.solver("uno", {}, {"preset": "filtersqp"})                     # Opti
```

Every Uno option can be passed as `"uno.<option>"`, for example `"uno.logger": "SILENT"` or `"uno.QP_solver": "BQPD"`. The plugin bundles its own Uno build: casadi 3.8.1 bundles Uno 2.7.2.

**Route B: call `unopy` directly, using CasADi as the derivative engine.** `02_uno_direct.py` shows how. Build CasADi `Function`s for:
- $f$ and $\nabla f$,
- $g$ and the Jacobian $J_g$, and
- the lower triangle of $\nabla^2 L = \sigma\nabla^2 f + \sum_i\lambda_i\nabla^2 g_i$.

Then hand them to `unopy.Model` as callbacks with sparse `(row, col)` triplets from `Sparsity.get_triplet()`. Uno's `MULTIPLIER_POSITIVE` sign convention matches CasADi's, so the multipliers agree. Route B gives you the newest Uno from pip and every option of the C++ API, and it also works for models that are not written in CasADi.

`python 02_uno_direct.py` checks that the two routes agree:

```
route                               ok  iters  time [s]      objective
A: nlpsol ipopt (reference)       True     60     0.078   855.85842580
A: nlpsol uno/filtersqp           True     20     0.323   855.85843406
B: unopy  uno/filtersqp           True     20     0.051   855.85843406
route A vs route B: max|w_A - w_B| = 2.84e-14  (same algorithm, same derivatives -> same iterates)
```

Both Uno routes reproduce the Ipopt solution to solver tolerance (about 1e-5). That gives you a template for running any CasADi model through Uno, including the MPCC formulations used elsewhere in this summer school.
