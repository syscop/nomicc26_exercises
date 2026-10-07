# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Teaching exercises for NOMICC 2026, a summer school on nonlinear optimization with mixed-integer and complementarity constraints. It is educational code, not a package. There is no build, no lint config and no test suite. Each exercise is a standalone Python script (or a stub that students complete) in its own directory under `exercises/`, and each one has a `description.md` that states the problem and the tasks.

## Environment

```bash
source venv/bin/activate          # the venv already exists; Python >= 3.12
python test_install.py --ccopt --camino --nosnoc   # smoke test; add --gurobi / --libhsl if installed
```

- To rebuild from scratch, run `./install_linux.sh` or `./install_macos_aarch64.sh`. They clone the nosnoc repo into `../nosnoc_py` (branch `v1.0.0-rc`, installed with `pip install -e`), download the libMad binaries into `../libMad-*`, and append `LD_LIBRARY_PATH` / `LD_PRELOAD` exports to `venv/bin/activate`. `install_linux_local.sh` is a variant that is safe to re-run and can be launched from inside the checkout.
- The `ccopt` CasADi plugin only loads if the venv has been activated, because the activation script sets the libMad library path. On macOS, `DYLD_LIBRARY_PATH` also needs SIP disabled, or you can use `otool` to patch `libcasadi_nlpsol_ccopt.dylib`.
- Optional: Gurobi (for the CAMINO `s-b-miqp` solver) and LibHSL (set `JULIA_HSL_LIBRARY_PATH` to get `Ma27Solver` etc. in MadNLP).
- `unopy` (Uno) is **not** in the venv. The `ballthrow` and `water-network` exercises expect Uno and may need it installed first.

## Running exercises

Run each script from inside its own exercise directory. The scripts use sibling imports such as `from utils import ...` and `from model import ...`.

```bash
cd exercises/ell0-portfolio && python main.py -N 10 --ccopt --bonmin [--daqp --gurobi --warmstart-miqp]
cd exercises/soft-docking   && python main.py {nlp|mpcc|minlp} [--show-plots]
cd exercises/gearbox        && python main.py minlp {true|false} [--show-plots]   # true = CAMINO s-b-miqp (needs Gurobi), false = Bonmin
cd exercises/contacts-example && python main.py   # nosnoc MPCC, writes pushing.gif
cd exercises/ballthrow      && python ball_throw.py [--N 200] [--check-jac] [--verify] [--plot]
```

`water-network` has no Python code yet. The reference is the AMPL models in `ampl/` (`waterMINLP.mod`, `waterMPEC.mod`, `waterNLP.mod`, and the `.dat` files) together with the formulation in `WaterNetwork.tex`. The task is to build CasADi/Uno models from these.

## Modeling patterns shared across exercises

All the solvers are reached through `casadi.nlpsol`. Every exercise builds a problem as `{"x", "f", "g", "p"}` and then chooses a plugin according to the formulation:

- **NLP**: `ipopt`.
- **MPCC**: `ccopt` (CCOpt.jl/MadNLP through libMad). Complementarity is passed as options: `cc_pairs` (a list of index pairs) and `cc_types`, which uses the libMad enum `VARVAR=0, VARCON=1, CONVAR=2, CONCON=3`. The options that recur are `"ccopt.relaxation_update.TYPE": "RolloffRelaxationUpdate"` and `"madnlp.bound_relax_factor": 0`. `test_install.py:test_ccopt` is the minimal working example.
- **MINLP**: `bonmin`, with `{"discrete": [0/1 per variable]}`. Alternatively, a CAMINO `MinlpSolver("s-b-miqp", problem, data, stats, settings)` can be used.

The problem is built in one of two ways:
- **CAMINO `Description`** (`camino.problems.dsc`), used in `soft-docking` and `gearbox`. It adds variables, constraints and integer flags to a `Description`, then calls `dsc.build()`, which returns `(problem, data)`. `problem.idx_x_integer` gives the discrete mask. It also uses `camino.utils.integrators.integrate_rk4` for the dynamics.
- **nosnoc `MPCC` with vdx types** (`Primal`, `Constraint`, `CConstraint`), used in `contacts-example`. Complementarity is written as `mpcc.G.name[...]` ⟂ `mpcc.H.name[...]`, and the problem is solved with `mpcc.solve(casadi_opts=..., plugin="ccopt" | "reg_homotopy")`.

`ell0-portfolio/model.py` is a student template. Its `_build_ccopt/_build_daqp/_build_bonmin/_build_gurobi` methods are stubs built around a shared `_build_common` (objective and budget constraint).

## Local-only files — do not commit or expose to students

Some files are reference solutions or instructor notes. They are excluded through `.git/info/exclude`, not `.gitignore`: `exercises/ell0-portfolio/model_commented.py`, `exercises/ell0-portfolio/math_to_code.md` and `install_linux_local.sh`. Other untracked files at the root, such as `AGENTS.md` and `plan-OC-exercises-*.md`, are instructor planning notes. When you edit student-facing templates, do not paste solutions into them.
