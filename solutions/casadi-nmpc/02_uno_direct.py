"""Stretch goal: two ways to connect CasADi to Uno (https://github.com/cvanaret/Uno).

We solve ONE cart-pole swing-up OCP, the NLP (5) of the notes: the first MPC problem of
nmpc_cartpole.py, from the hanging rest state. It is written in CasADi's low-level form

    min_w f(w)   s.t.  lbg <= g(w) <= ubg,  lbw <= w <= ubw,
    w = (x_0, u_0, x_1, u_1, ..., u_{N-1}, x_N)       (the stacked vector w of Section 3),

with f = J from (5a), g(w) = the equality constraints (5b) and (5c), and the simple bounds
lbw <= w <= ubw = (5d) and (5e). Equation numbers in the comments, such as (5c), refer to
nmpc_cartpole.tex / nmpc_cartpole.md.

Route A -- the CasADi "uno" nlpsol plugin (bundled with recent casadi wheels; 3.8.1 tested):
    ca.nlpsol("S", "uno", nlp, {"uno.preset": "filtersqp"})
  CasADi computes all derivatives and hands them to Uno internally.

Route B -- call Uno's Python API (unopy) yourself, using CasADi only as an AD engine:
  generate f, grad f, g, Jacobian of g and the Lagrangian Hessian as CasADi Functions and
  wrap them as unopy callbacks with explicit sparse (row, col) patterns. This is what the
  plugin does under the hood, and it is the way to go if you want Uno features that the
  plugin does not expose, or a non-CasADi model.

    python 02_uno_direct.py [--N 40] [--preset filtersqp|ipopt] [--verbose]
"""
import argparse
import io
import time

import casadi as ca
import numpy as np
import unopy

from nmpc_cartpole import NX, NU, cartpole_ode, rk4_step

Inf = float("inf")


def build_nlp(N=40, dt=0.05, F_max=20.0, p_max=1.5):
    """Multiple-shooting NLP (5) of the cart-pole OCP as a plain {x, f, g} dict plus bounds."""
    Phi = rk4_step(cartpole_ode(), dt, n_sub=2)             # Phi from (4), with f from (1)
    x_ref = ca.DM([0.0, np.pi, 0.0, 0.0])                   # upright equilibrium
    Q = ca.diag([10.0, 10.0, 0.1, 0.1])                     # weight in (3a)
    R, P = 0.01, 10 * Q                                     # weights in (3a), (3b)
    x_init = np.zeros(NX)                                   # xhat in (5b): hanging at rest

    Xs = [ca.SX.sym(f"x{k}", NX) for k in range(N + 1)]
    Us = [ca.SX.sym(f"u{k}", NU) for k in range(N)]
    w, lbw, ubw, w0 = [], [], [], []
    J, g, lbg, ubg = 0, [], [], []
    for k in range(N):
        w += [Xs[k], Us[k]]                                 # w = (..., x_k, u_k, ...)
        pb = Inf if k == 0 else p_max                       # (5e) for k >= 1; no bound on p_0 (data)
        lbw += [-pb, -Inf, -Inf, -Inf, -F_max]              # (5e) on p_k, (5d) on u_k
        ubw += [pb, Inf, Inf, Inf, F_max]                   # (5e) on p_k, (5d) on u_k
        w0 += list(x_init) + [0.0]                          # guess w^(0): x_k = xhat, u_k = 0
        e = Xs[k] - x_ref
        J += ca.bilin(Q, e, e) + R * Us[k] ** 2             # stage cost (3a)
        g.append(Xs[k + 1] - Phi(Xs[k], Us[k]))             # (5c), continuity constraints
        lbg += [0.0] * NX                                   # (5c) is an equality: 0 <= g <= 0
        ubg += [0.0] * NX
    w.append(Xs[N])
    lbw += [-p_max, -Inf, -Inf, -Inf]                       # (5e) on p_N
    ubw += [p_max, Inf, Inf, Inf]
    w0 += list(x_init)
    eN = Xs[N] - x_ref
    J += ca.bilin(P, eN, eN)                                # terminal cost (3b); J is now (5a)
    g.insert(0, Xs[0] - x_init)                             # (5b), initial-state constraint
    lbg = [0.0] * NX + lbg                                  # (5b) is an equality: 0 <= g <= 0
    ubg = [0.0] * NX + ubg

    nlp = {"x": ca.vertcat(*w), "f": J, "g": ca.vertcat(*g)}   # x = w, f = (5a), g = (5b), (5c)
    return nlp, dict(lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, x0=w0)


# ----------------------------------------------------------------------------- route A
def solve_plugin(nlp, b, solver, preset=None, verbose=False):
    opts = {"print_time": False}
    if solver == "uno":
        opts["uno.preset"] = preset
        if not verbose:
            opts["uno.logger"] = "SILENT"
    elif not verbose:
        opts.update({"ipopt.print_level": 0, "ipopt.sb": "yes"})
    S = ca.nlpsol("S", solver, nlp, opts)
    tic = time.perf_counter()
    sol = S(**b)
    t = time.perf_counter() - tic
    st = S.stats()
    return dict(w=np.array(sol["x"]).ravel(), f=float(sol["f"]), lam_g=np.array(sol["lam_g"]).ravel(),
                ok=st["success"], iters=st.get("iter_count", -1), time=t, status=st["return_status"])


# ----------------------------------------------------------------------------- route B
class CasadiUnoCallbacks:
    """Wrap a CasADi {x, f, g} NLP as the callbacks unopy expects.

    Sparse matrices are passed as coordinate (row, col) triplets. CasADi stores nonzeros
    column-major, and Sparsity.get_triplet() lists (row, col) in the same order, so the
    nonzeros of a DM can be copied straight into Uno's value arrays.
    """

    def __init__(self, nlp):
        x, f, g = nlp["x"], nlp["f"], nlp["g"]              # w, J from (5a), g from (5b), (5c)
        self.n, self.m = x.numel(), g.numel()
        sigma = ca.SX.sym("sigma")
        lam = ca.SX.sym("lam", self.m)
        # Uno's MULTIPLIER_POSITIVE convention: L = sigma f + lam' g  (same sign as CasADi)
        lag = sigma * f + ca.dot(lam, g)
        H = ca.tril(ca.hessian(lag, x)[0])                  # lower triangle only
        Jg = ca.jacobian(g, x)

        self.f_fun = ca.Function("f", [x], [f])
        self.grad_fun = ca.Function("grad_f", [x], [ca.gradient(f, x)])
        self.g_fun = ca.Function("g", [x], [g])
        self.jac_fun = ca.Function("jac_g", [x], [Jg])
        self.hess_fun = ca.Function("hess_l", [x, sigma, lam], [H])
        self.jac_rows, self.jac_cols = Jg.sparsity().get_triplet()
        self.hess_rows, self.hess_cols = H.sparsity().get_triplet()

    # unopy fills the output arrays in place
    def objective(self, x):
        return float(self.f_fun(x))                         # J, (5a)

    def gradient(self, x, grad):
        grad[:] = np.array(self.grad_fun(x)).ravel()        # gradient of (5a)

    def constraints(self, x, c):
        c[:] = np.array(self.g_fun(x)).ravel()              # residuals of (5b), (5c)

    def jacobian(self, x, vals):
        vals[:] = self.jac_fun(x).nonzeros()                # Jacobian of (5b), (5c)

    def hessian(self, x, sigma, lam, vals):
        vals[:] = self.hess_fun(x, sigma, lam).nonzeros()   # Hessian of sigma (5a) + lam' [(5b); (5c)]


def solve_unopy(nlp, b, preset, verbose=False):
    cb = CasadiUnoCallbacks(nlp)
    model = unopy.Model(unopy.PROBLEM_NONLINEAR, cb.n, unopy.ZERO_BASED_INDEXING)
    model.set_variables_lower_bounds(b["lbx"])                      # (5d), (5e)
    model.set_variables_upper_bounds(b["ubx"])                      # (5d), (5e)
    model.set_objective(unopy.MINIMIZE, cb.objective, cb.gradient)  # (5a)
    model.set_constraints(cb.m, cb.constraints, b["lbg"], b["ubg"], # (5b), (5c)
                          len(cb.jac_rows), cb.jac_rows, cb.jac_cols, cb.jacobian)
    model.set_lagrangian_hessian(len(cb.hess_rows), unopy.LOWER_TRIANGLE,
                                 cb.hess_rows, cb.hess_cols, cb.hessian)
    model.set_lagrangian_sign_convention(unopy.MULTIPLIER_POSITIVE)
    model.set_initial_primal_iterate(b["x0"])                       # guess w^(0)

    solver = unopy.UnoSolver()
    if not verbose:
        solver.set_logger_stream(io.StringIO())
    solver.set_preset(preset)
    tic = time.perf_counter()
    r = solver.optimize(model)
    t = time.perf_counter() - tic
    return dict(w=np.asarray(r.primal_solution, dtype=float), f=r.solution_objective,
                lam_g=np.asarray(r.constraint_dual_solution, dtype=float),
                ok=int(r.optimization_status) == unopy.SUCCESS, iters=r.number_iterations,
                time=t, status=str(r.solution_status))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=40)
    ap.add_argument("--preset", choices=["filtersqp", "ipopt"], default="filtersqp")
    ap.add_argument("--verbose", action="store_true", help="show the solver logs")
    args = ap.parse_args()

    nlp, b = build_nlp(args.N)
    print(f"Uno {unopy.current_uno_version()}, casadi {ca.__version__}; "
          f"NLP: n = {nlp['x'].numel()} variables, m = {nlp['g'].numel()} constraints\n")

    runs = {
        "A: nlpsol ipopt (reference)": solve_plugin(nlp, b, "ipopt", verbose=args.verbose),
        f"A: nlpsol uno/{args.preset}": solve_plugin(nlp, b, "uno", args.preset, args.verbose),
        f"B: unopy  uno/{args.preset}": solve_unopy(nlp, b, args.preset, args.verbose),
    }
    print(f"{'route':32s} {'ok':>5s} {'iters':>6s} {'time [s]':>9s} {'objective':>14s}")
    for name, r in runs.items():
        print(f"{name:32s} {str(r['ok']):>5s} {r['iters']:6d} {r['time']:9.3f} {r['f']:14.8f}")

    ref = runs["A: nlpsol ipopt (reference)"]
    print()
    for name, r in list(runs.items())[1:]:
        print(f"{name}: |f - f_ipopt| = {abs(r['f'] - ref['f']):.2e}, "
              f"max|w - w_ipopt| = {np.abs(r['w'] - ref['w']).max():.2e}, "
              f"max|lam_g - lam_g_ipopt| = {np.abs(r['lam_g'] - ref['lam_g']).max():.2e}")
    a, bb = list(runs.values())[1:]
    print(f"route A vs route B: max|w_A - w_B| = {np.abs(a['w'] - bb['w']).max():.2e}  "
          "(same algorithm, same derivatives -> same iterates)")


if __name__ == "__main__":
    main()
