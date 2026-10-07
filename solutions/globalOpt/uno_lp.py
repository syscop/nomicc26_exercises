"""Small Uno (unopy) wrappers for the relaxations in TutorialGO.

solve_lp   min c^T x  s.t.  lb_g <= A x <= ub_g,  lb_x <= x <= ub_x   (cross-checked with HiGHS)
solve_qcqp min x^T Q x  s.t.  x^T x = 1                                  (Q4, local solves)

The callback layout follows exercises/ballthrow/ball_throw.py.
"""
import io

import numpy as np
import unopy
from scipy.optimize import linprog

INF = 1e20


def _solver(preset="filtersqp", verbose=False):
    solver = unopy.UnoSolver()
    if not verbose:
        solver.set_logger_stream(io.StringIO())
    solver.set_preset(preset)
    return solver


def solve_lp(c, A, lb_g, ub_g, lb_x, ub_x, check=True, tol=1e-6, verbose=False):
    """Solve a dense LP with Uno; return (f*, x*). If check, compare with scipy linprog (HiGHS)."""
    c = np.asarray(c, float)
    A = np.atleast_2d(np.asarray(A, float))
    n, m = c.size, A.shape[0]
    lb_g = np.clip(np.asarray(lb_g, float), -INF, INF)
    ub_g = np.clip(np.asarray(ub_g, float), -INF, INF)
    lb_x = np.clip(np.asarray(lb_x, float), -INF, INF)
    ub_x = np.clip(np.asarray(ub_x, float), -INF, INF)

    rows, cols = np.nonzero(A)
    vals = A[rows, cols]

    def objective(x):
        return float(c @ np.asarray(x[:n]))

    def gradient(x, grad):
        grad[:] = c

    def constraints(x, g):
        g[:] = A @ np.asarray(x[:n])

    def jacobian(x, jac):
        jac[:] = vals

    def hessian(x, sigma, lam, hess):
        pass  # linear problem: Hessian of the Lagrangian is zero

    model = unopy.Model(unopy.PROBLEM_LINEAR, n, unopy.ZERO_BASED_INDEXING)
    model.set_variables_lower_bounds(lb_x.tolist())
    model.set_variables_upper_bounds(ub_x.tolist())
    model.set_objective(unopy.MINIMIZE, objective, gradient)
    model.set_constraints(m, constraints, lb_g.tolist(), ub_g.tolist(),
                          len(vals), rows.tolist(), cols.tolist(), jacobian)
    model.set_lagrangian_hessian(0, unopy.LOWER_TRIANGLE, [], [], hessian)
    x0 = np.clip(np.zeros(n), lb_x, ub_x)
    model.set_initial_primal_iterate(x0.tolist())

    result = _solver(verbose=verbose).optimize(model)
    if int(result.optimization_status) != unopy.SUCCESS:
        raise RuntimeError(f"Uno failed: {result.optimization_status}, {result.solution_status}")
    x = np.asarray(result.primal_solution[:n], float)
    f = float(c @ x)

    if check:
        A_ub = np.vstack([A[ub_g < INF], -A[lb_g > -INF]])
        b_ub = np.concatenate([ub_g[ub_g < INF], -lb_g[lb_g > -INF]])
        bounds = [(None if l <= -INF else l, None if u >= INF else u) for l, u in zip(lb_x, ub_x)]
        ref = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not ref.success or abs(ref.fun - f) > tol * max(1.0, abs(f)):
            raise RuntimeError(f"Uno ({f}) and HiGHS ({ref.fun}) disagree")
    return f, x


def solve_qcqp_sphere(Q, x0, sense=unopy.MINIMIZE, verbose=False):
    """Local solve of min/max x^T Q x s.t. x^T x = 1 with Uno; return (f*, x*, lambda)."""
    Q = np.asarray(Q, float)
    n = Q.shape[0]
    lower = [(i, j) for i in range(n) for j in range(i + 1)]
    hr, hc = [i for i, _ in lower], [j for _, j in lower]

    def objective(x):
        x = np.asarray(x[:n])
        return float(x @ Q @ x)

    def gradient(x, grad):
        grad[:] = 2 * Q @ np.asarray(x[:n])

    def constraints(x, g):
        x = np.asarray(x[:n])
        g[0] = x @ x

    def jacobian(x, jac):
        jac[:] = 2 * np.asarray(x[:n])

    def hessian(x, sigma, lam, hess):
        # MULTIPLIER_POSITIVE: sigma * 2Q + lam * 2I
        H = sigma * 2 * Q + lam[0] * 2 * np.eye(n)
        hess[:] = [H[i, j] for i, j in lower]

    model = unopy.Model(unopy.PROBLEM_NONLINEAR, n, unopy.ZERO_BASED_INDEXING)
    model.set_objective(sense, objective, gradient)
    model.set_constraints(1, constraints, [1.0], [1.0], n, [0] * n, list(range(n)), jacobian)
    model.set_lagrangian_hessian(len(lower), unopy.LOWER_TRIANGLE, hr, hc, hessian)
    model.set_lagrangian_sign_convention(unopy.MULTIPLIER_POSITIVE)
    model.set_initial_primal_iterate(list(np.asarray(x0, float)))

    result = _solver(verbose=verbose).optimize(model)
    if int(result.optimization_status) != unopy.SUCCESS:
        raise RuntimeError(f"Uno failed: {result.optimization_status}")
    x = np.asarray(result.primal_solution[:n], float)
    return objective(x), x, np.asarray(result.constraint_dual_solution, float)


def solve_casadi(x, f, g, lbg, ubg, lbx, ubx, x0, sense=unopy.MINIMIZE, preset="filtersqp",
                 options=None, verbose=False):
    """Solve min f(x) s.t. lbg <= g(x) <= ubg, lbx <= x <= ubx with Uno.

    x is a CasADi SX column, f and g are SX expressions; exact derivatives come from CasADi.
    Returns (f*, x*, result).
    """
    import casadi as ca

    n = x.numel()
    if g is None or g.numel() == 0:
        # Uno (0.4.x) fails with zero constraints (ALGORITHMIC_ERROR, or a segfault with logging
        # on), so pass one free row -inf <= x_0 <= inf instead.
        g, lbg, ubg = x[0], [-INF], [INF]
    g = ca.vec(g)
    m = g.numel()
    lam = ca.SX.sym("lam", m)
    sig = ca.SX.sym("sig")

    F = ca.Function("f", [x], [f])
    G = ca.Function("g", [x], [g])
    dF = ca.Function("df", [x], [ca.gradient(f, x)])
    Jg = ca.jacobian(g, x)
    Jsp = Jg.sparsity()
    J = ca.Function("J", [x], [ca.densify(Jg)])
    H = ca.tril(ca.hessian(sig * f + ca.dot(lam, g), x)[0])
    Hsp = H.sparsity()
    HF = ca.Function("H", [x, sig, lam], [H])

    jr, jc = Jsp.get_triplet()
    hr, hc = Hsp.get_triplet()

    def objective(z):
        return float(F(np.asarray(z[:n])))

    def gradient(z, grad):
        grad[:] = np.asarray(dF(np.asarray(z[:n]))).ravel()

    def constraints(z, c):
        c[:] = np.asarray(G(np.asarray(z[:n]))).ravel()

    def jacobian(z, jac):
        Jd = np.asarray(J(np.asarray(z[:n])))
        jac[:] = Jd[jr, jc]

    def hessian(z, sigma, mult, hess):
        hess[:] = np.asarray(HF(np.asarray(z[:n]), sigma, np.asarray(mult[:m])).nonzeros())

    model = unopy.Model(unopy.PROBLEM_NONLINEAR, n, unopy.ZERO_BASED_INDEXING)
    model.set_variables_lower_bounds(np.clip(np.broadcast_to(lbx, n), -INF, INF).tolist())
    model.set_variables_upper_bounds(np.clip(np.broadcast_to(ubx, n), -INF, INF).tolist())
    model.set_objective(sense, objective, gradient)
    model.set_constraints(m, constraints,
                          np.clip(np.broadcast_to(lbg, m), -INF, INF).tolist(),
                          np.clip(np.broadcast_to(ubg, m), -INF, INF).tolist(),
                          len(jr), list(jr), list(jc), jacobian)
    model.set_lagrangian_hessian(len(hr), unopy.LOWER_TRIANGLE, list(hr), list(hc), hessian)
    model.set_lagrangian_sign_convention(unopy.MULTIPLIER_POSITIVE)
    model.set_initial_primal_iterate(list(np.asarray(x0, float)))

    solver = _solver(preset=preset, verbose=verbose)
    for k, v in (options or {}).items():
        solver.set_option(k, v)
    result = solver.optimize(model)
    if int(result.optimization_status) != unopy.SUCCESS:
        raise RuntimeError(f"Uno failed: {result.optimization_status}, {result.solution_status}")
    z = np.asarray(result.primal_solution[:n], float)
    return objective(z), z, result


if __name__ == "__main__":
    # smoke test: min -x - y  s.t.  x + 2y <= 4, 3x + y <= 6, 0 <= x, y  ->  -2.8 at (1.6, 1.2)
    f, x = solve_lp([-1, -1], [[1, 2], [3, 1]], [-INF, -INF], [4, 6], [0, 0], [INF, INF])
    print("LP:", f, x)
    Q = np.diag([3.0, -1.0, 2.0])
    print("sphere min:", solve_qcqp_sphere(Q, [1, 1, 1])[0])
