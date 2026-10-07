"""TutorialGO Question 1: the smallest RLT relaxation of x^T Q_i x + b^T x + a <= 0.

A relaxation of the constraint is only as good as the lower bound it gives for
q(x) = x^T Q x + b^T x over the box, so we compare lower bounds of q on [-1,1]^4:

  mcc       McCormick RLT, every term of Q lifted (X_ij for each nonzero Q_ij), an LP
  sdp-full  McCormick + one PSD cone [1 x^T; x X] of size 5x5
  sdp-red   McCormick + one small cone per clique of the sparsity graph
            (Q_1: {1,2},{3,4}; Q_2: {1,2},{2,3},{3,4})
  sdp-blk   (Q_1 only) the convex block (3,4) is kept as a quadratic, only {1,2} is lifted
  split     convex split x^T Q x = x^T (Q + d e_1 e_1^T) x - d x_1^2 with the smallest d:
            one lifted variable X_11 and its McCormick (secant) inequalities, a convex QP
  abb       uniform shift Q - lambda_min I (alphaBB): four lifted X_ii, a convex QP

PSD cones are imposed through all principal minors >= 0 and solved with Uno. The
feasible set is convex and the objective linear, so a local solution is global; we
still use several starts and check the eigenvalues of the solution.

Run:  conda run -n optimization python q1_rlt.py
"""
import itertools

import casadi as ca
import numpy as np

from plotstyle import C, INK2, plt, save
from uno_lp import INF, solve_casadi, solve_lp

Q1 = np.array([[-2, -1, 0, 0], [-1, 2, 0, 0], [0, 0, 3, -1], [0, 0, -1, 2]], float)
Q2 = np.array([[-2, 1, 0, 0], [1, 2, -1, 0], [0, -1, 3, -1], [0, 0, -1, 2]], float)
N = 4
LO, UP = -np.ones(N), np.ones(N)
CHECKS = []


def check(name, ok):
    CHECKS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")


# ---------------------------------------------------------------- true minimum
def true_min(Q, b, lo=LO, up=UP):
    """Global min of x^T Q x + b^T x on a box: enumerate all faces (each x_i at lo, up or free)
    and solve the stationarity system on the free variables."""
    best, xbest = np.inf, None
    for pattern in itertools.product((0, 1, 2), repeat=N):
        x = np.zeros(N)
        free = [i for i, p in enumerate(pattern) if p == 2]
        fixed = [i for i, p in enumerate(pattern) if p != 2]
        for i in fixed:
            x[i] = lo[i] if pattern[i] == 0 else up[i]
        if free:
            A = 2 * Q[np.ix_(free, free)]
            r = -(2 * Q[np.ix_(free, fixed)] @ x[fixed] + b[free])
            try:
                x[free] = np.linalg.solve(A, r)
            except np.linalg.LinAlgError:
                continue
        if np.all(x >= lo - 1e-12) and np.all(x <= up + 1e-12):
            v = x @ Q @ x + b @ x
            if v < best:
                best, xbest = v, x.copy()
    return best, xbest


# ---------------------------------------------------------------- McCormick rows
def mccormick_rows(pairs, nvar, col, lo=LO, up=UP):
    """Rows (a, lb, ub) of the McCormick/RLT inequalities for X_ij ~ x_i x_j, written as
    lb <= a^T z <= ub with z = (x, X). col maps a pair (i,j) to its column in z."""
    rows = []

    def row(coef, lb, ub):
        a = np.zeros(nvar)
        for k, v in coef.items():
            a[k] += v
        rows.append((a, lb, ub))

    for (i, j) in pairs:
        k = col[(i, j)]
        li, ui, lj, uj = lo[i], up[i], lo[j], up[j]
        # (x_i - l_i)(x_j - l_j) >= 0,  (u_i - x_i)(u_j - x_j) >= 0
        row({k: 1, i: -lj, j: -li}, -li * lj, INF)
        row({k: 1, i: -uj, j: -ui}, -ui * uj, INF)
        # (x_i - l_i)(u_j - x_j) >= 0,  (u_i - x_i)(x_j - l_j) >= 0
        if i != j:
            row({k: 1, i: -uj, j: -li}, -INF, -li * uj)
            row({k: 1, i: -lj, j: -ui}, -INF, -ui * lj)
        else:
            row({k: 1, i: -(li + ui)}, -INF, -li * ui)
            row({k: 1}, 0.0, INF)  # X_ii >= 0
    return rows


def pattern(Q):
    return [(i, j) for i in range(N) for j in range(i, N) if Q[i, j] != 0]


def lifted_objective(Q, b, pairs, col, nvar):
    c = np.zeros(nvar)
    c[:N] = b
    for (i, j) in pairs:
        c[col[(i, j)]] += Q[i, j] if i == j else 2 * Q[i, j]
    return c


# ---------------------------------------------------------------- relaxations
def relax_mcc(Q, b):
    pairs = pattern(Q)
    col = {p: N + k for k, p in enumerate(pairs)}
    nvar = N + len(pairs)
    rows = mccormick_rows(pairs, nvar, col)
    A = np.array([r[0] for r in rows])
    c = lifted_objective(Q, b, pairs, col, nvar)
    f, _ = solve_lp(c, A, [r[1] for r in rows], [r[2] for r in rows],
                    np.r_[LO, -INF * np.ones(len(pairs))], np.r_[UP, INF * np.ones(len(pairs))])
    return f


def relax_sdp(Q, b, cliques, convex_keep=None, starts=4, seed=0):
    """McCormick on the pattern of Q (restricted to the lifted cliques) plus one PSD cone
    [1 x_C^T; x_C X_CC] per clique C (all principal minors >= 0).
    convex_keep: index set whose quadratic block is kept as a convex term (not lifted)."""
    keep = list(convex_keep or [])
    lifted = sorted({(i, j) for Cq in cliques for i in Cq for j in Cq if i <= j})
    col = {p: N + k for k, p in enumerate(lifted)}
    nvar = N + len(lifted)
    z = ca.SX.sym("z", nvar)
    x = z[:N]

    def X(i, j):
        return z[col[(min(i, j), max(i, j))]]

    # objective: lifted part + convex part
    Ql = Q.copy()
    if keep:
        Ql[np.ix_(keep, keep)] = 0
    f = ca.dot(ca.DM(b), x)
    for (i, j) in pattern(Ql):
        f += (Q[i, j] if i == j else 2 * Q[i, j]) * X(i, j)
    if keep:
        xk = x[keep]
        f += ca.mtimes([xk.T, ca.DM(Q[np.ix_(keep, keep)]), xk])

    g, lbg, ubg = [], [], []
    mc_pairs = [p for p in pattern(Ql) if p in col]
    for a, lb, ub in mccormick_rows(mc_pairs, nvar, col):
        g.append(ca.dot(ca.DM(a), z))
        lbg.append(lb)
        ubg.append(ub)
    for Cq in cliques:
        M = ca.SX(len(Cq) + 1, len(Cq) + 1)
        M[0, 0] = 1
        for a, i in enumerate(Cq):
            M[0, a + 1] = M[a + 1, 0] = x[i]
            for bb, j in enumerate(Cq):
                M[a + 1, bb + 1] = X(i, j)
        for r in range(1, M.shape[0] + 1):
            for S in itertools.combinations(range(M.shape[0]), r):
                g.append(ca.det(M[list(S), list(S)]))
                lbg.append(0.0)
                ubg.append(INF)
    g = ca.vertcat(*g)

    rng = np.random.default_rng(seed)
    best = (np.inf, None)
    for s in range(starts):
        x0 = np.zeros(nvar) if s == 0 else np.r_[rng.uniform(-0.3, 0.3, N), np.zeros(len(lifted))]
        for (i, j), k in col.items():
            x0[k] = x0[i] * x0[j] + (0.5 if i == j else 0.0)
        # filterSQP first; the interior-point preset with tight tolerances as a fallback
        for preset, opts in (("filtersqp", {}),
                             ("ipopt", {"primal_tolerance": 1e-10, "dual_tolerance": 1e-10})):
            try:
                val, zs, _ = solve_casadi(z, f, g, lbg, ubg, np.r_[LO, -INF * np.ones(len(lifted))],
                                          np.r_[UP, INF * np.ones(len(lifted))], x0,
                                          preset=preset, options=opts)
            except RuntimeError:
                continue
            if val < best[0]:
                best = (val, zs)
            break
    val, zs = best
    if zs is None:
        raise RuntimeError("all Uno starts failed")
    # verify: every cone PSD at the solution
    mineig = np.inf
    for Cq in cliques:
        M = np.eye(len(Cq) + 1)
        for a, i in enumerate(Cq):
            M[0, a + 1] = M[a + 1, 0] = zs[i]
            for bb, j in enumerate(Cq):
                M[a + 1, bb + 1] = zs[col[(min(i, j), max(i, j))]]
        mineig = min(mineig, np.linalg.eigvalsh(M).min())
    return val, mineig


def shift_split(Q):
    """Smallest d >= 0 with Q + d e_1 e_1^T PSD (Schur complement on the trailing block)."""
    Cm, bv = Q[1:, 1:], Q[1:, 0]
    assert np.linalg.eigvalsh(Cm).min() > 0
    return max(0.0, -Q[0, 0] + bv @ np.linalg.solve(Cm, bv))


def relax_convex_shift(Q, b, D):
    """min x^T (Q + D) x + b^T x - sum_i D_ii X_ii, X_ii <= secant, a convex QP (D >= 0 diagonal)."""
    idx = [i for i in range(N) if D[i] > 0]
    nvar = N + len(idx)
    z = ca.SX.sym("z", nvar)
    x = z[:N]
    f = ca.mtimes([x.T, ca.DM(Q + np.diag(D)), x]) + ca.dot(ca.DM(b), x)
    col = {(i, i): N + k for k, i in enumerate(idx)}
    g, lbg, ubg = [], [], []
    for k, i in enumerate(idx):
        f -= D[i] * z[N + k]
    for a, lb, ub in mccormick_rows([(i, i) for i in idx], nvar, col):
        g.append(ca.dot(ca.DM(a), z))
        lbg.append(lb)
        ubg.append(ub)
    val, _, _ = solve_casadi(z, f, ca.vertcat(*g), lbg, ubg,
                             np.r_[LO, -INF * np.ones(len(idx))], np.r_[UP, INF * np.ones(len(idx))],
                             np.zeros(nvar))
    return val


# ---------------------------------------------------------------- main
def analyse(name, Q, b, cliques, keep_block=None):
    print(f"\n{name}, b = {b}:")
    ev = np.linalg.eigvalsh(Q)
    print(f"  eig = {np.round(ev, 4)}")
    tm, xm = true_min(Q, b)
    print(f"  true min q = {tm:.6f} at x = {np.round(xm, 4)}")
    d = shift_split(Q)
    print(f"  smallest d with Q + d e1e1^T PSD: d* = {d:.6f}")
    check(f"{name}: Q + d* e1e1^T is PSD and singular",
          abs(np.linalg.eigvalsh(Q + d * np.diag([1, 0, 0, 0])).min()) < 1e-10)

    res = {}
    res["mcc"] = relax_mcc(Q, b)
    full, eig_full = relax_sdp(Q, b, [list(range(N))])
    red, eig_red = relax_sdp(Q, b, cliques)
    res["sdp-full"], res["sdp-red"] = full, red
    # The relaxations are exact here, so the optimal moment matrix has rank one and the
    # principal-minor constraints are degenerate (no LICQ): Uno only reaches ~1e-5 feasibility.
    check(f"{name}: cone solutions PSD up to 1e-4 (min eig {min(eig_full, eig_red):.1e})",
          min(eig_full, eig_red) > -1e-4)
    check(f"{name}: chordal cones give the same bound as the full cone ({red:.6f} vs {full:.6f})",
          abs(full - red) < 1e-5)
    if keep_block is not None:
        res["sdp-blk"], _ = relax_sdp(Q, b, [c for c in cliques if not set(c) <= set(keep_block)],
                                      convex_keep=keep_block)
    res["split"] = relax_convex_shift(Q, b, np.array([d, 0, 0, 0]))
    res["abb"] = relax_convex_shift(Q, b, -min(ev.min(), 0) * np.ones(N))
    for k, v in res.items():
        print(f"  {k:9s} lower bound {v:10.6f}   gap {tm - v:8.4f}")
        check(f"{name}: {k} is a valid lower bound", v <= tm + 1e-6)
    check(f"{name}: convex split is exact (one lifted X_11 suffices)", abs(res["split"] - tm) < 1e-6)
    return tm, res


def main():
    cases = []
    for b in (np.zeros(N), np.array([1.0, -1.0, 1.0, -1.0])):
        tag = "b=0" if not b.any() else "b=(1,-1,1,-1)"
        cases.append((f"Q1, {tag}", *analyse("Q1", Q1, b, [[0, 1], [2, 3]], keep_block=[2, 3])))
        cases.append((f"Q2, {tag}", *analyse("Q2", Q2, b, [[0, 1], [1, 2], [2, 3]])))

    # figure: dot plot of lower bounds per relaxation, one panel per case
    labels = {"mcc": "McCormick, all terms", "sdp-full": "SDP + McC, 5x5 cone",
              "sdp-red": "SDP + McC, clique cones", "sdp-blk": "SDP {1,2} + convex block",
              "split": "convex split, one $X_{11}$", "abb": r"$\alpha$BB shift, four $X_{ii}$"}
    order = list(labels)
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 4.6), sharey=True)
    for ax, (title, tm, res) in zip(axes.ravel(), cases):
        ys = np.arange(len(order))[::-1]
        for y, k in zip(ys, order):
            if k in res:
                ax.plot([res[k], tm], [y, y], color=C[0], lw=1, alpha=0.4)
                ax.plot(res[k], y, "o", ms=6, color=C[0], mec="white", mew=1.5)
                if tm - res[k] > 1e-3:  # exact bounds sit on the true-min line, no label needed
                    ax.annotate(f"{res[k]:.3f}", (res[k], y), xytext=(0, 5),
                                textcoords="offset points", ha="center", fontsize=7, color=INK2)
        ax.axvline(tm, color=C[1], lw=2)
        ax.annotate(f"true min {tm:.3f}", (tm, ys[-1] - 0.6), xytext=(-4, 0),
                    textcoords="offset points", ha="right", fontsize=7, color=INK2)
        ax.set_yticks(ys, [labels[k] for k in order])
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-1, len(order))
        ax.set_xlim(right=tm + 0.6)
    for ax in axes[1]:
        ax.set_xlabel(r"lower bound on $x^TQx + b^Tx$ over $[-1,1]^4$")
    fig.tight_layout()
    save(fig, "q1_bounds.pdf")

    nfail = sum(not ok for _, ok in CHECKS)
    print(f"\n{len(CHECKS) - nfail}/{len(CHECKS)} checks passed")
    return nfail


if __name__ == "__main__":
    raise SystemExit(main())
