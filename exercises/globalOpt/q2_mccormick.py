"""TutorialGO Question 2: one bilinear term instead of many.

    (1 - x1 + 2x2 - 2x4)(4x2 - 5x3 + x4 + 2) <= 2,   -1 <= x_i <= 1.

With u = 1 - x1 + 2x2 - 2x4 and v = 4x2 - 5x3 + x4 + 2 the constraint is w <= 2, w = u v,
a single bilinear term. Interval arithmetic gives u in [-4, 6], v in [-8, 12], and the
McCormick relaxation is

    w >= -8u - 4v - 32        from (u + 4)(v + 8) >= 0
    w >= 12u + 6v - 72        from (6 - u)(12 - v) >= 0
    w <= 12u - 4v + 48        from (u + 4)(12 - v) >= 0
    w <= -8u + 6v + 48        from (6 - u)(v + 8) >= 0
    w <= 2.

We compare it with the textbook alternative: expand the product and lift each of the
eight quadratic monomials separately (term-wise McCormick).

Run:  conda run -n optimization python q2_mccormick.py
"""
import itertools

import numpy as np
import sympy as sp
import casadi as ca

from plotstyle import C, INK2, SEQ, plt, save
from uno_lp import INF, solve_casadi, solve_lp

CHECKS = []


def check(name, ok):
    CHECKS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")


xs = sp.symbols("x1:5")
x1, x2, x3, x4 = xs
U = 1 - x1 + 2 * x2 - 2 * x4
V = 4 * x2 - 5 * x3 + x4 + 2
PROD = sp.expand(U * V)
ua, va = (np.array([float(e.coeff(xi)) for xi in xs]) for e in (U, V))
u0, v0 = float(U.subs({xi: 0 for xi in xs})), float(V.subs({xi: 0 for xi in xs}))


def interval(c0, a):
    return c0 - np.abs(a).sum(), c0 + np.abs(a).sum()


UL, UU = interval(u0, ua)
VL, VU = interval(v0, va)


def u_of(X):
    return u0 + X @ ua


def v_of(X):
    return v0 + X @ va


# ---------------------------------------------------------------- single-term relaxation
def single_rows():
    """Rows over z = (x1..x4, w): lb <= a^T z <= ub (the 4 McCormick rows and w <= 2)."""
    rows = []
    # w - (a u + b v) >= c  for under-, <= c for over-estimators; u = u0 + ua.x, v = v0 + va.x
    for (a, b, cst, sense) in [(VL, UL, -UL * VL, ">="), (VU, UU, -UU * VU, ">="),
                               (VU, UL, -UL * VU, "<="), (VL, UU, -UU * VL, "<=")]:
        coef = np.r_[-(a * ua + b * va), 1.0]
        rhs = cst + a * u0 + b * v0
        rows.append((coef, rhs, INF) if sense == ">=" else (coef, -INF, rhs))
    rows.append((np.r_[np.zeros(4), 1.0], -INF, 2.0))
    return rows


def single_accepts(X):
    """x is in the projection of the single-term relaxation iff max(underestimators) <= 2."""
    u, v = u_of(X), v_of(X)
    lo = np.maximum(VL * u + UL * v - UL * VL, VU * u + UU * v - UU * VU)
    return lo <= 2 + 1e-12


# ---------------------------------------------------------------- term-wise relaxation
QUAD = [m for m in sp.Poly(PROD, *xs).monoms() if sum(m) == 2]


def monomial_pair(m):
    idx = [i for i, e in enumerate(m) for _ in range(e)]
    return tuple(idx)


PAIRS = [monomial_pair(m) for m in QUAD]
PCOEF = np.array([float(sp.Poly(PROD, *xs).coeff_monomial(m)) for m in QUAD])
LIN = np.array([float(PROD.coeff(xi).subs({xj: 0 for xj in xs})) for xi in xs])
CONST = float(PROD.subs({xi: 0 for xi in xs}))


def mcc_bounds_pair(X, i, j):
    """Lowest and highest value of X_ij allowed by McCormick on [-1,1]^2 given x (vectorised)."""
    a, b = X[:, i], X[:, j]
    if i == j:
        lo = np.maximum(np.maximum(2 * a - 1, -2 * a - 1), 0.0)
        hi = np.ones_like(a)
    else:
        lo = np.maximum(-a - b - 1, a + b - 1)
        hi = np.minimum(a - b + 1, -a + b + 1)
    return lo, hi


def termwise_accepts(X):
    total = CONST + X @ LIN
    for c, (i, j) in zip(PCOEF, PAIRS):
        lo, hi = mcc_bounds_pair(X, i, j)
        total += c * (lo if c > 0 else hi)
    return total <= 2 + 1e-12


def termwise_rows():
    """Rows over z = (x1..x4, X_pairs): McCormick rows per pair and the linearised constraint."""
    n = 4 + len(PAIRS)
    rows = []
    for k, (i, j) in enumerate(PAIRS):
        K = 4 + k

        def row(coef, lb, ub):
            a = np.zeros(n)
            for idx, v in coef.items():
                a[idx] += v
            rows.append((a, lb, ub))

        row({K: 1, i: 1, j: 1}, -1, INF)        # (x_i + 1)(x_j + 1) >= 0
        row({K: 1, i: -1, j: -1}, -1, INF)      # (1 - x_i)(1 - x_j) >= 0
        if i != j:
            row({K: 1, i: -1, j: 1}, -INF, 1)   # (x_i + 1)(1 - x_j) >= 0
            row({K: 1, i: 1, j: -1}, -INF, 1)
        else:
            row({K: 1}, 0, 1)                   # 0 <= X_ii <= 1 (secant)
    a = np.r_[LIN, PCOEF]
    rows.append((a, -INF, 2 - CONST))
    return rows


# ---------------------------------------------------------------- main
def main():
    print("Q2: hidden bilinear term")
    print(f"  expanded product: {PROD}")
    check("product of the two affine factors equals u*v (sympy)", sp.expand(PROD - U * V) == 0)
    print(f"  u in [{UL:g}, {UU:g}], v in [{VL:g}, {VU:g}]")
    V16 = np.array(list(itertools.product((-1, 1), repeat=4)), float)
    check("interval bounds of u and v are attained at box vertices",
          np.isclose(u_of(V16).min(), UL) and np.isclose(u_of(V16).max(), UU)
          and np.isclose(v_of(V16).min(), VL) and np.isclose(v_of(V16).max(), VU))
    print(f"  {len(PAIRS)} quadratic monomials in the expansion: "
          + ", ".join(f"{c:+g} x{i+1}x{j+1}" for c, (i, j) in zip(PCOEF, PAIRS)))

    # validity of the McCormick inequalities for w = u v, by sampling the (u,v) box
    rng = np.random.default_rng(1)
    uu, vv = rng.uniform(UL, UU, 10**6), rng.uniform(VL, VU, 10**6)
    w = uu * vv
    ok = (np.all(w >= VL * uu + UL * vv - UL * VL - 1e-9) and np.all(w >= VU * uu + UU * vv - UU * VU - 1e-9)
          and np.all(w <= VU * uu + UL * vv - UL * VU + 1e-9) and np.all(w <= VL * uu + UU * vv - UU * VL + 1e-9))
    check("McCormick inequalities hold for w = uv on 1e6 samples of [-4,6]x[-8,12]", ok)

    # projections onto x: sample the box
    X = rng.uniform(-1, 1, (10**6, 4))
    feas = u_of(X) * v_of(X) <= 2
    acc1, acc2 = single_accepts(X), termwise_accepts(X)
    check("single-term relaxation contains every sampled feasible x", np.all(acc1[feas]))
    check("term-wise relaxation contains every sampled feasible x", np.all(acc2[feas]))
    print(f"  share of [-1,1]^4 accepted:  feasible {feas.mean():.3f}   "
          f"single-term {acc1.mean():.3f}   term-wise {acc2.mean():.3f}")
    print(f"  single-term accepts a sample the term-wise one rejects: {np.any(acc1 & ~acc2)}")
    print(f"  term-wise accepts a sample the single-term one rejects: {np.any(acc2 & ~acc1)}")

    # lower bound on the left-hand side over the whole box: Uno LPs vs. the true minimum (Uno NLP)
    r1 = [r for r in single_rows() if not (r[0][4] == 1 and np.all(r[0][:4] == 0))]  # drop w <= 2
    r2 = termwise_rows()[:-1]                                                          # drop the cut
    n2 = 4 + len(PAIRS)
    lb1, _ = solve_lp(np.r_[np.zeros(4), 1.0], np.array([r[0] for r in r1]), [r[1] for r in r1],
                      [r[2] for r in r1], np.r_[-np.ones(4), -INF], np.r_[np.ones(4), INF])
    lb2, _ = solve_lp(np.r_[LIN, PCOEF], np.array([r[0] for r in r2]), [r[1] for r in r2],
                      [r[2] for r in r2], np.r_[-np.ones(4), -INF * np.ones(len(PAIRS))],
                      np.r_[np.ones(4), INF * np.ones(len(PAIRS))])
    lb2 += CONST
    xv = ca.SX.sym("x", 4)
    prod = (u0 + ca.dot(ca.DM(ua), xv)) * (v0 + ca.dot(ca.DM(va), xv))
    true_lo = np.inf
    for x0 in itertools.chain(V16, rng.uniform(-1, 1, (20, 4))):
        try:
            f, _, _ = solve_casadi(xv, prod, None, [], [], -np.ones(4), np.ones(4), x0)
            true_lo = min(true_lo, f)
        except RuntimeError:
            pass
    sample_lo = (u_of(X) * v_of(X)).min()
    print(f"\n  min over [-1,1]^4 of the product: true {true_lo:.4f} (Uno multistart; "
          f"best sample {sample_lo:.4f}),  single-term LP {lb1:.4f},  term-wise LP {lb2:.4f}")
    check("both LP bounds are below the true minimum", lb1 <= true_lo + 1e-6 and lb2 <= true_lo + 1e-6)
    check("Uno multistart minimum is no worse than the best sample", true_lo <= sample_lo + 1e-9)

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8))
    ax = axes[0]
    ug, vg = np.meshgrid(np.linspace(UL, UU, 400), np.linspace(VL, VU, 400))
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from scipy.spatial import ConvexHull
    ax.contourf(ug, vg, (ug * vg <= 2).astype(float), levels=[0.5, 1.5], colors=[C[0]], alpha=0.18)
    ax.contour(ug, vg, ug * vg, levels=[2], colors=[C[0]], linewidths=2)
    us = np.linspace(UL, UU, 200)
    ax.plot(us, (2 - VL * us + UL * VL) / UL, color=C[1], lw=2)    # -8u - 4v - 32 = 2
    ax.plot(us, (2 - VU * us + UU * VU) / UU, color=C[1], lw=2)    # 12u + 6v - 72 = 2
    hull = np.array([[u_of(p), v_of(p)] for p in V16])
    h = ConvexHull(hull)
    ax.fill(*hull[h.vertices].T, fill=False, ec=INK2, lw=1, ls="--")
    ax.set_xlim(UL, UU)
    ax.set_ylim(VL, VU)
    ax.set_xlabel("$u = 1 - x_1 + 2x_2 - 2x_4$")
    ax.set_ylabel("$v = 4x_2 - 5x_3 + x_4 + 2$")
    ax.set_title("(a) $uv \\leq 2$ and its McCormick cuts", loc="left")
    ax.legend(handles=[Patch(fc=C[0], alpha=0.18, label="$uv \\leq 2$"),
                       Line2D([], [], color=C[0], label="$uv = 2$"),
                       Line2D([], [], color=C[1], label="McCormick cuts $= 2$"),
                       Line2D([], [], color=INK2, lw=1, ls="--", label="image of $[-1,1]^4$")],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)

    ax = axes[1]
    names = ["feasible", "single-term\nMcCormick", "term-wise\nMcCormick"]
    vals = [feas.mean(), acc1.mean(), acc2.mean()]
    ax.bar(range(3), vals, width=0.55, color=[C[0], C[1], C[2]], edgecolor="white", linewidth=2)
    for k, v in enumerate(vals):
        ax.annotate(f"{v:.3f}", (k, v), xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=8, color=INK2)
    ax.set_xticks(range(3), names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("share of $[-1,1]^4$ (10$^6$ samples)")
    ax.set_title("(b) size of the projected sets", loc="left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "q2_mccormick.pdf")

    nfail = sum(not ok for _, ok in CHECKS)
    print(f"\n{len(CHECKS) - nfail}/{len(CHECKS)} checks passed")
    return nfail


if __name__ == "__main__":
    raise SystemExit(main())
