"""TutorialGO Question 3: two RLT relaxations of a nonconvex quartic.

    min f(x) = -x1^2 x2^2 + 2 x1 x2^3 - x2^4   s.t.  -1 <= x_i <= 1.

RLT-A  lift every monomial x1^i x2^j of degree 2..4 to X_{1..12..2} and add all bound-factor
       products (x1 - l1)^a (u1 - x1)^b (x2 - l2)^c (u2 - x2)^d >= 0 of degree 2..4,
       linearised. Objective  -X_1122 + 2 X_1222 - X_2222.
RLT-B  use f = -(x1 x2 - x2^2)^2 = -w^2 with w = X_12 - X_22, the secant (convex envelope)
       of -w^2 on [wL, wU], and McCormick/secant inequalities for X_12, X_22.
       "interval": wL, wU from interval arithmetic,  "exact": the true range of w on the box.

The sheet writes X_211 for x1^2 x2; here it is X_112 (indices sorted).

Run:  conda run -n optimization python q3_quartic.py
"""
import itertools

import casadi as ca
import numpy as np
import sympy as sp

from plotstyle import C, INK2, SEQ, plt, save
from uno_lp import INF, solve_casadi, solve_lp

CHECKS = []


def check(name, ok):
    CHECKS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")


x1, x2 = sp.symbols("x1 x2")
F = -x1**2 * x2**2 + 2 * x1 * x2**3 - x2**4
MONOS = [(i, d - i) for d in range(2, 5) for i in range(d, -1, -1)]   # (power of x1, power of x2)
NAME = {m: "X" + "1" * m[0] + "2" * m[1] for m in MONOS}
COL = {m: 2 + k for k, m in enumerate(MONOS)}                         # z = (x1, x2, X...)
NA = 2 + len(MONOS)


def f_num(a, b):
    return -a**2 * b**2 + 2 * a * b**3 - b**4


def linearise(poly):
    """Map a polynomial in x1, x2 (degree <= 4) to (coefficient vector over z, constant)."""
    P = sp.Poly(sp.expand(poly), x1, x2)
    a, c0 = np.zeros(NA), 0.0
    for (i, j), c in zip(P.monoms(), P.coeffs()):
        if i + j == 0:
            c0 += float(c)
        elif i + j == 1:
            a[0 if i else 1] += float(c)
        else:
            a[COL[(i, j)]] += float(c)
    return a, c0


# ---------------------------------------------------------------- RLT-A
def rlt_a_rows(lo, up, maxdeg=4, even_bounds=True):
    """Bound-factor products of degree 2..maxdeg; even_bounds adds X >= 0 for even powers
    (X_11, X_22, X_1111, X_1122, X_2222), which the products alone do not imply."""
    factors = [x1 - lo[0], up[0] - x1, x2 - lo[1], up[1] - x2]
    rows = []
    if even_bounds:
        for m in MONOS:
            if m[0] % 2 == 0 and m[1] % 2 == 0:
                a = np.zeros(NA)
                a[COL[m]] = 1
                rows.append((a, 0.0, INF))
    for d in range(2, maxdeg + 1):
        for combo in itertools.combinations_with_replacement(range(4), d):
            a, c0 = linearise(sp.Mul(*[factors[k] for k in combo]))
            rows.append((a, -c0, INF))                                   # a^T z + c0 >= 0
    return rows


OBJ_A, _ = linearise(F)


def bound_a(lo, up, fix=None, rows=None, c=None):
    rows = rows or rlt_a_rows(lo, up)
    lbx = np.r_[lo, -INF * np.ones(len(MONOS))]
    ubx = np.r_[up, INF * np.ones(len(MONOS))]
    if fix is not None:
        lbx[:2] = ubx[:2] = fix
    obj = OBJ_A if c is None else OBJ_A + np.r_[c, np.zeros(len(MONOS))]
    f, z = solve_lp(obj, np.array([r[0] for r in rows]), [r[1] for r in rows],
                    [r[2] for r in rows], lbx, ubx)
    return f, z


# ---------------------------------------------------------------- RLT-B
def w_range_exact(lo, up):
    """Exact range of w = x2 (x1 - x2) on a box: vertices and the edge maxima x2 = x1 / 2."""
    pts = [(a, b) for a in (lo[0], up[0]) for b in (lo[1], up[1])]
    pts += [(a, np.clip(a / 2, lo[1], up[1])) for a in (lo[0], up[0])]
    vals = [b * (a - b) for a, b in pts]
    return min(vals), max(vals)


def w_range_interval(lo, up):
    p = [lo[0] * lo[1], lo[0] * up[1], up[0] * lo[1], up[0] * up[1]]
    s = [lo[1] ** 2, up[1] ** 2]
    sq_lo = 0.0 if lo[1] <= 0 <= up[1] else min(s)
    return min(p) - max(s), max(p) - sq_lo


def bound_b(lo, up, wr, fix=None, c=None):
    """z = (x1, x2, X12, X22, w, t); minimise t."""
    wl, wu = wr
    l1, u1, l2, u2 = lo[0], up[0], lo[1], up[1]
    rows = []

    def row(coef, lb, ub):
        a = np.zeros(6)
        for k, v in coef.items():
            a[k] += v
        rows.append((a, lb, ub))

    row({2: 1, 0: -l2, 1: -l1}, -l1 * l2, INF)       # McCormick X12
    row({2: 1, 0: -u2, 1: -u1}, -u1 * u2, INF)
    row({2: 1, 0: -u2, 1: -l1}, -INF, -l1 * u2)
    row({2: 1, 0: -l2, 1: -u1}, -INF, -u1 * l2)
    row({3: 1, 1: -2 * l2}, -l2**2, INF)              # X22 >= tangents at l2, u2
    row({3: 1, 1: -2 * u2}, -u2**2, INF)
    row({3: 1, 1: -(l2 + u2)}, -INF, -l2 * u2)        # X22 <= secant
    row({3: 1}, 0.0, INF)                             # X22 >= 0
    row({4: 1, 2: -1, 3: 1}, 0.0, 0.0)                # w = X12 - X22
    row({5: 1, 4: wl + wu}, wl * wu, INF)             # t >= -(wl + wu) w + wl wu  (secant of -w^2)
    lbx = np.r_[lo, -INF, -INF, wl, -INF]
    ubx = np.r_[up, INF, INF, wu, INF]
    if fix is not None:
        lbx[:2] = ubx[:2] = fix
    obj = np.r_[0, 0, 0, 0, 0, 1.0] + (0 if c is None else np.r_[c, 0, 0, 0, 0])
    f, z = solve_lp(obj, np.array([r[0] for r in rows]),
                    [r[1] for r in rows], [r[2] for r in rows], lbx, ubx)
    return f, z


# ---------------------------------------------------------------- true minimum
_x = ca.SX.sym("x", 2)
_f = -_x[0]**2 * _x[1]**2 + 2 * _x[0] * _x[1]**3 - _x[1]**4


def true_min(lo, up, n=41, c=(0.0, 0.0)):
    """min f + c^T x on a box: dense grid, then a Uno polish from the best grid points."""
    a, b = np.meshgrid(np.linspace(lo[0], up[0], n), np.linspace(lo[1], up[1], n))
    v = f_num(a, b) + c[0] * a + c[1] * b
    best = v.min()
    for k in np.argsort(v.ravel())[:3]:
        try:
            val, _, _ = solve_casadi(_x, _f + c[0] * _x[0] + c[1] * _x[1], None, [], [], lo, up,
                                     [a.ravel()[k], b.ravel()[k]])
            best = min(best, val)
        except RuntimeError:
            pass
    return best


# ---------------------------------------------------------------- main
def main():
    print("Q3: quartic")
    check("f = -(x1 x2 - x2^2)^2 (sympy)", sp.expand(F + (x1 * x2 - x2**2) ** 2) == 0)
    sheet = (1 - 2 * x1 - 2 * x2 + 4 * sp.Symbol("X12") + sp.Symbol("X11") + sp.Symbol("X22")
             - 2 * sp.Symbol("X122") - 2 * sp.Symbol("X112") + sp.Symbol("X1122"))
    lin = sp.expand((1 - x1)**2 * (1 - x2)**2)
    lin = lin.subs({x1**2 * x2**2: sp.Symbol("X1122"), x1 * x2**2: sp.Symbol("X122"),
                    x1**2 * x2: sp.Symbol("X112"), x1 * x2: sp.Symbol("X12"),
                    x1**2: sp.Symbol("X11"), x2**2: sp.Symbol("X22")})
    check("sheet's linearisation of (1-x1)^2 (1-x2)^2 >= 0 is correct", sp.expand(lin - sheet) == 0)

    lo, up = -np.ones(2), np.ones(2)
    rowsA = rlt_a_rows(lo, up)
    rows_pure = rlt_a_rows(lo, up, even_bounds=False)
    print(f"  RLT-A: {len(MONOS)} lifted monomials, {len(rows_pure)} bound-factor inequalities "
          f"(degree 2: 10, 3: 20, 4: 35) + {len(rowsA) - len(rows_pure)} even-power bounds X >= 0")
    # implied bounds of the intermediate powers: min / max of each over the pure RLT polytope
    print("  bounds implied by the degree<=4 bound-factor products alone (no X >= 0 rows):")
    for m in [(2, 0), (1, 1), (0, 2), (1, 2), (2, 1)]:
        e = np.zeros(NA)
        e[COL[m]] = 1
        A = np.array([r[0] for r in rows_pure])
        lb_, _ = solve_lp(e, A, [r[1] for r in rows_pure], [r[2] for r in rows_pure],
                          np.r_[lo, -INF * np.ones(len(MONOS))], np.r_[up, INF * np.ones(len(MONOS))])
        ub_, _ = solve_lp(-e, A, [r[1] for r in rows_pure], [r[2] for r in rows_pure],
                          np.r_[lo, -INF * np.ones(len(MONOS))], np.r_[up, INF * np.ones(len(MONOS))])
        print(f"    {NAME[m]:6s} in [{lb_:+.3f}, {-ub_:+.3f}]")

    # full box
    tm = true_min(lo, up)
    fa, za = bound_a(lo, up, rows=rowsA)
    wi, we = w_range_interval(lo, up), w_range_exact(lo, up)
    fbi, _ = bound_b(lo, up, wi)
    fbe, _ = bound_b(lo, up, we)
    print(f"\n  full box: true min {tm:.4f}")
    print(f"    w range: interval [{wi[0]:g}, {wi[1]:g}], exact [{we[0]:g}, {we[1]:g}]")
    print(f"    RLT-A {fa:.4f}   RLT-B (interval w) {fbi:.4f}   RLT-B (exact w) {fbe:.4f}")
    for name, v in (("RLT-A", fa), ("RLT-B interval", fbi), ("RLT-B exact", fbe)):
        check(f"full box: {name} bound is valid", v <= tm + 1e-6)

    # sub-boxes: a 4x4 partition of [-1,1]^2
    edges = np.linspace(-1, 1, 5)
    G = {k: np.zeros((4, 4)) for k in ("true", "A", "Bi", "Be")}
    for i, j in itertools.product(range(4), range(4)):
        l, u = np.array([edges[i], edges[j]]), np.array([edges[i + 1], edges[j + 1]])
        G["true"][j, i] = true_min(l, u, n=21)
        G["A"][j, i], _ = bound_a(l, u)
        G["Bi"][j, i], _ = bound_b(l, u, w_range_interval(l, u))
        G["Be"][j, i], _ = bound_b(l, u, w_range_exact(l, u))
    gaps = {k: G["true"] - G[k] for k in ("A", "Bi", "Be")}
    for k, lab in (("A", "RLT-A"), ("Bi", "RLT-B interval"), ("Be", "RLT-B exact")):
        check(f"sub-boxes: {lab} bounds are valid", np.all(gaps[k] >= -1e-6))
        print(f"    {lab:15s} mean gap {gaps[k].mean():.4f}  max gap {gaps[k].max():.4f}")
    print(f"    RLT-A tighter than RLT-B exact on {(gaps['A'] < gaps['Be'] - 1e-6).sum()}/16 boxes, "
          f"looser on {(gaps['A'] > gaps['Be'] + 1e-6).sum()}/16")

    # pointwise underestimators on the full box: phi(x) = min of the relaxation with x fixed
    n = 21
    g1 = np.linspace(-1, 1, n)
    PA, PB = np.zeros((n, n)), np.zeros((n, n))
    for i, j in itertools.product(range(n), range(n)):
        xf = np.array([g1[i], g1[j]])
        PA[j, i], _ = bound_a(lo, up, fix=xf, rows=rowsA)
        PB[j, i], _ = bound_b(lo, up, we, fix=xf)
    a, b = np.meshgrid(g1, g1)
    FF = f_num(a, b)
    check("pointwise: both underestimators are below f on the grid",
          np.all(PA <= FF + 1e-6) and np.all(PB <= FF + 1e-6))
    print(f"\n  pointwise gap f - phi on the full box: RLT-A mean {np.mean(FF - PA):.4f} "
          f"max {np.max(FF - PA):.4f};  RLT-B exact mean {np.mean(FF - PB):.4f} max {np.max(FF - PB):.4f}")
    print(f"    RLT-A tighter at {(PA > PB + 1e-6).sum()}/{n * n} grid points, "
          f"RLT-B tighter at {(PB > PA + 1e-6).sum()}/{n * n}")

    # tilted objectives min f + c^T x on the full box: the minimiser moves off the vertices
    rng = np.random.default_rng(3)
    tilt = []
    for c in rng.uniform(-2, 2, (12, 2)):
        t = true_min(lo, up, c=c)
        fa_, _ = bound_a(lo, up, rows=rowsA, c=c)
        fb_, _ = bound_b(lo, up, we, c=c)
        tilt.append((c, t, fa_, fb_))
    ga = np.array([t - fa_ for _, t, fa_, _ in tilt])
    gb = np.array([t - fb_ for _, t, _, fb_ in tilt])
    check("tilted objectives: both bounds are valid", np.all(ga >= -1e-6) and np.all(gb >= -1e-6))
    print(f"  tilted objectives f + c^T x, 12 random c in [-2,2]^2: mean gap RLT-A {ga.mean():.4f}, "
          f"RLT-B exact {gb.mean():.4f}; A tighter {(ga < gb - 1e-6).sum()}, "
          f"B tighter {(gb < ga - 1e-6).sum()}")
    print("    (every tilted minimiser is a box vertex, where RLT relaxations are exact)")

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(2, 3, figsize=(7.8, 5.0))
    vmax = max(np.max(FF - PA), np.max(FF - PB))
    panels = [(FF, "(a) $f(x)$", "viridis_r", None, None),
              (FF - PA, "(b) gap $f - \\varphi_A$", SEQ, 0, vmax),
              (PA - PB, "(c) $\\varphi_A - \\varphi_B$", SEQ, 0, None)]
    for ax, (Z, title, cmap, vmn, vm) in zip(axes[0], panels):
        im = ax.imshow(Z, origin="lower", extent=(-1, 1, -1, 1), cmap=cmap, vmin=vmn, vmax=vm)
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8)
    gmax = max(g.max() for g in gaps.values())
    for ax, k, title in zip(axes[1], ("A", "Bi", "Be"),
                            ("(d) box gap, RLT-A", "(e) RLT-B, interval $w$", "(f) RLT-B, exact $w$")):
        im = ax.imshow(gaps[k], origin="lower", extent=(-1, 1, -1, 1), cmap=SEQ, vmin=0, vmax=gmax)
        for i, j in itertools.product(range(4), range(4)):
            v = abs(gaps[k][j, i]) if abs(gaps[k][j, i]) < 5e-3 else gaps[k][j, i]
            ax.text(-0.75 + 0.5 * i, -0.75 + 0.5 * j, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v > 0.6 * gmax else INK2)
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8)
    for ax in axes.ravel():
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
    fig.tight_layout()
    save(fig, "q3_quartic.pdf")

    nfail = sum(not ok for _, ok in CHECKS)
    print(f"\n{len(CHECKS) - nfail}/{len(CHECKS)} checks passed")
    return nfail


if __name__ == "__main__":
    raise SystemExit(main())
