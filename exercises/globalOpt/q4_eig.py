"""TutorialGO Question 4: min / max x^T Q x  s.t.  x^T x = 1.

KKT: 2 Q x - 2 lambda x = 0, x^T x = 1, so every KKT point is a unit eigenvector, and the
objective there is x^T Q x = lambda x^T x = lambda. Hence min = lambda_min, max = lambda_max.

We solve both problems with Uno from many random starts and compare with numpy.

Run:  conda run -n optimization python q4_eig.py
"""
import numpy as np
import sympy as sp
import unopy

from plotstyle import C, INK2, plt, save
from uno_lp import solve_qcqp_sphere

Q = np.array([[4, -1, 0, 0], [-1, 4, -2, 0], [0, -2, 3, 2], [0, 0, 2, 1]], float)
CHECKS = []


def check(name, ok):
    CHECKS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")


def main():
    print("Q4: eigenvalue QCQP")
    lam = sp.Symbol("lambda")
    charpoly = sp.expand((sp.Matrix(Q.astype(int)) - lam * sp.eye(4)).det())
    print(f"  det(Q - lambda I) = {charpoly}")
    ev, V = np.linalg.eigh(Q)
    print(f"  eigenvalues (numpy): {np.round(ev, 6)}")
    roots = sorted(float(r) for r in sp.Poly(charpoly, lam).nroots())
    check("roots of the characteristic polynomial match numpy", np.allclose(roots, ev))
    check("Q is symmetric, so all eigenvalues are real", np.allclose(Q, Q.T))

    rng = np.random.default_rng(0)
    results = {}
    for sense, name in ((unopy.MINIMIZE, "min"), (unopy.MAXIMIZE, "max")):
        vals = []
        for _ in range(40):
            x0 = rng.normal(size=4)
            x0 /= np.linalg.norm(x0)
            try:
                f, x, _ = solve_qcqp_sphere(Q, x0, sense=sense)
            except RuntimeError:
                continue
            lam_x = x @ Q @ x / (x @ x)   # Rayleigh quotient: removes the tiny ||x|| = 1 violation
            res = np.linalg.norm(Q @ x - lam_x * x)
            vals.append((lam_x, res, np.linalg.norm(x)))
        vals = np.array(vals)
        results[name] = vals
        best = vals[:, 0].min() if name == "min" else vals[:, 0].max()
        target = ev[0] if name == "min" else ev[-1]
        print(f"  {name}: {len(vals)}/40 Uno runs converged, best {best:.6f}, "
              f"numpy {target:.6f}, max KKT residual |Qx - (x^TQx) x| = {vals[:, 1].max():.1e}, "
              f"max | ||x|| - 1 | = {np.abs(vals[:, 2] - 1).max():.1e}")
        check(f"{name}: best Uno value equals the {'smallest' if name == 'min' else 'largest'} eigenvalue",
              abs(best - target) < 1e-6)
        check(f"{name}: every Uno solution is a unit eigenvector",
              vals[:, 1].max() < 1e-5 and np.allclose(vals[:, 2], 1, atol=1e-5))
        found = sorted({round(float(v), 4) for v in vals[:, 0]})
        print(f"    distinct values found: {found}")
        check(f"{name}: every value found is an eigenvalue",
              all(np.min(np.abs(ev - v)) < 1e-3 for v in found))

    print(f"  x_min = {np.round(V[:, 0], 4)},  x_max = {np.round(V[:, -1], 4)}")

    # figure: where the multistart runs land, against the spectrum
    fig, ax = plt.subplots(figsize=(6.5, 1.9))
    for k, e in enumerate(ev):
        ax.axvline(e, color=INK2, lw=1, ls=":")
        last = k == len(ev) - 1
        ax.annotate(f"$\\lambda_{k+1} = {e:.4f}$", (e, 1.45), xytext=(-4 if last else 4, 0),
                    textcoords="offset points", ha="right" if last else "left", fontsize=8, color=INK2)
    for y, (name, col) in enumerate((("min", C[0]), ("max", C[1]))):
        v = results[name][:, 0]
        jitter = np.random.default_rng(1).uniform(-0.12, 0.12, v.size)
        ax.plot(v, y + jitter, "o", ms=6, color=col, mec="white", mew=1, alpha=0.8)
    ax.set_yticks([0, 1], ["min $x^TQx$", "max $x^TQx$"])
    ax.set_ylim(-0.5, 1.65)
    ax.set_xlabel("objective value at the Uno solution (40 random starts each)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "q4_eig.pdf")

    nfail = sum(not ok for _, ok in CHECKS)
    print(f"\n{len(CHECKS) - nfail}/{len(CHECKS)} checks passed")
    return nfail


if __name__ == "__main__":
    raise SystemExit(main())
