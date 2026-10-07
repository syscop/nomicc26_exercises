"""Generate the two illustrations used in nmpc_cartpole.tex / nmpc_cartpole.md.

    python make_figures.py        # writes figures/multiple_shooting.{pdf,png}
                                  #        figures/nmpc_principle.{pdf,png}

Both figures are computed from the real cart-pole model and solver, not drawn by hand.
"""
import os

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

from nmpc_cartpole import NX, build_ocp, cartpole_ode, make_parser, run_closed_loop

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3
INK, INK2, GRID, SHADE = "#0b0b0b", "#52514e", "#e4e3df", "#f1f0ec"
OUT = "figures"

plt.rcParams.update({
    "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.dpi": 220, "lines.linewidth": 1.6,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    print(f"wrote {OUT}/{name}.pdf/.png")


def arc(f, x0, u, dt, n=25):
    """Fine-grid trajectory of the ODE on one shooting interval from x0 with constant u."""
    xs, us = ca.SX.sym("x", NX), ca.SX.sym("u")
    tg = np.linspace(0, dt, n)
    F = ca.integrator("F", "cvodes", {"x": xs, "u": us, "ode": f(xs, us)}, 0.0, tg[1:])
    xa = np.hstack([np.asarray(x0, float).reshape(NX, 1), np.array(F(x0=x0, u=u)["xf"])])
    return tg, xa


# ----------------------------------------------------------------------------- figure 1
def fig_multiple_shooting():
    """Multiple shooting on a coarse grid: initial guess with gaps vs. converged solution."""
    args = make_parser().parse_args(["--N", "10", "--dt", "0.2"])
    opti, h = build_ocp(args)
    X, U, x0 = h["X"], h["U"], h["x0"]
    f, N, dt = h["f"], args.N, args.dt
    t_nodes = np.arange(N + 1) * dt

    # initial guess: theta interpolated linearly from hanging to upright, zero force
    Xg = np.zeros((NX, N + 1))
    Xg[1] = np.linspace(0, np.pi, N + 1)
    Xg[3] = np.pi / (N * dt)
    Ug = np.zeros((1, N))

    opti.set_value(x0, np.zeros(NX))
    opti.set_initial(X, Xg)
    opti.set_initial(U, Ug)
    sol = opti.solve()
    Xs, Us = np.atleast_2d(sol.value(X)), np.atleast_2d(sol.value(U))

    fig, axes = plt.subplots(2, 2, figsize=(6.4, 3.6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.3]})
    for col, (Xn, Un, title) in enumerate([
            (Xg, Ug, "Initial guess: nodes $s_k$ set freely, gaps $\\neq 0$"),
            (Xs, Us, "Solution: continuity $s_{k+1} = \\Phi(s_k, u_k)$ holds")]):
        ax, axu = axes[0, col], axes[1, col]
        for k in range(N):
            tg, xa = arc(f, Xn[:, k], float(Un[0, k]), dt)
            ax.plot(t_nodes[k] + tg, xa[1], c=BLUE, lw=1.8,
                    label="integrated arc $x(t; s_k, u_k)$" if k == 0 else None)
            gap_end = xa[1, -1]
            if abs(gap_end - Xn[1, k + 1]) > 0.02:
                ax.plot([t_nodes[k + 1]] * 2, [gap_end, Xn[1, k + 1]], c=ORANGE, lw=1.6,
                        ls=(0, (2, 1.5)), label="gap $\\Phi(s_k,u_k)-s_{k+1}$" if k == 0 else None)
                ax.plot(t_nodes[k + 1], gap_end, "o", ms=4, mfc="white", mec=BLUE, mew=1.2)
        ax.plot(t_nodes, Xn[1], "o", ms=7, c=INK, mec="white", mew=1.5, zorder=5,
                label="shooting node $s_k$ (decision variable)")
        ax.axhline(np.pi, c=INK2, ls=":", lw=1)
        ax.text(t_nodes[0], np.pi + 0.08, "upright $\\theta=\\pi$", ha="left", va="bottom",
                color=INK2, fontsize=7.5)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        axu.step(t_nodes, np.r_[Un[0], Un[0, -1]], where="post", c=BLUE, lw=1.6)
        axu.set_xlabel("t [s]")
        axu.set_ylim(-23, 23)
        axu.set_yticks([-20, 0, 20])
        for tk in t_nodes:
            axu.axvline(tk, c=GRID, lw=0.8, zorder=0)
    axes[0, 0].set_ylabel(r"pole angle $\theta$ [rad]")
    axes[1, 0].set_ylabel("$u_k = F$ [N]")
    lo = min(axes[0, 0].get_ylim()[0], axes[0, 1].get_ylim()[0])
    hi = max(axes[0, 0].get_ylim()[1], axes[0, 1].get_ylim()[1])
    for ax in axes[0]:
        ax.set_ylim(lo, hi)
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=7.5)
    save(fig, "multiple_shooting")


# ----------------------------------------------------------------------------- figure 2
def fig_nmpc_principle(j=10):
    """Receding horizon at sample j of the closed loop: past, now, predicted future."""
    args = make_parser().parse_args([])
    res = run_closed_loop(args)
    dt, N = args.dt, args.N
    t, x, u = res["t"], res["x"], res["u"]
    tj, T = j * dt, N * dt
    tp = tj + np.arange(N + 1) * dt
    Xp, Up = res["X_pred"][j], res["U_pred"][j]

    fig, (ax, axu) = plt.subplots(2, 1, figsize=(6.0, 3.9), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1.3]})
    for a in (ax, axu):
        a.axvspan(tj, tj + T, color=SHADE, zorder=0)
        a.axvline(tj, c=INK, lw=1.2)
        a.axvline(tj + dt, c=INK2, lw=0.8, ls=(0, (1, 2)))

    # state
    ax.plot(t[: j + 1], x[1, : j + 1], c=BLUE, label="past: measured state")
    ax.plot(tp, Xp[1], c=ORANGE, ls="--", label="prediction at $t_j$ (solution of the NLP)")
    ax.plot(tj, x[1, j], "o", ms=8, c=INK, mec="white", mew=1.5, zorder=5)
    ax.annotate("$\\hat x_j$", (tj, x[1, j]), xytext=(-18, 8), textcoords="offset points", color=INK)
    ax.axhline(np.pi, c=INK2, ls=":", lw=1)
    ax.text(t[0] + 0.02, np.pi + 0.1, "$x_{ref}$: $\\theta=\\pi$", color=INK2, fontsize=7.5)
    ax.set_ylabel(r"pole angle $\theta$ [rad]")
    ax.legend(loc="lower right", fontsize=7.5)

    # horizon annotations
    ytop = ax.get_ylim()[1]
    ax.set_ylim(top=ytop + 1.4)
    yb = ytop + 0.45
    ax.annotate("", xy=(tj, yb), xytext=(tj + T, yb),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1))
    ax.text(tj + T / 2, yb + 0.08, "prediction horizon  $T = N\\Delta t$", ha="center",
            va="bottom", color=INK, fontsize=8)
    ax.annotate("", xy=(0, yb), xytext=(tj, yb), arrowprops=dict(arrowstyle="<-", color=INK2, lw=1))
    ax.text(tj / 2, yb + 0.08, "past", ha="center", va="bottom", color=INK2, fontsize=8)
    yn = ytop + 1.0                    # next horizon, one sample later
    ax.annotate("", xy=(tj + dt, yn), xytext=(tj + dt + T, yn),
                arrowprops=dict(arrowstyle="<->", color=AQUA, lw=1))
    ax.text(tj + dt + T / 2, yn + 0.06, "next horizon at $t_{j+1} = t_j + \\Delta t$ (receding)",
            ha="center", va="bottom", color=INK2, fontsize=7.5)
    ax.text(tj, ax.get_ylim()[0], " now $t_j$", ha="left", va="bottom", color=INK, fontsize=8)

    # control
    axu.step(t[: j + 1], np.r_[u[:j], u[j - 1]], where="post", c=BLUE, lw=1.6, label="past: applied force")
    axu.step(tp, np.r_[Up[0], Up[0, -1]], where="post", c=ORANGE, ls="--", lw=1.6,
             label="predicted $u_0^\\star,\\dots,u_{N-1}^\\star$")
    axu.fill_between([tj, tj + dt], 0, Up[0, 0], color=ORANGE, alpha=0.35, lw=0, step="post")
    axu.plot([tj, tj + dt], [Up[0, 0]] * 2, c=ORANGE, lw=3, solid_capstyle="butt")
    axu.annotate("only $u_0^\\star$ is applied,\nthen measure, shift, re-solve",
                 xy=(tj + dt, Up[0, 0] / 2), xytext=(tj + 0.9, -18), color=INK, fontsize=7.5,
                 arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
    axu.axhline(args.F_max, c=INK2, ls=":", lw=1)
    axu.axhline(-args.F_max, c=INK2, ls=":", lw=1)
    axu.set_ylim(-24, 24)
    axu.set_yticks([-20, 0, 20])
    axu.set_ylabel("F [N]")
    axu.set_xlabel("t [s]")
    axu.set_xlim(0, tj + T + dt + 0.05)
    axu.legend(loc="center right", bbox_to_anchor=(1, 0.66), fontsize=7.5)
    fig.align_ylabels([ax, axu])
    fig.tight_layout()
    save(fig, "nmpc_principle")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_multiple_shooting()
    fig_nmpc_principle()
