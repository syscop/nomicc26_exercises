"""Nonlinear MPC for the cart-pole swing-up with CasADi.

State x = (p, theta, v, omega): cart position, pole angle (theta = 0 hangs down,
theta = pi is upright), and their rates. Control u = F, the horizontal force on the cart.

At every sampling instant we solve the optimal control problem (OCP)

    min   sum_{k=0}^{N-1} |x_k - x_ref|_Q^2 + R u_k^2  +  |x_N - x_ref|_P^2   (5a)
    s.t.  x_0 = xhat                         (measured state, a parameter)   (5b)
          x_{k+1} = Phi(x_k, u_k)            (RK4 step of the ODE)           (5c)
          -F_max <= u_k <= F_max                                             (5d)
          -p_max <= p_k <= p_max             (k = 1..N; p_0 is data)         (5e)

apply u_0 to the "plant" (a CVODES integration of the true ODE), measure the new state
(optionally with Gaussian noise), shift, and repeat.

Equation numbers in the comments, such as (5c), refer to nmpc_cartpole.tex / nmpc_cartpole.md.
"Alg. 1" is the NMPC loop (Algorithm 1 in the PDF, the pseudocode in Section 4 of the .md).

Usage:
    python nmpc_cartpole.py [--N 40] [--dt 0.05] [--T-sim 5]
                            [--solver ipopt|uno|sqpmethod] [--uno-preset filtersqp|ipopt]
                            [--F-max 20] [--no-terminal-cost] [--no-warmstart]
                            [--mismatch] [--noise SIGMA] [--seed 0] [--plot] [--animate]
"""
import argparse
import time

import casadi as ca
import numpy as np

NX, NU = 4, 1


# ----------------------------------------------------------------------------- model
def cartpole_ode(M=1.0, m=0.2, l=0.5, g=9.81):
    """Return the CasADi Function f(x, u) -> xdot of the cart-pole (point mass at the tip), eq. (1)."""
    x = ca.SX.sym("x", NX)                 # x = (p, theta, v, omega)
    u = ca.SX.sym("u", NU)                 # u = F
    p, th, v, om = x[0], x[1], x[2], x[3]
    F = u[0]
    s, c = ca.sin(th), ca.cos(th)
    den = M + m * s**2
    pdd = (F + m * s * (l * om**2 + g * c)) / den                          # (1), row 3
    thdd = (-F * c - m * l * om**2 * c * s - (M + m) * g * s) / (l * den)  # (1), row 4
    xdot = ca.vertcat(v, om, pdd, thdd)    # (1), rows 1-2 are v and omega
    return ca.Function("f", [x, u], [xdot], ["x", "u"], ["xdot"])   # f(x, u) in (1)


def rk4_step(f, dt, n_sub=2):
    """Return Phi(x, u): n_sub explicit RK4 steps (4) of length dt/n_sub (control held constant)."""
    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", NU)
    h = dt / n_sub
    xk = x
    for _ in range(n_sub):
        k1 = f(xk, u)                      # (4a)
        k2 = f(xk + h / 2 * k1, u)         # (4a)
        k3 = f(xk + h / 2 * k2, u)         # (4a)
        k4 = f(xk + h * k3, u)             # (4a)
        xk = xk + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)   # (4b)
    return ca.Function("Phi", [x, u], [xk], ["x", "u"], ["xnext"])   # Phi in (5c)


def plant(f, dt):
    """'True' system for the closed loop: an adaptive CVODES integration over one sample.

    This is the plant map Phi^pl in Alg. 1, line 6 (see "Plant vs. model" in Section 4).
    """
    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", NU)
    return ca.integrator("plant", "cvodes", {"x": x, "u": u, "ode": f(x, u)}, 0.0, dt)


# ----------------------------------------------------------------------------- OCP
def solver_options(name, uno_preset):
    """(plugin options, solver options) for Opti.solver(name, ...)."""
    if name == "ipopt":
        return {"print_time": False}, {"print_level": 0, "sb": "yes", "max_iter": 500}
    if name == "uno":
        # options in the second dict are forwarded to Uno; 'preset' picks the method:
        #   ipopt     -> primal-dual interior point + filter line search
        #   filtersqp -> trust-region SQP (BQPD QP solver) + filter
        return {"print_time": False}, {"preset": uno_preset, "print_solution": False, "logger": "SILENT"}
    if name == "sqpmethod":
        # CasADi's own SQP: line search, exact Hessian, QPs solved by qrqp. qrqp needs a convex
        # QP, so the (indefinite) Lagrangian Hessian is convexified by clipping its eigenvalues.
        return {"qpsol": "qrqp", "qpsol_options": {"print_iter": False, "print_header": False,
                                                    "error_on_fail": False},
                "print_header": False, "print_iteration": False, "print_status": False, "print_time": False,
                "max_iter": 200, "convexify_strategy": "eigen-clip"}, {}
    raise ValueError(name)


def build_ocp(args):
    """Build the multiple-shooting NLP (5) with Opti; return (opti, handles)."""
    f = cartpole_ode()
    Phi = rk4_step(f, args.dt, n_sub=2)
    N = args.N

    opti = ca.Opti()
    X = opti.variable(NX, N + 1)       # state trajectory  x_0 ... x_N
    U = opti.variable(NU, N)           # control trajectory u_0 ... u_{N-1}
    x0 = opti.parameter(NX)            # measured initial state xhat

    x_ref = ca.DM([0.0, np.pi, 0.0, 0.0])   # upright equilibrium
    Q = ca.diag([10.0, 10.0, 0.1, 0.1])     # weight in (3a)
    R = 0.01                                # weight in (3a)
    P = 10 * Q                              # terminal weight in (3b)

    J = 0
    for k in range(N):
        e = X[:, k] - x_ref
        J += ca.bilin(Q, e, e) + R * U[:, k] ** 2               # stage cost (3a)
        opti.subject_to(X[:, k + 1] == Phi(X[:, k], U[:, k]))   # (5c), continuity ("gap") constraints
    if not args.no_terminal_cost:
        eN = X[:, N] - x_ref
        J += ca.bilin(P, eN, eN)                                # terminal cost (3b)
    opti.minimize(J)                                            # (5a)

    opti.subject_to(X[:, 0] == x0)                              # (5b)
    opti.subject_to(opti.bounded(-args.F_max, U, args.F_max))   # (5d)
    # no bound on p_0: it is fixed by the measurement, and a noisy measurement just outside
    # the track limits would otherwise make the NLP infeasible
    opti.subject_to(opti.bounded(-args.p_max, X[0, 1:], args.p_max))   # (5e), k = 1..N

    p_opts, s_opts = solver_options(args.solver, args.uno_preset)
    opti.solver(args.solver, p_opts, s_opts)
    return opti, dict(X=X, U=U, x0=x0, f=f)


# ----------------------------------------------------------------------------- closed loop
def run_closed_loop(args):
    opti, h = build_ocp(args)
    X, U, x0 = h["X"], h["U"], h["x0"]

    f_plant = cartpole_ode(m=0.2 * 1.3) if args.mismatch else h["f"]
    sim = plant(f_plant, args.dt)
    rng = np.random.default_rng(args.seed)

    n_steps = int(round(args.T_sim / args.dt))
    xs = np.zeros((NX, n_steps + 1))   # true plant state
    ys = np.zeros((NX, n_steps))       # measured state handed to the controller
    us = np.zeros(n_steps)
    t_solve, n_iter, ok = np.zeros(n_steps), np.zeros(n_steps, int), np.zeros(n_steps, bool)
    xs[:, 0] = [0.0, 0.0, 0.0, 0.0]    # Alg. 1, line 1: hanging down at rest

    X_guess = np.tile(xs[:, [0]], (1, args.N + 1))   # Alg. 1, line 2: guess w^(0), x_k = x_0
    U_guess = np.zeros((NU, args.N))                 #                              u_k = 0
    X_pred, U_pred = [], []            # open-loop predictions of every MPC step

    for k in range(n_steps):           # Alg. 1, line 3 (the loop index j of the notes is k here)
        ys[:, k] = xs[:, k] + args.noise * rng.standard_normal(NX)   # Alg. 1, line 4: measure
        opti.set_value(x0, ys[:, k])
        opti.set_initial(X, X_guess)
        opti.set_initial(U, U_guess)
        tic = time.perf_counter()
        try:
            sol = opti.solve()         # Alg. 1, line 5: solve the NLP (5)
            ok[k] = True
            val = sol.value
        except RuntimeError:           # failed solve: use the last iterate anyway
            val = opti.debug.value
        t_solve[k] = time.perf_counter() - tic
        stats = opti.stats()
        n_iter[k] = stats.get("iter_count", -1)
        U_opt = np.atleast_2d(val(U))
        X_opt = np.atleast_2d(val(X))
        X_pred.append(X_opt)
        U_pred.append(U_opt)

        us[k] = U_opt[0, 0]            # u_0^*, the only control sent to the plant
        xs[:, k + 1] = np.asarray(sim(x0=xs[:, k], u=us[k])["xf"]).ravel()   # Alg. 1, line 6

        if not args.no_warmstart:      # Alg. 1, line 7: shift the previous solution by one sample
            X_guess = np.hstack([X_opt[:, 1:], X_opt[:, -1:]])
            U_guess = np.hstack([U_opt[:, 1:], U_opt[:, -1:]])

    return dict(t=np.arange(n_steps + 1) * args.dt, x=xs, y=ys, u=us,
                t_solve=t_solve, n_iter=n_iter, ok=ok, X_pred=X_pred, U_pred=U_pred)


# ----------------------------------------------------------------------------- output
def report(res, args):
    x, u = res["x"], res["u"]
    tail = x[1, -int(round(1.0 / args.dt)):]   # last second of the simulation
    err = np.abs(tail - np.pi).max()
    label = args.solver + (f"/{args.uno_preset}" if args.solver == "uno" else "")
    print(f"solver            : {label}")
    print(f"successful solves : {res['ok'].sum()} / {len(res['ok'])}")
    print(f"iterations        : total {res['n_iter'].sum()}, first {res['n_iter'][0]}, "
          f"mean after first {res['n_iter'][1:].mean():.1f}")
    print(f"solve time        : total {res['t_solve'].sum():.2f} s, "
          f"max {1e3 * res['t_solve'].max():.1f} ms, mean {1e3 * res['t_solve'].mean():.1f} ms")
    print(f"max |F|           : {np.abs(u).max():.2f} N   (bound {args.F_max})")
    print(f"max |p|           : {np.abs(x[0]).max():.3f} m   (bound {args.p_max})")
    verdict = "UPRIGHT" if err < 0.05 else "balanced, with jitter" if err < 0.5 else "NOT upright"
    print(f"|theta - pi|, last 1 s: max {err:.2e} rad  ->  {verdict}")


def plot(res, args, fname="cartpole_closed_loop.png"):
    import matplotlib.pyplot as plt

    t, x, u = res["t"], res["x"], res["u"]
    fig, ax = plt.subplots(5, 1, figsize=(7, 9), sharex=True)
    labels = ["p [m]", r"$\theta$ [rad]", "v [m/s]", r"$\omega$ [rad/s]"]
    for i in range(NX):
        if args.noise > 0:
            ax[i].plot(t[:-1], res["y"][i], ".", c="0.6", ms=3, label="measured")
        ax[i].plot(t, x[i], "b", label="true")
        ax[i].set_ylabel(labels[i])
        ax[i].grid(True)
    ax[0].axhline(args.p_max, ls="--", c="r")
    ax[0].axhline(-args.p_max, ls="--", c="r")
    ax[1].axhline(np.pi, ls=":", c="k")
    if args.noise > 0:
        ax[0].legend(loc="lower right")
    ax[4].step(t[:-1], u, "g", where="post")
    ax[4].axhline(args.F_max, ls="--", c="r")
    ax[4].axhline(-args.F_max, ls="--", c="r")
    ax[4].set_ylabel("F [N]")
    ax[4].set_xlabel("t [s]")
    ax[4].grid(True)
    fig.suptitle(f"Cart-pole NMPC, N={args.N}, dt={args.dt}, solver={args.solver}"
                 + (f", noise={args.noise}" if args.noise > 0 else ""))
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    print(f"wrote {fname}")
    if args.show_plots:
        plt.show()


def animate(res, args, fname="cartpole.gif", l=0.5):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    x = res["x"]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.set_xlim(-args.p_max - 0.6, args.p_max + 0.6)
    ax.set_ylim(-0.8, 0.8)
    ax.set_aspect("equal")
    ax.axvline(-args.p_max, c="r", ls="--")
    ax.axvline(args.p_max, c="r", ls="--")
    ax.axhline(0, c="k", lw=0.5)
    cart, = ax.plot([], [], "s", ms=18, c="tab:blue")
    pole, = ax.plot([], [], "-o", lw=3, c="tab:orange")
    txt = ax.text(0.02, 0.9, "", transform=ax.transAxes)

    def update(k):
        p, th = x[0, k], x[1, k]
        tip = (p + l * np.sin(th), -l * np.cos(th))
        cart.set_data([p], [0])
        pole.set_data([p, tip[0]], [0, tip[1]])
        txt.set_text(f"t = {k * args.dt:.2f} s")
        return cart, pole, txt

    anim = FuncAnimation(fig, update, frames=x.shape[1], blit=True)
    anim.save(fname, writer=PillowWriter(fps=int(round(1 / args.dt))))
    print(f"wrote {fname}")


def make_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=40, help="prediction horizon (samples)")
    ap.add_argument("--dt", type=float, default=0.05, help="sampling time [s]")
    ap.add_argument("--T-sim", type=float, default=5.0, help="closed-loop simulation time [s]")
    ap.add_argument("--F-max", type=float, default=20.0, help="force bound [N]")
    ap.add_argument("--p-max", type=float, default=1.5, help="track half-length [m]")
    ap.add_argument("--solver", choices=["ipopt", "uno", "sqpmethod"], default="ipopt")
    ap.add_argument("--uno-preset", choices=["filtersqp", "ipopt"], default="filtersqp")
    ap.add_argument("--no-terminal-cost", action="store_true")
    ap.add_argument("--no-warmstart", action="store_true", help="cold start every MPC solve")
    ap.add_argument("--mismatch", action="store_true", help="plant pole mass +30%% vs. the model")
    ap.add_argument("--noise", type=float, default=0.0, metavar="SIGMA",
                    help="std. dev. of Gaussian noise added to each measured state component "
                         "(m, rad, m/s, rad/s)")
    ap.add_argument("--seed", type=int, default=0, help="random seed for --noise")
    ap.add_argument("--plot", action="store_true", help="save cartpole_closed_loop.png")
    ap.add_argument("--show-plots", action="store_true", help="also open the plot window")
    ap.add_argument("--animate", action="store_true", help="save cartpole.gif")
    return ap


def main():
    args = make_parser().parse_args()
    res = run_closed_loop(args)
    report(res, args)
    if args.plot or args.show_plots:
        plot(res, args)
    if args.animate:
        animate(res, args)


if __name__ == "__main__":
    main()
