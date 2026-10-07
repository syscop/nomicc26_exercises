# Nonlinear MPC with CasADi: the cart-pole swing-up

This is the Markdown version of `nmpc_cartpole.tex`. The companion script is `nmpc_cartpole.py`, and each equation names the function that implements it.

## 1. The model

A cart of mass $M$ moves on a horizontal track. A pole of length $l$ is hinged to the cart, with a point mass $m$ at its tip.

$$
x = (p,\ \theta,\ v,\ \omega)^T \in \mathbb{R}^4, \qquad u = F \in \mathbb{R}
$$

- $p$ is the cart position.
- $\theta$ is the pole angle from the **downward** vertical: $\theta=0$ is hanging, $\theta=\pi$ is upright.
- $v=\dot p$ and $\omega=\dot\theta$.
- $F$ is the horizontal force on the cart.

The equations of motion (`cartpole_ode`) are

$$
\dot x = f(x,u) =
\begin{pmatrix}
  v \\ \omega \\
  \dfrac{F + m \sin\theta\,(l\omega^2 + g\cos\theta)}{M + m\sin^2\theta} \\
  \dfrac{-F\cos\theta - m l \omega^2 \cos\theta\sin\theta - (M+m) g \sin\theta}{l\,(M + m\sin^2\theta)}
\end{pmatrix}
\tag{1}
$$

with $M=1$ kg, $m=0.2$ kg, $l=0.5$ m and $g=9.81$ m/s². The upright equilibrium $x_\text{ref}=(0,\pi,0,0)$ is unstable. The system is also underactuated, with one input for two degrees of freedom, so the controller has to "pump" energy into the pole before it can catch it.

## 2. The optimal control problem

At each sample, NMPC takes the measured state $\hat x$ and solves

$$
\begin{aligned}
  \min_{x(\cdot),u(\cdot)} \quad & \int_0^{T} \ell(x,u)\,d\tau + V_f(x(T))\\
  \text{s.t.}\quad & \dot x = f(x,u),\quad x(0)=\hat x,\quad |u|\le F_{\max},\quad |p|\le p_{\max}.
\end{aligned}
\tag{2}
$$

It applies the first input and repeats. The stage and terminal costs are

$$
\begin{aligned}
  \ell(x,u) &= \|x-x_\text{ref}\|_Q^2 + R\,u^2, &&\qquad\text{(3a)}\\
  V_f(x)    &= \|x-x_\text{ref}\|_P^2,           &&\qquad\text{(3b)}
\end{aligned}
$$

where $\|e\|_Q^2 = e^TQe$, with the following parameters:
- $Q=\mathrm{diag}(10,10,0.1,0.1)$
- $R=0.01$
- $P=10Q$
- $F_{\max}=20$ N
- $p_{\max}=1.5$ m

## 3. Discretization: direct multiple shooting

We use a sampling time of $\Delta t = 0.05$ s and $N=40$ samples, so the horizon is $T = 2$ s. The controls are piecewise constant.

$\Phi(x,u)$ is the state after one sample. It is computed with 2 RK4 steps of length $h=\Delta t/2$ (`rk4_step`). One RK4 step maps $x$ to $x^+$ by

$$
\begin{aligned}
  k_1 &= f(x,u),\quad k_2 = f(x+\tfrac h2 k_1,u),\quad k_3 = f(x+\tfrac h2 k_2,u),\quad k_4=f(x+hk_3,u), &&\qquad\text{(4a)}\\
  x^+ &= x + \tfrac h6 (k_1+2k_2+2k_3+k_4), &&\qquad\text{(4b)}
\end{aligned}
$$

and $\Phi(x,u)$ applies this step twice.

Multiple shooting keeps the states at the sampling instants as variables, and turns the dynamics into **continuity constraints**. The NLP is (`build_ocp`)

$$
\begin{aligned}
  \min_{\substack{x_0,\dots,x_N\\ u_0,\dots,u_{N-1}}}\quad & J = \sum_{k=0}^{N-1} \Big(\|x_k-x_\text{ref}\|_Q^2 + R u_k^2\Big) + \|x_N-x_\text{ref}\|_P^2 && &&\text{(5a)}\\
  \text{s.t.}\quad & x_0 = \hat x, && &&\text{(5b)}\\
  & x_{k+1} = \Phi(x_k,u_k), && k=0,\dots,N-1, &&\text{(5c)}\\
  & -F_{\max}\le u_k \le F_{\max}, && k=0,\dots,N-1, &&\text{(5d)}\\
  & -p_{\max}\le p_k \le p_{\max}, && k=1,\dots,N. &&\text{(5e)}
\end{aligned}
$$

The objective (5a) is the sum of the stage costs (3a) and the terminal cost (3b). The constraints (5c) are the continuity constraints.

The track bound is not imposed at $k=0$. $p_0=\hat p$ is data, not a decision, and a noisy measurement just outside the track would otherwise make the NLP infeasible.

A solver sees all unknowns stacked into one vector $w = (x_0,u_0,x_1,u_1,\dots,x_{N-1},u_{N-1},x_N)\in\mathbb{R}^{5N+4}$. The problem above is then an NLP $\min_w F(w)$ subject to $G(w)=0$ and $w_L\le w\le w_U$. For $N=40$ this has 204 variables and 164 equality constraints.

The figure illustrates the idea on a coarse grid ($N=10$, $\Delta t=0.2$ s), writing $s_k$ for the shooting nodes $x_k$. Each node starts its own short integration, and the solver is free to place the nodes anywhere. The initial guess may therefore have gaps, and the continuity constraints close them only at the solution.

![Direct multiple shooting: initial guess with gaps (left) and converged solution (right)](figures/multiple_shooting.png)

*Direct multiple shooting for the cart-pole (only θ shown).*
- **Left:** the initial guess, with nodes $s_k$ on a straight line from hanging to upright and $u_k = 0$. Integrating the ODE from each node over one interval (blue arcs) does not reach the next node. The dashed gaps $\Phi(s_k,u_k)-s_{k+1}$ are the violated continuity constraints.
- **Right:** the NLP solution. All gaps are closed, so the arcs join into one trajectory of the ODE.

The force is piecewise constant (bottom panels).

**Why multiple shooting rather than single shooting?** Single shooting eliminates the states by forward simulation. The problem gets smaller, but for an unstable system it becomes highly nonlinear and badly conditioned. Multiple shooting has three advantages:
1. It spreads the nonlinearity over many small constraints.
2. It lets you initialize the state trajectory directly, for example with the previous solution.
3. It keeps the KKT system sparse and block-banded.

## 4. The receding-horizon loop (`run_closed_loop`)

NMPC turns the open-loop optimal control problem into a feedback law by re-solving it at every sample:
1. At time $t_j$, measure the state $\hat x_j$.
2. Solve the NLP over the horizon $[t_j, t_j+T]$.
3. Apply only the first control $u^\star_0$ for one sample, and discard the rest of the plan.
4. At $t_{j+1}=t_j+\Delta t$, measure again and solve over the horizon shifted by one sample.

The horizon "recedes" in front of the system, which gives the method its name.

![The NMPC principle: past, now, and the receding prediction horizon](figures/nmpc_principle.png)

*The NMPC principle, at sample $j=10$ ($t_j=0.5$ s) of the cart-pole swing-up. To the left of "now" is the past: the measured state and the applied force. The shaded region is the prediction horizon, with the predicted trajectory and control sequence from the NLP solved at $t_j$. Only $u^\star_0$ (filled bar) is sent to the plant, and the whole problem is solved again on the next horizon starting at $t_{j+1}$.*

```
x^pl_0 = (0, 0, 0, 0)                    # true plant state: hanging at rest
w      = guess: x_k = x^pl_0, u_k = 0
for j = 0, 1, 2, ...
    x̂_j  = x^pl_j + η_j,  η_j ~ N(0, σ² I)        # measurement (σ = 0 by default)
    w*   = solve NLP with x_0 = x̂_j, starting from w
    apply u_j = u*_0:  x^pl_{j+1} = CVODES integration of the true plant
    w = shift(w*) = (x*_1..x*_N, x*_N ; u*_1..u*_{N-1}, u*_{N-1})
```

**Warm start.** Consecutive NLPs differ only in $\hat x$, so the shifted solution is a very good initial guess.

**Plant vs. model.** In control, the *plant* is the real physical system. The *model* is the controller's mathematical approximation of it, which the NLP uses to predict the future. In the script both are simulated, but they are deliberately kept apart:

| | **model** (inside the NLP) | **plant** (the "real" cart-pole) |
|---|---|---|
| role | predicts $x_1,\dots,x_N$ through $x_{k+1}=\Phi(x_k,u_k)$ | produces the next true state after $u_0$ is applied |
| integration | fixed-step RK4, 2 substeps (`rk4_step`) | adaptive CVODES, tight tolerances (`plant`) |
| pole mass | $m = 0.2$ kg | 0.2 kg; 0.26 kg with `--mismatch` |
| measurement | — | $\hat x = x^\text{pl} + \eta$, with `--noise σ` |

Three kinds of discrepancy can be switched on:
1. **Discretization error** (always present). RK4 with $h = 0.025$ s is accurate, so prediction and reality agree closely, and even the open-loop plan would nearly work.
2. **Parametric model error** (`--mismatch`). The controller plans with the wrong pole mass, so the measured state after one sample differs from the predicted $x^\star_1$. Executing the first plan open loop would miss the upright position. Re-solving from the measured state at every sample (*feedback*) corrects the error.
3. **Measurement noise** (`--noise σ`). The controller sees $\hat x_j = x^\text{pl}_j + \eta_j$, with independent Gaussian noise of standard deviation σ in every component (m, rad, m/s, rad/s), and treats it as exact. Each NLP is therefore solved for a slightly wrong initial state. The pole stays balanced but jitters, and the solver needs more iterations because the shifted guess is less accurate.

The setup is still idealized: the full state is measured and there are no external disturbances. A real implementation estimates $\hat x$ from noisy partial measurements with a state estimator, such as an extended Kalman filter or moving-horizon estimation. Moving-horizon estimation itself solves an optimization problem much like the NLP above, backwards in time.

## 5. The CasADi code

The listing below is complete and runs as is. It builds the model (1), the RK4 map (4), and the NLP (5), and solves the first MPC problem. The comment at the end of each line names the equation it implements. `nmpc_cartpole.py` contains the same code, split into functions.

```python
import casadi as ca
import numpy as np

# parameters
M, m, l, g = 1.0, 0.2, 0.5, 9.81           # cart, pole mass, length, gravity
N, dt = 40, 0.05                           # horizon N, sampling time Delta t
F_max, p_max = 20.0, 1.5                   # bounds in (5d), (5e)
x_ref = ca.DM([0, np.pi, 0, 0])            # upright equilibrium
Q, R = ca.diag([10, 10, 0.1, 0.1]), 0.01   # weights in (3a)
P = 10 * Q                                 # weight in (3b)

# the model xdot = f(x, u)
x = ca.SX.sym("x", 4)                      # x = (p, theta, v, omega)
u = ca.SX.sym("u")                         # u = F
th, v, om = x[1], x[2], x[3]
s, c = ca.sin(th), ca.cos(th)
den = M + m * s**2
xdot = ca.vertcat(v, om,                   # (1), rows 1-2
    (u + m * s * (l * om**2 + g * c)) / den,               # (1), row 3
    (-u * c - m * l * om**2 * c * s - (M + m) * g * s) / (l * den))  # (1), row 4
f = ca.Function("f", [x, u], [xdot])       # f(x, u) in (1)

# Phi(x, u): two RK4 steps of length h = dt/2
h = dt / 2
xk = x
for _ in range(2):
    k1 = f(xk, u)                          # (4a)
    k2 = f(xk + h / 2 * k1, u)             # (4a)
    k3 = f(xk + h / 2 * k2, u)             # (4a)
    k4 = f(xk + h * k3, u)                 # (4a)
    xk = xk + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)          # (4b)
Phi = ca.Function("Phi", [x, u], [xk])     # Phi in (5c)

# the NLP
opti = ca.Opti()
X = opti.variable(4, N + 1)                # x_0, ..., x_N
U = opti.variable(1, N)                    # u_0, ..., u_{N-1}
x0 = opti.parameter(4)                     # measured state xhat
J = 0
for k in range(N):
    e = X[:, k] - x_ref
    J += ca.bilin(Q, e, e) + R * U[:, k]**2                # stage cost (3a)
    opti.subject_to(X[:, k + 1] == Phi(X[:, k], U[:, k]))  # (5c)
eN = X[:, N] - x_ref
J += ca.bilin(P, eN, eN)                   # terminal cost (3b)
opti.minimize(J)                           # (5a)
opti.subject_to(X[:, 0] == x0)             # (5b)
opti.subject_to(opti.bounded(-F_max, U, F_max))            # (5d)
opti.subject_to(opti.bounded(-p_max, X[0, 1:], p_max))     # (5e), k = 1..N
opti.solver("ipopt")                       # or "uno", "sqpmethod", ...

# one MPC step (the loop in Section 4), from the hanging rest state
xhat = np.zeros(4)                         # measured state
opti.set_value(x0, xhat)
opti.set_initial(X, np.tile(xhat, (N + 1, 1)).T)           # guess w^(0): x_k = xhat
opti.set_initial(U, 0)                     # guess w^(0): u_k = 0
sol = opti.solve()
u_apply = sol.value(U)[0]                  # u_0^*, the only control sent to the plant
```

`Opti` collects the variables into $w$ and the constraints (5b)–(5e) into $g(w)$ with bounds, giving the low-level form `{x: w, f: J, g: g(w), p: x̂}` for `nlpsol`. CasADi builds the exact gradient, Jacobian and Lagrangian Hessian by algorithmic differentiation (AD).

## 6. Solvers

All of these are bundled with the `casadi` pip wheel.

| solver | method | behaviour in MPC |
|---|---|---|
| `ipopt` | primal-dual interior point method with filter line search | robust from cold starts; benefits less from warm starts |
| `uno`, preset `ipopt` | Uno's Ipopt-like interior point method | similar to Ipopt |
| `uno`, preset `filtersqp` | trust-region SQP with filter, QPs solved by BQPD (active set) | re-uses the active set, so warm starts work very well |
| `sqpmethod` | CasADi's line-search SQP with `qrqp` | needs `convexify_strategy` because qrqp requires convex QPs |

Closed-loop results with $N=40$, 100 steps and shift warm start, from one laptop run (your times will differ):

| solver | 1st solve (cold) | mean iters, steps 2–100 | total iterations |
|---|---:|---:|---:|
| Ipopt | 60 | 5.3 | 581 |
| Uno / `ipopt` | 64 | 5.4 | 596 |
| Uno / `filtersqp` | 21 | 1.4 | 155 |
| sqpmethod + qrqp | 59 | 1.5 | 204 |

## 7. Results

![closed loop](cartpole_closed_loop.png)

Run `python nmpc_cartpole.py --plot` to generate the figure. The controller works in three phases:
1. It pushes the cart left at full force.
2. It reverses to swing the pole up.
3. It catches the pole in about 1 s.

The cart reaches $|p|\approx 1.33 < 1.5$.

## 8. Nonconvexity and local solutions

The NLP is nonconvex, so solvers return **local** solutions, and the starting point decides which one. Run `--no-warmstart` to see this. Each NLP then starts from "stay where you are, zero force". The first solve still finds the swing-up. Later cold-started solves converge to a different local minimizer that leaves the pole hanging, so the loop never swings up. Warm starting is therefore part of the controller design, not just a speed-up.

## 9. Outlook

Replacing $|F|\le F_\max$ by an on/off actuator $F\in\{-F_\max,0,F_\max\}$ gives an MINLP, which `bonmin` or CAMINO can solve. Contact and friction give MPCCs. Both are themes of NOMICC.
