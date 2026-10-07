"""A ten-minute tour of CasADi. Run it section by section and read the printout.

    python 01_casadi_basics.py

1. Symbols: SX (scalar expression graphs) vs MX (matrix expression graphs)
2. Functions: turning an expression into a callable, compiled-in-memory object
3. Algorithmic differentiation: gradient, Jacobian, Hessian
4. Solving an NLP with nlpsol (low-level interface), with Ipopt and with Uno
5. The same NLP with Opti (high-level interface)
6. Integrating an ODE
"""
import casadi as ca
import numpy as np


def banner(s):
    print("\n" + "=" * 70 + f"\n{s}\n" + "=" * 70)


# ----------------------------------------------------------------------------- 1. symbols
banner("1. Symbols")
x = ca.SX.sym("x", 2)                  # a 2-vector of scalar symbols
e = ca.sin(x[0]) * x[1] ** 2
print("SX expression        :", e)
X = ca.MX.sym("X", 2)                  # one 2-vector symbol; operations are matrix-valued
print("MX expression        :", ca.sin(X[0]) * X[1] ** 2)
# Rule of thumb: SX is faster for small, scalar-heavy expressions (model equations);
# MX is needed for calls to other Functions, integrators, solvers, large matrices.

# ----------------------------------------------------------------------------- 2. functions
banner("2. Functions")
f = ca.Function("f", [x], [e, ca.jacobian(e, x)], ["x"], ["e", "de_dx"])
print(f)
print("f([1, 2])            :", f([1, 2]))          # numerical evaluation -> DM
print("f(X) is symbolic     :", f(X)[0])            # Functions can be called on MX symbols too
print("named call           :", f(x=[1, 2])["de_dx"])

# ----------------------------------------------------------------------------- 3. AD
banner("3. Algorithmic differentiation (Rosenbrock)")
rosen = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
g = ca.gradient(rosen, x)
H, _ = ca.hessian(rosen, x)            # hessian returns (H, gradient)
dfun = ca.Function("d", [x], [rosen, g, H])
val, grad, hess = dfun([-1.2, 1.0])
print("f      =", val)
print("grad f =", grad.T)
print("hess f =\n", hess)
# Check the gradient against central finite differences
x0, h = np.array([-1.2, 1.0]), 1e-6
fd = [(dfun(x0 + h * ei)[0] - dfun(x0 - h * ei)[0]) / (2 * h) for ei in np.eye(2)]
print("finite differences  :", np.array(fd, dtype=float).ravel())

# ----------------------------------------------------------------------------- 4. nlpsol
banner("4. nlpsol: min Rosenbrock  s.t.  x0^2 + x1^2 <= 1.5")
nlp = {"x": x, "f": rosen, "g": x[0] ** 2 + x[1] ** 2}
for name, opts in [
    ("ipopt", {"ipopt.print_level": 0, "print_time": False, "ipopt.sb": "yes"}),
    ("uno", {"uno.preset": "ipopt", "uno.logger": "SILENT", "print_time": False}),
    ("uno", {"uno.preset": "filtersqp", "uno.logger": "SILENT", "print_time": False}),
]:
    S = ca.nlpsol("S", name, nlp, opts)
    sol = S(x0=[-1.2, 1.0], lbg=-ca.inf, ubg=1.5)
    st = S.stats()
    label = name + (f"/{opts['uno.preset']}" if name == "uno" else "")
    print(f"{label:15s} x* = {np.array(sol['x']).ravel()}, f* = {float(sol['f']):.6f}, "
          f"lam_g = {float(sol['lam_g']):.4f}, success = {st['success']}, iters = {st.get('iter_count')}")

# ----------------------------------------------------------------------------- 5. Opti
banner("5. Opti: the same problem, written like the math")
opti = ca.Opti()
y = opti.variable(2)
opti.minimize((1 - y[0]) ** 2 + 100 * (y[1] - y[0] ** 2) ** 2)
circle = y[0] ** 2 + y[1] ** 2 <= 1.5
opti.subject_to(circle)
opti.set_initial(y, [-1.2, 1.0])
opti.solver("ipopt", {"print_time": False}, {"print_level": 0, "sb": "yes"})
sol = opti.solve()
print("Opti x* =", sol.value(y), " multiplier =", sol.value(opti.dual(circle)))
# Opti is a thin layer over nlpsol: it collects variables/constraints into {x, f, g, p}.

# ----------------------------------------------------------------------------- 6. integrator
banner("6. Integrating the Van der Pol oscillator with CVODES")
z = ca.SX.sym("z", 2)
ode = ca.vertcat((1 - z[1] ** 2) * z[0] - z[1], z[0])
tgrid = np.linspace(0, 10, 6)
I = ca.integrator("I", "cvodes", {"x": z, "ode": ode}, 0.0, tgrid[1:])
traj = I(x0=[0, 1])["xf"]
for t, zz in zip(tgrid[1:], np.array(traj).T):
    print(f"t = {t:4.1f}   z = {zz}")
# Integrators are differentiable too: ca.jacobian(I(x0=z)['xf'], z) gives sensitivities.
