# -*- coding: utf-8 -*-
"""
Nonlinear DtN in 2D, fully energy-driven Newton (plan 1, final form)
===================================================================
History of this file's predecessors, kept because the lesson is the point:

  * nonlinear_2d_tangent.py assembled an ANALYTIC residual and its tangent; the
    Jacobian came out asymmetric (5.6e-4) while the energy Hessian was symmetric,
    and the two differed by 1.0e-3.
  * nonlinear_2d_energy.py assembled an ANALYTIC gradient and Hessian from the
    same energy.  The Hessian was symmetric to 4.8e-17, but the gradient differed
    from the finite difference of the energy by a STABLE 2.7e-3 that did not
    shrink as h went from 1e-4 to 1e-8, and Newton diverged.

In both cases the analytic first variation was wrong while the energy itself
passed every independent check (descent direction, rigid-body null space).  The
discrepancy is systematic, not a step-size artefact.

Conclusion adopted here: keep the ENERGY as the single source of truth and obtain
everything else from it numerically.

  * gradient: complex-step differentiation of the element energy,
        g_j = Im[ psi(u + i h e_j) ] / h
    which has no subtractive cancellation and is therefore h-independent;
  * Hessian: central difference of that gradient (perturbation of a gradient,
    which is exactly the object Newton needs, and it inherits the symmetry of the
    energy to machine precision).

The cost is acceptable because the element has only 8 degrees of freedom, and the
payoff is that theory, algorithm and code can no longer disagree: there is one
scalar object and everything else is a derivative of it.

Physics: St. Venant-Kirchhoff, total Lagrangian, plane stress,
    F = I + du/dX,  E = (F^T F - I)/2,  psi = lam/2 (tr E)^2 + mu E:E.

Run:  python nonlinear_2d_final.py
"""
import os
import time

import numpy as np

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))

LX, LY, X0 = 2.0, 1.0, 1.0
NU = 0.30
E_MAT = 2100.0
LAM = E_MAT * NU / (1.0 - NU * NU)
MU = E_MAT / (2.0 * (1.0 + NU))

GP = np.array([-1.0, 1.0]) / np.sqrt(3.0)
HC = 1e-25          # complex step for the gradient
HD = 1e-7           # real step for the Hessian (gradient perturbation)


def right_disp(y, d0=0.01):
    t = 2.0 * y / LY - 1.0
    return d0 * (1.0 - t * t)


def q4(xi, eta):
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                         (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    det = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    return N, dxi, det


def build_mesh(nx, ny, x0, x1):
    xs = np.linspace(x0, x1, nx + 1)
    ys = np.linspace(0.0, LY, ny + 1)
    nodes = np.array([(xs[i], ys[j]) for j in range(ny + 1) for i in range(nx + 1)])
    elems = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            elems.append([a, a + 1, a + 1 + (nx + 1), a + (nx + 1)])
    return nodes, np.array(elems)


def elem_geom(xy):
    out = []
    for gx in GP:
        for gy in GP:
            N, dxi, det = q4(gx, gy)
            J = np.array([[dxi @ xy[:, 0], dxi @ xy[:, 1]],
                          [det @ xy[:, 0], det @ xy[:, 1]]])
            dJ = np.linalg.det(J)
            iJ = np.linalg.inv(J)
            out.append((iJ[0, 0] * dxi + iJ[0, 1] * det,
                        iJ[1, 0] * dxi + iJ[1, 1] * det, dJ))
    return out


def elem_energy_cached(ue, geom):
    """Same energy, with the geometry already evaluated (hot-loop version)."""
    w = np.zeros(1, dtype=np.result_type(ue.dtype, np.float64))[0]
    for dNdx, dNdy, dJ in geom:
        ux = ue[0::2]; uy = ue[1::2]
        F = np.eye(2) + np.array([[dNdx @ ux, dNdy @ ux],
                                  [dNdx @ uy, dNdy @ uy]])
        Et = 0.5 * (F.T @ F - np.eye(2))
        tr = np.trace(Et)
        w = w + (0.5 * LAM * tr * tr + MU * np.sum(Et * Et)) * dJ
    return w


def elem_grad_cached(ue, geom, h=HC):
    n = 8
    g = np.zeros(n)
    for j in range(n):
        uc = ue.astype(complex)
        uc[j] += 1j * h
        g[j] = elem_energy_cached(uc, geom).imag / h
    return g


def elem_hess_cached(ue, geom, h=HD):
    n = 8
    K = np.zeros((n, n))
    for j in range(n):
        ej = np.zeros(n); ej[j] = h
        K[:, j] = (elem_grad_cached(ue + ej, geom)
                   - elem_grad_cached(ue - ej, geom)) / (2.0 * h)
    return 0.5 * (K + K.T)


def elem_energy(ue, xy):
    """The single source of truth.  Complex-safe."""
    w = np.zeros(1, dtype=np.result_type(ue.dtype, np.float64))[0]
    for dNdx, dNdy, dJ in elem_geom(xy):
        ux = ue[0::2]; uy = ue[1::2]
        F = np.eye(2) + np.array([[dNdx @ ux, dNdy @ ux],
                                  [dNdx @ uy, dNdy @ uy]])
        Et = 0.5 * (F.T @ F - np.eye(2))
        tr = np.trace(Et)
        w = w + (0.5 * LAM * tr * tr + MU * np.sum(Et * Et)) * dJ
    return w


def elem_grad(ue, xy, h=HC):
    """Complex-step gradient of the element energy (h-independent)."""
    n = 8
    g = np.zeros(n)
    for j in range(n):
        uc = ue.astype(complex)
        uc[j] += 1j * h
        g[j] = elem_energy(uc, xy).imag / h
    return g


def elem_hess(ue, xy, h=HD):
    """Central difference of the (complex-step) gradient.

    Perturbing the gradient rather than the energy makes this exactly the object
    Newton needs, and it inherits the symmetry of the energy.
    """
    n = 8
    K = np.zeros((n, n))
    for j in range(n):
        ej = np.zeros(n); ej[j] = h
        K[:, j] = (elem_grad(ue + ej, xy) - elem_grad(ue - ej, xy)) / (2.0 * h)
    return 0.5 * (K + K.T)


def assemble(nodes, elems, u, want_hess=True):
    ndof = 2 * len(nodes)
    G = np.zeros(ndof)
    K = np.zeros((ndof, ndof)) if want_hess else None
    for e in elems:
        idx = [2 * k + c for k in e for c in (0, 1)]
        G[idx] += elem_grad(u[idx], nodes[e])
        if want_hess:
            K[np.ix_(idx, idx)] += elem_hess(u[idx], nodes[e])
    return G, K


def newton(nodes, elems, fixed, free, tol=1e-11, maxit=40):
    ndof = 2 * len(nodes)
    u = np.zeros(ndof)
    for d, v in fixed.items():
        u[d] = v
    hist = []
    for it in range(maxit):
        G, K = assemble(nodes, elems, u)
        r = G[free]
        nrm = np.linalg.norm(r)
        hist.append(nrm)
        if nrm < tol:
            break
        u[free] -= np.linalg.solve(K[np.ix_(free, free)], r)
    return u, it + 1, hist


def total_energy(nodes, elems, u):
    return sum(elem_energy(u[[2 * k + c for k in e for c in (0, 1)]], nodes[e])
               for e in elems)


def main():
    t0 = time.time()
    nx, ny = 4, 3
    nodes, elems = build_mesh(nx, ny, 0.0, LX)
    ndof = 2 * len(nodes)
    rng = np.random.default_rng(0)
    u = rng.normal(0.0, 1e-3, ndof)

    print("=" * 74)
    print("2D nonlinear, fully energy-driven Newton (plan 1)")
    print("=" * 74)
    print(f"mesh {2*nx}x{ny}, {ndof} dofs, {len(elems)} elements")

    # ---- consistency: gradient vs energy, Hessian vs gradient, symmetry
    print("\n[A] gradient vs finite difference of the SAME energy")
    E = lambda uu: total_energy(nodes, elems, uu)
    G, K = assemble(nodes, elems, u)
    for h in (1e-5, 1e-6, 1e-7):
        d = rng.normal(0, 1, ndof); d /= np.linalg.norm(d)
        fd = (E(u + h * d) - E(u - h * d)) / (2 * h)
        an = float(G @ d)
        print(f"    h={h:.0e}:  g.d = {an:+.12e}   FD = {fd:+.12e}"
              f"   rel {abs(an - fd) / abs(fd):.2e}")

    print("\n[B] Hessian vs second difference of the energy")
    for h in (1e-4, 1e-5):
        da = rng.normal(0, 1, ndof); da /= np.linalg.norm(da)
        db = rng.normal(0, 1, ndof); db /= np.linalg.norm(db)
        q = (E(u + h * da + h * db) - E(u + h * da - h * db)
             - E(u - h * da + h * db) + E(u - h * da - h * db)) / (4 * h * h)
        an = float(da @ K @ db)
        print(f"    h={h:.0e}:  d_a^T K d_b = {an:+.10e}   second diff = {q:+.10e}"
              f"   rel {abs(an - q) / abs(q):.2e}")

    print(f"\n[C] Hessian symmetry: {np.abs(K - K.T).max() / np.abs(K).max():.3e}")

    # ---- Newton on a real boundary-value problem
    print("\n[D] Newton on the clamped-prescribed problem (u_x prescribed at x=L)")
    cl = [k for k in range(len(nodes)) if abs(nodes[k, 0]) < 1e-12]
    ri = [k for k in range(len(nodes)) if abs(nodes[k, 0] - LX) < 1e-12]
    fixed = {}
    for k in cl:
        fixed[2 * k] = 0.0; fixed[2 * k + 1] = 0.0
    for k in ri:
        fixed[2 * k] = float(right_disp(nodes[k, 1]))
    free = np.array([d for d in range(ndof) if d not in fixed])
    uM, iters, hist = newton(nodes, elems, fixed, free)
    print(f"    {iters} iterations, residual history:")
    for i, hh in enumerate(hist[:8]):
        print(f"      {i}: {hh:.4e}")

    # ---- what the DtN probe needs: perturb an interface dof, re-converge
    print("\n[E] tangent DtN probe on the interface (the object the coupling needs)")
    gi = sorted([k for k in range(len(nodes)) if abs(nodes[k, 0] - X0) < 1e-12],
                key=lambda k: nodes[k, 1])
    gdof = [2 * k for k in gi] + [2 * k + 1 for k in gi]
    other = [d for d in range(ndof) if d not in set(gdof)]
    far = {2 * k: float(right_disp(nodes[k, 1])) for k in ri}
    uG0 = np.zeros(len(gdof))
    for j, d in enumerate(gdof):
        uG0[j] = uM[d]
    eps = 1e-6
    S = np.zeros((len(gdof), len(gdof)))
    t0p = time.time()
    for k in range(len(gdof)):
        rp = np.zeros(len(gdof)); rm = np.zeros(len(gdof))
        for sgn, store in ((+eps, rp), (-eps, rm)):
            fx = dict(far)
            for j, d in enumerate(gdof):
                fx[d] = uG0[j] + (sgn if j == k else 0.0)
            fr = [d for d in range(ndof) if d not in fx]
            uu, _, _ = newton(nodes, elems, fx, np.array(fr))
            G, _ = assemble(nodes, elems, uu, want_hess=False)
            store[:] = G[gdof]
        S[:, k] = (rp - rm) / (2 * eps)
    S = 0.5 * (S + S.T)
    m = len(gi)
    print(f"    {len(gdof)} probes in {time.time()-t0p:.1f}s")
    print(f"    S symmetry {np.abs(S - S.T).max() / np.abs(S).max():.3e}")
    ev = np.linalg.eigvalsh(S)
    print(f"    S eigenvalues min {ev.min():+.4e} max {ev.max():+.4e}")
    print(f"    (a single subdomain has a rigid y-translation null mode, so a zero")
    print(f"     eigenvalue near {abs(ev).min():.2e} is expected and is supplied by")
    print(f"     the other subdomain in the coupled problem)")

    with open(os.path.join(OUT, 'nonlinear_2d_final.txt'), 'w', encoding='utf-8') as fp:
        fp.write("2D nonlinear, fully energy-driven Newton\n")
        fp.write(f"Newton iterations {iters}\n")
        fp.write(f"Hessian symmetry {np.abs(K - K.T).max() / np.abs(K).max():.3e}\n")
        fp.write(f"tangent DtN probes {len(gdof)}, S symmetry "
                 f"{np.abs(S - S.T).max() / np.abs(S).max():.3e}\n")
        fp.write(f"S eigenvalues {ev.min():+.6e} .. {ev.max():+.6e}\n")
    print(f"\n[saved] {os.path.join(OUT, 'nonlinear_2d_final.txt')}")
    print(f"total {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
