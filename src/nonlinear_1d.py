# -*- coding: utf-8 -*-
"""
Nonlinear DtN by Newton-linearised probing -- 1D calibration
============================================================
The paper's formulation is linear, and the Limitations section states that the
extension to nonlinear elasticity requires "either a Newton-linearised probe or
an iterative (Picard) scheme, and the energy-matching law is then a local
statement about the tangent operator".  This script establishes the 1D facts
before any multidimensional work, in the same way a 1D bar was used to pin down
the linear interface formulation.

What has to be settled before 2D
--------------------------------
1. Where the tangent operator comes from.  The DtN operator is now a linearisation:
   S_N^T = d(interface residual)/d(interface displacement) at a converged state.
   Its columns are obtained by perturbing one interface degree of freedom,
   re-converging the interior, and reading the change of the interface residual.
   That is the nonlinear analogue of "one probe per trace mode".
2. Which finite-difference step is usable.  The probe step must be small enough
   for the linearisation and large enough to stay above Newton-tolerance noise.
3. That the interface equation closes at the converged state.  For a Newton step
   the tangent system is

       (S_L^T + S_N^T) du_Gamma = -(r_L + r_N)

   so at a converged state the tangent equation must reproduce the tangent
   increment, and the residual itself must vanish.

Physics
-------
Total Lagrangian 1D bar, St. Venant-Kirchhoff material, which is the simplest
nonlinear model that still exercises geometric nonlinearity:
    stretch      lambda = 1 + du/dx
    Green strain E = (lambda^2 - 1)/2
    stress       S = E_mat * E          (second Piola-Kirchhoff)
    internal force (weak form): r_i = int S * dN_i/dx dX
For a uniform bar this has the closed-form solution  E = const  =>  u linear,
which is a useful sanity check: the nonlinear solution of a uniformly stretched
bar with no body force IS linear, so the 1D test verifies the machinery rather
than the physics.  A body force  b(X) is therefore added to make the solution
genuinely nonlinear in X.

Run:  python nonlinear_1d.py
"""
import numpy as np

L, X0 = 2.0, 1.0
E_MAT, BETA = 1000.0, 1.0
BODY = 2.0                      # constant body force per unit length
U_END = 2e-3                    # prescribed displacement at X = L


# ------------------------------------------------------------------ bar model
def mesh(n):
    """n elements on [0, L], returns node coordinates."""
    return np.linspace(0.0, L, n + 1)


def element_residual_tangent(ue, xe):
    """Residual and tangent stiffness of one 2-node bar element.

    ue = (u_left, u_right), xe = (x_left, x_right).
    Exact integrals via 2-point Gauss on the reference element.
    """
    h = xe[1] - xe[0]
    gp = np.array([-1.0, 1.0]) / np.sqrt(3.0)
    gw = np.array([1.0, 1.0])
    dN = np.array([-1.0, 1.0]) / h            # dN/dX
    N = np.array([0.5 * (1 - gp), 0.5 * (1 + gp)])   # (2,2)
    re = np.zeros(2)
    ke = np.zeros((2, 2))
    for a in range(2):
        w = gw[a] * (h / 2.0)
        # d u / dX = sum_i u_i dN_i/dX  (constant over the element)
        dudX = dN @ ue
        lam = 1.0 + dudX
        Egl = 0.5 * (lam * lam - 1.0)
        S = E_MAT * Egl
        # residual: int_0^1 (S dN_i/dX - b N_i) J dxi
        re += (S * dN - BODY * N[a]) * w
        # tangent: dS/dE = E_MAT, dE/dlam = lam, dlam/du_j = dN_j/dX
        dSdlam = E_MAT * lam
        ke += np.outer(dN, dN) * dSdlam * w
    return re, ke


def assemble(u, x):
    """Global residual and tangent for the bar (no BC applied yet)."""
    n = len(x) - 1
    ndof = n + 1
    R = np.zeros(ndof)
    K = np.zeros((ndof, ndof))
    for e in range(n):
        idx = np.array([e, e + 1])
        re, ke = element_residual_tangent(u[idx], x[idx])
        R[idx] += re
        K[np.ix_(idx, idx)] += ke
    return R, K


def solve_bar(u_Gamma, n=40, tol=1e-12, maxit=50):
    """Solve the bar with u(0)=0 and u(X0)=u_Gamma, return the full field."""
    x = mesh(n)
    gi = int(round(X0 / L * n))          # interface node index
    free = [i for i in range(n + 1) if i not in (0, gi)]
    u = np.zeros(n + 1)
    u[0] = 0.0
    u[gi] = u_Gamma
    for it in range(maxit):
        R, K = assemble(u, x)
        r = R[free]
        if np.linalg.norm(r) < tol:
            break
        du = np.linalg.solve(K[np.ix_(free, free)], -r)
        u[free] += du
    return x, u, gi, it


def interface_residual(u_Gamma, n=40, X_far=None, u_far=None):
    """Residual at the interface node for a subdomain solve.

    Subdomains: left = [0, X0] with u(0)=0, u(X0)=u_Gamma
                right = [X0, L] with u(X0)=u_Gamma, u(L)=u_far
    """
    if X_far is None:                     # left subdomain
        x = np.linspace(0.0, X0, n + 1)
        gi = n
        fixed = {0: 0.0, gi: u_Gamma}
    else:                                  # right subdomain
        x = np.linspace(X0, L, n + 1)
        gi = 0
        fixed = {gi: u_Gamma, n: u_far}
    free = [i for i in range(n + 1) if i not in fixed]
    u = np.zeros(n + 1)
    for i, v in fixed.items():
        u[i] = v
    for _ in range(60):
        R, K = assemble(u, x)
        r = R[free]
        if np.linalg.norm(r) < 1e-13:
            break
        u[free] += np.linalg.solve(K[np.ix_(free, free)], -r)
    R, K = assemble(u, x)
    return R[gi], u, x, gi


def tangent_dtn(u_Gamma, side, n=40, eps=1e-7, u_far=None):
    """Newton-linearised DtN operator: d r_Gamma / d u_Gamma, by probing."""
    r0, _, _, _ = interface_residual(u_Gamma, n, X_far=(L if side == 'R' else None),
                                     u_far=u_far)
    rp, _, _, _ = interface_residual(u_Gamma + eps, n,
                                     X_far=(L if side == 'R' else None), u_far=u_far)
    rm, _, _, _ = interface_residual(u_Gamma - eps, n,
                                     X_far=(L if side == 'R' else None), u_far=u_far)
    return (rp - rm) / (2.0 * eps), r0


def main():
    print("=" * 72)
    print("Nonlinear DtN by Newton-linearised probing: 1D calibration")
    print("=" * 72)

    # ---- 1. sanity: uniform bar without body force has a LINEAR solution
    global BODY
    bsave = BODY
    BODY = 0.0
    x, u, gi, it = solve_bar(U_END, n=40)
    # the bar spans [0, L] with u(L) = U_END and u(0) = 0, so the exact
    # (linear) solution is u = U_END * x / L  -- note L, not X0
    lin = U_END * x / L
    print(f"\n[1] no body force: max|u - linear| = {np.abs(u - lin).max():.3e}"
          f"   (exact solution is linear)  Newton iters {it}")
    BODY = bsave

    # ---- 2. converged nonlinear solution WITH body force, and the DtN tangent
    uG = U_END / 2.0
    rL, _, _, _ = interface_residual(uG, 40, X_far=None)
    rR, _, _, _ = interface_residual(uG, 40, X_far=L, u_far=U_END)
    print(f"\n[2] interface residual at the trial u_Gamma = {uG:.6e}:")
    print(f"      left  r = {rL:+.6e}")
    print(f"      right r = {rR:+.6e}   (outward normals are opposite)")
    print(f"      sum   r = {rL + rR:+.6e}   <- must vanish at the coupled solution")

    # ---- 3. solve the coupled 1D problem by Newton on u_Gamma
    def coupled_residual(uG):
        rl, _, _, _ = interface_residual(uG, 40, X_far=None)
        rr, _, _, _ = interface_residual(uG, 40, X_far=L, u_far=U_END)
        return rl + rr

    uG = 0.0
    print("\n[3] coupled Newton on u_Gamma alone:")
    for k in range(20):
        r = coupled_residual(uG)
        SL, _ = tangent_dtn(uG, 'L', 40)
        SR, _ = tangent_dtn(uG, 'R', 40, u_far=U_END)
        print(f"      iter {k+1}: u_Gamma = {uG:.12e}   r = {r:+.3e}"
              f"   S_L+S_R = {SL + SR:+.6e}")
        if abs(r) < 1e-14:
            break
        uG -= r / (SL + SR)
    print(f"      converged u_Gamma = {uG:.12e}")

    # ---- 4. step-size study of the probe
    print("\n[4] probe step study (tangent should be step independent):")
    for eps in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
        SL, _ = tangent_dtn(uG, 'L', 40, eps=eps)
        SR, _ = tangent_dtn(uG, 'R', 40, eps=eps, u_far=U_END)
        print(f"      eps = {eps:.0e}:  S_L = {SL:+.10e}   S_R = {SR:+.10e}"
              f"   sum = {SL + SR:+.10e}")

    # ---- 5. the tangent equation must close at the converged state
    print("\n[5] tangent closure: (S_L+S_R) du_Gamma = -(r_L + r_R) at convergence")
    SL, _ = tangent_dtn(uG, 'L', 40)
    SR, _ = tangent_dtn(uG, 'R', 40, u_far=U_END)
    r = coupled_residual(uG)
    du = -r / (SL + SR)
    print(f"      r = {r:+.3e}   Newton increment du_Gamma = {du:+.3e}")
    print(f"      a converged state must have |r| at machine level and du ~ 0")

    # ---- 6. compare with a monolithic nonlinear solve of the whole bar
    print("\n[6] check against a monolithic solve of the whole bar")
    nm = 80
    xm = mesh(nm)
    gim = int(round(X0 / L * nm))
    last = nm                     # node index nm is the right end (nm+1 nodes)
    free = [i for i in range(nm + 1) if i not in (0, last)]
    um = np.zeros(nm + 1)
    um[last] = U_END
    for _ in range(60):
        R, K = assemble(um, xm)
        rr = R[free]
        if np.linalg.norm(rr) < 1e-13:
            break
        um[free] += np.linalg.solve(K[np.ix_(free, free)], -rr)
    print(f"      monolithic u_Gamma = {um[gim]:.12e}")
    print(f"      coupled   u_Gamma = {uG:.12e}")
    print(f"      relative difference = {abs(um[gim] - uG) / abs(um[gim]):.3e}")


if __name__ == '__main__':
    main()
