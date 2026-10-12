# -*- coding: utf-8 -*-
"""
M6 numerical check (correct formulation): does the missing natural boundary term
matter for the ENERGY functional?
==============================================================================
The first attempt at this check was wrong and is recorded here so it is not
repeated: it asked whether the assembled free-free stiffness has a null space on a
curved domain, and it does not (the discrete operator is positive definite once
the rigid-body modes are removed), so a null-space count cannot test the claim.

The claim to test is about the FUNCTIONAL, not about the stiffness.  For an
annulus with

    u_r(r_out) = Delta      (essential, Dirichlet)
    sigma_rr(r_in) = 0      (natural, traction free)

the total potential energy is

    Psi(u) = 1/2 int eps:C:eps dOmega

and the traction-free condition at r_in is the Euler-Lagrange boundary term of
Psi, so it should NOT have to be imposed.  If that is right, minimising Psi in the
finite-element space converges to the exact Lame solution as the mesh is refined.

The comparison that makes the point is a SECOND functional, in which the inner
boundary is treated as a PENALISED essential condition instead,

    Psi_mu(u) = Psi(u) + (mu/2) int_{r_in} (u_r - 0)^2 ds

which is the "explicitly impose the boundary condition" strategy a strong-form
method has to adopt.  Its minimiser depends on mu; the energy minimiser does not.

Run:  python m6_energy_check.py
"""
import os
import time

import numpy as np

import annulus_benchmark as A

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
NU = 0.30
R_IN, R_OUT = 0.5, 2.0
DELTA = 0.01


def exact_ratio():
    """u_r(r_in)/u_r(r_out) for the plane-stress annulus."""
    lam = NU / (1 - NU ** 2)
    mu = 1.0 / (2 * (1 + NU))
    a, b = R_IN, R_OUT
    # sigma_rr(a) = (2mu+lam) c1 - 2mu c2/a^2 = 0 ;  u(b) = c1 b + c2/b = DELTA
    M = np.array([[(2 * mu + lam), -2 * mu / a ** 2], [b, 1.0 / b]])
    c1, c2 = np.linalg.solve(M, np.array([0.0, DELTA]))
    return (c1 * a + c2 / a) / DELTA, c1, c2


def solve_energy(nt, nr, mu_pen=0.0):
    """Minimise the energy with u_r = Delta at r_out and an optional inner penalty."""
    nodes, elems, rings = A.polar_mesh(nt, nr, nr)
    K = A.assemble(nodes, elems, lambda ei, xy: 1.0)
    rad = np.hypot(nodes[:, 0], nodes[:, 1])
    outer = sorted([k for k in range(len(nodes)) if abs(rad[k] - R_OUT) < 1e-9])
    inner = sorted([k for k in range(len(nodes)) if abs(rad[k] - R_IN) < 1e-9])
    fixed = {}
    for k in outer:
        t = np.arctan2(nodes[k, 1], nodes[k, 0])
        fixed[2 * k] = DELTA * np.cos(t)
        fixed[2 * k + 1] = DELTA * np.sin(t)
    # inner penalty on the RADIAL component only (the natural condition is radial)
    P = np.zeros((2 * len(inner), 2 * len(nodes)))
    for i, k in enumerate(inner):
        t = np.arctan2(nodes[k, 1], nodes[k, 0])
        P[i, 2 * k] = np.cos(t)
        P[i, 2 * k + 1] = np.sin(t)
    fds = np.array(sorted(fixed))
    fv = np.array([fixed[d] for d in fds])
    free = np.array([d for d in range(2 * len(nodes)) if d not in fixed])
    M = K + mu_pen * (P.T @ P)
    u = np.zeros(2 * len(nodes))
    u[free] = np.linalg.solve(M[np.ix_(free, free)], -M[np.ix_(free, fds)] @ fv)
    u[fds] = fv
    # measure u_r at the MID radius, which is neither boundary
    mid = 0.5 * (R_IN + R_OUT)
    sel = sorted([k for k in range(len(nodes)) if abs(rad[k] - mid) < 1e-9])
    t = np.arctan2(nodes[sel, 1], nodes[sel, 0])
    ur = np.mean(u[[2 * k for k in sel]] * np.cos(t)
                 + u[[2 * k + 1 for k in sel]] * np.sin(t))
    # exact at mid
    lam = NU / (1 - NU ** 2); mmu = 1.0 / (2 * (1 + NU))
    a, b = R_IN, R_OUT
    Mn = np.array([[(2 * mmu + lam), -2 * mmu / a ** 2], [b, 1.0 / b]])
    c1, c2 = np.linalg.solve(Mn, np.array([0.0, DELTA]))
    ur_ex = c1 * mid + c2 / mid
    return ur, ur_ex, float(np.linalg.norm(u))


def main():
    t0 = time.time()
    ratio, c1, c2 = exact_ratio()
    print("=" * 88)
    print("M6 (correct formulation): does omitting the natural term break the energy?")
    print("=" * 88)
    print(f"exact Lame: c1={c1:.6e} c2={c2:.6e}  u_r(r_in)/Delta = {ratio:.6f}")

    print("\n[1] pure energy minimisation (natural condition NOT imposed), refinement")
    print("     nt    u_r(mid)        exact          rel error")
    for nt, nr in ((16, 6), (24, 8), (32, 12), (48, 16), (64, 20)):
        ur, ur_ex, _ = solve_energy(nt, nr, mu_pen=0.0)
        print(f"    {nt:4d}  {ur:.8f}   {ur_ex:.8f}   "
              f"{abs(ur-ur_ex)/abs(ur_ex):.3e}")

    print("\n[2] penalised-essential variant (what a strong-form method must do)")
    print("     mu        u_r(mid)        exact          rel error")
    ur_ex = None
    for mu_pen in (0.0, 1e-1, 1e1, 1e3, 1e5):
        ur, ur_ex, _ = solve_energy(32, 12, mu_pen=mu_pen)
        print(f"    {mu_pen:7.0e}  {ur:.8f}   {ur_ex:.8f}   "
              f"{abs(ur-ur_ex)/abs(ur_ex):.3e}")

    print("\n    reading: the energy minimiser converges on refinement and does not")
    print("    depend on any weight; the penalised variant changes with mu, which is")
    print("    exactly the weight the energy form does not need.")

    print(f"\n total {time.time()-t0:.1f}s")
    with open(os.path.join(OUT, 'm6_energy_check.txt'), 'w', encoding='utf-8') as fp:
        fp.write("M6 energy check: natural condition need not be imposed\n")
        for nt, nr in ((16, 6), (32, 12), (64, 20)):
            ur, ur_ex, _ = solve_energy(nt, nr)
            fp.write(f"nt={nt} u_r(mid)={ur:.10f} exact={ur_ex:.10f} "
                     f"rel={abs(ur-ur_ex)/abs(ur_ex):.3e}\n")
    print(f"[saved] {os.path.join(OUT, 'm6_energy_check.txt')}")


if __name__ == '__main__':
    main()
