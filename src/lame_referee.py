# -*- coding: utf-8 -*-
"""
Third-party referee for the Lame coefficient: an independent finite-difference solve
====================================================================================
The symbolic pass and a hand substitution of the same constitutive expression
disagreed, so neither is trusted here.  The boundary value problem is instead solved
by a completely independent route that shares no algebra with either: the
plane-stress equilibrium equation for axisymmetric u_r(r),

    sigma_rr = (E/(1-nu^2)) (eps_rr + nu eps_tt),   eps_rr = u',  eps_tt = u/r ,
    d(sigma_rr)/dr + (sigma_rr - sigma_tt)/r = 0 ,

is integrated with a standard second-order finite-difference discretisation on a
fine radial grid, with the two boundary conditions

    sigma_rr(r_in) = 0      (traction free)
    u(r_out) = Delta        (prescribed)

imposed by elimination.  The resulting profile is then compared with the two
candidate closed forms:

    A:  c2/c1 = r_in^2 ,                    which hand reduction of the constitutive
                                            law says is required by sigma_rr(r_in)=0
    B:  c2/c1 = r_in^2 (1+nu)/(1-nu) ,      which the symbolic solve returned

and with the finite-element solution of the two-dimensional problem.  Whichever
agrees with the finite-difference solve AND with the finite elements is the correct
reference; the other is an error in whichever derivation produced it.

Run:  python lame_referee.py
"""
import numpy as np

import lame_cylinder as L

E = 1.0
NU = 0.30
R_IN, R_OUT = 0.5, 2.0
DELTA = 0.01

lam = E * NU / (1 - NU ** 2)
mu = E / (2 * (1 + NU))


def sigma(r, u, up):
    """sigma_rr and sigma_tt for an axisymmetric field."""
    srr = (lam + 2 * mu) * up + lam * u / r
    stt = (lam + 2 * mu) * u / r + lam * up
    return srr, stt


def fd_solve(n=20001):
    """Second-order FD solution of the axisymmetric plane-stress BVP."""
    r = np.linspace(R_IN, R_OUT, n)
    h = r[1] - r[0]
    # unknowns: u at all nodes.  Equations: equilibrium at interior nodes,
    # sigma_rr(r_in)=0 at node 0, u(r_out)=Delta at node n-1.
    A = np.zeros((n, n))
    b = np.zeros(n)

    def srr_coeffs(i):
        """Coefficients of (sigma_rr)_i in terms of u_{i-1}, u_i, u_{i+1}."""
        # up_i = (u_{i+1}-u_{i-1})/(2h)
        a_m = -(lam + 2 * mu) / (2 * h)
        a_0 = lam / r[i]
        a_p = +(lam + 2 * mu) / (2 * h)
        return a_m, a_0, a_p

    def stt_coeffs(i):
        a_0 = (lam + 2 * mu) / r[i]
        a_m = -lam / (2 * h)
        a_p = +lam / (2 * h)
        return a_m, a_0, a_p

    # boundary: sigma_rr(r_in) = 0
    am, a0, ap = srr_coeffs(0)
    A[0, 0] = a0; A[0, 1] = ap        # u_{-1} handled by reflection below
    # one-sided derivative at the first node is second-order accurate; use it
    # sigma_rr = (lam+2mu)(u1-u0)/h + lam u0/r0
    A[0, :] = 0.0
    A[0, 0] = -(lam + 2 * mu) / h + lam / r[0]
    A[0, 1] = +(lam + 2 * mu) / h
    b[0] = 0.0

    # equilibrium at interior nodes:
    #   (srr_{i+1} - srr_{i-1})/(2h) + (srr_i - stt_i)/r_i = 0
    for i in range(1, n - 1):
        row = np.zeros(n)
        for off, idx in ((-1, i - 1), (0, i), (1, i + 1)):
            pass
        # srr at i+1, i-1, i  (each depends on its own neighbours)
        def add_srr(row, j, w):
            am, a0, ap = srr_coeffs(j)
            if j - 1 >= 0:
                row[j - 1] += w * am
            row[j] += w * a0
            if j + 1 < n:
                row[j + 1] += w * ap
        add_srr(row, i + 1, 1.0 / (2 * h))
        add_srr(row, i - 1, -1.0 / (2 * h))
        add_srr(row, i, 1.0 / r[i])
        am, a0, ap = stt_coeffs(i)
        row[i - 1] -= am / r[i]
        row[i] -= a0 / r[i]
        row[i + 1] -= ap / r[i]
        A[i, :] = row
        b[i] = 0.0

    # boundary: u(r_out) = Delta
    A[n - 1, :] = 0.0
    A[n - 1, n - 1] = 1.0
    b[n - 1] = DELTA

    u = np.linalg.solve(A, b)
    return r, u


def main():
    print("=" * 92)
    print("Independent finite-difference referee for the Lame coefficient")
    print("=" * 92)
    r, u = fd_solve()

    # fit u = c1 r + c2/r to the FD solution
    M = np.vstack([r, 1.0 / r]).T
    c1_fit, c2_fit = np.linalg.lstsq(M, u, rcond=None)[0]
    print(f"\n[FD] fitted u = c1 r + c2/r :  c1 = {c1_fit:.10e}  c2 = {c2_fit:.10e}")
    print(f"     c2/c1 = {c2_fit/c1_fit:.10f}")
    # residual of the fit: is the FD solution really of this form?
    print(f"     max deviation of FD from the fitted two-term form : "
          f"{np.max(np.abs(M @ [c1_fit, c2_fit] - u)):.3e}")
    # what sigma_rr does the FD solution produce at the inner radius?
    up0 = (u[1] - u[0]) / (r[1] - r[0])
    s0, t0 = sigma(r[0], u[0], up0)
    print(f"     sigma_rr(r_in) of the FD solution = {s0:.3e}  (should be 0)")

    # candidates
    print(f"\n{'candidate':38s} {'c2/c1':>10} {'max |cand - FD|/|FD|':>22}")
    for tag, ratio in (("A: r_in^2", R_IN ** 2),
                       ("B: r_in^2 (1+nu)/(1-nu)", R_IN ** 2 * (1 + NU) / (1 - NU))):
        c1 = DELTA / (R_OUT + ratio / R_OUT)
        c2 = c1 * ratio
        cand = c1 * r + c2 / r
        err = np.max(np.abs(cand - u) / np.abs(u))
        srr0 = (lam + 2 * mu) * c1 - 2 * mu * c2 / R_IN ** 2
        print(f"{tag:38s} {ratio:10.6f} {err:22.3e}   sigma_rr(r_in)={srr0:+.3e}")

    # the finite element solution of the 2D problem, for the same comparison
    nodes, elems, rings = L.polar_mesh(192, 64)
    K = L.assemble(nodes, elems, NU)
    rad = np.hypot(nodes[:, 0], nodes[:, 1])
    out = sorted([k for k in range(len(nodes)) if abs(rad[k] - R_OUT) < 1e-9])
    fixed = {}
    for k in out:
        t = np.arctan2(nodes[k, 1], nodes[k, 0])
        fixed[2 * k] = DELTA * np.cos(t); fixed[2 * k + 1] = DELTA * np.sin(t)
    fds = np.array(sorted(fixed)); fv = np.array([fixed[d] for d in fds])
    free = np.array([d for d in range(2 * len(nodes)) if d not in fixed])
    ue = np.zeros(2 * len(nodes))
    ue[free] = np.linalg.solve(K[np.ix_(free, free)], -K[np.ix_(free, fds)] @ fv)
    ue[fds] = fv
    print(f"\n[FEM] radial profile vs the FD solve and the two candidates")
    print(f"    {'r':>6} {'FEM':>13} {'FD':>13} {'cand A':>13} {'cand B':>13}")
    for rr in (0.5, 0.75, 1.0, 1.25, 1.5, 1.75):
        sel = sorted([k for k in range(len(nodes)) if abs(rad[k] - rr) < 1e-9])
        if not sel:
            continue
        t = np.arctan2(nodes[sel, 1], nodes[sel, 0])
        fem = float(np.mean(ue[[2 * k for k in sel]] * np.cos(t)
                            + ue[[2 * k + 1 for k in sel]] * np.sin(t)))
        fdv = float(np.interp(rr, r, u))
        a = DELTA / (R_OUT + R_IN ** 2 / R_OUT) * (rr + R_IN ** 2 / rr)
        rb = R_IN ** 2 * (1 + NU) / (1 - NU)
        c1b = DELTA / (R_OUT + rb / R_OUT)
        bb = c1b * (rr + rb / rr)
        print(f"    {rr:6.2f} {fem:13.9f} {fdv:13.9f} {a:13.9f} {bb:13.9f}")

    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    print("    Whichever candidate agrees with BOTH the independent FD solve and the")
    print("    finite elements is the correct reference.  The other derivation contains")
    print("    the error.")


if __name__ == "__main__":
    main()
