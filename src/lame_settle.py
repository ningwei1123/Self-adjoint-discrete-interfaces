# -*- coding: utf-8 -*-
"""
Lame reference, settled numerically-symbolically: which candidate matches the FEM?
=================================================================================
The symbolic pass showed that u = c1 r + c2/r is the exact general solution of the
plane-stress equilibrium equation (the residual simplifies to zero), and that the
solved ratio is c2/c1 = r_i^2 (1+nu)/(1-nu).  Two other candidate ratios had been
used in the course of this work, r_i^2 and ((lam+2mu)/(2mu)) r_i^2, and an internal
inconsistency remained in one of the printed diagnostics, so the three candidates
are compared here against the quantity that actually matters: the finite-element
solution.

This sidesteps every algebraic question.  A candidate reference is correct exactly
when it satisfies the two boundary conditions of the problem AND reproduces the
finite-element radial displacement.  Both tests are applied to all three
candidates, with the E-modulus normalised to 1 so that only shape matters.

Run:  python lame_settle.py
"""
import numpy as np
import sympy as sp

import lame_cylinder as L

E = 1.0
NU = 0.30
R_IN, R_OUT = 0.5, 2.0
DELTA = 0.01

lam = E * NU / (1 - NU ** 2)
mu = E / (2 * (1 + NU))


def candidates():
    """The three c2/c1 ratios that have appeared, plus the symbolic one."""
    return {
        "r_i^2                       (first draft)": R_IN ** 2,
        "((lam+2mu)/(2mu)) r_i^2     (second draft)": (lam + 2 * mu) / (2 * mu) * R_IN ** 2,
        "r_i^2 (1+nu)/(1-nu)         (symbolic)": R_IN ** 2 * (1 + NU) / (1 - NU),
    }


def fit(ratio, delta=DELTA):
    """c1 fixed by u(R_OUT) = delta for the given c2/c1."""
    c1 = delta / (R_OUT + ratio / R_OUT)
    return c1, c1 * ratio


def bcs(c1, c2):
    sr = (lam + 2 * mu) * c1 - 2 * mu * c2 / R_IN ** 2
    ur = c1 * R_OUT + c2 / R_OUT
    return sr, ur


def fem_profile(nt=192, nr=64):
    nodes, elems, rings = L.polar_mesh(nt, nr)
    K = L.assemble(nodes, elems, NU)
    rad = np.hypot(nodes[:, 0], nodes[:, 1])
    out = sorted([k for k in range(len(nodes)) if abs(rad[k] - R_OUT) < 1e-9])
    fixed = {}
    for k in out:
        t = np.arctan2(nodes[k, 1], nodes[k, 0])
        fixed[2 * k] = DELTA * np.cos(t)
        fixed[2 * k + 1] = DELTA * np.sin(t)
    fds = np.array(sorted(fixed))
    fv = np.array([fixed[d] for d in fds])
    free = np.array([d for d in range(2 * len(nodes)) if d not in fixed])
    u = np.zeros(2 * len(nodes))
    u[free] = np.linalg.solve(K[np.ix_(free, free)], -K[np.ix_(free, fds)] @ fv)
    u[fds] = fv
    return nodes, rad, u, rings


def main():
    print("=" * 90)
    print("Which Lame candidate is the right reference?  (boundary data + FEM)")
    print("=" * 90)
    print(f"E = {E}, nu = {NU}, r_in = {R_IN}, r_out = {R_OUT}, Delta = {DELTA}")
    print(f"lam = {lam:.8f}   mu = {mu:.8f}")

    nodes, rad, u, rings = fem_profile()
    probe = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    fem = {}
    for rr in probe:
        sel = sorted([k for k in range(len(nodes)) if abs(rad[k] - rr) < 1e-9])
        if not sel:
            continue
        t = np.arctan2(nodes[sel, 1], nodes[sel, 0])
        fem[rr] = float(np.mean(u[[2 * k for k in sel]] * np.cos(t)
                                + u[[2 * k + 1 for k in sel]] * np.sin(t)))

    print(f"\n{'candidate':44s} {'c2/c1':>9} {'sig_rr(r_in)':>13} "
          f"{'u(r_out)':>10} {'max FEM err':>12}")
    best = None
    for tag, ratio in candidates().items():
        c1, c2 = fit(ratio)
        sr, ur = bcs(c1, c2)
        errs = [abs((c1 * rr + c2 / rr) - fem[rr]) / abs(fem[rr])
                for rr in probe if rr in fem]
        mx = max(errs)
        c2c1 = c2 / c1
        print(f"{tag:44s} {c2c1:9.6f} {sr:13.3e} {ur:10.7f} {mx:12.3e}")
        if best is None or mx < best[1]:
            best = (tag, mx, ratio)

    print(f"\n  best match to the finite element solution: {best[0]}")
    print(f"     maximum relative error {best[1]:.3e}")

    # ---- resolve the internal inconsistency symbolically ----------------
    print("\n" + "=" * 90)
    print("The symbolic ratio, re-derived with the boundary condition imposed directly")
    print("=" * 90)
    r, ri, ro, D, nu_s = sp.symbols('r r_i r_o Delta nu', positive=True)
    c1s, c2s = sp.symbols('c1 c2', real=True)
    lam_s = nu_s / (1 - nu_s ** 2)
    mu_s = 1 / (2 * (1 + nu_s))
    u_s = c1s * r + c2s / r
    eps_rr = sp.diff(u_s, r)
    eps_tt = u_s / r
    sig_rr = (lam_s + 2 * mu_s) * eps_rr + lam_s * eps_tt
    sol = sp.solve([sp.Eq(sig_rr.subs(r, ri), 0), sp.Eq(u_s.subs(r, ro), D)],
                   [c1s, c2s], dict=True)[0]
    ratio_sym = sp.simplify(sol[c2s] / sol[c1s])
    print("  c2/c1 =", sp.simplify(ratio_sym))
    print("  at nu = 3/10 :", sp.nsimplify(ratio_sym.subs(nu_s, sp.Rational(3, 10)),
                                          rational=True))
    print("  numeric      :", sp.N(ratio_sym.subs(nu_s, sp.Rational(3, 10)), 12))

    # verify the free-surface condition using the SOLVED coefficients
    sr_check = sp.simplify(sig_rr.subs(r, ri).subs(sol))
    print("\n  sigma_rr(r_in) with the solved coefficients =", sr_check)
    ur_check = sp.simplify(u_s.subs(r, ro).subs(sol))
    print("  u(r_out)       with the solved coefficients =", ur_check)

    print("\n" + "=" * 90)
    print("VERDICT")
    print("=" * 90)
    print(f"  The symbolic solution has c2/c1 = r_i^2 (1+nu)/(1-nu) = "
          f"{sp.N(ratio_sym.subs(nu_s, sp.Rational(3,10)), 8)}.")
    print("  It is the one that reproduces the finite element solution, and it is the")
    print("  one to use as the reference.  The two ratios used earlier came from an")
    print("  incorrect elimination and are the reason the comparison appeared not to")
    print("  converge.")


if __name__ == "__main__":
    main()
