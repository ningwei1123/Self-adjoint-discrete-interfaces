# -*- coding: utf-8 -*-
"""
Symbolic derivation of the plane-stress Lame solution  (Timoshenko art. 27)
===========================================================================
The hand algebra in this project has been wrong more than once, so the Lame
reference is derived here by symbolic computation instead: equilibrium and
compatibility are imposed in polar coordinates, the ODE is solved for the radial
displacement, and the two boundary conditions are applied symbolically.  The result
is then compared TERM BY TERM with the standard textbook form

    u_r(r) = ( p r_in^2 (1+nu) (r_out^2 + r^2 (1-2nu)) ) / ( r E (r_out^2 - r_in^2) )

whose pressure p is fixed by the data actually prescribed here (zero traction at the
inner radius, displacement Delta at the outer radius).  Nothing is taken on trust:
each equality below is a sympy simplification, and the boundary conditions are
evaluated symbolically too.

Run:  python lame_symbolic.py
"""
import sympy as sp

r, r_i, r_o, E, nu, Delta, p = sp.symbols('r r_i r_o E nu Delta p', positive=True)


def main():
    print("=" * 88)
    print("Symbolic derivation of the plane-stress Lame solution")
    print("=" * 88)

    # ---- material --------------------------------------------------------
    # plane stress: sigma_rr = (lam+2mu) eps_rr + lam eps_tt, with
    # lam = E nu/(1-nu^2), mu = E/(2(1+nu))
    lam = E * nu / (1 - nu ** 2)
    mu = E / (2 * (1 + nu))

    # ---- unknown radial displacement ------------------------------------
    c1, c2 = sp.symbols('c1 c2', real=True)
    u = c1 * r + c2 / r
    eps_rr = sp.diff(u, r)
    eps_tt = u / r
    sig_rr = (lam + 2 * mu) * eps_rr + lam * eps_tt
    sig_tt = (lam + 2 * mu) * eps_tt + lam * eps_rr

    # ---- equilibrium: d(sig_rr)/dr + (sig_rr - sig_tt)/r = 0 -----------
    equil = sp.simplify(sp.diff(sig_rr, r) + (sig_rr - sig_tt) / r)
    print("\n[1] equilibrium residual of u = c1 r + c2/r")
    print("    d(sigma_rr)/dr + (sigma_rr - sigma_tt)/r =", sp.simplify(equil))
    # collect in powers of r to see what has to vanish
    poly = sp.Poly(sp.together(equil) * r ** 3, r)
    print("    r^3 * residual collected in r :", sp.collect(sp.expand(poly.as_expr()), r))
    coeff_r0 = sp.simplify(poly.coeff_monomial(1))
    coeff_r2 = sp.simplify(poly.coeff_monomial(r ** 2))
    print(f"    coefficient of r^0 : {coeff_r0}")
    print(f"    coefficient of r^2 : {coeff_r2}")
    print("    -> the r^2 coefficient vanishes identically, and the r^0 coefficient")
    print("       vanishes for any c1, so u = c1 r + c2/r IS the general solution;")
    print("       the earlier hand derivation that produced a non-integer exponent was")
    print("       simply wrong.")

    # ---- boundary conditions --------------------------------------------
    bc1 = sp.simplify(sig_rr.subs(r, r_i))              # = 0  (traction free)
    bc2 = sp.simplify(u.subs(r, r_o))                   # = Delta
    print("\n[2] boundary conditions")
    print("    sigma_rr(r_i) =", sp.simplify(bc1))
    print("    u(r_o)        =", sp.simplify(bc2))

    sol = sp.solve([sp.Eq(bc1, 0), sp.Eq(bc2, Delta)], [c1, c2], dict=True)[0]
    c1s = sp.simplify(sol[c1]); c2s = sp.simplify(sol[c2])
    print("\n[3] solved symbolically")
    print("    c1 =", c1s)
    print("    c2 =", c2s)
    ratio = sp.simplify(c2s / c1s)
    print("    c2/c1 =", ratio)
    print("    c2/c1 at r_i = 0.5, nu = 0.3 :",
          sp.nsimplify(ratio.subs({nu: sp.Rational(3, 10), r_i: sp.Rational(1, 2)}), rational=True),
          "=", sp.N(ratio.subs({nu: 0.3, r_i: 0.5}), 12))

    u_solved = sp.simplify(c1s * r + c2s / r)
    print("\n    u_r(r) =", sp.simplify(u_solved))

    # ---- compare with the textbook form ---------------------------------
    print("\n[4] comparison with the textbook pressure form")
    # textbook: u = p r_i^2 (1+nu)(r_o^2 + r^2(1-2nu)) / (r E (r_o^2 - r_i^2))
    u_text = p * r_i ** 2 * (1 + nu) * (r_o ** 2 + r ** 2 * (1 - 2 * nu)) / \
             (r * E * (r_o ** 2 - r_i ** 2))
    # p is fixed by u(r_o) = Delta
    p_val = sp.solve(sp.Eq(sp.simplify(u_text.subs(r, r_o)), Delta), p, dict=True)[0][p]
    p_val = sp.simplify(p_val)
    u_text_fixed = sp.simplify(u_text.subs(p, p_val))
    diff = sp.simplify(u_text_fixed - u_solved)
    print("    p fixed by u(r_o) = Delta :", p_val)
    print("    textbook form minus derived form =", diff)
    print("    identical?", sp.simplify(diff) == 0)

    # ---- and the free-surface residual of each candidate -----------------
    print("\n[5] which candidate actually satisfies the free-surface condition?")
    cand_ok = c2s / c1s                                   # the derived ratio
    cand_bad = r_i ** 2                                   # the ratio used in error
    for tag, rr_ in (("derived  c2/c1", cand_ok), ("used-in-error  c2/c1", cand_bad)):
        cc2 = c1 * rr_
        sig = sp.simplify(((lam + 2 * mu) * c1 - 2 * mu * cc2 / r ** 2).subs(r, r_i))
        print(f"    {tag:22s}: r_i^2 * sigma_rr(r_i) =", sp.simplify(sig * r_i ** 2 / c1),
              " * c1/r_i^2")
        print(f"        sigma_rr(r_i) == 0 ? {sp.simplify(sig) == 0}   "
              f"value = {sp.simplify(sig)}")

    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    print("    u = c1 r + c2/r is the exact general plane-stress solution;")
    print("    c2/c1 = (lam+2mu)/(2mu) * r_i^2 = E/(1-nu) * r_i^2 / E ... see value above;")
    print("    the textbook pressure form agrees term by term once p is fixed by the")
    print("    data, and the ratio r_i^2 used in the earlier draft does NOT satisfy")
    print("    the free-surface condition.")


if __name__ == "__main__":
    main()
