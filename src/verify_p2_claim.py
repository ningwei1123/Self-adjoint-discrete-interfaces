# -*- coding: utf-8 -*-
"""
RETRACTION RECORD: a negative claim in an earlier draft of V5 was false
=====================================================================
ORIGINAL CLAIM (now withdrawn): that the energy-matching law requires

    (P2)  M_i is the compliance of subdomain i with its non-interface end
          TRACTION FREE,

and that violating it costs 38%.

WHY IT WAS FALSE.  The two candidate compliances of a bar segment -- measured with
the far end clamped, and measured with the far end free -- are the SAME number in
one dimension, EA/L.  The law therefore holds under both, so (P2) has no content,
and it cannot be what distinguishes a valid configuration from an invalid one.  The
script below demonstrates that equality exactly.

WHAT THE 38% ACTUALLY WAS.  A bookkeeping error in the condensed interface load:
the term -K_{G,i} K_{ii}^{-1} K_{i,D} u_D was dropped, which is required whenever
the interface and the Dirichlet boundary are not directly connected by an element.
With an incomplete load the coupled solve returns the wrong interface displacement
while the compliance ratio is unaffected -- precisely the signature that misled the
draft into formulating (P2).  The same omission, and the same misdiagnosis, had
already occurred once before in this project (the annulus interface load).

Historical title follows.
Verification of a withdrawn negative claim (1D, exact)
=================================================================================
The V5 manuscript asserts, in section 5.8, that the energy-matching law requires

    (P2)  M_i is the compliance of subdomain i with its non-interface end
          TRACTION FREE,

and reports that violating (P2) costs 38% on a 2D benchmark.  That claim is in
tension with the paper's own Table 1, whose construction has a CLAMPED far end on
one subdomain and a PRESCRIBED far end on the other, yet reproduces the law to
0.23%.

At most one of those can be right, so the question is settled here on an exactly
solvable model where no discretisation or bookkeeping can intervene: two bars in
series, one end clamped, the other end displaced by Delta.  This is the same
configuration as Table 1, restricted to one dimension.

Exact solution: with EA and lengths L1, L2,
    u(x) = Delta x / (L1 + L2)  for the whole bar,
    u_Gamma = Delta L1/(L1+L2).
The law in the form used in the paper is
    u_Gamma/Delta = M_F/(M_F + M_N),   chi = M_N/M_F,
so it holds iff M_F/(M_F+M_N) = L1/(L1+L2), i.e. iff M_i is proportional to L_i.
The compliance of a bar SEGMENT of length L under the loads it actually carries is
L/(EA) regardless of what holds its far end, so the law must hold exactly here.

The script evaluates the three candidate definitions of M_i that one could
plausibly adopt:
    (a) force/displacement with the far end CLAMPED   -> k = EA/L       (L/(EA))
    (b) force/displacement with the far end FREE      -> k = EA/L       (same)
    (c) the value that would make (P2) matter, i.e. treating the clamped and free
        cases as different
and shows that (a) and (b) coincide, so (P2) cannot be what distinguishes the
valid from the invalid configurations.  The 38% reported in V5 must therefore have
a different cause, and that is the finding this script supports.

Run:  python verify_p2_claim.py
"""
import numpy as np

EA = 1.0


def exact(L1, L2, delta=1.0):
    """Clamped at x=0, u(L1+L2) = delta."""
    uG = delta * L1 / (L1 + L2)
    return uG, delta, L1 / (L1 + L2)


def compliance_clamped(L):
    """Compliance of a bar of length L with its far end CLAMPED, loaded at the
    driven end: k = EA/L, so M = L/EA."""
    return L / EA


def compliance_free(L):
    """Compliance of a bar of length L with its far end FREE, loaded at the
    driven end: the bar carries a constant force, so the same k = EA/L."""
    return L / EA


def main():
    print("=" * 84)
    print("Is the matching law conditional on a TRACTION-FREE far end?  (exact 1D)")
    print("=" * 84)
    print(f"{'L1':>6} {'L2':>6} {'exact u/D':>12} {'law via (a)':>13} "
          f"{'law via (b)':>13} {'diff(a)':>10} {'diff(b)':>10}")
    worst_a = worst_b = 0.0
    for L1, L2 in ((1.0, 1.0), (1.0, 2.0), (2.0, 1.0), (0.5, 2.0), (3.0, 0.25)):
        _, _, ex = exact(L1, L2)
        Ma, Mb = compliance_clamped(L1), compliance_free(L1)
        # the two subdomains' compliances, under either definition
        Na, Nb = compliance_clamped(L2), compliance_free(L2)
        law_a = Ma / (Ma + Na)
        law_b = Mb / (Mb + Nb)
        da, db = abs(law_a - ex), abs(law_b - ex)
        worst_a = max(worst_a, da); worst_b = max(worst_b, db)
        print(f"{L1:6.2f} {L2:6.2f} {ex:12.6f} {law_a:13.6f} {law_b:13.6f} "
              f"{da:10.2e} {db:10.2e}")

    print(f"\nworst deviation: clamped-far-end definition {worst_a:.2e}, "
          f"free-far-end definition {worst_b:.2e}")
    print("\nReading: in one dimension the two definitions of the compliance are the")
    print("SAME number, EA/L, so the law holds under BOTH and condition (P2) has no")
    print("content here.  This is consistent with Table 1 of the paper, whose")
    print("construction has a clamped far end and still reproduces the law, and it")
    print("means the 38% reported in V5 cannot be attributed to (P2).")

    print("\n" + "=" * 84)
    print("What DOES distinguish the two 2D configurations then?")
    print("=" * 84)
    print("In 2D the relevant difference is not the far-end condition of the")
    print("subdomain but whether the COMPLIANCE SCALAR is evaluated in the same")
    print("interface mode on both sides, and whether the prescribed data are carried")
    print("in the condensed load.  Both are bookkeeping properties of the")
    print("implementation, not mechanical conditions on the problem.  A test of that")
    print("hypothesis is given in verify_p2_claim.py's companion, chi_scalar.py:")
    print("there the interface load was formed as -K_{G,D} u_D only, dropping the")
    print("term -K_{G,i} K_{ii}^{-1} K_{i,D} u_D that static condensation requires")
    print("whenever the interface and the Dirichlet boundary are not directly")
    print("connected by an element.  With the load incomplete, the coupled solve")
    print("returns the wrong interface displacement while the compliance ratio is")
    print("unaffected, which is exactly the pattern of a false premise rather than a")
    print("mechanical restriction.")


if __name__ == "__main__":
    main()
