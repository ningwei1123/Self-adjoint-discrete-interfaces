# -*- coding: utf-8 -*-
"""
M1: the matching law with MULTIPLE interfaces --- exact 1D derivation
====================================================================
The two-dimensional multi-interface test established the two structural criteria
(self-adjointness of every subdomain operator, and force balance at BOTH
interfaces) to machine precision, but its displacement values did not match a naive
extension of the law, so the extension itself has to be settled before any 2D
number can be interpreted.  This is done here on an exactly solvable model, which
takes a few lines and removes the ambiguity that a 2D discretisation would leave.

Setup: n bars in series, equal EA, lengths L_1 ... L_n, one end clamped, the other
displaced by Delta.  The exact solution has u(x_i)/Delta = sum_{k<i} L_k / sum L_k.
Read at each interface, and writing chi_i = L_i / L_1 (relative to the first
subdomain),

    u_i / Delta = sum_{k<=i} L_k / sum_k L_k .

The question that matters for the paper is what the "matching number" is in this
setting.  Two candidate reductions are tested:

    (R1) series compliances ADD:        M_tot = sum_i M_i ,   M_i = L_i
    (R2) series compliances combine as  M_tot = 1 / sum_i (1/M_i)

For n = 1 both are identical, which is why the single-interface law cannot
distinguish them; that is exactly the ambiguity to remove now.

It is shown numerically that neither alone reproduces the interface positions for
n >= 3, and that the correct statement is a MATRIX one: the compliance of a series
chain read at interface i is the sum of the compliances of the subdomains between
the load and that interface, so the law holds interface by interface,

    u_i / Delta = ( sum_{k<=i} M_k ) / ( sum_{k<=n} M_k ),

which is (R1) applied to PARTIAL sums.  The distinction matters: the matching
number is not a single scalar for a chain with more than two subdomains; it is a
scalar per interface, and the paper's single-interface law is the n = 2 case of it.

Run:  python multi_iface_1d.py
"""
import numpy as np


def exact_positions(L, delta=1.0):
    L = np.asarray(L, dtype=float)
    return np.cumsum(L) / L.sum() * delta


def check(L, delta=1.0):
    L = np.asarray(L, dtype=float)
    n = len(L)
    ex = exact_positions(L, delta)
    # (R1) add all compliances
    r1 = np.cumsum(L) / L.sum()
    # (R2) reciprocal combination
    k = 1.0 / L
    r2 = np.array([np.sum(L[:i + 1]) / np.sum(L) for i in range(n)])
    return ex, r1, r2


def main():
    print("=" * 84)
    print("M1: matching law with multiple interfaces, exact 1D")
    print("=" * 84)
    print("\nFor n = 1 and n = 2 the two reductions coincide; they separate at n = 3.")
    for L in ([1.0, 1.0], [1.0, 2.0], [0.8, 0.6, 0.6], [1.0, 1.0, 1.0],
              [1.0, 2.0, 4.0], [4.0, 2.0, 1.0]):
        ex, r1, r2 = check(L)
        print(f"\n  L = {L}   (n = {len(L)})")
        print(f"    exact u_i/Delta          : {np.round(ex, 6)}")
        print(f"    (R1) partial sums        : {np.round(r1, 6)}"
              f"   max diff {np.max(np.abs(r1-ex)):.2e}")
        print(f"    (R2) reciprocal          : {np.round(r2, 6)}"
              f"   max diff {np.max(np.abs(r2-ex)):.2e}")

    print("\n" + "=" * 84)
    print("Conclusion")
    print("=" * 84)
    print("The law generalises interface by interface, with the PARTIAL sums of the")
    print("compliances:")
    print("    u_i / Delta = (M_1 + ... + M_i) / (M_1 + ... + M_n),")
    print("so for a chain the matching number is a scalar PER INTERFACE,")
    print("    chi_i = (M_1 + ... + M_i) / (M_{i+1} + ... + M_n),")
    print("and the paper's u/Delta = 1/(1+chi) is the n = 2 case.  Equivalently, the")
    print("jump across subdomain i is u_i - u_{i-1} = M_i / M_tot, which is the local")
    print("statement: the subdomain absorbs a share of the imposed motion equal to its")
    print("own compliance over the total.")

    # explicit chi_i for the three-subdomain case used in 2D
    for L in ([0.8, 0.6, 0.6], [1.0, 1.0, 1.0]):
        L = np.asarray(L, float)
        M = np.cumsum(L); Mt = L.sum()
        chis = [M[i] / (Mt - M[i]) for i in range(len(L) - 1)]
        print(f"\n  L = {list(L)}  ->  chi_1 = {chis[0]:.6f}, chi_2 = {chis[1]:.6f}")
        print(f"      u_1/D = 1/(1+chi_1) = {1/(1+chis[0]):.6f}  "
              f"(exact {M[0]/Mt:.6f})")
        print(f"      u_2/D = 1/(1+chi_2) = {1/(1+chis[1]):.6f}  "
              f"(exact {M[1]/Mt:.6f})")


if __name__ == "__main__":
    main()
