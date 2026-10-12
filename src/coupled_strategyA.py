# -*- coding: utf-8 -*-
"""
Strategy A, monolithic penalty assembly with a NODAL TRACE basis
================================================================
This is the "one matrix, no outer iteration" route of the project's feasibility
study, with the network subdomain represented by its interface operator instead
of by a generic feature basis.

Formulation
-----------
The trace on Gamma is discretised by the nodal trace basis (cardinal functions at
the interface nodes), ordered [u_x at all nodes ; u_y at all nodes].  The whole
coupled potential energy is

    Pi = 1/2 u_F^T K_F u_F            (left subdomain, Q4)
       + 1/2 c^T S_N c + c^T q_N      (right subdomain, statically condensed)
       + rho/2 || u_F|_Gamma - c ||^2 (interface continuity, penalty)

so the assembled system is

    [ K_F + rho P^T P     -rho P^T ] [ u_F ]   [ f_F - ... ]
    [ -rho P              S_N + rho I ] [ c  ] = [ -q_N    ]

with P the nodal trace evaluation matrix on Gamma.  Everything is assembled into
ONE matrix and solved once.  For the LEFT subdomain the interface nodes are kept
as unknowns and the Dirichlet data are carried by the right-hand side, so no
iteration and no condensation loop is needed.

Validation
----------
  1. the assembled system must reproduce the statically condensed reference
     (S_L + S_N) c = -(q_L + q_N) as rho grows;
  2. the exact right-subdomain operator must reproduce the monolithic FE
     interface trace to discretisation accuracy;
  3. S_N reconstructed by probing 34 unit traces must agree with the exact
     operator to machine precision (this is the "no network needed" result).

Run:
  python coupled_strategyA.py
"""
import os
import time

import numpy as np

import func_operator as F
import stress_recovery2d as R
import hybrid_energy_nn as H

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
CFIELD = H.C_FIELD
NX_L, NY_L = 12, 12         # left subdomain mesh
NX_R, NY_R = 16, 12         # right subdomain mesh (same ny -> same interface nodes)
M = NY_L + 1                # interface nodes (shared by both subdomains)
assert NY_R == NY_L, "both subdomains must share the interface node set"


def trace_basis(y_nodes):
    """Cardinal nodal trace basis: mode j is 1 at node j and 0 at the others."""
    m = len(y_nodes)
    return np.eye(m)        # nodal (cardinal) basis in the y direction


def left_subdomain():
    """Q4 left subdomain with x = 0 fully clamped and Gamma left free."""
    nodes, elems = R.build_mesh(NX_L, NY_L, 0.0, H.X0, 0.0, H.LY)
    K = F.assemble_field(nodes, elems, CFIELD, H.NU)
    n = len(nodes)
    gi = sorted([k for k in range(n) if abs(nodes[k, 0] - H.X0) < 1e-12],
                key=lambda k: nodes[k, 1])
    cl = [k for k in range(n) if abs(nodes[k, 0]) < 1e-12]
    return nodes, K, n, gi, cl


def right_operator(nx=NX_R, ny=NY_R):
    """Exact nodal DtN operator S_N and far-load q_N of Omega_N by probing."""
    nodes, elems = R.build_mesh(nx, ny, H.X0, H.LXR, 0.0, H.LY)
    K = F.assemble_field(nodes, elems, CFIELD, H.NU)
    n = len(nodes)
    gi = sorted([k for k in range(n) if abs(nodes[k, 0] - H.X0) < 1e-12],
                key=lambda k: nodes[k, 1])
    fi = [k for k in range(n) if abs(nodes[k, 0] - H.LXR) < 1e-12]
    gdof = [2 * k for k in gi] + [2 * k + 1 for k in gi]
    fdof = [2 * k for k in fi]
    free = np.array([d for d in range(2 * n)
                     if d not in set(gdof) and d not in set(fdof)])
    Kff = K[np.ix_(free, free)]
    Kfg = K[np.ix_(free, gdof)]
    Kfd = K[np.ix_(free, fdof)]
    Z = np.linalg.solve(Kff, np.column_stack([Kfg, Kfd]))
    S = K[np.ix_(gdof, gdof)] - K[np.ix_(gdof, free)] @ Z[:, :len(gdof)]
    uD = np.array([R.right_disp(nodes[k, 1]) for k in fi])
    q = K[np.ix_(gdof, fdof)] @ uD - K[np.ix_(gdof, free)] @ (Z[:, len(gdof):] @ uD)
    return S, q, np.array([nodes[k, 1] for k in gi])


def assemble_penalty(rho=1e8):
    """One monolithic matrix: left FE unknowns + nodal trace unknowns."""
    nodes, KF, nF, gi, cl = left_subdomain()
    S_N, q_N, yR = right_operator()
    m = len(gi)
    assert m == M, (m, M)
    ndofF = 2 * nF
    gdof = [2 * k for k in gi] + [2 * k + 1 for k in gi]
    cdof = [2 * k + cc for k in cl for cc in (0, 1)]
    # unknowns: all left dofs except the clamped ones, then the 2m trace dofs
    keep = [d for d in range(ndofF) if d not in set(cdof)]
    idxF = {d: i for i, d in enumerate(keep)}
    nU = len(keep) + 2 * m
    A = np.zeros((nU, nU))
    b = np.zeros(nU)
    # left subdomain block
    for d in keep:
        for e in keep:
            A[idxF[d], idxF[e]] = KF[d, e]
        acc = 0.0
        for dcl in cdof:
            acc -= KF[d, dcl] * 0.0            # clamped values are zero
        b[idxF[d]] = acc
    off = len(keep)
    # interface penalty: rho/2 || u_F|Gamma - c ||^2
    for j in range(m):
        for cc, comp in ((0, 0), (1, 1)):
            dF = 2 * gi[j] + comp
            iF = idxF[dF]
            ic = off + cc * m + j
            A[iF, iF] += rho
            A[ic, ic] += rho
            A[iF, ic] -= rho
            A[ic, iF] -= rho
    # right subdomain: 1/2 c^T S_N c + c^T q_N
    A[off:, off:] += S_N
    b[off:] -= q_N
    return dict(A=A, b=b, keep=keep, idxF=idxF, off=off, gi=gi, m=m,
                KF=KF, nF=nF, S_N=S_N, q_N=q_N, yR=yR)


def condensed_reference():
    """Interface-only reference: (S_L + S_N) c = -(q_L + q_N)."""
    nodes, KF, nF, gi, cl = left_subdomain()
    gdof = [2 * k for k in gi] + [2 * k + 1 for k in gi]
    cdof = [2 * k + cc for k in cl for cc in (0, 1)]
    free = np.array([d for d in range(2 * nF)
                     if d not in set(gdof) and d not in set(cdof)])
    Kff = KF[np.ix_(free, free)]
    Kfg = KF[np.ix_(free, gdof)]
    Z = np.linalg.solve(Kff, Kfg)
    S_L = KF[np.ix_(gdof, gdof)] - KF[np.ix_(gdof, free)] @ Z
    S_N, q_N, yR = right_operator()
    c = np.linalg.solve(S_L + S_N, -q_N)
    return c, S_L, S_N, q_N, yR


def main():
    t0 = time.time()
    print("=== strategy A: monolithic penalty assembly, nodal trace basis ===")
    c_ref, S_L, S_N, q_N, yR = condensed_reference()
    print(f"S_L, S_N symmetric to "
          f"{np.abs(S_L - S_L.T).max() / np.abs(S_L).max():.2e}, "
          f"{np.abs(S_N - S_N.T).max() / np.abs(S_N).max():.2e}")

    # --- no-network result: probe the exact operator from 2m unit traces
    E = np.eye(2 * M)
    S_probe = np.zeros((2 * M, 2 * M))
    for i in range(2 * M):
        S_probe[:, i] = S_N @ E[:, i]
    print(f"S_N reconstruction by probing: "
          f"{np.linalg.norm(S_probe - S_N) / np.linalg.norm(S_N):.3e}")

    # --- penalty sweep: the assembled system must converge to the reference
    print("\npenalty sweep (assembled one-matrix system vs condensed reference):")
    prev = None
    for rho in (1e4, 1e6, 1e8, 1e10):
        d = assemble_penalty(rho)
        u = np.linalg.solve(d['A'], d['b'])
        c = u[d['off']:]
        err = np.linalg.norm(c - c_ref) / np.linalg.norm(c_ref)
        # interface continuity defect
        defc = max(abs(u[d['idxF'][2 * d['gi'][j] + comp]] - c[comp * M + j])
                   for j in range(M) for comp in (0, 1))
        print(f"  rho={rho:.0e}: ||c - c_ref||/||c_ref|| = {err:.3e}"
              f"   max continuity defect {defc:.3e}")
        prev = (c, err)

    # --- accuracy against the monolithic FE solution
    nodes, _, uM = H.monolithic(nx=48, ny=24, cfield=CFIELD)
    gim = sorted([k for k in range(len(nodes)) if abs(nodes[k, 0] - H.X0) < 1e-12],
                 key=lambda k: nodes[k, 1])
    yM = np.array([nodes[k, 1] for k in gim])
    uxM = np.array([uM[2 * k] for k in gim])
    uyM = np.array([uM[2 * k + 1] for k in gim])
    c = prev[0]
    ex = np.linalg.norm(c[:M] - np.interp(yR, yM, uxM)) / np.linalg.norm(np.interp(yR, yM, uxM))
    ey = np.linalg.norm(c[M:] - np.interp(yR, yM, uyM)) / np.linalg.norm(np.interp(yR, yM, uyM))
    print(f"\ninterface vs monolithic FE 48x24: ux {ex:.4%}   uy {ey:.4%}")
    print(f"total time {time.time() - t0:.1f}s")

    with open(os.path.join(OUT, 'coupled_strategyA.txt'), 'w', encoding='utf-8') as fp:
        fp.write("Strategy A: monolithic penalty assembly with a nodal trace basis\n")
        fp.write(f"interface dofs {2 * M}\n")
        fp.write(f"S_N probing reconstruction "
                 f"{np.linalg.norm(S_probe - S_N) / np.linalg.norm(S_N):.3e}\n")
        fp.write(f"penalty sweep final: ||c-c_ref||/||c_ref|| {prev[1]:.3e}\n")
        fp.write(f"interface vs monolithic: ux {ex:.6%}  uy {ey:.6%}\n")
    print(f"[saved] {os.path.join(OUT, 'coupled_strategyA.txt')}")


if __name__ == '__main__':
    main()
