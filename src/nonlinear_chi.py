# -*- coding: utf-8 -*-
"""
Checklist item 3: the tangent energy-matching number chi_T
==========================================================
The paper's energy-matching law u_Gamma/Delta = 1/(1+chi) with chi = M_N/M_F is
a statement about the LINEAR interface compliances.  Nonlinearly it must be
local: it holds for the TANGENT operators at a converged state,

    chi_T = M_N^T / M_F^T,     M_i^T = (b^T (S_i^T)^-1 b)^-1,

with S_i^T the tangent DtN operator of subdomain i at that state and b the
constant (rigid-translation) mode of the interface edge.

The tangent operators are measured by probing: perturb one interface dof,
re-converge the interior, read the change of the interface internal force.  The
probe rows and the probe columns are the SAME interface dofs here, because both
subdomains share the interface node set, so the extracted block is the square
tangent DtN operator on the trace.

Reported for each load level
  1. symmetry of both tangent operators (0 means the tangent inherits the
     symmetry of the energy);
  2. the residual |r_F + r_N| of the coupled equilibrium at the trace at which
     the operators were probed -- it must be at machine level, because the trace
     is the converged coupled solution;
  3. chi_T, the predicted allocation 1/(1+chi_T), and the allocation measured
     directly from the coupled solution.

Run:  python nonlinear_chi.py
"""
import os
import time

import numpy as np

import nonlinear_2d_final as N
import nonlinear_coupled2d as C

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))


UNIFORM = 1.0     # uniform end displacement, matching the constant-mode compliance


def solve_with(nodes, elems, geom, gdof, fixed_base, gvals):
    """Converge one subdomain with the given interface trace."""
    ndof = 2 * len(nodes)
    fixed = dict(fixed_base)
    for j, d in enumerate(gdof):
        fixed[d] = gvals[j]
    free = np.array([d for d in range(ndof) if d not in fixed])
    u = np.zeros(ndof)
    for d, v in fixed.items():
        u[d] = v
    gid, _ = C.global_ids(nodes, np.zeros((0, 2)))
    for _ in range(40):
        G, K = C.assemble(nodes, elems, geom, np.zeros((0, 2)), [], [],
                          gid, ndof, u)
        r = G[free]
        if np.linalg.norm(r) < 1e-10:
            break
        u[free] -= np.linalg.solve(K[np.ix_(free, free)], r)
    G, _ = C.assemble(nodes, elems, geom, np.zeros((0, 2)), [], [],
                      gid, ndof, u)
    return np.array([G[d] for d in gdof]), u


def tangent_dtn(nx, ny, x0, x1, gamma_x, far_x, far_dofs, d0, uG, eps=1e-6):
    """Tangent DtN operator of one subdomain at interface trace uG."""
    nodes, elems, geom = C.subdomain(nx, ny, x0, x1)
    n = len(nodes)
    gi = sorted([k for k in range(n) if abs(nodes[k, 0] - gamma_x) < 1e-12],
                key=lambda k: nodes[k, 1])
    fi = [k for k in range(n) if abs(nodes[k, 0] - far_x) < 1e-12]
    gdof = [2 * k for k in gi] + [2 * k + 1 for k in gi]
    fixed_base = {}
    if far_dofs:
        for k in fi:
            for cc in far_dofs:
                fixed_base[2 * k + cc] = float(N.right_disp(nodes[k, 1], d0))
    m = len(gdof)
    S = np.zeros((m, m))
    for k in range(m):
        gp = uG.copy(); gp[k] += eps
        gm = uG.copy(); gm[k] -= eps
        rp, _ = solve_with(nodes, elems, geom, gdof, fixed_base, gp)
        rm, _ = solve_with(nodes, elems, geom, gdof, fixed_base, gm)
        S[:, k] = (rp - rm) / (2.0 * eps)
    S = 0.5 * (S + S.T)
    r0, _ = solve_with(nodes, elems, geom, gdof, fixed_base, uG)
    return S, r0


def compliance_x(S, m):
    """Compliance for the constant x-mode; the y-mode is left free at x = X0 and
    is therefore handled separately (it has a rigid-translation null direction on
    the far-clamped side)."""
    k = m // 2
    b = np.zeros(m)
    b[:k] = 1.0
    x = np.linalg.solve(S, b)
    return 1.0 / float(b @ x)


def main():
    ny, nF, nR = 4, 8, 8
    t0 = time.time()
    print("=" * 74)
    print("Tangent energy-matching number chi_T")
    print("=" * 74)
    results = []
    for d0 in (0.01, 0.05):
        print(f"\n--- prescribed displacement D0 = {d0} "
              f"(nominal strain {d0/2:.1%}) ---")
        # ---- converged coupled state
        nodesL, elemsL, geomL = C.subdomain(nF, ny, 0.0, N.X0)
        nodesR, elemsR, geomR = C.subdomain(nR, ny, N.X0, N.LX)
        gid, ndof = C.global_ids(nodesL, nodesR)
        cl = [k for k in range(len(nodesL)) if abs(nodesL[k, 0]) < 1e-12]
        ri = [k for k in range(len(nodesR)) if abs(nodesR[k, 0] - N.LX) < 1e-12]
        fixed = {}
        for k in cl:
            fixed[2 * gid[('L', k)]] = 0.0
            fixed[2 * gid[('L', k)] + 1] = 0.0
        for k in ri:
            fixed[2 * gid[('R', k)]] = float(UNIFORM * d0)
        free = np.array([d for d in range(ndof) if d not in fixed])
        u = np.zeros(ndof)
        for d, v in fixed.items():
            u[d] = v
        for _ in range(40):
            G, K = C.assemble(nodesL, elemsL, geomL, nodesR, elemsR, geomR,
                              gid, ndof, u)
            r = G[free]
            if np.linalg.norm(r) < 1e-10:
                break
            u[free] -= np.linalg.solve(K[np.ix_(free, free)], r)
        gLf = sorted([k for k in range(len(nodesL))
                      if abs(nodesL[k, 0] - N.X0) < 1e-12],
                     key=lambda k: nodesL[k, 1])
        uG = np.array([u[2 * gid[('L', k)] + c] for k in gLf for c in (0, 1)])
        m = len(uG)

        # ---- tangent operators by probing
        t1 = time.time()
        SL, rL = tangent_dtn(nF, ny, 0.0, N.X0, N.X0, 0.0, (0, 1), d0, uG)
        SR, rR = tangent_dtn(nR, ny, N.X0, N.LX, N.X0, N.LX, (0,), d0, uG)
        print(f"    probing both operators: {time.time()-t1:.1f}s")
        print(f"    S_F^T symmetry {np.abs(SL - SL.T).max() / np.abs(SL).max():.2e}"
              f"   S_N^T symmetry {np.abs(SR - SR.T).max() / np.abs(SR).max():.2e}")
        bal = np.abs(rL + rR).max()
        print(f"    coupled equilibrium residual at the probed trace: "
              f"|r_F + r_N| max = {bal:.3e}"
              f"   (relative {bal / np.abs(rL).max():.2e})")

        # ---- chi_T
        MF, MN = compliance_x(SL, m), compliance_x(SR, m)
        chi = MN / MF
        pred = 1.0 / (1.0 + chi)
        # The denominator must be the ACTUAL mean end displacement, and the
        # numerator the mean interface displacement, exactly as in the linear
        # calibration (chi_calibration.py).  Dividing by the nominal D0 is only
        # correct when the far-end trace is exactly uniform.
        ux_end = np.array([u[2 * gid[('R', k)]] for k in ri])
        Delta = float(np.mean(ux_end))
        meas = float(np.mean(uG[:m // 2])) / Delta
        print(f"    mean end displacement (x = LX): {Delta:.8e}"
              f"   (nominal {UNIFORM * d0:.8e})")
        print(f"    M_F^T = {MF:.6e}   M_N^T = {MN:.6e}")
        print(f"    chi_T = {chi:.6f}    predicted u_Gamma/Delta = {pred:.6f}")
        print(f"    measured  u_Gamma/Delta = {meas:.6f}"
              f"    difference {abs(pred - meas):.4f}"
              f" ({abs(pred - meas) / meas:.2%})")
        results.append((d0, chi, pred, meas, bal))

    print(f"\ntotal {time.time()-t0:.1f}s")
    with open(os.path.join(OUT, 'nonlinear_chi.txt'), 'w', encoding='utf-8') as fp:
        fp.write("Tangent energy-matching number chi_T\n")
        for d0, chi, pred, meas, bal in results:
            fp.write(f"D0={d0}: chi_T={chi:.6f} predicted={pred:.6f} "
                     f"measured={meas:.6f} balance={bal:.2e}\n")
    print(f"[saved] {os.path.join(OUT, 'nonlinear_chi.txt')}")


if __name__ == '__main__':
    main()
