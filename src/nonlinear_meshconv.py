# -*- coding: utf-8 -*-
"""
Checklist item 2: grid convergence of the nonlinear coupled solve
=================================================================
Item 1 established that the two-subdomain nonlinear coupling converges and that
its interface force balance closes to 6.4e-14.  That result was obtained on ONE
mesh, which is not enough: the 0.16% difference from the monolithic solve could be
discretisation error of the hybrid grid, or a coupling defect.

This script varies the mesh and reports
  * the coupled-versus-monolithic interface error, which must fall at the
    discretisation rate as the mesh refines (a coupling defect would stall);
  * the interface force balance, which must stay at machine level on every mesh;
  * the Newton iteration count, which must stay bounded.

A larger prescribed displacement is also used, so that the geometric
nonlinearity is actually exercised rather than being a perturbation of the linear
solution.  (At the default D0 = 0.01 the interface displacement is 3.1e-3 on a
domain of length 2, i.e. a strain of 0.16%, where the nonlinear and linearised
solutions are numerically indistinguishable.)

Run:  python nonlinear_meshconv.py
"""
import os
import time

import numpy as np

import nonlinear_2d_final as N
import nonlinear_coupled2d as C

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))


def coupled_solve(nF, nR, ny, d0):
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
        fixed[2 * gid[('R', k)]] = float(N.right_disp(nodesR[k, 1], d0))
    free = np.array([d for d in range(ndof) if d not in fixed])
    u = np.zeros(ndof)
    for d, v in fixed.items():
        u[d] = v
    it = 0
    for it in range(40):
        G, K = C.assemble(nodesL, elemsL, geomL, nodesR, elemsR, geomR,
                          gid, ndof, u)
        r = G[free]
        if np.linalg.norm(r) < 1e-9:
            break
        u[free] -= np.linalg.solve(K[np.ix_(free, free)], r)
    # interface force balance
    GL = C.subdomain_forces(nodesL, elemsL, geomL, 'L', gid, u, ndof)
    GR = C.subdomain_forces(nodesR, elemsR, geomR, 'R', gid, u, ndof)
    gLf = sorted([k for k in range(len(nodesL)) if abs(nodesL[k, 0] - N.X0) < 1e-12],
                 key=lambda k: nodesL[k, 1])
    gRf = sorted([k for k in range(len(nodesR)) if abs(nodesR[k, 0] - N.X0) < 1e-12],
                 key=lambda k: nodesR[k, 1])
    rL = np.array([GL[2 * gid[('L', k)] + c] for k in gLf for c in (0, 1)])
    rR = np.array([GR[2 * gid[('R', k)] + c] for k in gRf for c in (0, 1)])
    bal = np.abs(rL + rR).max() / max(np.abs(rL).max(), 1e-30)
    ys = np.array([nodesL[k, 1] for k in gLf])
    ux = np.array([u[2 * gid[('L', k)]] for k in gLf])
    uy = np.array([u[2 * gid[('L', k)] + 1] for k in gLf])
    return ys, ux, uy, bal, it + 1, ndof


def monolithic(nx, ny, d0):
    nodes, elems, geom = C.subdomain(nx, ny, 0.0, N.LX)
    gid, ndof = C.global_ids(nodes, np.zeros((0, 2)))
    cl = [k for k in range(len(nodes)) if abs(nodes[k, 0]) < 1e-12]
    ri = [k for k in range(len(nodes)) if abs(nodes[k, 0] - N.LX) < 1e-12]
    fixed = {}
    for k in cl:
        fixed[2 * gid[('L', k)]] = 0.0
        fixed[2 * gid[('L', k)] + 1] = 0.0
    for k in ri:
        fixed[2 * gid[('L', k)]] = float(N.right_disp(nodes[k, 1], d0))
    free = np.array([d for d in range(ndof) if d not in fixed])
    u = np.zeros(ndof)
    for d, v in fixed.items():
        u[d] = v
    it = 0
    for it in range(40):
        G, K = C.assemble(nodes, elems, geom, np.zeros((0, 2)), [], [],
                          gid, ndof, u)
        r = G[free]
        if np.linalg.norm(r) < 1e-9:
            break
        u[free] -= np.linalg.solve(K[np.ix_(free, free)], r)
    gMc = sorted([k for k in range(len(nodes)) if abs(nodes[k, 0] - N.X0) < 1e-12],
                 key=lambda k: nodes[k, 1])
    ys = np.array([nodes[k, 1] for k in gMc])
    ux = np.array([u[2 * gid[('L', k)]] for k in gMc])
    uy = np.array([u[2 * gid[('L', k)] + 1] for k in gMc])
    return ys, ux, uy, it + 1, ndof


def err(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def main():
    for d0 in (0.01, 0.05):
        print("=" * 74)
        print(f"grid convergence, prescribed displacement D0 = {d0}"
              f"  (strain scale ~{d0/2:.1%})")
        print("=" * 74)
        print(" nF   ny  coupled dofs  Newton  force balance   "
              "ux err vs mono   uy err vs mono")
        prev = None
        rows = []
        for nF, nR, ny in ((4, 4, 3), (8, 8, 4), (12, 12, 6), (16, 16, 8)):
            t0 = time.time()
            ys, ux, uy, bal, it, nd = coupled_solve(nF, nR, ny, d0)
            ysM, uxM, uyM, itM, ndM = monolithic(2 * nF + 2 * nR, ny, d0)
            # both traces live on the same y values (the interface is refined
            # identically on both sides), so compare directly
            if len(ys) == len(ysM):
                ex, ey = err(ux, uxM), err(uy, uyM)
            else:
                ex = ey = np.nan
            print(f" {nF:3d}  {ny:3d}  {nd:9d}  {it:5d}  {bal:.2e}      "
                  f"{ex:12.4%}  {ey:12.4%}   [{time.time()-t0:.0f}s]")
            rows.append((nF, ny, ex, ey, bal))
            prev = (ex, ey)
        # observed rate between the last two meshes
        if len(rows) >= 2:
            (n1, y1, e1x, e1y, _), (n2, y2, e2x, e2y, _) = rows[-2], rows[-1]
            h1, h2 = 1.0 / (2 * n1), 1.0 / (2 * n2)
            if e2x > 0 and e1x > 0:
                rate = np.log(e1x / e2x) / np.log(h1 / h2)
                print(f"  observed convergence rate (ux, last two meshes): "
                      f"{rate:.2f}")
        print()

    with open(os.path.join(OUT, 'nonlinear_meshconv.txt'), 'w', encoding='utf-8') as fp:
        fp.write("Nonlinear coupled solve: grid convergence\n")
        for r in rows:
            fp.write(f"nF={r[0]} ny={r[1]} ux_err={r[2]:.4%} uy_err={r[3]:.4%} "
                     f"balance={r[4]:.2e}\n")


if __name__ == '__main__':
    main()
