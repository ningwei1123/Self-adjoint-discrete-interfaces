# -*- coding: utf-8 -*-
"""
Nonlinear two-subdomain coupling in 2D (checklist item 1)
========================================================
Two Q4 subdomains share ONE set of unknowns on the interface x = X0, so the
coupled problem is a single Newton solve of the union:

    Omega_F = [0, X0]  : u = 0 at x = 0
    Omega_N = [X0, LX] : u_x = g(y) at x = LX, u_y free
    Gamma              : shared unknowns (strong displacement continuity)

Implementation note: rather than juggling two independent numbering schemes with
coupling blocks (which is where the first draft of this script went wrong), every
global unknown is derived from one table,

    dof(key, comp) = 2 * gid(key) + comp,

where the interface keys are shared between the subdomains.  A subdomain then
assembles by mapping its own local numbering into the global one, and the tangent
of the union is the plain sum of the element contributions -- no coupling blocks
to get wrong.

The quantity of interest is the interface force balance.  Because the two
subdomains are assembled independently, cancellation of their internal forces on
the shared interface nodes is NOT automatic; it is the nonlinear analogue of the
linear interface-equilibrium residual, and it must be verified.

Run:  python nonlinear_coupled2d.py
"""
import os
import time

import numpy as np

import nonlinear_2d_final as N

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
LX, LY, X0 = N.LX, N.LY, N.X0


def subdomain(nx, ny, x0, x1):
    nodes, elems = N.build_mesh(nx, ny, x0, x1)
    geom = [N.elem_geom(nodes[e]) for e in elems]
    return nodes, elems, geom


def global_ids(nodesL, nodesR):
    """One id per unique NODE; interface nodes are shared.

    Keys are the rounded (x, y) coordinates, so a node appearing in both
    subdomains maps to a single unknown.  This makes the total number of dofs an
    output of the assignment rather than a number computed separately, which is
    what went wrong in the first draft.
    """
    keymap = {}          # (x,y) -> global node id
    gid = {}             # (side, local index) -> global node id
    for side, nodes in (('L', nodesL), ('R', nodesR)):
        for k, (x, y) in enumerate(nodes):
            key = (round(float(x), 10), round(float(y), 10))
            if key not in keymap:
                keymap[key] = len(keymap)
            gid[(side, k)] = keymap[key]
    return gid, 2 * len(keymap)


def assemble(nodesL, elemsL, geomL, nodesR, elemsR, geomR, gid, ndof, u):
    G = np.zeros(ndof)
    K = np.zeros((ndof, ndof))
    for side, nodes, elems, geom in (('L', nodesL, elemsL, geomL),
                                     ('R', nodesR, elemsR, geomR)):
        for ei, e in enumerate(elems):
            gi = [gid[(side, int(k))] for k in e]
            gdof = [2 * g + c for g in gi for c in (0, 1)]
            ue = u[gdof]
            ge = N.elem_grad_cached(ue, geom[ei])
            Ke = N.elem_hess_cached(ue, geom[ei])
            G[gdof] += ge
            K[np.ix_(gdof, gdof)] += Ke
    return G, K


def subdomain_forces(nodes, elems, geom, side, gid, u, ndof):
    """Internal force vector of ONE subdomain in the global numbering."""
    G = np.zeros(ndof)
    for ei, e in enumerate(elems):
        gi = [gid[(side, int(k))] for k in e]
        gdof = [2 * g + c for g in gi for c in (0, 1)]
        G[gdof] += N.elem_grad_cached(u[gdof], geom[ei])
    return G


def main():
    t0 = time.time()
    nF, nR, ny = 8, 8, 4
    nodesL, elemsL, geomL = subdomain(nF, ny, 0.0, X0)
    nodesR, elemsR, geomR = subdomain(nR, ny, X0, LX)
    gid, ndof = global_ids(nodesL, nodesR)

    iface_keys = sorted({round(y, 12) for k, (x, y) in enumerate(nodesL)
                         if abs(x - X0) < 1e-12})
    print("=" * 74)
    print("Nonlinear two-subdomain coupling in 2D")
    print("=" * 74)
    print(f"left {2*nF}x{ny}, right {2*nR}x{ny}, interface nodes "
          f"{len(iface_keys)}, global dofs {ndof}")

    # boundary conditions
    cl = [k for k in range(len(nodesL)) if abs(nodesL[k, 0]) < 1e-12]
    ri = [k for k in range(len(nodesR)) if abs(nodesR[k, 0] - LX) < 1e-12]
    fixed = {}
    for k in cl:
        fixed[2 * gid[('L', k)] + 0] = 0.0
        fixed[2 * gid[('L', k)] + 1] = 0.0
    for k in ri:
        fixed[2 * gid[('R', k)] + 0] = float(N.right_disp(nodesR[k, 1]))
    free = np.array([d for d in range(ndof) if d not in fixed])

    u = np.zeros(ndof)
    for d, v in fixed.items():
        u[d] = v

    print("\n[C] coupled Newton convergence")
    hist = []
    for it in range(40):
        G, K = assemble(nodesL, elemsL, geomL, nodesR, elemsR, geomR,
                        gid, ndof, u)
        r = G[free]
        hist.append(np.linalg.norm(r))
        print(f"      iter {it}: |r| = {hist[-1]:.4e}")
        if hist[-1] < 1e-9:
            break
        u[free] -= np.linalg.solve(K[np.ix_(free, free)], r)

    # ---- [A] interface force balance ------------------------------------
    GL = subdomain_forces(nodesL, elemsL, geomL, 'L', gid, u, ndof)
    GR = subdomain_forces(nodesR, elemsR, geomR, 'R', gid, u, ndof)
    gLf = sorted([k for k in range(len(nodesL)) if abs(nodesL[k, 0] - X0) < 1e-12],
                 key=lambda k: nodesL[k, 1])
    gRf = sorted([k for k in range(len(nodesR)) if abs(nodesR[k, 0] - X0) < 1e-12],
                 key=lambda k: nodesR[k, 1])
    rL = np.array([GL[2 * gid[('L', k)] + c] for k in gLf for c in (0, 1)])
    rR = np.array([GR[2 * gid[('R', k)] + c] for k in gRf for c in (0, 1)])
    bal = np.abs(rL + rR).max()
    rel = bal / max(np.abs(rL).max(), 1e-30)
    print("\n[A] interface force balance at the converged coupled state")
    print(f"      |r_F| max = {np.abs(rL).max():.3e}")
    print(f"      |r_N| max = {np.abs(rR).max():.3e}")
    print(f"      |r_F + r_N| max = {bal:.3e}   relative {rel:.3e}"
          f"   {'PASS' if rel < 1e-8 else 'CHECK'}")

    # ---- [B] monolithic reference ---------------------------------------
    nodesM, elemsM, geomM = subdomain(2 * nF + 2 * nR, ny, 0.0, LX)
    gidM, ndofM = global_ids(nodesM, np.zeros((0, 2)))
    clM = [k for k in range(len(nodesM)) if abs(nodesM[k, 0]) < 1e-12]
    riM = [k for k in range(len(nodesM)) if abs(nodesM[k, 0] - LX) < 1e-12]
    fixedM = {}
    for k in clM:
        fixedM[2 * gidM[('L', k)] + 0] = 0.0
        fixedM[2 * gidM[('L', k)] + 1] = 0.0
    for k in riM:
        fixedM[2 * gidM[('L', k)] + 0] = float(N.right_disp(nodesM[k, 1]))
    freeM = np.array([d for d in range(ndofM) if d not in fixedM])
    uM = np.zeros(ndofM)
    for d, v in fixedM.items():
        uM[d] = v
    for itM in range(40):
        GM, KM = assemble(nodesM, elemsM, geomM, np.zeros((0, 2)), [], [],
                          gidM, ndofM, uM)
        rM = GM[freeM]
        if np.linalg.norm(rM) < 1e-9:
            break
        uM[freeM] -= np.linalg.solve(KM[np.ix_(freeM, freeM)], rM)
    print(f"\n[B] monolithic reference ({2*nF+2*nR}x{ny}, {ndofM} dofs) "
          f"converged in {itM+1} iterations")

    gMc = sorted([k for k in range(len(nodesM)) if abs(nodesM[k, 0] - X0) < 1e-12],
                 key=lambda k: nodesM[k, 1])
    uxM = np.array([uM[2 * gidM[('L', k)]] for k in gMc])
    uyM = np.array([uM[2 * gidM[('L', k)] + 1] for k in gMc])
    uxL = np.array([u[2 * gid[('L', k)]] for k in gLf])
    uyL = np.array([u[2 * gid[('L', k)] + 1] for k in gLf])
    ex = np.linalg.norm(uxL - uxM) / np.linalg.norm(uxM)
    ey = np.linalg.norm(uyL - uyM) / np.linalg.norm(uyM)
    print(f"      coupled vs monolithic interface: ux {ex:.4%}   uy {ey:.4%}")

    # ---- [D] nonlinearity magnitude -------------------------------------
    print("\n[D] problem scales")
    print(f"      interface ux  {uxL.min():.6e} .. {uxL.max():.6e}")
    print(f"      interface uy  {uyL.min():.6e} .. {uyL.max():.6e}")
    print(f"      max|u| left subdomain {np.abs(u[:2*len(nodesL)]).max():.4e}")
    print(f"      Newton residuals {[f'{v:.2e}' for v in hist[:6]]}")

    print(f"\ntotal {time.time()-t0:.1f}s")
    with open(os.path.join(OUT, 'nonlinear_coupled2d.txt'), 'w', encoding='utf-8') as fp:
        fp.write("Nonlinear two-subdomain coupling in 2D\n")
        fp.write(f"global dofs {ndof}, interface nodes {len(iface_keys)}\n")
        fp.write(f"Newton residuals {hist[:8]}\n")
        fp.write(f"interface force balance |rL+rR|max {bal:.3e} rel {rel:.3e}\n")
        fp.write(f"coupled vs monolithic: ux {ex:.6%}  uy {ey:.6%}\n")
    print(f"[saved] {os.path.join(OUT, 'nonlinear_coupled2d.txt')}")


if __name__ == '__main__':
    main()
