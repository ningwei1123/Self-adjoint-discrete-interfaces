# -*- coding: utf-8 -*-
"""
Checklist item 4: performance of the nonlinear tangent machinery
===============================================================
Item 1-3 established that the energy-driven tangent works (symmetry 0.0, Newton
quadratic to 4e-13, interface force balance 6e-14, grid convergence rate 2.16).
Its cost, however, is dominated by the way the tangent is obtained:

    gradient : 8 complex-step energy evaluations per element, per call
    Hessian  : 2 x 8 gradient calls, i.e. 128 energy evaluations per element

and the Newton iteration calls that for every element on every step.  This script
(i) measures where the time goes, (ii) applies the cheap optimisations that do not
change any number, and (iii) verifies that the optimised path is bit-comparable
with the unoptimised one -- an optimisation that changes the answer is a new bug,
not an optimisation.

Optimisations applied (all exact)
  * geometry per element is computed once and reused (already done in item 1);
  * the per-element global dof index lists are built once;
  * the Hessian reuses one complex work array instead of allocating per column;
  * an element-level cache is avoided deliberately: it would not be hit, since u
    changes every Newton step.

Run:  python nonlinear_perf.py
"""
import os
import time

import numpy as np

import nonlinear_2d_final as N
import nonlinear_coupled2d as C

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))


def grad_opt(ue, geom, h=N.HC, work=None):
    """Complex-step gradient with a reused work array."""
    n = 8
    g = np.zeros(n)
    w = np.empty(n, dtype=complex) if work is None else work
    for j in range(n):
        w[:] = ue
        w[j] += 1j * h
        g[j] = N.elem_energy_cached(w, geom).imag / h
    return g


def hess_opt(ue, geom, h=N.HD, work=None):
    n = 8
    K = np.zeros((n, n))
    up = np.empty(n); um = np.empty(n)
    for j in range(n):
        up[:] = ue; um[:] = ue
        up[j] += h; um[j] -= h
        K[:, j] = (grad_opt(up, geom, work=work) - grad_opt(um, geom, work=work)) / (2.0 * h)
    return 0.5 * (K + K.T)


def assemble_opt(nodes, elems, geom, gid, ndof, u, side='L', want_hess=True,
                 idxlist=None):
    if idxlist is None:
        idxlist = [[2 * gid[(side, int(k))] + c for k in e for c in (0, 1)]
                   for e in elems]
    G = np.zeros(ndof)
    K = np.zeros((ndof, ndof)) if want_hess else None
    work = np.empty(8, dtype=complex)
    for ei, gdof in enumerate(idxlist):
        ue = u[gdof]
        G[gdof] += grad_opt(ue, geom[ei], work=work)
        if want_hess:
            K[np.ix_(gdof, gdof)] += hess_opt(ue, geom[ei], work=work)
    return G, K, idxlist


def main():
    t0 = time.time()
    nF = nR = 8
    ny = 4
    d0 = 0.01
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

    print("=" * 74)
    print("Performance of the energy-driven tangent")
    print("=" * 74)
    print(f"mesh: left {2*nF}x{ny}, right {2*nR}x{ny}; global dofs {ndof}")

    # ---- reference (item 1 path) ---------------------------------------
    t = time.time()
    G_ref, K_ref = C.assemble(nodesL, elemsL, geomL, nodesR, elemsR, geomR,
                              gid, ndof, u)
    t_ref = time.time() - t
    print(f"\n[1] reference assembly (C.assemble): {t_ref:.3f}s for one "
          f"gradient+Hessian")

    # ---- optimised path -------------------------------------------------
    t = time.time()
    G_opt, K_opt, idxlistL = assemble_opt(nodesL, elemsL, geomL, gid, ndof, u,
                                          side='L', want_hess=False)
    Gr_opt, _, idxlistR = assemble_opt(nodesR, elemsR, geomR, gid, ndof, u,
                                       side='R', want_hess=False)
    G_opt = G_opt + Gr_opt
    t_g = time.time() - t
    print(f"    optimised gradient only:            {t_g:.3f}s")

    # full Hessian on the free block only (that is all Newton needs)
    nfree = len(free)
    t = time.time()
    Kf = np.zeros((nfree, nfree))
    allidx = []
    for side, elems, geom, nodes in (('L', elemsL, geomL, nodesL),
                                     ('R', elemsR, geomR, nodesR)):
        for ei, e in enumerate(elems):
            allidx.append((side, ei, [2 * gid[(side, int(k))] + c
                                      for k in e for c in (0, 1)]))
    pos = {d: i for i, d in enumerate(free)}
    work = np.empty(8, dtype=complex)
    for side, ei, gdof in allidx:
        geom = geomL if side == 'L' else geomR
        ue = u[gdof]
        Ke = hess_opt(ue, geom[ei], work=work)
        loc = [pos[d] for d in gdof if d in pos]
        if loc:
            sub = [i for i, d in enumerate(gdof) if d in pos]
            Kf[np.ix_(loc, loc)] += Ke[np.ix_(sub, sub)]
    t_h = time.time() - t
    print(f"    optimised Hessian on free block:    {t_h:.3f}s")
    print(f"    total optimised:                    {t_g + t_h:.3f}s"
          f"   (reference {t_ref:.3f}s)")

    # ---- bit-comparability ---------------------------------------------
    dg = np.abs(G_opt - G_ref).max() / max(np.abs(G_ref).max(), 1e-30)
    dk = np.abs(Kf - K_ref[np.ix_(free, free)]).max() / max(np.abs(K_ref).max(), 1e-30)
    print(f"\n[2] optimised vs reference: gradient rel diff {dg:.3e}, "
          f"Hessian rel diff {dk:.3e}")
    print(f"    {'PASS (numerically identical)' if max(dg, dk) < 1e-12 else '*** DIFFERS ***'}")

    # ---- Newton with the optimised path, timed -------------------------
    print("\n[3] full coupled Newton with the optimised assembly")
    t = time.time()
    hist = []
    for it in range(40):
        G, _, _ = assemble_opt(nodesL, elemsL, geomL, gid, ndof, u,
                               side='L', want_hess=False, idxlist=idxlistL)
        Gr, _, _ = assemble_opt(nodesR, elemsR, geomR, gid, ndof, u,
                                side='R', want_hess=False, idxlist=idxlistR)
        # NOTE: the two subdomains must be assembled with their own id maps; the
        # shared ids make the sum correct, which is what item 1 verified.
        G = G + Gr
        r = G[free]
        hist.append(np.linalg.norm(r))
        if hist[-1] < 1e-9:
            break
        Kf = np.zeros((nfree, nfree))
        for side, elems, geom, nodes in (('L', elemsL, geomL, nodesL),
                                         ('R', elemsR, geomR, nodesR)):
            for ei, e in enumerate(elems):
                gdof = [2 * gid[(side, int(k))] + c for k in e for c in (0, 1)]
                Ke = hess_opt(u[gdof], geom[ei], work=work)
                loc = [pos[d] for d in gdof if d in pos]
                if loc:
                    sub = [i for i, d in enumerate(gdof) if d in pos]
                    Kf[np.ix_(loc, loc)] += Ke[np.ix_(sub, sub)]
        u[free] -= np.linalg.solve(Kf, r)
    t_newton = time.time() - t
    print(f"    converged in {len(hist)} iterations in {t_newton:.2f}s")
    print(f"    residuals {[f'{v:.2e}' for v in hist[:6]]}")

    with open(os.path.join(OUT, 'nonlinear_perf.txt'), 'w', encoding='utf-8') as fp:
        fp.write("Performance of the energy-driven tangent\n")
        fp.write(f"reference assembly {t_ref:.3f}s; optimised {t_g + t_h:.3f}s\n")
        fp.write(f"gradient rel diff {dg:.3e}, Hessian rel diff {dk:.3e}\n")
        fp.write(f"Newton {len(hist)} iterations in {t_newton:.2f}s\n")
    print(f"\n[saved] {os.path.join(OUT, 'nonlinear_perf.txt')}")
    print(f"total {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
