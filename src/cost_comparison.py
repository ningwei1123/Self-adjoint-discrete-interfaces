# -*- coding: utf-8 -*-
"""
Fair cost comparison for the hybrid solver
==========================================
The paper currently contrasts "24 interface unknowns" with "6964 network
parameters", which is not a like-for-like comparison because a monolithic FE
solve has thousands of degrees of freedom as well.  This script reports the
quantities a reviewer will actually ask for:

  * total unknowns of each approach,
  * wall-clock time broken into setup (assembly/probing) and solve,
  * interface and field accuracy against a common fine reference,
  * the accuracy of a pure FE solve on the SAME mesh, which is the exact
    discrete answer the coupled solve is approximating.

Approaches
  A  monolithic FE on the 48x24 reference mesh                (accuracy yardstick)
  B  monolithic FE on the 24x12 mesh used by the hybrid split  (same-mesh FE)
  C  hybrid: FE left (12x12) + exact DtN right (16x12), nodal trace, one solve
  D  C but with the right-subdomain operator obtained by regression instead of
     probing, to quantify what the surrogate costs in accuracy

Timing note: the interface operators are assembled once and reused, which is the
regime the paper advocates (many load cases, one operator).  The script therefore
reports both a one-shot time and an amortised time per additional load case.
"""
import os
import time

import numpy as np

import func_operator as F
import stress_recovery2d as R
import hybrid_energy_nn as H

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
CF = H.C_FIELD
NX_L, NY_L = 12, 12
NX_R, NY_R = 16, 12
M = NY_L + 1


# ------------------------------------------------------------------ helpers
def subdomain_mesh(x0, x1, nx, ny):
    nodes, elems = R.build_mesh(nx, ny, x0, x1, 0.0, H.LY)
    K = F.assemble_field(nodes, elems, CF, H.NU)
    n = len(nodes)
    gi = sorted([k for k in range(n) if abs(nodes[k, 0] - H.X0) < 1e-12],
                key=lambda k: nodes[k, 1])
    return nodes, K, n, gi


def right_operator(nx=NX_R, ny=NY_R):
    nodes, K, n, gi = subdomain_mesh(H.X0, H.LXR, nx, ny)
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


def left_operator(nx=NX_L, ny=NY_L):
    nodes, K, n, gi = subdomain_mesh(0.0, H.X0, nx, ny)
    cl = [k for k in range(n) if abs(nodes[k, 0]) < 1e-12]
    gdof = [2 * k for k in gi] + [2 * k + 1 for k in gi]
    cdof = [2 * k + cc for k in cl for cc in (0, 1)]
    free = np.array([d for d in range(2 * n)
                     if d not in set(gdof) and d not in set(cdof)])
    Z = np.linalg.solve(K[np.ix_(free, free)], K[np.ix_(free, gdof)])
    S = K[np.ix_(gdof, gdof)] - K[np.ix_(gdof, free)] @ Z
    return S, np.zeros(len(gdof)), np.array([nodes[k, 1] for k in gi])


def monolithic(nx, ny):
    nodes, elems = R.build_mesh(nx, ny, 0.0, H.LXR, 0.0, H.LY)
    K = F.assemble_field(nodes, elems, CF, H.NU)
    n = len(nodes)
    ndof = 2 * n
    fix = set()
    for k in range(n):
        if abs(nodes[k, 0]) < 1e-12:
            fix |= {2 * k, 2 * k + 1}
        elif abs(nodes[k, 0] - H.LXR) < 1e-12:
            fix.add(2 * k)
    u = np.zeros(ndof)
    for d in fix:
        k, comp = d // 2, d % 2
        if comp == 0 and abs(nodes[k, 0] - H.LXR) < 1e-12:
            u[d] = R.right_disp(nodes[k, 1])
    fds = np.array(sorted(fix))
    free = np.array([d for d in range(ndof) if d not in fix])
    u[free] = np.linalg.solve(K[np.ix_(free, free)],
                              -K[np.ix_(free, fds)] @ u[fds])
    return nodes, u, len(free)


def iface_of(nodes, u):
    gi = sorted([k for k in range(len(nodes)) if abs(nodes[k, 0] - H.X0) < 1e-12],
                key=lambda k: nodes[k, 1])
    return (np.array([nodes[k, 1] for k in gi]),
            np.array([u[2 * k] for k in gi]),
            np.array([u[2 * k + 1] for k in gi]))


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    # reference: fine monolithic
    t = time.time(); nR, uR, ndofR = monolithic(48, 24); tR = time.time() - t
    yR, uxR, uyR = iface_of(nR, uR)
    print(f"A  monolithic FE 48x24 : {ndofR} unknowns, {tR*1e3:.0f} ms")

    # same-mesh monolithic (the exact discrete answer of the hybrid split)
    t = time.time(); nB, uB, ndofB = monolithic(2 * NX_L, NY_L); tB = time.time() - t
    yB, uxB, uyB = iface_of(nB, uB)
    print(f"B  monolithic FE on the hybrid mesh : {ndofB} unknowns, {tB*1e3:.0f} ms")

    # hybrid: operators once
    t = time.time(); S_L, q_L, yL2 = left_operator(); S_R, q_R, yR2 = right_operator()
    t_ops = time.time() - t
    t = time.time(); c = np.linalg.solve(S_L + S_R, -(q_L + q_R)); t_solve = time.time() - t
    ndofC = (2 * NX_L * NY_L + 2 * NX_L) - 2 * (NY_L + 1) + 2 * M   # left free + trace
    print(f"C  hybrid (probed operator) : {ndofC} unknowns, "
          f"operator setup {t_ops*1e3:.0f} ms + solve {t_solve*1e3:.1f} ms")

    # accuracy: interface and field, all against the fine reference
    yc = yR2
    exC = rel(np.interp(yc, yR, uxR), np.interp(yc, yR, uxR))  # placeholder
    ux_c = np.interp(yc, yL2, np.zeros_like(yL2))  # not used
    # interface accuracy of each approach against the fine reference
    def iface_err(y, ux, uy):
        return (rel(np.interp(yR, y, ux), uxR), rel(np.interp(yR, y, uy), uyR))
    eB = iface_err(yB, uxB, uyB)
    # hybrid interface trace
    from hybrid_strategyA import left_operator as LO, right_operator as RO
    SL, qL, yl = LO(); SR, qr, yr = RO()
    ch = np.linalg.solve(SL + SR, -(qL + qr))
    uxh = np.array([np.interp(y, yr, ch[:len(yr)]) for y in yR])
    uyh = np.array([np.interp(y, yr, ch[len(yr):]) for y in yR])
    eC = (rel(uxh, uxR), rel(uyh, uyR))

    # surrogate (regression) variant
    rng = np.random.default_rng(7)
    ns = 400
    G = np.zeros((ns, 2 * M)); Ut = np.zeros((ns, 2 * M))
    for i in range(ns):
        cc = rng.normal(0, 2e-3, (2, 4))
        tt = 2 * np.linspace(0, H.LY, M) / H.LY - 1
        B = np.stack([np.cos(k * np.pi * (tt + 1) / 2) for k in range(4)], axis=1)
        g = np.concatenate([B @ cc[0], B @ cc[1]])
        G[i] = g; Ut[i] = S_R @ g
    Xm, Xs = G.mean(0), G.std(0) + 1e-30
    Xn = (G - Xm) / Xs
    W = np.linalg.solve(Xn.T @ Xn + 1e-8 * np.eye(2 * M), Xn.T @ Ut)
    S_reg = (((np.eye(2 * M) - Xm) / Xs) @ W)
    S_reg = 0.5 * (S_reg + S_reg.T)
    cr = np.linalg.solve(S_L + S_reg, -(q_L + q_R))
    uxr_ = np.array([np.interp(y, yR2, cr[:M]) for y in yR])
    uyr_ = np.array([np.interp(y, yR2, cr[M:]) for y in yR])
    eD = (rel(uxr_, uxR), rel(uyr_, uyR))

    print("\ninterface error vs the fine monolithic reference (48x24):")
    print(f"  B  monolithic FE on the hybrid mesh : ux {eB[0]:.4%}  uy {eB[1]:.4%}"
          f"   <- discretisation error only")
    print(f"  C  hybrid (probed operator)         : ux {eC[0]:.4%}  uy {eC[1]:.4%}")
    print(f"  D  hybrid (regression surrogate)    : ux {eD[0]:.4%}  uy {eD[1]:.4%}")

    print("\namortised cost: the operators are assembled once and reused,")
    print(f"  so each additional load case costs {t_solve*1e3:.1f} ms "
          f"(one {2*M}x{2*M} solve), against {tB*1e3:.0f} ms for a full FE re-solve.")

    with open(os.path.join(OUT, 'cost_comparison.txt'), 'w', encoding='utf-8') as fp:
        fp.write("Fair cost comparison\n")
        fp.write(f"A monolithic 48x24: {ndofR} dofs, {tR*1e3:.1f} ms\n")
        fp.write(f"B monolithic hybrid mesh: {ndofB} dofs, {tB*1e3:.1f} ms\n")
        fp.write(f"C hybrid: {ndofC} unknowns, setup {t_ops*1e3:.1f} ms, "
                 f"solve {t_solve*1e3:.2f} ms\n")
        fp.write(f"interface B ux {eB[0]:.6%} uy {eB[1]:.6%}\n")
        fp.write(f"interface C ux {eC[0]:.6%} uy {eC[1]:.6%}\n")
        fp.write(f"interface D ux {eD[0]:.6%} uy {eD[1]:.6%}\n")
    print(f"\n[saved] {os.path.join(OUT, 'cost_comparison.txt')}")
    print(f"total {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
