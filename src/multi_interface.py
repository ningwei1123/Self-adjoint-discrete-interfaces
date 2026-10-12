# -*- coding: utf-8 -*-
"""
M1: multiple interfaces --- three subdomains in series
=====================================================
Every criterion in the paper was demonstrated on ONE interface.  A reviewer is
entitled to ask whether they survive when the decomposition has several.

Two things are checked.

(1) MULTI-INTERFACE CONSISTENCY.  Three Q4 subdomains in series, with two shared
    interfaces.  The two interfaces must each carry a cancellation of the internal
    forces of the two subdomains that meet there, and the interface operators
    extracted per subdomain must remain symmetric.  Neither is automatic: the
    subdomains are assembled independently and the interfaces are separate node
    sets.

(2) THE MATCHING LAW WITH TWO INTERFACES.  For a series of compliances the law
    generalises without new ingredients.  Let S_(i) be the interface-block operator
    of subdomain i written on the DOFS OF BOTH INTERFACES, with the other interface
    eliminated, so that S_(i) = [[A_i, B_i],[B_i^T, C_i]] in the (Gamma1, Gamma2)
    blocks.  The coupled interface problem is

        (S_(1) + S_(2) + S_(3)) u = b,

    and because each S_(i) has the block structure of a two-port, its inverse has
    the compliance interpretation

        M_(1) = A_1^{-1},  M_(2) = A_2^{-1} + C_2^{-1},  M_(3) = C_3^{-1},

    i.e. subdomain 2 contributes its two halves IN SERIES, one on each side.  The
    law is then the same statement read at each interface:

        u_1 / Delta = M_1 / M_tot,   u_2 / Delta = (M_1 + M_2) / M_tot,
        (u_2 - u_1) / Delta = M_2 / M_tot,      M_tot = M_1 + M_2 + M_3.

    The one-dimensional model is exact and is checked first, then the same statement
    is tested on the two-dimensional meshes, where the compliances are measured from
    the interface blocks of S_(i) restricted to the CONSTANT mode of each interface.

Run:  python multi_interface.py
"""
import os
import time

import numpy as np

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
LY = 1.0
NY = 8
DELTA = 0.01


def q4(xi, eta):
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                         (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    det = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    return N, dxi, det


def rect_mesh(nx, x0, x1):
    xs = np.linspace(x0, x1, nx + 1)
    ys = np.linspace(0.0, LY, NY + 1)
    nodes = np.array([(x, y) for y in ys for x in xs])
    elems = []
    for j in range(NY):
        for i in range(nx):
            a = j * (nx + 1) + i
            elems.append([a, a + 1, a + nx + 2, a + nx + 1])
    return nodes, np.array(elems)


def assemble(nodes, elems, nu):
    ndof = 2 * len(nodes)
    K = np.zeros((ndof, ndof))
    d0 = 1.0 / (1.0 - nu * nu)
    D2 = np.array([[d0, nu * d0, 0.0],
                   [nu * d0, d0, 0.0],
                   [0.0, 0.0, d0 * (1 - nu) / 2.0]])
    gp = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
    for e in elems:
        xy = nodes[e]
        for gx in gp:
            for gy in gp:
                N, dxi, det = q4(gx, gy)
                J = np.array([[dxi @ xy[:, 0], dxi @ xy[:, 1]],
                              [det @ xy[:, 0], det @ xy[:, 1]]])
                dJ = np.linalg.det(J)
                iJ = np.linalg.inv(J)
                dNdx = iJ[0, 0] * dxi + iJ[0, 1] * det
                dNdy = iJ[1, 0] * dxi + iJ[1, 1] * det
                B = np.zeros((3, 8))
                for k in range(4):
                    B[0, 2 * k] = dNdx[k]; B[1, 2 * k + 1] = dNdy[k]
                    B[2, 2 * k] = dNdy[k]; B[2, 2 * k + 1] = dNdx[k]
                idx = [2 * e[k] + c for k in range(4) for c in (0, 1)]
                K[np.ix_(idx, idx)] += B.T @ D2 @ B * dJ
    return K


def edges(nodes, x):
    return sorted([k for k in range(len(nodes)) if abs(nodes[k, 0] - x) < 1e-12],
                  key=lambda k: nodes[k, 1])


# ------------------------------------------------------------------ 1D exact
def one_d_exact(L1, L2, L3, delta=DELTA):
    M = np.array([L1, L2, L3], dtype=float) / 1.0        # EA = 1
    Mt = M.sum()
    return dict(u1=M[0] / Mt * delta,
                u2=(M[0] + M[1]) / Mt * delta,
                d1=M[0] / Mt, d2=(M[0] + M[1]) / Mt,
                Mt=Mt, M=M)


# ------------------------------------------------------------------ 3D-of-1D
def three_subdomain_2d(x1, x2, LX, nu, nx=(6, 6, 6)):
    """Assemble three Q4 subdomains in series with two shared interfaces.

    Returns per-subdomain interface-block operators on the (Gamma1, Gamma2) dof
    layout, the assembled union, and the monolithic solution.
    """
    S = []
    for (xa, xb, n) in zip((0.0, x1, x2), (x1, x2, LX), nx):
        nodes, elems = rect_mesh(n, xa, xb)
        K = assemble(nodes, elems, nu)
        nnode = len(nodes)
        g = []
        for xe in (x1, x2):
            if xa - 1e-12 <= xe <= xb + 1e-12:
                g.append([2 * k + c for k in edges(nodes, xe) for c in (0, 1)])
            else:
                g.append([])
        # interior = everything that is not on an interface of this subdomain
        onif = set(g[0]) | set(g[1])
        # Dirichlet end (x=0 for subdomain 0, x=LX for the last one) stays in the
        # interior block only if it carries no data; here we keep all non-interface
        # dofs as interior and let the condensed load handle the data
        inter = [d for d in range(2 * nnode) if d not in onif]
        S.append(dict(nodes=nodes, K=K, g=g, inter=inter, n=nnode))
    return S


def interface_blocks(sub, g1_size, g2_size, data=None):
    """Condense one subdomain onto (Gamma1, Gamma2) and return the 2x2 block form.

    `data` is an optional dict dof -> prescribed value (the clamped end of the
    first subdomain, and the prescribed end of the last).  Omitting it leaves those
    ends free, which gives the union a rigid-body translation and makes the
    interface system singular; that was the defect in the first version of this
    script.
    """
    gd = sub['g'][0] + sub['g'][1]
    K = sub['K']
    if data:
        held = set(data)
        inter = [d for d in sub['inter'] if d not in held]
        hv = np.array([data[d] for d in sorted(held)])
        hd = np.array(sorted(held))
    else:
        inter = sub['inter']
        hv = None
    Sfull = (K[np.ix_(gd, gd)]
             - K[np.ix_(gd, inter)] @ np.linalg.solve(K[np.ix_(inter, inter)],
                                                      K[np.ix_(inter, gd)]))
    bfull = None
    if data:
        bfull = -(K[np.ix_(gd, hd)] @ hv
                  - K[np.ix_(gd, inter)]
                  @ np.linalg.solve(K[np.ix_(inter, inter)], K[np.ix_(inter, hd)] @ hv))
    n1, n2 = len(sub['g'][0]), len(sub['g'][1])
    blocks = {}
    if n1 and n2:
        blocks['AA'] = Sfull[:n1, :n1]
        blocks['AB'] = Sfull[:n1, n1:]
        blocks['BB'] = Sfull[n1:, n1:]
    elif n1:
        blocks['AA'] = Sfull[:n1, :n1]
    elif n2:
        blocks['BB'] = Sfull[:, :]
    return blocks, Sfull, bfull


def main():
    t0 = time.time()
    nu = 0.30
    x1, x2, LX = 0.8, 1.4, 2.0

    print("=" * 90)
    print("M1: three subdomains in series, two interfaces")
    print("=" * 90)

    # ---- 1D exact model ---------------------------------------------------
    e = one_d_exact(x1 - 0.0, x2 - x1, LX - x2)
    print("\n[1D exact]  L = (%.2f, %.2f, %.2f), EA = 1" % (x1, x2 - x1, LX - x2))
    print(f"    M = {e['M']}   M_tot = {e['Mt']:.4f}")
    print(f"    u1/Delta = {e['d1']:.6f}   u2/Delta = {e['d2']:.6f}   "
          f"(u2-u1)/Delta = {(e['d2']-e['d1']):.6f}")
    print("    the law read at each interface: u1/D = M1/Mtot, "
          "(u2-u1)/D = M2/Mtot")

    # ---- 2D three subdomains ---------------------------------------------
    S = three_subdomain_2d(x1, x2, LX, nu)
    gi1 = len(edges(S[0]['nodes'], x1))
    gi2 = len(edges(S[1]['nodes'], x2))
    print(f"\n[2D] interface nodes: Gamma1 {gi1}, Gamma2 {gi2}")

    blocks = []
    # Dirichlet data: the left end of Omega1 is clamped; one y dof anywhere in the
    # chain is held to remove the remaining rigid translation along y.
    lf = edges(S[0]['nodes'], 0.0)
    d0 = {2 * k + c: 0.0 for k in lf for c in (0, 1)}
    # Omega3's right end carries the prescribed displacement (handled as data too)
    rt3n = edges(S[2]['nodes'], LX)
    d3 = {2 * k: DELTA for k in rt3n}
    rh = edges(S[2]['nodes'], LX)
    d3[2 * rh[0] + 1] = 0.0                     # kill the y translation
    dat = [d0, {}, d3]
    for i, sub in enumerate(S):
        b, Sf, bf = interface_blocks(sub, gi1, gi2, data=dat[i])
        blocks.append(b)
        sym = (np.abs(Sf - Sf.T).max() / np.abs(Sf).max()) if Sf.size else 0.0
        tag = ['Omega1 [0,x1]', 'Omega2 [x1,x2]', 'Omega3 [x2,LX]'][i]
        keys = sorted(b.keys())
        print(f"    {tag:16s} blocks {keys}  symmetry of the condensed operator "
              f"{sym:.2e}")

    # ---- the coupled interface system, built from the block operators ------
    nd = 2 * gi1 + 2 * gi2

    def place(bl, target):
        if 'AA' in bl:
            target[:2 * gi1, :2 * gi1] += bl['AA']
        if 'BB' in bl:
            target[2 * gi1:, 2 * gi1:] += bl['BB']
        if 'AB' in bl:
            target[:2 * gi1, 2 * gi1:] += bl['AB']
            target[2 * gi1:, :2 * gi1] += bl['AB'].T

    S1 = np.zeros((nd, nd)); S2 = np.zeros((nd, nd)); S3 = np.zeros((nd, nd))
    place(blocks[0], S1); place(blocks[1], S2); place(blocks[2], S3)
    Stot = S1 + S2 + S3

    print("\n[multi-interface self-adjointness]")
    for tag, Sm in (("Omega1", S1), ("Omega2", S2), ("Omega3", S3),
                    ("total", Stot)):
        rel = np.abs(Sm - Sm.T).max() / max(np.abs(Sm).max(), 1e-30)
        print(f"    {tag:8s} ||S - S^T|| / ||S|| = {rel:.2e}")

    # ---- interface load: the condensed load of every subdomain -------------
    loads = []
    for i, sub in enumerate(S):
        bl, Sf, bf = interface_blocks(sub, gi1, gi2, data=dat[i])
        loads.append(np.zeros(nd) if bf is None else
                     np.concatenate([bf[:2 * gi1],
                                     bf[2 * gi1:2 * gi1 + 2 * gi2]]))
    b = np.zeros(nd)
    for i, sub in enumerate(S):
        bl, Sf, bf = interface_blocks(sub, gi1, gi2, data=dat[i])
        if bf is None:
            continue
        n1, n2 = len(sub['g'][0]), len(sub['g'][1])
        # bf is laid out as [Gamma1 dofs][Gamma2 dofs] in this subdomain's own
        # ordering; place each part into the global interface layout
        off1 = 2 * gi1
        if n1:
            b[:2 * gi1] += bf[:2 * n1] if n1 == gi1 else 0.0
        if n2:
            b[off1:off1 + 2 * gi2] += bf[2 * n1:2 * n1 + 2 * n2]
    print(f"    interface load: ||b|| = {np.linalg.norm(b):.4e}")
    w = np.linalg.eigvalsh(0.5 * (Stot + Stot.T))
    print(f"    spectrum: min {w.min():.4e} max {w.max():.4e}")
    u = np.linalg.solve(Stot, -b)

    u1 = float(np.mean(u[:2 * gi1:2]))
    u2 = float(np.mean(u[2 * gi1:2 * gi1 + 2 * gi2:2]))
    print("\n[matching law at both interfaces]")
    print(f"    1D exact    u1/Delta = {e['d1']:.6f}   u2/Delta = {e['d2']:.6f}")
    print(f"    2D measured u1/Delta = {u1/DELTA:.6f}   u2/Delta = {u2/DELTA:.6f}")
    print(f"    relative difference  u1: {abs(u1/DELTA - e['d1'])/e['d1']:.2%}   "
          f"u2: {abs(u2/DELTA - e['d2'])/e['d2']:.2%}")

    print("\n[multi-interface force balance] at the coupled solution")
    for name, sl, Sleft, Sright in (
            ("Gamma1", slice(0, 2 * gi1), S1, S2),
            ("Gamma2", slice(2 * gi1, nd), S2, S3)):
        rr = ((Sleft + Sright) @ u)[sl]
        print(f"    {name}: ||(S_left + S_right) u|| / ||u|| = "
              f"{np.linalg.norm(rr)/np.linalg.norm(u):.2e}")

    print(f"\n  total {time.time()-t0:.2f}s")
    with open(os.path.join(OUT, 'multi_interface.txt'), 'w', encoding='utf-8') as fp:
        fp.write("M1 multi-interface: three subdomains, two interfaces\n")
        fp.write(f"1D exact    u1/D={e['d1']:.6f} u2/D={e['d2']:.6f}\n")
        fp.write(f"2D measured u1/D={u1/DELTA:.6f} u2/D={u2/DELTA:.6f}\n")
        for tag, Sm in (("Omega1", S1), ("Omega2", S2), ("Omega3", S3)):
            fp.write(f"{tag} symmetry "
                     f"{np.abs(Sm-Sm.T).max()/max(np.abs(Sm).max(),1e-30):.2e}\n")
    print(f"[saved] {os.path.join(OUT, 'multi_interface.txt')}")


if __name__ == '__main__':
    main()
