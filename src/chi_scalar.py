# -*- coding: utf-8 -*-
"""
M5 (final attempt): the matching law read as ONE scalar equation
================================================================
Every earlier attempt at this sweep went wrong in the same place: converting
between nodal interface values and a modal amplitude.  The trace basis is not
orthogonal, so the coefficient returned by a least-squares projection is not the
physical amplitude of the constant mode, and three successive normalisations
produced three different wrong numbers.

This version removes that conversion entirely.  Write b for the vector of nodal
values of the CONSTANT interface mode, i.e. b = (1,...,1) for the x-component and
zero for the y-component.  Then for a load that is itself a constant far-field
displacement,

    b^T (S_F + S_N) b  *  w   =   b^T (load vector)      =>   w = the amplitude,

and the compliance of subdomain i in this mode is

    M_i = (b^T b)^2 / (b^T S_i b . b^T b)  ->  more simply  k_i = b^T S_i b / b^T b,
    M_i = 1 / k_i .

No projection, no basis, no least squares: only the scalar b^T S b for each
subdomain and b^T b.  The law is then

    u_Gamma / Delta  =  M_F / (M_F + M_N)  =  k_N / (k_F + k_N) = 1 / (1 + chi),
    chi = M_N / M_F = k_F / k_N,

and the MEASURED ratio is read directly as the mean of the nodal interface
displacement divided by the imposed far-field value.  If the law holds, the two
agree without any further interpretation.

The only structural requirement is that the two subdomains share the same
interface nodes, which is enforced by construction below.

Run:  python chi_scalar.py
"""
import os
import time

import numpy as np

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
LX, LY = 2.0, 1.0
NY = 8


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


def case(nu=0.30, x0=1.0, nxF=8, nxN=8, delta=0.01):
    """Returns (chi, predicted ratio, measured ratio) with no modal projection."""
    # subdomain on [x0, LX]: interface at x0, far edge at LX
    nF, eF = rect_mesh(nxF, x0, LX)
    KF = assemble(nF, eF, nu)
    iF = edges(nF, x0)
    oF = edges(nF, LX)
    gF = [2 * k + c for k in iF for c in (0, 1)]
    hF = set(gF) | {2 * k + c for k in oF for c in (0, 1)}
    intF = [d for d in range(2 * len(nF)) if d not in hF]
    SF = (KF[np.ix_(gF, gF)]
          - KF[np.ix_(gF, intF)] @ np.linalg.solve(KF[np.ix_(intF, intF)],
                                                   KF[np.ix_(intF, gF)]))
    # subdomain on [0, x0]: interface at x0, clamped at 0
    nN, eN = rect_mesh(nxN, 0.0, x0)
    KN = assemble(nN, eN, nu)
    iN = edges(nN, x0)
    zN = edges(nN, 0.0)
    gN = [2 * k + c for k in iN for c in (0, 1)]
    hN = set(gN) | {2 * k + c for k in zN for c in (0, 1)}
    intN = [d for d in range(2 * len(nN)) if d not in hN]
    SN = (KN[np.ix_(gN, gN)]
          - KN[np.ix_(gN, intN)] @ np.linalg.solve(KN[np.ix_(intN, intN)],
                                                   KN[np.ix_(intN, gN)]))

    # b: nodal values of the CONSTANT interface mode (x-component only)
    m = len(iF)
    b = np.zeros(2 * m)
    b[:m] = 1.0
    nb2 = float(b @ b)

    kF = float(b @ (SF @ b)) / nb2
    kN = float(b @ (SN @ b)) / nb2
    MF, MN = 1.0 / kF, 1.0 / kN
    chi = MN / MF                     # = kF / kN
    pred = 1.0 / (1.0 + chi)

    # ---- coupled solve: constant far displacement, interface free ----------
    fixed = {}
    for k in oF:
        fixed[2 * k] = delta
    fds = np.array(sorted(fixed))
    fv = np.array([fixed[d] for d in fds])
    bF = (-KF[np.ix_(gF, fds)] @ fv
          - KF[np.ix_(gF, intF)]
          @ np.linalg.solve(KF[np.ix_(intF, intF)], KF[np.ix_(intF, fds)] @ fv))
    uG = np.linalg.solve(SF + SN, bF)
    ux = uG[:m]
    meas = float(np.mean(ux)) / delta
    return dict(chi=chi, pred=pred, meas=meas, kF=kF, kN=kN,
                rel=abs(pred - meas) / max(abs(meas), 1e-30),
                nu=nu, x0=x0)


def main():
    t0 = time.time()
    print("=" * 92)
    print("M5: matching law as one scalar equation (no modal projection)")
    print("=" * 92)

    print("\n(a) Poisson ratio (x0 = 1.0, constant far displacement)")
    print("      nu       chi        predicted   measured     rel")
    rows = []
    for nu in (0.00, 0.15, 0.30, 0.40, 0.45, 0.49):
        r = case(nu=nu)
        print(f"    {nu:5.2f}  {r['chi']:9.4f}   {r['pred']:.6f}   "
              f"{r['meas']:.6f}   {r['rel']:.3%}")
        rows.append((f"nu={nu}", r))

    print("\n(b) split position x0 (nu = 0.30)")
    print("      x0       chi        predicted   measured     rel")
    for x0 in (0.5, 0.75, 1.0, 1.25, 1.5):
        r = case(nu=0.30, x0=x0, nxF=max(2, int(8 * (LX - x0))),
                 nxN=max(2, int(8 * x0)))
        print(f"    {x0:5.2f}  {r['chi']:9.4f}   {r['pred']:.6f}   "
              f"{r['meas']:.6f}   {r['rel']:.3%}")
        rows.append((f"x0={x0}", r))

    print("\n(c) mesh refinement at nu = 0.30, x0 = 1.0")
    print("      n        chi        predicted   measured     rel")
    for n in (4, 8, 12, 16):
        r = case(nu=0.30, nxF=n, nxN=n)
        print(f"    {n:5d}  {r['chi']:9.4f}   {r['pred']:.6f}   "
              f"{r['meas']:.6f}   {r['rel']:.3%}")
        rows.append((f"n={n}", r))

    rels = [r['rel'] for _, r in rows]
    print(f"\n  worst relative discrepancy over {len(rows)} cases: {max(rels):.3%}")
    print(f"  total {time.time()-t0:.2f}s")
    with open(os.path.join(OUT, 'chi_scalar.txt'), 'w', encoding='utf-8') as fp:
        fp.write("M5: matching law as a scalar equation\n")
        for tag, r in rows:
            fp.write(f"{tag:10s} chi={r['chi']:.6f} pred={r['pred']:.6f} "
                     f"meas={r['meas']:.6f} rel={r['rel']:.4%}\n")
    print(f"[saved] {os.path.join(OUT, 'chi_scalar.txt')}")


if __name__ == '__main__':
    main()
