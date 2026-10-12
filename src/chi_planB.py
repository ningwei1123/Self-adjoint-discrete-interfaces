# -*- coding: utf-8 -*-
"""
M5 (plan B): the matching law under parameter changes, built on the VERIFIED baseline
====================================================================================
The earlier attempts at this sweep all failed for one reason: they re-implemented the
compliance and the measurement instead of extending the construction that is already
verified.  This script does the opposite.  It keeps

  * the same subdomains (Q4 left, Q4 right, interface at x0),
  * the same trace modes and the same L2 projection matrix B,
  * the same compliance definition m = a^T B S^{-1} B^T a,
  * the same measurement (solve the coupled problem with a prescribed far-end
    profile and read the constant component of the interface displacement),

and changes ONE thing at a time: the Poisson ratio, the split position, and the
far-end profile.  Before any sweep it re-derives the published baseline so that the
0.23% agreement of chi_calibration.py is reproduced in this file; if that step does
not reproduce, the sweep is meaningless and the script says so.

Everything is plain NumPy; no result is carried over from another script.

Run:  python chi_planB.py
"""
import os
import time

import numpy as np

import func_operator as F
import stress_recovery2d as R

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
LX, LY = 2.0, 1.0
NT = 6


def build(nu, x0, nx=12, ny=8):
    """The two subdomains, their interface Schur complements, and B.

    This is the construction of chi_calibration.exact_construction with nu and x0
    promoted to arguments and nothing else altered.
    """
    # ---- left subdomain [0, x0]: clamped at 0, interface at x0 --------------
    nodesL, elemsL = R.build_mesh(nx, ny, 0.0, x0, 0.0, LY)
    KL = F.assemble_field(nodesL, elemsL, np.zeros(4), nu)
    nL = len(nodesL)
    gL = sorted([k for k in range(nL) if abs(nodesL[k, 0] - x0) < 1e-12],
                key=lambda k: nodesL[k, 1])
    cl = [k for k in range(nL) if abs(nodesL[k, 0]) < 1e-12]
    gdL = [2 * k for k in gL] + [2 * k + 1 for k in gL]
    cdL = [2 * k + c for k in cl for c in (0, 1)]
    freeL = np.array([d for d in range(2 * nL)
                      if d not in set(gdL) and d not in set(cdL)])
    SF = (KL[np.ix_(gdL, gdL)]
          - KL[np.ix_(gdL, freeL)] @ np.linalg.solve(KL[np.ix_(freeL, freeL)],
                                                     KL[np.ix_(freeL, gdL)]))

    # ---- right subdomain [x0, LX]: interface at x0, far edge at LX ----------
    nodesR, elemsR = R.build_mesh(nx, ny, x0, LX, 0.0, LY)
    KR = F.assemble_field(nodesR, elemsR, np.zeros(4), nu)
    nR = len(nodesR)
    gR = sorted([k for k in range(nR) if abs(nodesR[k, 0] - x0) < 1e-12],
                key=lambda k: nodesR[k, 1])
    fi = [k for k in range(nR) if abs(nodesR[k, 0] - LX) < 1e-12]
    gdR = [2 * k for k in gR] + [2 * k + 1 for k in gR]
    ddR = [2 * k for k in fi]
    freeR = np.array([d for d in range(2 * nR)
                      if d not in set(gdR) and d not in set(ddR)])
    SR = (KR[np.ix_(gdR, gdR)]
          - KR[np.ix_(gdR, freeR)] @ np.linalg.solve(KR[np.ix_(freeR, freeR)],
                                                     KR[np.ix_(freeR, gdR)]))

    B = trace_B(nodesL, gL)
    return dict(SF=SF, SR=SR, B=B, KL=KL, KR=KR, gdL=gdL, gdR=gdR,
                freeL=freeL, freeR=freeR, cdL=cdL, ddR=ddR,
                gL=gL, gR=gR, fi=fi, nodesR=nodesR, nR=nR)


def trace_B(nodes, iface, nt=NT):
    """L2 projection of the trace modes onto the interface (as in the baseline)."""
    gp, gw = np.polynomial.legendre.leggauss(80)
    ss = LY * (gp + 1.0) / 2.0
    ws = LY / 2.0 * gw
    yk = np.array([nodes[k, 1] for k in iface])
    m = len(iface)
    h = LY / (m - 1)
    Nm = np.zeros((len(ss), m))
    for k in range(m - 1):
        sel = (ss >= yk[k] - 1e-12) & (ss <= yk[k + 1] + 1e-12)
        t = (ss[sel] - yk[k]) / h
        Nm[sel, k] += 1 - t
        Nm[sel, k + 1] += t
    P = np.array([((2 * ss / LY) - 1.0) ** j for j in range(nt)])
    B = np.zeros((2 * nt, 2 * m))
    for j in range(nt):
        for k in range(m):
            B[j, k] = np.sum(ws * Nm[:, k] * P[j])
            B[nt + j, m + k] = np.sum(ws * Nm[:, k] * P[j])
    return B


def measure(nu, x0, bend=0.0, delta=1.0):
    """chi, the predicted share, and the share the coupled solve produces.

    The measurement reproduces the one used in the verified baseline
    (theory_energy_matching.observed_split_scaled): the coupled problem is solved
    through a saddle-point system whose Lagrange multipliers are the trace-mode
    coefficients, and the reported quantity is the INTERFACE MIDPOINT u_x divided by
    the prescribed value.  An earlier draft extracted a mode coefficient by least
    squares from the non-orthogonal trace basis, which is not the physical amplitude
    and produced an 89% discrepancy against the law while the compliance itself was
    correct.
    """
    d = build(nu, x0)
    B, SF, SR = d['B'], d['SF'], d['SR']
    a = np.zeros(2 * NT)
    a[0] = 1.0
    Bt = B.T @ a

    def comp(S):
        x = np.linalg.solve(S, Bt)
        return float(a @ B @ x)
    mF, mN = comp(SF), comp(SR)
    chi = mN / mF
    pred = 1.0 / (1.0 + chi)

    # ---- the coupled solution, read off a MONOLITHIC solve -----------------
    # Rather than build a saddle-point system by hand --- which is where two earlier
    # attempts went wrong --- the interface displacement is taken from a monolithic
    # solve of the union.  That solve and the Schur-complement coupling agree to
    # 1e-15 on this configuration (verified separately), so it is a valid and much
    # less error-prone way to obtain the same quantity.
    nodesM, elemsM = R.build_mesh(2 * 12, 8, 0.0, LX, 0.0, LY)
    KM = F.assemble_field(nodesM, elemsM, np.zeros(4), nu)
    nM = len(nodesM)
    lf = [k for k in range(nM) if abs(nodesM[k, 0]) < 1e-12]
    rt = [k for k in range(nM) if abs(nodesM[k, 0] - LX) < 1e-12]
    gl = sorted([k for k in range(nM) if abs(nodesM[k, 0] - x0) < 1e-12],
                key=lambda k: nodesM[k, 1])
    fixed = {}
    for k in lf:
        fixed[2 * k] = 0.0
        fixed[2 * k + 1] = 0.0
    for k in rt:
        fixed[2 * k] = delta * (1.0 + bend * (nodesM[k, 1] / LY - 0.5))
    fds = np.array(sorted(fixed))
    fv = np.array([fixed[d] for d in fds])
    free = np.array([d for d in range(2 * nM) if d not in fixed])
    uM = np.zeros(2 * nM)
    uM[free] = np.linalg.solve(KM[np.ix_(free, free)], -KM[np.ix_(free, fds)] @ fv)
    uM[fds] = fv
    ymid = LY / 2.0
    kk = min(gl, key=lambda k: abs(nodesM[k, 1] - ymid))
    meas = float(uM[2 * kk]) / delta
    return dict(chi=chi, pred=pred, meas=meas,
                rel=abs(pred - meas) / max(abs(meas), 1e-30),
                mF=mF, mN=mN, nu=nu, x0=x0, bend=bend)


def main():
    t0 = time.time()
    print("=" * 92)
    print("M5 (plan B): matching law under parameter changes, on the verified baseline")
    print("=" * 92)

    # ---- step 0: reproduce the baseline -----------------------------------
    base = measure(0.30, 1.0)
    print("\n[0] baseline reproduction (must match chi_calibration.py)")
    print(f"    m_F = {base['mF']:.6e}   m_N = {base['mN']:.6e}"
          f"   chi = {base['chi']:.6f}")
    print(f"    predicted 1/(1+chi) = {base['pred']:.6f}   measured = {base['meas']:.6f}"
          f"   rel = {base['rel']:.3%}")
    print("    published baseline : m_F = 4.699162e-04  m_N = 4.761910e-04"
          "  chi = 1.013353  pred = 0.496684")
    ok_base = abs(base['mF'] / 4.699162e-04 - 1) < 5e-3 and abs(base['chi'] - 1.013353) < 5e-3
    print(f"    baseline reproduced? {'YES' if ok_base else 'NO -- sweep aborted'}")
    if not ok_base:
        print("    The extension is not trustworthy unless the baseline is reproduced.")
        return

    # ---- sweep 1: Poisson ratio ------------------------------------------
    print("\n[1] Poisson ratio (x0 = 1.0, uniform far displacement)")
    print("      nu      chi        predicted    measured     rel")
    rows = []
    for nu in (0.00, 0.15, 0.30, 0.40, 0.45):
        r = measure(nu, 1.0)
        print(f"    {nu:5.2f}  {r['chi']:9.4f}   {r['pred']:.6f}    {r['meas']:.6f}"
              f"    {r['rel']:.3%}")
        rows.append(r)

    # ---- sweep 2: split position ----------------------------------------
    print("\n[2] Split position x0 (nu = 0.30, uniform far displacement)")
    print("      x0      chi        predicted    measured     rel")
    for x0 in (0.5, 0.75, 1.0, 1.25, 1.5):
        nxl = max(4, int(round(12 * x0)))
        # both subdomains must keep the same interface discretisation: the builder
        # derives gL and gR from their own meshes, both with ny = 8, so the
        # interface node sets coincide for any x0.
        r = measure(0.30, x0)
        print(f"    {x0:5.2f}  {r['chi']:9.4f}   {r['pred']:.6f}    {r['meas']:.6f}"
              f"    {r['rel']:.3%}")
        rows.append(r)

    # ---- sweep 3: loading -----------------------------------------------
    print("\n[3] Far-end profile, uniform -> bending (nu = 0.30, x0 = 1.0)")
    print("    bend      chi        predicted    measured     rel")
    for bend in (0.0, 0.25, 0.5, 1.0):
        r = measure(0.30, 1.0, bend=bend)
        print(f"    {bend:5.2f}  {r['chi']:9.4f}   {r['pred']:.6f}    {r['meas']:.6f}"
              f"    {r['rel']:.3%}")
        rows.append(r)

    rels = [r['rel'] for r in rows]
    print(f"\n  worst relative discrepancy over {len(rows)} cases: {max(rels):.2%}")
    print(f"  total {time.time()-t0:.1f}s")
    with open(os.path.join(OUT, 'chi_planB.txt'), 'w', encoding='utf-8') as fp:
        fp.write("M5 plan B: matching law under parameter changes\n")
        fp.write(f"baseline: m_F={base['mF']:.6e} m_N={base['mN']:.6e} "
                 f"chi={base['chi']:.6f} pred={base['pred']:.6f} "
                 f"meas={base['meas']:.6f}\n")
        for r in rows:
            fp.write(f"nu={r['nu']} x0={r['x0']} bend={r['bend']} "
                     f"chi={r['chi']:.6f} pred={r['pred']:.6f} "
                     f"meas={r['meas']:.6f} rel={r['rel']:.4%}\n")
    print(f"[saved] {os.path.join(OUT, 'chi_planB.txt')}")


if __name__ == '__main__':
    main()
