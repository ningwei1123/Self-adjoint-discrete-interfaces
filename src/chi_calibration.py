# -*- coding: utf-8 -*-
"""
Calibration of the compliance extraction against the LINEAR case
===============================================================
Before trusting any nonlinear chi_T, the same extraction must reproduce the
linear energy-matching law, for which the paper already has an independent
verification (theory_energy_matching.py, Table 1, u_Gamma/Delta = 1/(1+chi)).

Two constructions are compared here, deliberately, because they differ in how the
far end of the right subdomain is treated:

  (P) the paper's construction: the right subdomain is the polynomial feature
      space on [X0, Lx] with the far data imposed by a PENALTY rho at x = Lx, and
      with Tikhonov regularisation in its stiffness;
  (D) this work's construction: the far data are an EXACT Dirichlet condition
      (the x-component is prescribed, u_y is free), so the subdomain operator is
      a genuine Schur complement.

Both are evaluated with the SAME trace modes as the paper,
P_j(y) = ((2y/Ly) - 1)^j for j = 0..5, and the constant x-mode a = e_0.

Reported
  1. M_F, M_N and chi for construction (P), to be checked against Table 1;
  2. the same quantities for construction (D);
  3. whether the compliance so defined predicts the measured allocation of an
     actual coupled solve.

Run:  python chi_calibration.py
"""
import os
import time

import numpy as np

import func_operator as F
import stress_recovery2d as R
import coupled_solver as CS
import interface_methods as IM
import theory_energy_matching as TEM

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
LY, X0, LX = 1.0, 1.0, 2.0
NT = 6


def trace_modes(y):
    return np.array([(2.0 * y / LY - 1.0) ** j for j in range(NT)])


# ---------------------------------------------------------------- (P) paper
def paper_construction(rho=1e6, max_deg=5, fnx=12, fny=8):
    """Exactly theory_energy_matching.operators, re-derived so it can be reused."""
    d = TEM.operators(KF_scale=1.0, max_deg=max_deg, fnx=fnx, fny=fny)
    return d


def paper_measured_split(Delta=0.01):
    return TEM.observed_split_scaled(KF_scale=1.0, max_deg=5, Delta=Delta)


# ---------------------------------------------------------------- (D) exact
def exact_construction(d0=0.01, nx=12, ny=8):
    """Left Q4 + right Q4 with exact far Dirichlet; trace modes as in the paper."""
    # --- left subdomain, interface Schur complement
    nodesL, elemsL = R.build_mesh(nx, ny, 0.0, X0, 0.0, LY)
    KL = F.assemble_field(nodesL, elemsL, np.zeros(4), 0.30)
    nL = len(nodesL)
    gL = sorted([k for k in range(nL) if abs(nodesL[k, 0] - X0) < 1e-12],
                key=lambda k: nodesL[k, 1])
    cl = [k for k in range(nL) if abs(nodesL[k, 0]) < 1e-12]
    gdL = [2 * k for k in gL] + [2 * k + 1 for k in gL]
    cdL = [2 * k + c for k in cl for c in (0, 1)]
    freeL = np.array([d for d in range(2 * nL)
                      if d not in set(gdL) and d not in set(cdL)])
    SF = (KL[np.ix_(gdL, gdL)]
          - KL[np.ix_(gdL, freeL)] @ np.linalg.solve(KL[np.ix_(freeL, freeL)],
                                                     KL[np.ix_(freeL, gdL)]))

    # --- right subdomain, exact u_x at x = LX
    nodesR, elemsR = R.build_mesh(nx, ny, X0, LX, 0.0, LY)
    KR = F.assemble_field(nodesR, elemsR, np.zeros(4), 0.30)
    nR = len(nodesR)
    gR = sorted([k for k in range(nR) if abs(nodesR[k, 0] - X0) < 1e-12],
                key=lambda k: nodesR[k, 1])
    fi = [k for k in range(nR) if abs(nodesR[k, 0] - LX) < 1e-12]
    gdR = [2 * k for k in gR] + [2 * k + 1 for k in gR]
    ddR = [2 * k for k in fi]
    freeR = np.array([d for d in range(2 * nR)
                      if d not in set(gdR) and d not in set(ddR)])
    SR = (KR[np.ix_(gdR, gdR)]
          - KR[np.ix_(gdR, freeR)] @ np.linalg.solve(KR[np.ix_(freeR, freeR)],
                                                     KR[np.ix_(freeR, gdR)]))

    def compliance(S, nodes_i, g_i):
        """m = a^T B S^-1 B^T a with B the L2 projection onto the trace modes.

        B maps nodal interface values to the mode coefficients through the mass
        matrix of the trace basis, exactly as theory_energy_matching builds it.
        """
        gp, gw = np.polynomial.legendre.leggauss(80)
        ss = LY * (gp + 1.0) / 2.0
        ws = LY / 2.0 * gw
        yk = np.array([nodes_i[k, 1] for k in g_i])
        m = len(g_i)
        h = LY / (m - 1)
        Nm = np.zeros((len(ss), m))
        for k in range(m - 1):
            sel = (ss >= yk[k] - 1e-12) & (ss <= yk[k + 1] + 1e-12)
            t = (ss[sel] - yk[k]) / h
            Nm[sel, k] += 1 - t
            Nm[sel, k + 1] += t
        P = trace_modes(ss)
        B = np.zeros((2 * NT, 2 * m))
        for j in range(NT):
            for k in range(m):
                B[j, k] = np.sum(ws * Nm[:, k] * P[j])
                B[NT + j, m + k] = np.sum(ws * Nm[:, k] * P[j])
        M = B @ np.linalg.solve(S, B.T)
        a = np.zeros(2 * NT); a[0] = 1.0
        return float(a @ M @ a)

    mF = compliance(SF, nodesL, gL)
    mN = compliance(SR, nodesR, gR)
    return dict(m_F=mF, m_N=mN, chi=mN / mF)


def main():
    t0 = time.time()
    print("=" * 74)
    print("Calibration of the compliance extraction on the LINEAR problem")
    print("=" * 74)

    print("\n[1] construction (P): the paper's own verification script")
    try:
        dP = paper_construction()
        print(f"    m_F = {dP['m_F']:.6e}   m_N = {dP['m_N']:.6e}"
              f"   chi = {dP['chi']:.6f}   1/(1+chi) = {1/(1+dP['chi']):.6f}")
    except Exception as exc:
        print(f"    *** failed: {exc!r} ***")

    print("\n[2] construction (D): exact far Dirichlet, same trace modes")
    dD = exact_construction()
    print(f"    m_F = {dD['m_F']:.6e}   m_N = {dD['m_N']:.6e}"
          f"   chi = {dD['chi']:.6f}   1/(1+chi) = {1/(1+dD['chi']):.6f}")

    print("\n[3] measured allocation from an actual coupled solve (paper's method)")
    try:
        meas = paper_measured_split(Delta=0.01)
        print(f"    measured u_Gamma/Delta = {meas}")
    except Exception as exc:
        print(f"    *** failed: {exc!r} ***")

    print(f"\ntotal {time.time()-t0:.1f}s")
    with open(os.path.join(OUT, 'chi_calibration.txt'), 'w', encoding='utf-8') as fp:
        fp.write("Compliance extraction calibration (linear)\n")
        fp.write(f"(P) paper   : m_F={dP['m_F']:.6e} m_N={dP['m_N']:.6e} "
                 f"chi={dP['chi']:.6f} 1/(1+chi)={1/(1+dP['chi']):.6f}\n")
        fp.write(f"(D) exact   : m_F={dD['m_F']:.6e} m_N={dD['m_N']:.6e} "
                 f"chi={dD['chi']:.6f} 1/(1+chi)={1/(1+dD['chi']):.6f}\n")
    print(f"[saved] {os.path.join(OUT, 'chi_calibration.txt')}")


if __name__ == '__main__':
    main()
