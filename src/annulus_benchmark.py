# -*- coding: utf-8 -*-
"""
M1a: curved interface (concentric annulus, polar mesh) --- audited version
=========================================================================
This script replaces two earlier drafts of the same benchmark.  Both drafts were
correct in their physics and wrong in their bookkeeping, and the errors are
recorded here because each of them produced a numerically confident but
meaningless number:

  E1. Element ordering.  A polar Q4 has to be traced around its perimeter as
      (r_i, th_j) -> (r_{i+1}, th_j) -> (r_{i+1}, th_{j+1}) -> (r_i, th_{j+1}).
      Any other order traces a self-intersecting "bowtie" whose Jacobian changes
      sign inside the element; the assembled stiffness is then indefinite (three
      zero eigenvalues became five negative plus three zero).
  E2. Interface load after condensation.  The correct right-hand side is
          b = -K_{G,D} u_D - K_{G,i} K_{ii}^{-1} K_{i,D} u_D.
      In an annulus the interface and the Dirichlet boundary are separated by a
      ring of interior dofs, so K_{G,D} = 0 and dropping the second term drives
      the interface solution to exactly zero.
  E3. Per-subdomain interior sets.  The two subdomains have different interior
      dof sets; a single global "free" set makes both Schur complements singular.
  E4. Loop-variable shadowing.  Naming a node loop variable `k` and later using
      `k` for the number of interface nodes silently corrupted the mode vector and
      flipped the sign of the reported matching ratio.

Geometry, boundary data and the exact reference
-----------------------------------------------
    Omega_N : r_in  <= r <= r_split      (surrogate subdomain)
    Omega_F : r_split < r <= r_out       (finite-element subdomain)
    r_in = 0.5,  r_split = 1.0,  r_out = 2.0,  plane stress, nu = 0.3

    u_r(r_out) = Delta,  sigma_rr(r_in) = 0
    => u_r(r) = c1 r + c2 / r,
       c1 = Delta r_out / (r_out^2 + r_in^2),  c2 = c1 r_in^2
    => u_r(r_split)/Delta = (r_split^2 + r_in^2) / (r_out^2 + r_in^2) = 0.588235

The constant interface trace is a genuine 2D elasticity property here (the exact
solution is radial and theta-independent), so the energy-matching law can be
tested without any mode-truncation ambiguity.

Reported
  1. interface force balance  |r_F + r_N| / |r_F|      (must be at machine level)
  2. symmetry of the tangent DtN operator S_N
  3. reconstruction of the monolithic solution
  4. the monolithic interface displacement against the exact value
  5. the matching ratio on the constant radial mode, chi_T and 1/(1 + chi_T)
  6. material-contrast sweep (M1b) to 1e4

Run:  python annulus_benchmark.py
"""
import os
import time

import numpy as np

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))

R_IN, R_SPLIT, R_OUT = 0.5, 1.0, 2.0
NU = 0.30
DELTA = 0.01
TOL_R = 1e-9


def q4(xi, eta):
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                         (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    det = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    return N, dxi, det


def polar_mesh(nt, nr_in, nr_out):
    rings = list(np.linspace(R_IN, R_SPLIT, nr_in + 1)[:-1]) + \
            list(np.linspace(R_SPLIT, R_OUT, nr_out + 1))
    thetas = np.linspace(0.0, 2.0 * np.pi, nt + 1)[:-1]
    nodes = np.array([(r * np.cos(t), r * np.sin(t)) for r in rings for t in thetas])
    nr = len(rings)
    elems = []
    for i in range(nr - 1):
        for j in range(nt):
            j2 = (j + 1) % nt
            elems.append([i * nt + j,                       # (r_i,   th_j)
                          (i + 1) * nt + j,                 # (r_i+1, th_j)
                          (i + 1) * nt + j2,                # (r_i+1, th_j+1)
                          i * nt + j2])                     # (r_i,   th_j+1)
    return nodes, np.array(elems), rings


def assemble(nodes, elems, young):
    ndof = 2 * len(nodes)
    K = np.zeros((ndof, ndof))
    gp = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
    for ei, e in enumerate(elems):
        xy = nodes[e]
        E = young(ei, xy)
        d0 = E / (1 - NU * NU)
        D2 = np.array([[d0, NU * d0, 0.0],
                       [NU * d0, d0, 0.0],
                       [0.0, 0.0, d0 * (1 - NU) / 2]])
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
                for a in range(4):
                    B[0, 2 * a] = dNdx[a]; B[1, 2 * a + 1] = dNdy[a]
                    B[2, 2 * a] = dNdy[a]; B[2, 2 * a + 1] = dNdx[a]
                idx = [2 * e[a] + c for a in range(4) for c in (0, 1)]
                K[np.ix_(idx, idx)] += B.T @ D2 @ B * dJ
    return K


def solve_dirichlet(K, fixed):
    ndof = K.shape[0]
    fds = np.array(sorted(fixed))
    fv = np.array([fixed[d] for d in fds])
    free = np.array([d for d in range(ndof) if d not in fixed])
    u = np.zeros(ndof)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], -K[np.ix_(free, fds)] @ fv)
    u[fds] = fv
    return u


def exact_ratio():
    return (R_SPLIT ** 2 + R_IN ** 2) / (R_OUT ** 2 + R_IN ** 2)


def run(nt=32, nr_in=4, nr_out=5, contrast=1.0, delta=DELTA, verbose=True):
    nodes, elems, rings = polar_mesh(nt, nr_in, nr_out)
    ndof = 2 * len(nodes)
    rad = np.hypot(nodes[:, 0], nodes[:, 1])

    def young(ei, xy):
        rc = np.hypot(xy[:, 0].mean(), xy[:, 1].mean())
        return contrast if rc <= R_SPLIT + 1e-12 else 1.0

    Emask = np.array([np.hypot(nodes[e, 0].mean(), nodes[e, 1].mean())
                      <= R_SPLIT + 1e-12 for e in elems])
    K_N = assemble(nodes, elems[Emask], young)
    K_F = assemble(nodes, elems[~Emask], young)
    K = K_N + K_F

    iface_nodes = sorted([nd for nd in range(len(nodes))
                          if abs(rad[nd] - R_SPLIT) < TOL_R])
    outer_nodes = sorted([nd for nd in range(len(nodes))
                          if abs(rad[nd] - R_OUT) < TOL_R])
    set_N = {int(nd) for e in elems[Emask] for nd in e}
    set_F = {int(nd) for e in elems[~Emask] for nd in e}
    gdof = [2 * nd + c for nd in iface_nodes for c in (0, 1)]
    iN = [2 * nd + c for nd in sorted(set_N - set(iface_nodes)) for c in (0, 1)]
    iF = [2 * nd + c for nd in sorted(set_F - set(iface_nodes) - set(outer_nodes))
          for c in (0, 1)]

    fixed = {}
    for nd in outer_nodes:
        th = np.arctan2(nodes[nd, 1], nodes[nd, 0])
        fixed[2 * nd] = delta * np.cos(th)
        fixed[2 * nd + 1] = delta * np.sin(th)
    fds = np.array(sorted(fixed))
    fv = np.array([fixed[d] for d in fds])

    uM = solve_dirichlet(K, fixed)

    def schur(Kd, interior):
        return (Kd[np.ix_(gdof, gdof)]
                - Kd[np.ix_(gdof, interior)]
                @ np.linalg.solve(Kd[np.ix_(interior, interior)],
                                  Kd[np.ix_(interior, gdof)]))

    def condensed_load(Kd, interior):
        return (-Kd[np.ix_(gdof, fds)] @ fv
                - Kd[np.ix_(gdof, interior)]
                @ np.linalg.solve(Kd[np.ix_(interior, interior)],
                                  Kd[np.ix_(interior, fds)] @ fv))

    S_F, S_N = schur(K_F, iF), schur(K_N, iN)
    bF, bN = condensed_load(K_F, iF), condensed_load(K_N, iN)
    uG = np.linalg.solve(S_F + S_N, -(bF + bN))

    u = np.zeros(ndof)
    u[fds] = fv
    u[gdof] = uG
    u[iN] = np.linalg.solve(K_N[np.ix_(iN, iN)],
                            -(K_N[np.ix_(iN, gdof)] @ uG + K_N[np.ix_(iN, fds)] @ fv))
    u[iF] = np.linalg.solve(K_F[np.ix_(iF, iF)],
                            -(K_F[np.ix_(iF, gdof)] @ uG + K_F[np.ix_(iF, fds)] @ fv))

    def iface_force(Kd, interior):
        return (Kd[np.ix_(gdof, gdof)] @ uG
                + Kd[np.ix_(gdof, interior)] @ u[interior]
                + Kd[np.ix_(gdof, fds)] @ fv)
    rF, rN = iface_force(K_F, iF), iface_force(K_N, iN)
    bal = np.abs(rF + rN).max() / max(np.abs(rF).max(), 1e-30)
    sym = np.abs(S_N - S_N.T).max() / np.abs(S_N).max()
    mono = np.linalg.norm(u[gdof] - uM[gdof]) / max(np.linalg.norm(uM[gdof]), 1e-30)

    # the constant radial mode
    mm = len(iface_nodes)
    th = np.arctan2(nodes[iface_nodes, 1], nodes[iface_nodes, 0])
    bvec = np.zeros(2 * mm)
    bvec[:mm] = np.cos(th)
    bvec[mm:] = np.sin(th)

    # modal stiffness: drive the interface by b, far boundary FREE, rigid modes
    # removed by a least-squares solve; M = b.b / (b . K . u)
    def modal_compliance(Kd, pins_nodes):
        fixedm = dict(zip(gdof, bvec.tolist()))
        fdsm = np.array(sorted(fixedm))
        fvm = np.array([fixedm[d] for d in fdsm])
        hold = [2 * pins_nodes[0], 2 * pins_nodes[0] + 1, 2 * pins_nodes[1]]
        free = np.array([d for d in range(ndof)
                         if d not in fixedm and d not in set(hold)])
        fv_hold = np.zeros(ndof)
        rhs = -Kd[np.ix_(free, fdsm)] @ fvm - Kd[np.ix_(free, hold)] @ fv_hold[hold]
        uu = np.zeros(ndof)
        uu[fdsm] = fvm
        uu[free] = np.linalg.lstsq(Kd[np.ix_(free, free)], rhs, rcond=1e-13)[0]
        F = float(bvec @ (Kd @ uu)[gdof])
        return float(bvec @ bvec) / F

    pins_F = sorted(set_F - set(iface_nodes) - set(outer_nodes))
    pins_N = sorted(set_N - set(iface_nodes))
    MF = modal_compliance(K_F, pins_F)
    MN = modal_compliance(K_N, pins_N)
    chi = MN / MF
    pred = 1.0 / (1.0 + chi)

    # measured interface displacement along the constant radial mode
    ur = (uG[:mm] * np.cos(th) + uG[mm:] * np.sin(th))
    meas = float(ur.mean()) / delta
    mono_ur = (uM[gdof][:mm] * np.cos(th) + uM[gdof][mm:] * np.sin(th))
    mono_meas = float(mono_ur.mean()) / delta

    if verbose:
        print(f"  nt={nt:3d} nr={nr_in}+{nr_out} iface {mm:3d} | bal {bal:.2e} "
              f"sym {sym:.2e} mono {mono:.2e} | mono u_r(1)/D {mono_meas:.6f} "
              f"(exact {exact_ratio():.6f}) | chi {chi:.5f} pred {pred:.6f} "
              f"meas {meas:.6f}")
    return dict(nt=nt, nif=mm, bal=bal, sym=sym, mono=mono, chi=chi, pred=pred,
                meas=meas, mono_meas=mono_meas, exact=exact_ratio(),
                contrast=contrast)


def main():
    t0 = time.time()
    print("=" * 108)
    print("M1a: curved interface (concentric annulus);  exact u_r(r_split)/Delta =",
          f"{exact_ratio():.6f}")
    print("=" * 108)
    print("\n[1] refinement, unit contrast")
    rows = [run(nt=n, nr_in=max(2, n // 8), nr_out=max(3, n // 6))
            for n in (16, 24, 32, 48, 64)]
    print("\n[2] contrast sweep (M1b), nt=32")
    cres = [run(nt=32, nr_in=4, nr_out=5, contrast=c)
            for c in (1.0, 1e1, 1e2, 1e3, 1e4)]
    print(f"\ntotal {time.time()-t0:.1f}s")
    with open(os.path.join(OUT, 'annulus_benchmark.txt'), 'w', encoding='utf-8') as fp:
        fp.write("M1a annulus refinement (exact u_r(1)/Delta = %.6f)\n" % exact_ratio())
        for r in rows:
            fp.write(f"nt={r['nt']} iface={r['nif']} bal={r['bal']:.2e} sym={r['sym']:.2e} "
                     f"mono={r['mono']:.2e} mono_ur={r['mono_meas']:.6f} "
                     f"chi={r['chi']:.5f} pred={r['pred']:.6f} meas={r['meas']:.6f}\n")
        fp.write("M1b contrast sweep\n")
        for r in cres:
            fp.write(f"contrast={r['contrast']:.0e} bal={r['bal']:.2e} mono={r['mono']:.2e} "
                     f"chi={r['chi']:.5f} pred={r['pred']:.6f} meas={r['meas']:.6f}\n")
    print(f"[saved] {os.path.join(OUT, 'annulus_benchmark.txt')}")


if __name__ == '__main__':
    main()
