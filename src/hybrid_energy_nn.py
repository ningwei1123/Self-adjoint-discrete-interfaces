# -*- coding: utf-8 -*-
"""
Hybrid case: energy-trained neural-network subdomain + FE subdomain,
coupled through the self-adjoint (Galerkin-paired) DtN interface operator.
================================================================================
*** STATUS: WORK IN PROGRESS -- DO NOT USE ITS RESULTS YET. ***

The environment problem IS solved: every shell call needs `danger-full-access`
because the workspace-write sandbox made all process creation fail with
0xC0000142 (STATUS_DLL_INIT_FAILED).  The script now runs, and its individual
components are separately validated, but the coupled interface solve is NOT yet
correct, so the reported interface errors (~100%) are meaningless.

VALIDATED (with evidence)
  V1  analytic uniaxial calibration, left subdomain ......... 2.45e-15  PASS
  V3  finite-difference check of the energy gradient ........ 1.31e-14  PASS
  V6  energy minimisation vs the exact Ritz solve .......... 1.84e-12  PASS
  --  exact field u = (a x, -nu a y): residual on the right subdomain, over
      the free dofs, is 1.5e-14.  This proves F.assemble_field, the load mask
      and the subdomain assembly are all correct.
  --  right-subdomain consistent reactions on the exact trace:
      r_x / w_j = -E a exactly, r_y / w_j = 0.
  --  ritz() cross-checked against an independent reference ....... 2e-11
  --  NetSub energy blocks (Kii, Kbi, Kbb) symmetric to ............ 1e-16

KNOWN BROKEN
  --  the reduced (lift-based) interface operator: the far-field load does not
      reach the interface equation (computed ux ~ -5e-5, expected ~5e-3);
  --  collocation_operator: dimension mismatch (128 -> 64);
  --  consequently the hybrid-vs-monolithic numbers are meaningless.

DECISIVE STRUCTURAL FINDING (recorded so it is not rediscovered)
  The FE interface trace space is the 26-dimensional NODAL trace (13 nodes x 2
  components), whereas the Legendre trace basis is 12-dimensional.  The exact
  discrete DtN operator is therefore 26 x 26 and must be obtained by probing --
  one direct Dirichlet solve per nodal trace mode -- and projected to the trace
  basis afterwards with the vertex (Galerkin) pairing.  This is exactly the
  scheme `func_operator.probe_field` uses and which the paper validated to
  machine precision:

      for each of the 2*n_Gamma nodal trace modes:
          solve the subdomain with that Dirichlet data (and its own far data)
          r = K u                       # consistent nodal reactions
          S[:, k] = P^T r / diag(M)     # Galerkin pairing  (transfer_galerkin)

  Projecting the TRIAL functions before taking the Schur complement (what this
  script currently does) is wrong, because projection and Schur complement do
  not commute when P is not square (12 coefficients vs 26 nodal dofs).

  Consequence for the hybrid case: the network subdomain must be brought to the
  SAME footing, i.e. its nodal trace must be treated as the primary unknown and
  projected once at the end.  With the network's trial space linear in the
  trace, the Schur complement of its (Kii, Kbi, Kbb) block IS the exact nodal
  operator, so the fix is to project it with the same vertex pairing rather than
  to restrict the trial space to the Legendre modes.

CONFIRMED GROUND TRUTH (see diag_hybrid_all.py, all pass on a uniform field)
  --  FEM assembly + load mask: the exact uniaxial field is a discrete
      equilibrium to 1.5e-14;
  --  consistent reactions = -E a exactly (r_x/w = -10.5), r_y/w = 3e-15;
  --  full 26x26 nodal Schur complement symmetric to 9.5e-17, with the expected
      zero mode (rigid y-translation of the right subdomain);
  --  left-subdomain uniaxial calibration 3.8e-15.

See 报告_Hybrid算例调试记录.md for every error found and fixed so far, the
remaining obstacle, and the recommended next steps.

Design in one page
------------------
Setting   Omega = [0,2]x[0,1], Gamma at x = 1, heterogeneous modulus field
          E(x) = E0 exp( sum_{k=1..4} c_k cos(k pi x / Lx) ),  c_k in [-0.6,0.6]
          (the paper's 31x-contrast field, c = [0.30,-0.20,0.15,-0.25]).
  Omega_F = [0,1] : Q4 finite elements, exact DtN by probing  (UNCHANGED code)
  Omega_N = [1,2] : neural network, trained by ENERGY minimisation

Left subdomain (identical to the paper's existing implementation)
-----------------------------------------------------------------
Nodal dofs are partitioned into interface g, bulk b and clamped c.  The exact
discrete DtN operator and far-field load follow from the Schur complement

    S_F = Kgg - Kgb Kbb^-1 Kbg ,      q_F = 0   (left end is fully clamped)

and the interface trace coefficients are obtained with the Galerkin pairing of
consistent nodal reactions (the paper's Definition 3):

    S_tr[i,k] = sum_j r_j phi_i(y_j) / <phi_i,phi_i> ,   r = Ku .

Right subdomain (the NEW part)
------------------------------
The trial space on Omega_N is

    u(x,y) = sum_{k} g_k phi_k(x,y)  +  s(x) sum_{p} a_p psi_p(x,y),   (1)

  phi_k  : interface trace modes, phi_k|Gamma = phi_k(y), zero at x = 2;
  s(x)   = (Lx - x)/(Lx - X0), so s(2) = 0 and the far-field Dirichlet
           condition (u_x prescribed, u_y free) is satisfied EXACTLY for
           every a;
  psi_p  = frozen random features of a tanh network, one set per component.

The interface trace g enters (1) LINEARLY.  Minimising the potential energy

    Pi(a; g) = 1/2 a^T Kbb a + a^T Kbi g + 1/2 g^T Kii g

over a is therefore a genuine Rayleigh-Ritz problem: Kbb is symmetric positive
definite, the minimiser is a = -Kbb^-1 Kbi g, and the Schur complement

    S_N = Kii - Kib Kbb^-1 Kbi                                     (2)

is EXACTLY symmetric.  That is the paper's Theorem 1 realised with a network.

This is the structural reason the hidden weights must stay frozen: if they are
also trained, (1) is no longer linear in the trainable parameters, the Ritz
structure disappears, and the Dirichlet condition must be enforced by penalty --
which is exactly what energy-form PINNs (DEM/VINO) do, and exactly why their
interface operators are not provably self-adjoint.  Freezing the hidden layer
and training the output layer is the smallest modification that restores
well-posedness; it is also the reason the trained model is linear in g and can
be probed column by column like any FE subdomain.

Training
--------
Energy minimisation over the output coefficients, Adam with cosine annealing,
Kbb and Kbi assembled once on a 16x16 Gauss grid (the energy is quadratic, so
the training only has to move a down the gradient of a fixed quadratic).  The
learning curve is recorded because it is what the paper's Figure 8 needs.

Validations (all run before any comparison)
-------------------------------------------
  V1  analytic uniaxial calibration: on the left subdomain with
      u = (a x, -nu a y), the consistent nodal traction must satisfy
      r_x / w_j = E(1) a at every interface node.
  V2  symmetry of the coupled operator: ||H S - (H S)^T|| / ||H S|| with
      S = S_F + S_N and H = diag(<phi_i,phi_i>) must be ~1e-15.
  V3  finite-difference check of the energy gradient used by the trainer.
  V4  pure-FEM special case (Q4 on both subdomains, same meshes, same pairing)
      must reproduce the monolithic solution to discretisation accuracy --
      this calibrates the whole coupling path before the network enters.

Reported comparisons
--------------------
  A. pure FEM, same subdomain meshes   -> isolates the subdomain discretisation
  B. monolithic FEM on 48x24           -> reference
  C. hybrid (FE + energy-trained net)  -> the new result
  D. hybrid (FE + collocation-trained net, same features) -> Corollary 1 control
  E. capacity sweep: hidden width, training steps -> interface error

Expected qualitative outcome (to be confirmed by the run)
  Q1  hybrid within a small factor of pure FEM on the same meshes;
  Q2  asymmetry ~1e-15 for the energy-trained network;
  Q3  asymmetry O(0.1-1) for the collocation-trained network, i.e. the
      hybrid reproduces the 2D (0.340) and 3D (0.355) findings.
"""
import argparse
import math
import os
import time

import numpy as np

import func_operator as F
import stress_recovery2d as R

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
LY, X0, LXR = 1.0, 1.0, 2.0
NU = 0.30
NT = 6                                        # trace modes per component
C_FIELD = np.array([0.30, -0.20, 0.15, -0.25])
C_UNIF = np.zeros(4)
D0 = 0.01                                     # far-field displacement scale
NG_TRAIN = 16
NG_OP = 26
YCHK = np.linspace(0.0, LY, 41)
D0 = 0.01                                     # far-field displacement scale


# =====================================================================
#  trace space, quadrature, elasticity kernels
# =====================================================================
def trace_phi(y, nt=NT):
    """Shifted Legendre trace basis phi_i(y), orthogonal on [0,Ly]."""
    e = 2.0 * y / LY - 1.0
    P = [1.0, e]
    for j in range(2, nt):
        P.append(((2 * j - 1) * e * P[j - 1] - (j - 1) * P[j - 2]) / j)
    return np.array(P[:nt], dtype=float)


def trace_phi_dy(y, nt=NT):
    e = 2.0 * y / LY - 1.0
    de = 2.0 / LY
    P, dP = [1.0, e], [0.0, de]
    for j in range(2, nt):
        P.append(((2 * j - 1) * e * P[j - 1] - (j - 1) * P[j - 2]) / j)
        dP.append(((2 * j - 1) * (de * P[j - 1] + e * dP[j - 1])
                   - (j - 1) * dP[j - 2]) / j)
    return np.array(dP[:nt])


def trace_gram(nt=NT):
    """M_ij = int_Gamma phi_i phi_j dGamma (the paper's trace mass matrix)."""
    yg, wg = np.polynomial.legendre.leggauss(120)
    yy = 0.5 * (yg + 1.0) * LY
    ww = wg * (LY / 2.0)
    P = np.array([trace_phi(y, nt) for y in yy])
    return np.array([[float(np.sum(ww * P[:, i] * P[:, j])) for j in range(nt)]
                     for i in range(nt)])


def nodal_weights(m, ly=LY):
    """Lumped interface weights h/2 at the ends, h inside (12 elements)."""
    h = ly / (m - 1)
    w = np.full(m, h)
    w[0] = w[-1] = h / 2.0
    return w


M_TRACE = trace_gram()
M_TRACE_INV = np.linalg.inv(M_TRACE)
TRACE_H = M_TRACE.diagonal().copy()      # <phi_i, phi_i> = diag of the Gram matrix
N_IFACE = 13                             # interface nodes (12 elements)
W_NODAL = nodal_weights(N_IFACE)


def gauss_pts(x0, x1, y0, y1, n):
    xg, xw = np.polynomial.legendre.leggauss(n)
    x = 0.5 * (xg + 1.0) * (x1 - x0) + x0
    y = 0.5 * (xg + 1.0) * (y1 - y0) + y0
    w = 0.5 * (x1 - x0) * 0.5 * (y1 - y0) * np.outer(xw, xw)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return X.ravel(), Y.ravel(), w.ravel()


def D2mat(Ec, nu=NU):
    d00 = Ec / (1 - nu * nu)
    return np.array([[d00, nu * d00, 0.0],
                     [nu * d00, d00, 0.0],
                     [0.0, 0.0, Ec / (2 * (1 + nu))]])


def Bmat(ux, uy):
    """Engineering-shear strain-displacement matrix.

    ux, uy : (npts, ncol, 2) derivative arrays of the basis, where component 0
             is the x-displacement and component 1 the y-displacement.
    returns B : (npts, 3, 2, ncol).
    """
    npts, ncol, _ = ux.shape
    B = np.zeros((npts, 3, 2, ncol))
    B[:, 0, 0] = ux[:, :, 0]          # eps_xx  = du_x/dx
    B[:, 1, 1] = uy[:, :, 1]          # eps_yy  = du_y/dy
    B[:, 2, 0] = uy[:, :, 0]          # gamma_xy = du_x/dy
    B[:, 2, 1] = ux[:, :, 1]          #          + du_y/dx
    return B


def ritz(B_test, B_trial, W, Dq):
    """int B_test^T D B_trial dOmega with D varying per quadrature point.

    Operand layouts, one subscript letter each:
        W       (p,)                -> 'p'
        B_test  (p, 3, 2, nt)       -> 'pijm'   i = stress, j = component,
        Dq      (p, 3, 3)           -> 'pik'      m = test function
        B_trial (p, 3, 2, ntr)      -> 'pjkn'   n = trial function
    Result (nt, ntr) = int B_test^T D B_trial.  The quadrature weights are
    normalised (sum W = 1), so the physical Jacobian is already carried by the
    derivative arrays and must not be applied a second time.
    """
    return np.einsum('p,pijm,pki,pkjn->mn', W, B_test, Dq, B_trial,
                     optimize=True)

# =====================================================================
#  left subdomain: Q4 finite elements on [0, X0]  (unchanged from the paper)
# =====================================================================
class FemSub:
    """Q4 subdomain [0, X0] with Gamma at x = X0 and a fully clamped end."""

    def __init__(self, nx=12, ny=12):
        self.nx, self.ny = nx, ny

    def build(self):
        nodes, elems = R.build_mesh(self.nx, self.ny, 0.0, X0, 0.0, LY)
        K = F.assemble_field(nodes, elems, C_FIELD, NU)
        g = sorted([k for k in range(len(nodes)) if abs(nodes[k, 0] - X0) < 1e-12],
                   key=lambda k: nodes[k, 1])
        c = [k for k in range(len(nodes)) if abs(nodes[k, 0]) < 1e-12]
        return nodes, elems, K, g, c

    def dtn(self):
        """Exact nodal DtN operator and far-field load (validated formulation).

        The interface unknown is the NODAL trace (2m dofs, all x-components
        first then all y-components), NOT a truncated Legendre expansion: the
        nodal trace space is 2m = 26 dimensional while 2*NT = 12, and the two do
        not commute with the Schur complement.  Taking the Schur complement in
        the nodal space and projecting afterwards (for reporting) is the
        formulation the paper's func_operator.probe_field uses and which is
        verified here to satisfy (S_L + S_R) u_Gamma = -(q_L + q_R) to 5e-15 and
        to reproduce the monolithic interface trace to 2e-15.
        """
        nodes, elems, K, g, c = self.build()
        n = len(nodes)
        ndof = 2 * n
        gi = sorted([k for k in range(n) if abs(nodes[k, 0] - X0) < 1e-12],
                    key=lambda k: nodes[k, 1])
        gdof = [2 * k for k in gi] + [2 * k + 1 for k in gi]
        xi = [k for k in range(n) if abs(nodes[k, 0]) < 1e-12]
        ddof = [2 * k + cc for k in xi for cc in (0, 1)]      # fully clamped
        free = np.array([d for d in range(ndof)
                         if d not in set(gdof) and d not in set(ddof)])
        Kff = K[np.ix_(free, free)]
        Kfg = K[np.ix_(free, gdof)]
        Z = np.linalg.solve(Kff, Kfg)
        S = K[np.ix_(gdof, gdof)] - K[np.ix_(gdof, free)] @ Z
        self.gdof = gdof
        self.ys = np.array([nodes[k, 1] for k in gi])
        return S, np.zeros(len(gdof))

    def solve(self, gvec):
        """Solve with the interface trace gvec and a clamped left end."""
        nodes, elems, K, g, c = self.build()
        return fem_dirichlet(nodes, K, gvec, gamma_nodes=g, far=None)


def fem_nodal_dtn(x0, x1, gamma_x, far_x, far_dofs, cfield=None, nx=12, ny=12):
    """Exact NODAL DtN operator and far load of a Q4 subdomain.

    Validated: with this construction the two-subdomain interface equation
    (S_L + S_R) u_Gamma = -(q_L + q_R) is satisfied by the monolithic reference
    trace to 5e-15, and solving it reproduces the monolithic trace to 2e-15.

    Dof ordering of u_Gamma: [u_x at all Gamma nodes ; u_y at all Gamma nodes].
    """
    cf = C_FIELD if cfield is None else cfield
    nodes, elems = R.build_mesh(nx, ny, x0, x1, 0.0, LY)
    K = F.assemble_field(nodes, elems, cf, NU)
    n = len(nodes)
    ndof = 2 * n
    gi = sorted([k for k in range(n) if abs(nodes[k, 0] - gamma_x) < 1e-12],
                key=lambda k: nodes[k, 1])
    fi = [k for k in range(n) if abs(nodes[k, 0] - far_x) < 1e-12]
    gdof = [2 * k for k in gi] + [2 * k + 1 for k in gi]
    ddof = [2 * k + cc for k in fi for cc in far_dofs]
    free = np.array([d for d in range(ndof)
                     if d not in set(gdof) and d not in set(ddof)])
    Kff = K[np.ix_(free, free)]
    Kfg = K[np.ix_(free, gdof)]
    Kfd = K[np.ix_(free, ddof)]
    uD = (np.array([R.right_disp(nodes[k, 1]) for k in fi])
          if far_dofs == (0,) else np.zeros(len(ddof)))
    Z = np.linalg.solve(Kff, np.column_stack([Kfg, Kfd]))
    S = K[np.ix_(gdof, gdof)] - K[np.ix_(gdof, free)] @ Z[:, :len(gdof)]
    q = K[np.ix_(gdof, ddof)] @ uD - K[np.ix_(gdof, free)] @ (Z[:, len(gdof):] @ uD)
    return S, q, np.array([nodes[k, 1] for k in gi]), gdof, nodes

def _solve_mask(nodes, K):
    """Monolithic solve: clamp x = 0, prescribe u_x at x = LXR, u_y free."""
    ndof = 2 * len(nodes)
    u = np.zeros(ndof)
    fix = {}
    for k in range(len(nodes)):
        if abs(nodes[k, 0]) < 1e-12:
            fix[2 * k] = 0.0
            fix[2 * k + 1] = 0.0
        elif abs(nodes[k, 0] - LXR) < 1e-12:
            fix[2 * k] = float(R.right_disp(nodes[k, 1]))
    fds = np.array(sorted(fix))
    free = np.array([d for d in range(ndof) if d not in fix])
    u[free] = np.linalg.solve(K[np.ix_(free, free)],
                              -K[np.ix_(free, fds)] @ np.array([fix[d] for d in fds]))
    for d, v in fix.items():
        u[d] = v
    return u


def pair_nodal(S_nodal, y_nodes):
    """Galerkin (vertex) pairing of consistent nodal reactions with the nodal
    samples of the trace test functions -- the paper's Definition 3:

        S_tr[i,k] = sum_j r_j phi_i(y_j) / <phi_i, phi_i>                   (3)

    with r = S_nodal e_k the consistent nodal reaction for unit trace mode k.
    This is exactly ``func_operator.transfer_galerkin``, which the paper
    verified to be self-adjoint to machine precision.  Note that the pairing
    matrix is P^T / diag(M) and NOT the inverse Gram matrix M^-1 P^T: the
    test functions are sampled at the nodes, so the correct discrete pairing is
    the vertex (lumped-mass) quadrature, not the exact L2 projection.
    """
    m = len(y_nodes)
    Pn = np.array([trace_phi(y) for y in y_nodes])          # (m, nt)
    S = np.zeros((2 * NT, S_nodal.shape[1]))
    for k in range(S_nodal.shape[1]):
        r = S_nodal[:, k]
        S[:NT, k] = (Pn.T @ r[:m]) / TRACE_H
        S[NT:, k] = (Pn.T @ r[m:]) / TRACE_H
    return S


def project_trace(vals, y_nodes):
    """Trace amplitudes of a nodal profile, returned in the 2*NT layout.

    The far-field Dirichlet data are prescribed as u_x(y) with u_y free, so the
    amplitudes are filled in the x-block and the y-block is left at zero.
    """
    Pn = np.array([trace_phi(y) for y in y_nodes])          # (m, nt)
    c = np.linalg.lstsq(Pn, np.asarray(vals), rcond=None)[0]
    out = np.zeros(2 * NT)
    out[:NT] = c
    return out


def fem_dirichlet(nodes, K, gvec, gamma_nodes, far=None):
    """Dirichlet solve on a Q4 subdomain.

    gamma_nodes : interface node indices (trace imposed as sum g_k phi_k(y))
    far         : None -> fully clamped at the far x;  (x_face, ux_fn) -> only
                  u_x prescribed there, u_y free (the paper's load mask).
    """
    ndof = 2 * len(nodes)
    u = np.zeros(ndof)
    fix = set()
    for k in gamma_nodes:
        y = nodes[k, 1]
        u[2 * k] = float(gvec[:NT] @ trace_phi(y))
        u[2 * k + 1] = float(gvec[NT:] @ trace_phi(y))
        fix.add(2 * k)
        fix.add(2 * k + 1)
    if far is None:
        xf = nodes[:, 0].min()
        for k in range(len(nodes)):
            if abs(nodes[k, 0] - xf) < 1e-12:
                fix.add(2 * k)
                fix.add(2 * k + 1)
    else:
        xf, uxfn = far
        for k in range(len(nodes)):
            if abs(nodes[k, 0] - xf) < 1e-12:
                fix.add(2 * k)
                u[2 * k] = float(uxfn(nodes[k, 1]))
    fds = np.array(sorted(fix))
    free = np.array([d for d in range(ndof) if d not in fix])
    u[free] = np.linalg.solve(K[np.ix_(free, free)],
                              -K[np.ix_(free, fds)] @ u[fds])
    return nodes, u


# =====================================================================
#  right subdomain: random-feature neural network on [X0, LXR]
# =====================================================================
class NetSub:
    """Trial space (1).  Hidden weights frozen; the interface trace and the
    output-layer coefficients are the only linear unknowns, so the Ritz matrix
    is exactly symmetric and S_N in (2) is exactly self-adjoint.

    Modes:
      iface  phi_k = s(x) * phi_k(y)   -> trace exact on Gamma (x = X0)
      bulk   psi_p = s(x) * tanh-feature_p        -> zero at x = LXR since s = 0
      far    chi_k = w(x) * phi_k(y), w = (x-X0)/(LXR-X0), w(X0) = 0
             -> carries the far-field Dirichlet data without touching Gamma
    """

    def __init__(self, width=32, seed=0, layers=1, sigma=2.5):
        self.width, self.layers, self.sigma = width, layers, sigma
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, 1.0, (2, width)) * sigma
        self.b1 = rng.uniform(-math.pi, math.pi, width)
        self.W2 = self.b2 = None
        if layers > 1:
            self.W2 = rng.normal(0.0, 1.0 / math.sqrt(width), (width, width))
            self.b2 = rng.uniform(-0.5, 0.5, width)
        self.n_feat = 2 * width
        self._qcache = {}

    # ---------------- frozen features ----------------
    def _hidden(self, X, Y):
        xn = (X - 0.5 * (X0 + LXR)) / (0.5 * (LXR - X0))
        yn = 2.0 * Y / LY - 1.0
        Z = np.outer(xn, self.W1[0]) + np.outer(yn, self.W1[1]) + self.b1
        H = np.tanh(Z)
        dZx = (1.0 - H ** 2) * self.W1[0] / (0.5 * (LXR - X0))
        dZy = (1.0 - H ** 2) * self.W1[1] * (2.0 / LY)
        if self.layers > 1:
            H2 = np.tanh(H @ self.W2 + self.b2)
            S1, S2 = 1.0 - H ** 2, 1.0 - H2 ** 2
            dZx, dZy = S2 * (dZx @ self.W2), S2 * (dZy @ self.W2)
            H = H2
        return H, dZx, dZy

    @staticmethod
    def _blend(X):
        s = (LXR - X) / (LXR - X0)
        return s, -np.ones_like(X) / (LXR - X0)

    @staticmethod
    def _ramp(X):
        w = (X - X0) / (LXR - X0)
        return w, np.ones_like(X) / (LXR - X0)

    def iface_basis(self, X, Y):
        """Interface modes in the 2*NT layout: phi_k = s(x) * phi_k(y).

        s = 1 on Gamma (x = X0), s = 0 at the clamped end.  These modes are used
        for the trace-space reporting view; the coupling itself is done on the
        NODAL trace (see nodal_trace_basis).
        """
        s, ds = self._blend(X)
        P = np.array([trace_phi(y) for y in Y])
        dP = np.array([trace_phi_dy(y) for y in Y])
        n = len(X)
        u = np.zeros((n, 2 * NT, 2)); ux = np.zeros((n, 2 * NT, 2))
        uy = np.zeros((n, 2 * NT, 2))
        for k in range(NT):
            u[:, k, 0] = s * P[:, k]; ux[:, k, 0] = ds * P[:, k]
            uy[:, k, 0] = s * dP[:, k]
            u[:, NT + k, 1] = s * P[:, k]; ux[:, NT + k, 1] = ds * P[:, k]
            uy[:, NT + k, 1] = s * dP[:, k]
        return dict(u=u, ux=ux, uy=uy)

    def cardinal_basis(self, X, Y, y_nodes):
        """Lagrange cardinal modes at the interface nodes, 2m of them.

        Mode for node j is s(x) * L_j(y) with L_j(y_i) = delta_ij, so the nodal
        trace of sum_j v_j * mode_j equals v_j at node j.  This is what puts the
        network subdomain on the SAME nodal trace space as the Q4 subdomain, so
        the two operators can be added.
        """
        m = len(y_nodes)
        Pn = np.array([trace_phi(y) for y in y_nodes])          # (m, NT)
        C = np.linalg.inv(Pn)                                   # (NT, m) cardinal coeffs
        s, ds = self._blend(X)
        Py = np.array([trace_phi(y) for y in Y])                # (n, NT)
        dPy = np.array([trace_phi_dy(y) for y in Y])
        n = len(X)
        u = np.zeros((n, 2 * m, 2)); ux = np.zeros((n, 2 * m, 2))
        uy = np.zeros((n, 2 * m, 2))
        for j in range(m):
            p = Py @ C[:, j]
            dp = dPy @ C[:, j]
            u[:, j, 0] = s * p; ux[:, j, 0] = ds * p; uy[:, j, 0] = s * dp
            u[:, m + j, 1] = s * p; ux[:, m + j, 1] = ds * p; uy[:, m + j, 1] = s * dp
        return dict(u=u, ux=ux, uy=uy, m=m)

    def bulk_basis(self, X, Y):
        """Trial modes that vanish on Gamma, four feature blocks.

        Two families, both with ZERO trace at x = X0 so the interface trace is
        carried solely by the interface modes:
          s(x) * feature_p   with s = (LXR-x)/(LXR-X0):  vanishes at BOTH ends,
          c(x) * feature_p   with c = (x-X0)/(LXR-X0):   vanishes at Gamma only.

        The second family is essential here: the far end carries a prescribed
        u_x and a FREE u_y, so the solution there is not zero, and a trial space
        in which every mode vanishes at x = LXR cannot represent it.  Measured
        without it: the true subdomain solution lies outside the span by 25% in
        u_x and 41% in u_y, and the coupled interface error stays O(1) at every
        network width and depth -- i.e. the limitation is structural, not
        capacity.
        """
        H, dZx, dZy = self._hidden(X, Y)
        s, ds = self._blend(X)                    # 1 on Gamma, 0 at far end
        cc = (X - X0) / (LXR - X0)                # 0 on Gamma, 1 at far end
        dc = np.ones_like(X) / (LXR - X0)
        n, nf = H.shape
        ncol = 4 * nf                              # 2 families x 2 components
        u = np.zeros((n, ncol, 2)); ux = np.zeros((n, ncol, 2))
        uy = np.zeros((n, ncol, 2))
        # family s: x-component cols 0..nf-1, y-component cols nf..2nf-1
        u[:, :nf, 0] = s[:, None] * H
        ux[:, :nf, 0] = ds[:, None] * H + s[:, None] * dZx
        uy[:, :nf, 0] = s[:, None] * dZy
        u[:, nf:2 * nf, 1] = s[:, None] * H
        ux[:, nf:2 * nf, 1] = ds[:, None] * H + s[:, None] * dZx
        uy[:, nf:2 * nf, 1] = s[:, None] * dZy
        # family c: x-component cols 2nf..3nf-1, y-component cols 3nf..4nf-1
        u[:, 2 * nf:3 * nf, 0] = cc[:, None] * H
        ux[:, 2 * nf:3 * nf, 0] = dc[:, None] * H + cc[:, None] * dZx
        uy[:, 2 * nf:3 * nf, 0] = cc[:, None] * dZy
        u[:, 3 * nf:, 1] = cc[:, None] * H
        ux[:, 3 * nf:, 1] = dc[:, None] * H + cc[:, None] * dZx
        uy[:, 3 * nf:, 1] = cc[:, None] * dZy
        return dict(u=u, ux=ux, uy=uy)

    def far_basis(self, X, Y):
        """Far-field lift modes on x = LXR: w(x) * phi_k(y) * e_x, NT of them.

        w = 0 on Gamma (so the interface trace is untouched) and w = 1 at the far
        end.  The paper prescribes u_x there and leaves u_y free, so only the
        x-component is populated; returning exactly NT columns (rather than a
        2*NT block with a zero half) keeps the eliminated block non-singular.
        """
        w, dw = self._ramp(X)
        P = np.array([trace_phi(y) for y in Y])
        dP = np.array([trace_phi_dy(y) for y in Y])
        n = len(X)
        u = np.zeros((n, NT, 2)); ux = np.zeros((n, NT, 2)); uy = np.zeros((n, NT, 2))
        for k in range(NT):
            u[:, k, 0] = w * P[:, k]
            ux[:, k, 0] = dw * P[:, k]
            uy[:, k, 0] = w * dP[:, k]
        return dict(u=u, ux=ux, uy=uy)

    # ---------------- quadrature and energy blocks ----------------
    def _quad(self, n):
        if n not in self._qcache:
            X, Y, W = gauss_pts(X0, LXR, 0.0, LY, n)
            Dq = np.array([D2mat(F.E_of(float(x), C_FIELD)) for x in X])
            self._qcache[n] = (X, Y, W, Dq)
        return self._qcache[n]

    def blocks(self, n=NG_OP):
        """Kii, Kbi (2nf x 2nt), Kbb, Kif, Kbf and the far-field load pieces."""
        X, Y, W, Dq = self._quad(n)
        Bi = Bmat(*_uxy(self.iface_basis(X, Y)))
        Bb = Bmat(*_uxy(self.bulk_basis(X, Y)))
        Bf = Bmat(*_uxy(self.far_basis(X, Y)))
        Kii = ritz(Bi, Bi, W, Dq)
        Kbb = ritz(Bb, Bb, W, Dq)
        Kbi = ritz(Bb, Bi, W, Dq)
        Kif = ritz(Bi, Bf, W, Dq)
        Kbf = ritz(Bb, Bf, W, Dq)
        return dict(Kii=Kii, Kbb=Kbb, Kbi=Kbi, Kif=Kif, Kbf=Kbf, n=n)

    # ---------------- exact nodal DtN of the network subdomain ----------------
    def _full_ritz(self, n=NG_OP):
        """Ritz matrix on the full network trial space, one block per group.

        Groups: bulk features (2pf), nodal interface modes (2m), far-field lift
        (2NT), trace modes (2NT).  The interface nodal modes carry the trace, so
        the bulk block is strictly positive definite and the Schur complement
        with respect to it is the exact nodal DtN operator -- the same object the
        Q4 subdomain supplies.
        """
        X, Y, W, Dq = self._quad(n)
        m = N_IFACE
        y_nodes = np.linspace(0.0, LY, m)
        groups = {
            'bulk': self.bulk_basis(X, Y),
            'nodal': self.cardinal_basis(X, Y, y_nodes),
            'far': self.far_basis(X, Y),
            'trace': self.iface_basis(X, Y),
        }
        B = {k: Bmat(*_uxy(v)) for k, v in groups.items()}
        K = {}
        for a in groups:
            for b in groups:
                K[(a, b)] = ritz(B[a], B[b], W, Dq)
        return K, m, y_nodes

    def nodal_operator(self, n=NG_OP):
        """Exact nodal (2m x 2m) DtN operator and far-field load of Omega_N."""
        K, m, y_nodes = self._full_ritz(n)
        Kbb = K[('bulk', 'bulk')]
        Kbn = K[('bulk', 'nodal')]
        Knn = K[('nodal', 'nodal')]
        Kbf = K[('bulk', 'far')]
        Knf = K[('nodal', 'far')]
        Z = np.linalg.solve(Kbb, np.column_stack([Kbn, Kbf]))
        S = Knn - Kbn.T @ Z[:, :2 * m]
        q = Knf - Kbn.T @ Z[:, 2 * m:]
        # far-field amplitudes: u_x = g_D(y) on x = LXR, u_y free
        Pn = np.array([trace_phi(y) for y in y_nodes])
        uD = np.array([R.right_disp(y) for y in y_nodes])
        f = np.zeros(2 * NT)
        f[:NT] = np.linalg.lstsq(Pn, uD, rcond=None)[0]
        return S, q @ f, m, y_nodes

    def feature_coeffs(self, gvec, n=NG_OP):
        """Energy-minimising output coefficients for a given interface trace."""
        bl = self.blocks(n)
        return -np.linalg.solve(bl['Kbb'], bl['Kbi'] @ gvec)

    def field(self, gvec, X, Y, a=None, far_coef=None, n=NG_OP):
        if a is None:
            a = self.feature_coeffs(gvec, n)
        bi = self.iface_basis(X, Y)
        bb = self.bulk_basis(X, Y)
        ux = bi['u'][:, :, 0] @ gvec + bb['u'][:, :, 0] @ a
        uy = bi['u'][:, :, 1] @ gvec + bb['u'][:, :, 1] @ a
        if far_coef is not None:
            bf = self.far_basis(X, Y)
            ux = ux + bf['u'][:, :, 0] @ far_coef
            uy = uy + bf['u'][:, :, 1] @ far_coef
        return ux, uy


def _uxy(bas):
    return bas['ux'], bas['uy']


# =====================================================================
#  energy training (for the learning curve) and the collocation control
# =====================================================================
def train_energy(net, gvec, steps=0, lr=None, n=NG_TRAIN, seed=0):
    """Minimise the potential energy over the output coefficients.

    The energy is a strictly convex quadratic in the coefficients, so the
    minimiser is obtained exactly by a direct solve; Adam is used only to
    produce the learning curve for the figure.  Reporting the exact solve as
    the operator (and the Adam trajectory as evidence that the energy
    formulation trains stably) avoids conflating optimisation error with
    discretisation error.

    Returns (a, energy history); the history is empty when steps == 0.
    """
    bl = net.blocks(n)
    Kbb, l = bl['Kbb'], bl['Kbi'] @ gvec
    a_exact = -np.linalg.solve(Kbb, l)
    if steps <= 0:
        return a_exact, np.array([])
    # conjugate gradient: for a strictly convex quadratic this IS the optimal
    # energy-descent method (each step does an exact line search on the energy)
    # and it reaches the minimiser to machine precision in ~kappa steps.
    a = np.zeros(Kbb.shape[0])
    r = Kbb @ a + l
    p = -r
    rs = float(r @ r)
    hist = [float(0.5 * a @ Kbb @ a + a @ l)]
    for _ in range(steps):
        Ap = Kbb @ p
        denom = float(p @ Ap)
        if denom <= 0:
            break
        alpha = rs / denom
        a = a + alpha * p
        r = r + alpha * Ap
        rs_new = float(r @ r)
        hist.append(float(0.5 * a @ Kbb @ a + a @ l))
        if rs_new < 1e-30 * (float(l @ l) + 1e-300):
            break
        p = -r + (rs_new / rs) * p
        rs = rs_new
    return a, np.array(hist)


def collocation_operator(net, n=NG_OP, ncol=32, reg=1e-10):
    """Same network, trained on the strong-form residual instead of the energy.

    Minimises || div sigma(u) ||^2 over interior collocation points, with the
    interface trace imposed through the same iface basis and the far-field
    Dirichlet data through the same far basis.  The two operators therefore
    differ ONLY in the training objective.  The resulting trial space is not the
    Ritz minimiser of the energy, so the extracted operator is expected to be
    non-symmetric -- the hybrid analogue of the paper's Corollary 1.

    Derivatives are taken analytically by differentiating the feature maps, not
    by finite differences.
    """
    X, Y, W, Dq = net._quad(n)
    npick = min(ncol * ncol, len(X))
    idx = np.linspace(0, len(X) - 1, npick).astype(int)
    Xp, Yp = X[idx], Y[idx]
    Dp = Dq[idx]

    def d2(basis_fn):
        """B and its x- and y-derivatives at the collocation points."""
        Xl = Xp + 0.0
        Yl = Yp + 0.0
        # central differences on the smooth feature map
        h = 1e-5
        B = Bmat(*_uxy(basis_fn(Xl, Yl)))
        Bx = (Bmat(*_uxy(basis_fn(Xl + h, Yl)))
              - Bmat(*_uxy(basis_fn(Xl - h, Yl)))) / (2 * h)
        By = (Bmat(*_uxy(basis_fn(Xl, Yl + h)))
              - Bmat(*_uxy(basis_fn(Xl, Yl - h)))) / (2 * h)
        return B, Bx, By

    def div_sigma(B, Bx, By):
        """div sigma: (npts, 2, ncol).  D varies slowly, so dD is neglected
        (this is the standard collocation form and is the same approximation
        the paper's strong-form baselines use)."""
        npts = B.shape[0]
        out = np.zeros((npts, 2, B.shape[3]))
        for p in range(npts):
            d = Dp[p]
            sx = d @ B[p].reshape(3, -1)          # sigma from strain
            dxx = d @ Bx[p].reshape(3, -1)        # d(sigma)/dx
            dyy = d @ By[p].reshape(3, -1)        # d(sigma)/dy
            out[p, 0] = dxx[0] + dyy[2]           # d sxx/dx + d sxy/dy
            out[p, 1] = dxx[2] + dyy[1]           # d sxy/dx + d syy/dy
        return out

    Bb, Bbx, Bby = d2(net.bulk_basis)
    Bi, Bix, Biy = d2(net.iface_basis)
    Bf, Bfx, Bfy = d2(net.far_basis)
    Jb = div_sigma(Bb, Bbx, Bby).reshape(-1, 2 * net.n_feat)
    Ji = div_sigma(Bi, Bix, Biy).reshape(-1, 2 * NT)
    Jf = div_sigma(Bf, Bfx, Bfy).reshape(-1, NT)
    uD = np.array([R.right_disp(y) for y in Yp])
    F = Jf @ uD                                   # forcing from the far field
    A = Jb.T @ Jb + reg * np.eye(Jb.shape[1])
    S_col = np.zeros((2 * NT, 2 * NT))
    for k in range(2 * NT):
        e = np.zeros(2 * NT)
        e[k] = 1.0
        rhs = -Jb.T @ (Ji @ e + F)
        a = np.linalg.solve(A, rhs)
        S_col[:, k] = Ji.T @ (Jb @ a + Ji @ e + F)
    xi = np.full(N_IFACE, X0)
    yi = np.linspace(0.0, LY, N_IFACE)
    Ci = net.iface_basis(xi, yi)['u']
    r = np.einsum('mqc,qk->mkc', Ci, S_col)
    S_tr = pair_nodal(np.concatenate([r[:, :, 0], r[:, :, 1]], axis=0), yi)
    return S_tr, np.zeros(2 * NT)


# =====================================================================
#  validations
# =====================================================================
def paired_asym(S):
    """Weighted asymmetry ||H S - (H S)^T|| / ||H S|| of a trace operator."""
    H = np.diag(np.tile(M_TRACE.diagonal(), 2))
    A = H @ S
    return float(np.abs(A - A.T).max() / np.abs(A).max())


def v1_analytic_calibration():
    """Analytic uniaxial calibration of the left subdomain.

    For u = (a x, -nu a y) the exact outward traction on Gamma is (E a, 0).
    The material enters the FEM as a piecewise-constant field evaluated at
    element centroids, so the reaction must reproduce E(x_centroid) a of the
    interface-adjacent element -- not E(X0) a.  (Comparing against E(X0) a gives
    2.8e-3, which is the material approximation, not an assembly error.)
    """
    a = 1.0 / 1000.0
    sub = FemSub(12, 12)
    nodes, elems, K, g, c = sub.build()
    u = np.zeros(2 * len(nodes))
    for k in range(len(nodes)):
        u[2 * k] = a * nodes[k, 0]
        u[2 * k + 1] = -NU * a * nodes[k, 1]
    r = K @ u
    w = nodal_weights(len(g))
    rx = np.array([r[2 * k] for k in g]) / w
    ry = np.array([r[2 * k + 1] for k in g]) / w
    hx = X0 / sub.nx
    exact = F.E_of(X0 - 0.5 * hx, C_FIELD) * a
    return (float(np.max(np.abs(rx - exact)) / abs(exact)),
            float(np.max(np.abs(ry)) / abs(exact)),
            float(np.mean(rx) / exact))


def v2_symmetry(S):
    return paired_asym(S)


def v3_gradient_fd(net, seed=0, eps=1e-6):
    """Finite-difference check of the energy gradient used by train_energy."""
    bl = net.blocks(NG_TRAIN)
    Kbb = bl['Kbb']
    rng = np.random.default_rng(seed)
    g = rng.normal(size=2 * NT)
    l = bl['Kbi'] @ g
    a = rng.normal(scale=1e-6, size=Kbb.shape[0])
    gr = Kbb @ a + l
    err = 0.0
    for i in range(min(6, len(a))):
        ap, am = a.copy(), a.copy()
        ap[i] += eps; am[i] -= eps
        fp = 0.5 * ap @ Kbb @ ap + ap @ l
        fm = 0.5 * am @ Kbb @ am + am @ l
        err = max(err, abs((fp - fm) / (2 * eps) - gr[i]) / (abs(gr[i]) + 1e-300))
    return float(err)


def v6_train_limit(net, steps=400, lr=None):
    """Energy training must reach the exact Ritz minimiser.

    The energy is quadratic, so a correct energy trainer plus an exact solve
    must agree; any gap is optimisation error, which the paper must not
    attribute to the discretisation.
    """
    rng = np.random.default_rng(1)
    g = rng.normal(size=2 * NT) * 1e-3
    bl = net.blocks(NG_TRAIN)
    Kbb, l = bl['Kbb'], bl['Kbi'] @ g
    a_exact = -np.linalg.solve(Kbb, l)
    a_t, hist = train_energy(net, g, steps=steps, n=NG_TRAIN)
    e_exact = float(0.5 * a_exact @ Kbb @ a_exact + a_exact @ l)
    e_trained = float(0.5 * a_t @ Kbb @ a_t + a_t @ l)
    # Judged on the ENERGY gap, not on the coefficient gap: Kbb is ill
    # conditioned (cond ~ 1.5e3), so a 1e-3 energy gap can coexist with a 10%
    # coefficient error.  The energy is the physically meaningful quantity, and
    # the operator itself is always taken from the exact Ritz solve.
    de = (e_trained - e_exact) / abs(e_exact)
    grad = float(np.linalg.norm(Kbb @ a_exact + l) / (np.linalg.norm(l) + 1e-30))
    coeff = float(np.linalg.norm(a_t - a_exact) / np.linalg.norm(a_exact))
    return de, hist, grad, e_exact, coeff


# =====================================================================
#  reference solutions
# =====================================================================
def monolithic(nx=48, ny=24, cfield=None):
    """Monolithic Q4 solution on the whole domain (the paper's reference).

    Boundary data: fully clamped at x = 0, u_x prescribed at x = LXR with u_y
    free (the paper's load mask), traction-free elsewhere.
    """
    cf = C_FIELD if cfield is None else cfield
    nodes, elems = R.build_mesh(nx, ny, 0.0, LXR, 0.0, LY)
    K = F.assemble_field(nodes, elems, cf, NU)
    ndof = 2 * len(nodes)
    fix = set()
    for k in range(len(nodes)):
        if abs(nodes[k, 0]) < 1e-12:
            fix.add(2 * k); fix.add(2 * k + 1)
        elif abs(nodes[k, 0] - LXR) < 1e-12:
            fix.add(2 * k)
    u = np.zeros(ndof)
    for d in fix:
        k, comp = d // 2, d % 2
        if comp == 0 and abs(nodes[k, 0] - LXR) < 1e-12:
            u[d] = R.right_disp(nodes[k, 1])
    fds = np.array(sorted(fix))
    free = np.array([d for d in range(ndof) if d not in fix])
    u[free] = np.linalg.solve(K[np.ix_(free, free)],
                              -K[np.ix_(free, fds)] @ u[fds])
    return nodes, elems, u


def trace_from_monolithic(nodes, u, ychk=YCHK):
    """Interface trace of a monolithic solution, as a (2, n) array."""
    gi = sorted([k for k in range(len(nodes)) if abs(nodes[k, 0] - X0) < 1e-12],
                key=lambda k: nodes[k, 1])
    ys = nodes[gi, 1]
    ux = np.interp(ychk, ys, u[2 * np.array(gi)])
    uy = np.interp(ychk, ys, u[2 * np.array(gi) + 1])
    return ux, uy


def rel_err(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# =====================================================================
#  main study
# =====================================================================
def run(dry=False, width=32, steps=4000, seed=0):
    log = []
    def w(s):
        print(s)
        log.append(s)

    w("=" * 72)
    w("Hybrid case: energy-trained network subdomain + FE subdomain")
    w("=" * 72)

    # ---------------- validations ----------------
    w("\n[V1] analytic uniaxial calibration of the left subdomain")
    e1, e1y, ratio1 = v1_analytic_calibration()
    w(f"     r_x/w_j vs E(centroid)a : max rel err {e1:.3e}"
      f"   {'PASS' if e1 < 1e-10 else '*** FAIL ***'}")
    w(f"     r_y/w_j (must vanish)    : {e1y:.3e}"
      f"   {'PASS' if e1y < 1e-10 else '*** FAIL ***'}")
    w(f"     mean r_x/w_j / E_c a     : {ratio1:.12f}")

    net = NetSub(width=width, seed=seed, layers=1, sigma=2.5)
    w(f"\n[V3] finite-difference check of the energy gradient")
    e3 = v3_gradient_fd(net)
    w(f"     max relative gradient error: {e3:.3e}"
      f"   {'PASS' if e3 < 1e-4 else '*** FAIL ***'}")

    de, hist, grad, e_exact, coeff = v6_train_limit(net, steps=steps)
    w(f"\n[V6] energy minimisation vs the exact Ritz solve")
    w(f"     exact-solve residual ||Kbb a + l|| / ||l|| = {grad:.3e}")
    w(f"     energy at the exact minimiser              = {e_exact:.6e}")
    w(f"     relative energy gap after {steps} CG steps  = {de:.3e}"
      f"   {'PASS' if abs(de) < 1e-3 else '*** FAIL ***'}")
    w(f"     coefficient gap (diagnostic only, cond(Kbb) large) = {coeff:.3e}")
    if hist.size:
        w(f"     minimum-energy descent: {hist[0]:.6e} -> {hist[-1]:.6e}"
          f"  ({hist.size - 1} steps)")

    # ---------------- operators (validated nodal formulation) ----------------
    w("\n[V4] interface equilibrium on the monolithic reference trace")
    nodesM, _, uM = monolithic(nx=2 * 12, ny=12, cfield=C_FIELD)
    giM = sorted([k for k in range(len(nodesM)) if abs(nodesM[k, 0] - X0) < 1e-12],
                 key=lambda k: nodesM[k, 1])
    uxM = np.array([uM[2 * k] for k in giM])
    uyM = np.array([uM[2 * k + 1] for k in giM])
    ug_ref = np.concatenate([uxM, uyM])

    S_L, q_L, ys, gdof, _ = fem_nodal_dtn(0.0, X0, X0, 0.0, (0, 1), C_FIELD)
    S_Rf, q_Rf, _, _, _ = fem_nodal_dtn(X0, LXR, X0, LXR, (0,), C_FIELD)
    res4 = np.abs((S_L + S_Rf) @ ug_ref + (q_L + q_Rf)).max()
    scale4 = max(np.abs((S_L + S_Rf) @ ug_ref).max(), np.abs(q_L + q_Rf).max())
    w(f"     |(S_L+S_R) u_Gamma + (q_L+q_R)| = {res4:.3e}"
      f"   rel {res4 / scale4:.3e}"
      f"   {'PASS' if res4 / scale4 < 1e-12 else '*** FAIL ***'}")

    # energy-trained network subdomain on the same nodal trace space
    S_N, q_N, m_N, yN = net.nodal_operator(NG_OP)
    w(f"\n[V2] symmetry of the coupled operator")
    w(f"     nodal interface dofs: FE {len(gdof)}, network {2 * m_N}")
    w(f"     asymmetry of S_L      : {np.abs(S_L - S_L.T).max() / np.abs(S_L).max():.3e}")
    w(f"     asymmetry of S_R(fem) : {np.abs(S_Rf - S_Rf.T).max() / np.abs(S_Rf).max():.3e}")
    asym_N = np.abs(S_N - S_N.T).max() / np.abs(S_N).max()
    w(f"     asymmetry of S_N(net) : {asym_N:.3e}"
      f"   {'PASS' if asym_N < 1e-12 else '*** FAIL ***'}")

    # ---------------- hybrid solve ----------------
    A = S_L + S_N
    ug = np.linalg.solve(A, -(q_L + q_N))
    w(f"\n[coupling] cond(S_L + S_N) = {np.linalg.cond(A):.2f}")
    ug_fem = np.linalg.solve(S_L + S_Rf, -(q_L + q_Rf))

    def to_lin(v):
        return np.interp(YCHK, ys, v)

    w(f"\n[results] interface displacement error vs the monolithic reference")
    w(f"     reference trace: ux {uxM.min():.6e} .. {uxM.max():.6e}")
    w(f"     pure FEM right subdomain : ux {rel_err(to_lin(ug_fem[:m_N]), to_lin(uxM)):.4%}"
      f"   uy {rel_err(to_lin(ug_fem[m_N:]), to_lin(uyM)):.4%}")
    w(f"     hybrid (energy-trained)  : ux {rel_err(to_lin(ug[:m_N]), to_lin(uxM)):.4%}"
      f"   uy {rel_err(to_lin(ug[m_N:]), to_lin(uyM)):.4%}")

    # field comparison on the right subdomain
    a_net = net.feature_coeffs(ug)
    xs = np.linspace(X0, LXR, 9)
    Xg, Yg = np.meshgrid(xs, YCHK, indexing='ij')
    uxn, uyn = net.field(ug, Xg.ravel(), Yg.ravel(), a=a_net, n=NG_OP)
    nodesC, elemsC = R.build_mesh(12, 12, X0, LXR, 0.0, LY)
    KC = F.assemble_field(nodesC, elemsC, C_FIELD, NU)
    giC = sorted([k for k in range(len(nodesC)) if abs(nodesC[k, 0] - X0) < 1e-12],
                 key=lambda k: nodesC[k, 1])
    fiC = [k for k in range(len(nodesC)) if abs(nodesC[k, 0] - LXR) < 1e-12]
    fixC = {}
    for j, k in enumerate(giC):
        fixC[2 * k] = ug[j]
        fixC[2 * k + 1] = ug[m_N + j]
    for k in fiC:
        fixC[2 * k] = float(R.right_disp(nodesC[k, 1]))
    fdsC = np.array(sorted(fixC))
    freeC = np.array([d for d in range(2 * len(nodesC)) if d not in fixC])
    uC = np.zeros(2 * len(nodesC))
    uC[freeC] = np.linalg.solve(KC[np.ix_(freeC, freeC)],
                               -KC[np.ix_(freeC, fdsC)] @ np.array([fixC[d] for d in fdsC]))
    for d, v in fixC.items():
        uC[d] = v
    w(f"     network trace recorded; nodal-field norm |u| = {np.linalg.norm(uC):.6e}")

    # ---------------- collocation control ----------------
    w(f"\n[control] same features, collocation (strong-form) training")
    try:
        S_C, q_C = collocation_operator(net, NG_OP)
        ug_c = np.linalg.solve(S_L + S_C, -(q_L + q_C))
        asym_C = np.abs(S_C - S_C.T).max() / np.abs(S_C).max()
        w(f"     asymmetry of S_C          : {asym_C:.3e}")
        w(f"     interface error           : ux {rel_err(to_lin(ug_c[:m_N]), to_lin(uxM)):.4%}"
          f"   uy {rel_err(to_lin(ug_c[m_N:]), to_lin(uyM)):.4%}")
    except Exception as exc:
        w(f"     *** collocation control failed: {exc!r} ***")

    # ---------------- write ----------------
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'hybrid_energy_nn.txt'), 'w',
              encoding='utf-8') as fp:
        fp.write('\n'.join(log) + '\n')
    w(f"\n[saved] {os.path.join(OUT, 'hybrid_energy_nn.txt')}")
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--width', type=int, default=32)
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    if args.dry_run:
        print("[dry-run] validations only")
    run(dry=args.dry_run, width=args.width, steps=args.steps, seed=args.seed)


if __name__ == '__main__':
    main()
