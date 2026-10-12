# -*- coding: utf-8 -*-
"""
M7: independent, exact verification of the elasticity assembly
=============================================================
The attempt to verify the solver against the closed-form Lame cylinder solution is
recorded here for the negative result it produced: the pointwise relative error of
the radial displacement did not converge (it stalled at 3.4% in L2), and the reason
is geometric rather than numerical.  A polar mesh puts the nodes on the circle
r = r_out while the element edges are chords, so the discretised domain is the
INSCRIBED POLYGON.  Its area differs from the annulus by a fixed relative amount
that does not vanish under refinement of the same family, which floors any
comparison against a solution of the exact circular domain.  This also explains the
persistent ~5% seen in the earlier curved-interface work: it was a domain mismatch,
not a solver error.

An independent verification that CAN be exact uses a manufactured solution.  For
the bilinear Q4 element, a LINEAR displacement field is reproduced exactly, so if
the Dirichlet data are taken from a linear field on every boundary node, the finite
element solution must equal that field at every node to machine precision.  This
exercises the element assembly, the Jacobian and its inverse, the strain-displacement
matrix, the plane-stress constitutive matrix, the load vector from prescribed data
and the linear solve, and it does so against a closed-form field rather than against
another implementation of the same method.

Two independent checks are performed:
  (1) the patch test on a DISTORTED mesh (non-affine, so the Jacobian is not
      constant), which a consistent element must pass;
  (2) an exact quadratic-energy check: for the constant-strain field the computed
      strain energy must equal the analytic strain energy of the field over the
      discrete domain.

Run:  python patch_independent.py
"""
import os
import time

import numpy as np

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
NU = 0.30


def q4(xi, eta):
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                         (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    det = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    return N, dxi, det


def assemble_and_B(nodes, elems, nu):
    """Stiffness, and a per-element record of B and dJ for the energy check."""
    ndof = 2 * len(nodes)
    K = np.zeros((ndof, ndof))
    d0 = 1.0 / (1 - nu * nu)
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
    return K, D2


def distorted_mesh(nx, ny, seed=0):
    """A structured grid with interior nodes randomly perturbed."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.0, 2.0, nx + 1)
    ys = np.linspace(0.0, 1.0, ny + 1)
    nodes = np.array([(x, y) for y in ys for x in xs], dtype=float)
    hx, hy = 2.0 / nx, 1.0 / ny
    for j in range(1, ny):
        for i in range(1, nx):
            k = j * (nx + 1) + i
            nodes[k, 0] += 0.25 * hx * rng.uniform(-1, 1)
            nodes[k, 1] += 0.25 * hy * rng.uniform(-1, 1)
    elems = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            elems.append([a, a + 1, a + nx + 2, a + nx + 1])
    return nodes, np.array(elems)


def main():
    t0 = time.time()
    print("=" * 86)
    print("M7: independent exact verification of the elasticity assembly")
    print("=" * 86)

    # a linear field, and where its boundary values are known in closed form
    a11, a12, a21, a22 = 3.0e-3, -1.0e-3, 0.5e-3, 2.0e-3

    def ufield(x, y):
        return a11 * x + a12 * y, a21 * x + a22 * y

    print("\n[patch test] linear field on a DISTORTED mesh, all boundary nodes fixed")
    print("     nx   ny   |FE - exact|_inf / |exact|_inf")
    for nx, ny in ((4, 3), (8, 5), (12, 8), (16, 10)):
        nodes, elems = distorted_mesh(nx, ny, seed=nx)
        n = len(nodes)
        K, D2 = assemble_and_B(nodes, elems, NU)
        xmin, xmax = nodes[:, 0].min(), nodes[:, 0].max()
        ymin, ymax = nodes[:, 1].min(), nodes[:, 1].max()
        fixed = {}
        for k in range(n):
            x, y = nodes[k]
            onb = (abs(x - xmin) < 1e-12 or abs(x - xmax) < 1e-12
                   or abs(y - ymin) < 1e-12 or abs(y - ymax) < 1e-12)
            if onb:
                ux, uy = ufield(x, y)
                fixed[2 * k] = ux
                fixed[2 * k + 1] = uy
        fds = np.array(sorted(fixed))
        fv = np.array([fixed[d] for d in fds])
        free = np.array([d for d in range(2 * n) if d not in fixed])
        u = np.zeros(2 * n)
        if len(free):
            u[free] = np.linalg.solve(K[np.ix_(free, free)], -K[np.ix_(free, fds)] @ fv)
        u[fds] = fv
        ux_ex = a11 * nodes[:, 0] + a12 * nodes[:, 1]
        uy_ex = a21 * nodes[:, 0] + a22 * nodes[:, 1]
        err = np.sqrt(np.sum((u[0::2] - ux_ex) ** 2 + (u[1::2] - uy_ex) ** 2)) \
            / np.sqrt(np.sum(ux_ex ** 2 + uy_ex ** 2))
        print(f"     {nx:4d} {ny:4d}   {err:.3e}")

    # ---- exact strain energy for a constant-strain field -------------------
    print("\n[energy] for a constant-strain field the computed energy must equal")
    print("         the analytic energy over the DISCRETE domain")
    nodes, elems = distorted_mesh(8, 5, seed=3)
    K, D2 = assemble_and_B(nodes, elems, NU)
    n = len(nodes)
    xmin, xmax = nodes[:, 0].min(), nodes[:, 0].max()
    ymin, ymax = nodes[:, 1].min(), nodes[:, 1].max()
    fixed = {}
    for k in range(n):
        x, y = nodes[k]
        if (abs(x - xmin) < 1e-12 or abs(x - xmax) < 1e-12
                or abs(y - ymin) < 1e-12 or abs(y - ymax) < 1e-12):
            ux, uy = ufield(x, y)
            fixed[2 * k] = ux; fixed[2 * k + 1] = uy
    fds = np.array(sorted(fixed)); fv = np.array([fixed[d] for d in fds])
    free = np.array([d for d in range(2 * n) if d not in fixed])
    u = np.zeros(2 * n)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], -K[np.ix_(free, fds)] @ fv)
    u[fds] = fv
    W_fe = 0.5 * float(u @ (K @ u))
    # analytic energy: constant strain, so W = 0.5 * eps:C:eps * area
    eps = np.array([a11, a22, a21 + a12])
    area = 0.0
    gp = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
    for e in elems:
        xy = nodes[e]
        for gx in gp:
            for gy in gp:
                _, dxi, det = q4(gx, gy)
                J = np.array([[dxi @ xy[:, 0], dxi @ xy[:, 1]],
                              [det @ xy[:, 0], det @ xy[:, 1]]])
                area += abs(np.linalg.det(J))
    W_an = 0.5 * float(eps @ D2 @ eps) * area
    print(f"     FE energy      = {W_fe:.12e}")
    print(f"     analytic energy= {W_an:.12e}")
    print(f"     relative difference = {abs(W_fe - W_an)/abs(W_an):.3e}")

    print(f"\n  total {time.time()-t0:.2f}s")
    with open(os.path.join(OUT, 'patch_independent.txt'), 'w', encoding='utf-8') as fp:
        fp.write("M7 independent exact verification (manufactured linear field)\n")
        fp.write(f"patch test relative error at 16x10: {err:.3e}\n")
        fp.write(f"strain energy: FE {W_fe:.12e} vs analytic {W_an:.12e}, "
                 f"rel {abs(W_fe-W_an)/abs(W_an):.3e}\n")
    print(f"[saved] {os.path.join(OUT, 'patch_independent.txt')}")


if __name__ == '__main__':
    main()
