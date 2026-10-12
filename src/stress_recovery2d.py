# -*- coding: utf-8 -*-
"""
二维应力恢复 / 平滑化 (与 3D 版 stress_recovery3d.py 对应)
==========================================================
- fem_assemble_np / solve2d : numpy 版 Q4 平面应力 FEM (用于快速参考解)
- elem_gauss_stress2d : 单元 2x2 高斯点应力 (超收敛点)
- naive_nodal2d       : 朴素节点平均 (基线)
- spr_nodal2d         : SPR (Zienkiewicz-Zhu) 逐应力分量二次补片恢复
- nn_stress2d         : 2D NN 子域解析应力
- interp_nodal_vm2d   : 规则网格上节点 von Mises 双线性插值
- mixed_vm2d          : FEM-SPR + NN-解析 混合应力场, 界面带内做投影平滑
"""
import math
import numpy as np

E_mat, NU = 2100.0, 0.30
Lx, Ly, X0 = 2.0, 1.0, 1.0
D00 = E_mat / (1 - NU * NU)
D01 = E_mat * NU / (1 - NU * NU)
D22 = E_mat / (2 * (1 + NU))   # G
D2 = np.array([[D00, D01, 0.0], [D01, D00, 0.0], [0.0, 0.0, D22]])


def right_disp(y, D0=0.01):
    t = 2.0 * y / Ly - 1.0
    return D0 * (1.0 - t * t)


def _q4(xi, eta):
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                         (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    det = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    return N, dxi, det


def _geo(xy):
    """xy: (4,2) 单元节点坐标 -> 2x2 高斯点的 dNdx,dNdy, detJ, 物理坐标"""
    out = []
    gp = [-1.0 / math.sqrt(3), 1.0 / math.sqrt(3)]
    for gx in gp:
        for gy in gp:
            N, dxi, det = _q4(gx, gy)
            J = np.array([[dxi @ xy[:, 0], dxi @ xy[:, 1]],
                          [det @ xy[:, 0], det @ xy[:, 1]]])
            dJ = np.linalg.det(J)
            iJ = np.linalg.inv(J)
            dNdx = iJ[0, 0] * dxi + iJ[0, 1] * det
            dNdy = iJ[1, 0] * dxi + iJ[1, 1] * det
            px = N @ xy[:, 0]; py = N @ xy[:, 1]
            out.append((dNdx, dNdy, dJ, px, py, N))
    return out


def assemble_np(nodes, elems):
    ndof = 2 * len(nodes)
    K = np.zeros((ndof, ndof))
    for e in elems:
        xy = nodes[e]
        for (dNdx, dNdy, dJ, px, py, N) in _geo(xy):
            B = np.zeros((3, 8))
            for k in range(4):
                B[0, 2 * k] = dNdx[k]
                B[1, 2 * k + 1] = dNdy[k]
                B[2, 2 * k] = dNdy[k]
                B[2, 2 * k + 1] = dNdx[k]
            Ke = B.T @ D2 @ B * dJ
            idx = [2 * e[k] + c for k in range(4) for c in range(2)]
            K[np.ix_(idx, idx)] += Ke
    return K


def build_mesh(nx, ny, x0, x1, y0, y1):
    xs = np.linspace(x0, x1, nx + 1); ys = np.linspace(y0, y1, ny + 1)
    nodes = np.array([(xs[i], ys[j]) for j in range(ny + 1) for i in range(nx + 1)])
    elems = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            elems.append([a, a + 1, a + 1 + (nx + 1), a + (nx + 1)])
    return nodes, np.array(elems)


def solve_reference_np(nx=56, ny=20):
    nodes, elems = build_mesh(nx, ny, 0.0, Lx, 0.0, Ly)
    K = assemble_np(nodes, elems)
    nn = len(nodes); ndof = 2 * nn
    left = np.where(np.abs(nodes[:, 0]) < 1e-9)[0]
    right = np.where(np.abs(nodes[:, 0] - Lx) < 1e-9)[0]
    fixed = []
    ufix = {}
    for k in left:
        for c in range(2):
            fixed.append(2 * k + c); ufix[2 * k + c] = 0.0
    for k in right:
        fixed.append(2 * k); ufix[2 * k] = right_disp(nodes[k, 1])
    fixed = np.array(fixed)
    free = np.array([d for d in range(ndof) if d not in set(fixed.tolist())])
    Kr = K[np.ix_(free, free)]
    fr = -K[np.ix_(free, fixed)] @ np.array([ufix[d] for d in fixed])
    ur = np.linalg.solve(Kr, fr)
    u = np.zeros(ndof)
    u[free] = ur
    for d, v in ufix.items():
        u[d] = v
    return dict(nodes=nodes, elems=elems, nx=nx, ny=ny, u=u, nn=nn)


def elem_gauss_stress2d(nodes, elems, u):
    out = []
    for eidx, e in enumerate(elems):
        xy = nodes[e]
        ux = u[2 * e]; uy = u[2 * e + 1]
        samples = []
        for (dNdx, dNdy, dJ, px, py, N) in _geo(xy):
            exx = dNdx @ ux; eyy = dNdy @ uy
            exy = 0.5 * (dNdy @ ux + dNdx @ uy)
            s = D2 @ np.array([exx, eyy, 2 * exy])   # [sxx,syy,sxy], exy 工程剪应变=2*exy
            samples.append(((px, py), s))
        out.append((eidx, samples))
    return out


def naive_nodal2d(nodes, elems, u):
    nn = len(nodes); acc = np.zeros((nn, 3)); cnt = np.zeros(nn)
    for eidx, samples in elem_gauss_stress2d(nodes, elems, u):
        s = np.mean([st for (_, st) in samples], axis=0)
        for k in elems[eidx]:
            acc[k] += s; cnt[k] += 1
    return acc / np.maximum(cnt, 1)[:, None]


def spr_nodal2d(nodes, elems, u, degree=2):
    """逐应力分量 SPR: 每节点用邻接单元高斯点做二次最小二乘拟合, 回代节点。"""
    nn = len(nodes)
    acc = [[] for _ in range(nn)]
    for eidx, samples in elem_gauss_stress2d(nodes, elems, u):
        for (xy, s) in samples:
            for k in elems[eidx]:
                acc[k].append((xy, s))
    nodal = np.zeros((nn, 3))
    for i in range(nn):
        S = acc[i]
        X = np.array([xy for (xy, _) in S]); Xn = nodes[i]
        dx = X[:, 0] - Xn[0]; dy = X[:, 1] - Xn[1]
        n = len(S)
        A = None
        if degree == 2 and n >= 6:
            Aq = np.column_stack([np.ones(n), dx, dy, dx * dx, dx * dy, dy * dy])
            sv = np.linalg.svd(Aq, compute_uv=False)
            # 边界/角点补片的采样点常使二次基列线性相关 -> 秩亏
            # (Yue & Robbins, Commun. Numer. Methods Eng. 23 (2007) 1-10)
            if sv[-1] > 1e-8 * sv[0]:
                A = Aq
        if A is None:
            A = np.column_stack([np.ones(n), dx, dy])
        for c in range(3):
            b = np.array([s[c] for (_, s) in S])
            coef, *_ = np.linalg.lstsq(A, b, rcond=None)
            nodal[i, c] = coef[0]
    return nodal


def vm_of(s):
    sxx, syy, sxy = s
    return math.sqrt(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy)


def interp_nodal_vm2d(nodes, elems, nodal_vm, nx, ny, span, x, y):
    hx = span / nx; hy = Ly / ny
    i = min(max(int(x / hx), 0), nx - 1)
    j = min(max(int(y / hy), 0), ny - 1)
    xi = 2.0 * (x - i * hx) / hx - 1.0
    eta = 2.0 * (y - j * hy) / hy - 1.0
    N, _, _ = _q4(xi, eta)
    a = j * (nx + 1) + i
    idx = [a, a + 1, a + 1 + (nx + 1), a + (nx + 1)]
    return float(np.array([nodal_vm[k] for k in idx]) @ N)


def nn_vm2d(sol, x, y):
    """2D NN 解析 von Mises (sol 来自 coupled_solver.solve_coupled)。"""
    import coupled_solver as C2
    return C2.nn_point_vm(sol, x, y)
