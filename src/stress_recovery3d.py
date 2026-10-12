# -*- coding: utf-8 -*-
"""
三维应力恢复 / 平滑化
=====================
解决问题: 直接对位移做中心差分得到的应力噪声大(尤其 FEM 子域, 单元间应力不连续),
且 FEM 原始节点应力只是简单外推, 精度低。

本文提供三类应力场:
  (1) elem_gauss_stress : 单元 2x2x2 高斯点 FE 应力 (超收敛点, O(h^2) 精度)
  (2) spr_nodal         : SPR (Zienkiewicz-Zhu Superconvergent Patch Recovery)
                          对每个节点, 用其相关单元的高斯点应力做局部多项式最小二乘拟合,
                          再回代到节点 -> 得到光滑、超收敛的节点应力场
  (3) nn_stress_at      : NN 子域解析应力 (对多项式特征直接求导, 无差分噪声)

mixed_vm_spr 给出"FEM 子域 SPR + NN 子域解析"的混合 von Mises 场, 用于后处理与误差评估。
"""
import numpy as np
import coupled_solver3d as C

D = C.D


def _elem_geo(nodes, el):
    c0 = nodes[el[0]]
    c6 = nodes[el[6]]
    hx = (c6[0] - c0[0]) / 2.0
    hy = (c6[1] - c0[1]) / 2.0
    hz = (c6[2] - c0[2]) / 2.0
    return hx, hy, hz


def _BN(nodes, el, xi, eta, ze):
    hx, hy, hz = _elem_geo(nodes, el)
    N, dN = C.h8_shape(xi, eta, ze)
    dN = np.array(dN)
    dNx = dN[:, 0] / hx
    dNy = dN[:, 1] / hy
    dNz = dN[:, 2] / hz
    B = np.zeros((6, 24))
    for r in range(8):
        B[0, 3 * r] = dNx[r]
        B[1, 3 * r + 1] = dNy[r]
        B[2, 3 * r + 2] = dNz[r]
        B[3, 3 * r] = dNy[r];     B[3, 3 * r + 1] = dNx[r]
        B[4, 3 * r + 1] = dNz[r]; B[4, 3 * r + 2] = dNy[r]
        B[5, 3 * r + 2] = dNx[r]; B[5, 3 * r] = dNz[r]
    return N, B


def elem_gauss_stress(nodes, elems, u):
    """每个单元在 2x2x2 高斯点上的应力 (超收敛点)。
    返回 list[(eidx, [(xyz, stress6), ...8])]。"""
    g = np.polynomial.legendre.leggauss(2)[0]
    out = []
    for eidx, el in enumerate(elems):
        ue = np.array([u[3 * el[r] + c] for r in range(8) for c in range(3)])
        samples = []
        for xi in g:
            for eta in g:
                for ze in g:
                    N, B = _BN(nodes, el, xi, eta, ze)
                    x = sum(N[r] * nodes[el[r]][0] for r in range(8))
                    y = sum(N[r] * nodes[el[r]][1] for r in range(8))
                    z = sum(N[r] * nodes[el[r]][2] for r in range(8))
                    samples.append(((x, y, z), D @ (B @ ue)))
        out.append((eidx, samples))
    return out


def naive_nodal(nodes, elems, u):
    """朴素节点应力: 单元高斯点应力取平均后按节点简单算术平均 (未做任何拟合/光滑)。
    这是 SPR 要对比的基线 (经典做法)。返回 (nnodes, 6)。"""
    nn = len(nodes)
    acc = np.zeros((nn, 6))
    cnt = np.zeros(nn)
    for eidx, samples in elem_gauss_stress(nodes, elems, u):
        el = elems[eidx]
        s = np.mean([st for (_, st) in samples], axis=0)
        for r in range(8):
            acc[el[r]] += s
            cnt[el[r]] += 1
    return acc / np.maximum(cnt, 1)[:, None]


def _basis(X, Xn, degree):
    dx = X[:, 0] - Xn[0]
    dy = X[:, 1] - Xn[1]
    dz = X[:, 2] - Xn[2]
    one = np.ones_like(dx)
    if degree == 0:
        return np.column_stack([one])
    elif degree == 1:
        return np.column_stack([one, dx, dy, dz])
    elif degree == 2:
        return np.column_stack([one, dx, dy, dz, dx * dx, dy * dy, dz * dz,
                                dx * dy, dy * dz, dz * dx])
    else:
        raise ValueError(degree)


def _patch_basis(X, Xn, degree, fallback_degree=1, rtol=1e-8):
    """按秩安全地构造补片基。

    **仅检查样本点数是不够的**: 面/棱/角点补片的采样点会落在平面或直线上,
    使二次基的列线性相关 (例如 2x2x2 高斯点上 dx 只取 ±a, 于是 dx^2 ≡ a^2·1,
    与常数项共线 -> 二次基最多秩 7)。这与 Yue & Robbins, CNME 23 (2007) 1-10
    对 4 节点单元边界补片秩亏的讨论同源。
    因此这里用 SVD 显式判秩, 秩亏就自动降阶, 最差退化为常数补片。
    """
    for dg in sorted({degree, fallback_degree, 0}, reverse=True):
        A = _basis(X, Xn, dg)
        if A.shape[0] < A.shape[1]:
            continue
        sv = np.linalg.svd(A, compute_uv=False)
        if sv[0] <= 0.0 or sv[-1] > rtol * sv[0]:
            return A, dg
    return _basis(X, Xn, 0), 0


def spr_nodal(nodes, elems, u, degree=2, fallback_degree=1):
    """SPR 节点应力恢复: 每节点用其相关单元的高斯点应力做局部最小二乘拟合, 回代到节点。
    degree=2 需 >=10 个样本, 否则回退到 degree=1。返回 (nnodes, 6) 节点应力。"""
    nn = len(nodes)
    acc = [[] for _ in range(nn)]
    for eidx, samples in elem_gauss_stress(nodes, elems, u):
        el = elems[eidx]
        for (xyz, s) in samples:
            for r in range(8):
                acc[el[r]].append((xyz, s))
    nodal = np.zeros((nn, 6))
    degs = []
    for k in range(nn):
        S = acc[k]
        X = np.array([xyz for (xyz, _) in S])
        Xn = nodes[k]
        A, dg = _patch_basis(X, Xn, degree, fallback_degree)
        degs.append(dg)
        for comp in range(6):
            b = np.array([s[comp] for (_, s) in S])
            coef, *_ = np.linalg.lstsq(A, b, rcond=None)
            nodal[k, comp] = coef[0]      # 节点处 dx=dy=dz=0 -> 取常数项
    return nodal


def spr_nodal_stats(nodes, elems, u, degree=2, fallback_degree=1):
    """返回 (nodal, degs): 便于审计各节点实际使用的补片阶数。"""
    nn = len(nodes)
    acc = [[] for _ in range(nn)]
    for eidx, samples in elem_gauss_stress(nodes, elems, u):
        el = elems[eidx]
        for (xyz, s0) in samples:
            for r in range(8):
                acc[el[r]].append((xyz, s0))
    nodal = np.zeros((nn, 6))
    degs = np.zeros(nn, dtype=int)
    for k in range(nn):
        S = acc[k]
        X = np.array([xyz for (xyz, _) in S])
        A, dg = _patch_basis(X, nodes[k], degree, fallback_degree)
        degs[k] = dg
        for comp in range(6):
            b = np.array([s0[comp] for (_, s0) in S])
            coef, *_ = np.linalg.lstsq(A, b, rcond=None)
            nodal[k, comp] = coef[0]
    return nodal, degs


def interp_nodal_vm(nodes, elems, nodal_stress, x, y, z):
    """用 H8 形函数在单元内插值节点 von Mises 应力。"""
    nnx = None
    # 定位单元: 简单线性搜索 + 形函数局部坐标求解 (规则网格 -> 直接算)
    # 规则网格: 找到单元通过最近网格索引 (调用方保证网格规则)
    return None  # 见 interp_nodal_vm_regular


def interp_nodal_vm_regular(sol, nodal_stress, x, y, z):
    """针对规则 H8 网格: 定位单元 -> H8 形函数插值节点 von Mises。"""
    nodes = sol['nodes']
    nnx, nny, nnz = sol['nnx'], sol['nny'], sol['nnz']
    # 判断 x 跨度 (FEM 子域为 X0, 参考为 Lx)
    span = C.X0 if (sol.get('ufem') is not None and sol.get('ndofF') is not None) else C.Lx
    hx = span / (nnx - 1)
    hy = C.Ly / (nny - 1)
    hz = C.Lz / (nnz - 1)
    i = min(max(int(x / hx), 0), nnx - 2)
    j = min(max(int(y / hy), 0), nny - 2)
    k = min(max(int(z / hz), 0), nnz - 2)
    xl = 2.0 * (x - i * hx) / hx - 1.0
    yl = 2.0 * (y - j * hy) / hy - 1.0
    zl = 2.0 * (z - k * hz) / hz - 1.0
    N, _ = C.h8_shape(xl, yl, zl)
    idx = [k * nny * nnx + j * nnx + i,
           k * nny * nnx + j * nnx + (i + 1),
           k * nny * nnx + (j + 1) * nnx + (i + 1),
           k * nny * nnx + (j + 1) * nnx + i,
           (k + 1) * nny * nnx + j * nnx + i,
           (k + 1) * nny * nnx + j * nnx + (i + 1),
           (k + 1) * nny * nnx + (j + 1) * nnx + (i + 1),
           (k + 1) * nny * nnx + (j + 1) * nnx + i]
    vm = np.array([C.von_mises(nodal_stress[k]) for k in idx])
    return float(N @ vm)


def nn_stress_at(sol, x, y, z):
    """NN 子域解析应力 (对多项式特征直接求导)。返回 (6,) 应力分量。"""
    coeffs = sol['coeffs']; feats = sol['feats']; nf = len(feats)
    strain = np.zeros(6)
    for a in range(nf):
        _, dx, dy, dz = C.feat_grad_3d(feats[a], x, y, z)
        for c in range(3):
            strain += coeffs[a, c] * C.strain_vec(c, dx, dy, dz)
    return D @ strain


def build_mixed_vm(sol, nodal_spr_FEM):
    """返回函数 f(x,y,z)->von Mises, 耦合解用 FEM-SPR / NN-解析混合; 参考解用 SPR。"""
    is_coupled = sol.get('ufem') is not None and sol.get('ndofF') is not None
    if is_coupled:
        def f(x, y, z):
            if x <= C.X0:
                return interp_nodal_vm_regular(sol, nodal_spr_FEM, x, y, z)
            return C.von_mises(nn_stress_at(sol, x, y, z))
        return f
    else:
        def f(x, y, z):
            return interp_nodal_vm_regular(sol, nodal_spr_FEM, x, y, z)
        return f
