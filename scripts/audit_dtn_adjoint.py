# -*- coding: utf-8 -*-
"""
界面 DtN 算子自伴性专项审计
============================
发现: 由"点式角点面力 + 等权最小二乘投影"探测得到的 S_L 在 L2 加权意义下
      (H·S) 不自伴(相对 0.48), 最小特征值 < 0。需判定:
      (i) 是提取方式的离散缺陷, 还是理论错误?
      (ii) 是否影响耦合解精度?
      (iii) 哪种提取方式能恢复自伴?
对照 4 种面力→迹系数提取:
  A 现用: 点式角点(单侧) + 等权最小二乘
  B 点式角点(两侧平均) + 等权最小二乘
  C 点式角点(单侧) + 梯形加权最小二乘
  D 一致节点力(∫N_i t dΓ) + FE 质量阵求 t 的节点值 + 加权投影
"""
import _paths  # noqa: F401  (adds src/ to the path)
import math
import numpy as np
import stress_recovery2d as R

E, NU = 2100.0, 0.30
Lx, Ly, X0 = 2.0, 1.0, 1.0
G = E / (2 * (1 + NU))
D00 = E / (1 - NU * NU); D01 = NU * D00
D2 = np.array([[D00, D01, 0.0], [D01, D00, 0.0], [0.0, 0.0, G]])
NXS, NYS, NT = 12, 12, 6


def assemble(nodes, elems):
    ndof = 2 * len(nodes); K = np.zeros((ndof, ndof))
    gp = [-1 / math.sqrt(3), 1 / math.sqrt(3)]
    for e in elems:
        xy = nodes[e]
        for gx in gp:
            for gy in gp:
                N, dxi, det = R._q4(gx, gy)
                J = np.array([[dxi @ xy[:, 0], dxi @ xy[:, 1]], [det @ xy[:, 0], det @ xy[:, 1]]])
                dJ = np.linalg.det(J); iJ = np.linalg.inv(J)
                dNdx = iJ[0, 0] * dxi + iJ[0, 1] * det; dNdy = iJ[1, 0] * dxi + iJ[1, 1] * det
                B = np.zeros((3, 8))
                for k in range(4):
                    B[0, 2 * k] = dNdx[k]; B[1, 2 * k + 1] = dNdy[k]
                    B[2, 2 * k] = dNdy[k]; B[2, 2 * k + 1] = dNdx[k]
                idx = [2 * e[k] + c for k in range(4) for c in range(2)]
                K[np.ix_(idx, idx)] += B.T @ D2 @ B * dJ
    return K


def build(x0, x1, nx=NXS, ny=NYS):
    nodes, elems = R.build_mesh(nx, ny, x0, x1, 0.0, Ly)
    iface = sorted([k for k in range(len(nodes)) if abs(nodes[k, 0] - X0) < 1e-9],
                   key=lambda k: nodes[k, 1])
    return nodes, elems, iface


def solve_dirichlet(nodes, elems, K, iface, gfun, far=None):
    ndof = 2 * len(nodes); fix = {}
    for k in iface:
        fix[2 * k], fix[2 * k + 1] = gfun(nodes[k, 1])
    if far is None:
        xmin = nodes[:, 0].min()
        for k in [k for k in range(len(nodes)) if abs(nodes[k, 0] - xmin) < 1e-9]:
            fix[2 * k] = 0.0; fix[2 * k + 1] = 0.0
    else:
        xv, uxf = far
        for k in [k for k in range(len(nodes)) if abs(nodes[k, 0] - xv) < 1e-9]:
            fix[2 * k] = uxf(nodes[k, 1])
    fds = list(fix); fs = set(fds)
    free = [d for d in range(ndof) if d not in fs]
    u = np.zeros(ndof)
    if free:
        u[free] = np.linalg.solve(K[np.ix_(free, free)], -K[np.ix_(free, fds)] @ np.array([fix[d] for d in fds]))
    for d, v in fix.items():
        u[d] = v
    return u


def trace_basis(y, nt=NT):
    e = 2.0 * y / Ly - 1.0
    P = [1.0, e]
    for j in range(2, nt):
        P.append(((2 * j - 1) * e * P[j - 1] - (j - 1) * P[j - 2]) / j)
    return np.array(P[:nt], dtype=float)


def lam_edge(nodes, elems, u, iface, at_right, nx=NXS, ny=NYS, two_sided=False):
    """界面点式面力 (外法线约定)。two_sided=True 时对相邻两单元取平均。"""
    m = len(iface)
    acc = np.zeros((m, 2)); cnt = np.zeros(m)
    for j in range(ny):
        col = nx - 1 if at_right else 0
        e = elems[j * nx + col]; xy = nodes[e]
        pairs = [((1, -1), e[1]), ((1, 1), e[2])] if at_right else [((-1, -1), e[0]), ((-1, 1), e[3])]
        for nat, node in pairs:
            if node in iface:
                _, dxi, det = R._q4(*nat)
                J = np.array([[dxi @ xy[:, 0], dxi @ xy[:, 1]], [det @ xy[:, 0], det @ xy[:, 1]]])
                iJ = np.linalg.inv(J)
                dNdx = iJ[0, 0] * dxi + iJ[0, 1] * det; dNdy = iJ[1, 0] * dxi + iJ[1, 1] * det
                ux = u[2 * e]; uy = u[2 * e + 1]
                exx = dNdx @ ux; eyy = dNdy @ uy; exy = 0.5 * (dNdy @ ux + dNdx @ uy)
                sig = D2 @ np.array([exx, eyy, 2 * exy])
                sgn = 1.0 if at_right else -1.0
                k = iface.index(node)
                if two_sided:
                    acc[k] += np.array([sgn * sig[0], sgn * sig[2]]); cnt[k] += 1
                else:
                    acc[k] = np.array([sgn * sig[0], sgn * sig[2]]); cnt[k] = 1
    return acc / np.maximum(cnt, 1)[:, None]


def nodal_forces(nodes, elems, u, iface, at_right, nx=NXS, ny=NYS):
    """一致节点力: 对边界边做 1D 高斯积分 ∫ N_i t dΓ (t 由单元应力给出)。"""
    m = len(iface)
    acc = np.zeros((m, 2)); cnt = np.zeros(m)
    g1, w1 = np.polynomial.legendre.leggauss(3)
    for j in range(ny):
        col = nx - 1 if at_right else 0
        e = elems[j * nx + col]; xy = nodes[e]
        # 界面边: at_right -> 节点 (1,2); 否则 (0,3)
        nids = [e[1], e[2]] if at_right else [e[0], e[3]]
        pts = xy[1:3] if at_right else xy[[0, 3]]
        L = np.linalg.norm(pts[1] - pts[0])
        sgn = 1.0 if at_right else -1.0
        for gs, ws in zip(g1, w1):
            N1 = 0.5 * (1 - gs); N2 = 0.5 * (1 + gs)
            # 该积分点处的单元应力 (用单元形函数插值应变)
            xi = gs if at_right else -gs          # 边所在单元的局部坐标
            eta = -1.0 if at_right else -1.0
            # 用边界边的自然坐标 -> 单元自然坐标
            if at_right:
                xin, etan = 1.0, gs
            else:
                xin, etan = -1.0, gs
            _, dxi, det = R._q4(xin, etan)
            J = np.array([[dxi @ xy[:, 0], dxi @ xy[:, 1]], [det @ xy[:, 0], det @ xy[:, 1]]])
            iJ = np.linalg.inv(J)
            dNdx = iJ[0, 0] * dxi + iJ[0, 1] * det; dNdy = iJ[1, 0] * dxi + iJ[1, 1] * det
            ux = u[2 * e]; uy = u[2 * e + 1]
            exx = dNdx @ ux; eyy = dNdy @ uy; exy = 0.5 * (dNdy @ ux + dNdx @ uy)
            sig = D2 @ np.array([exx, eyy, 2 * exy])
            t = np.array([sgn * sig[0], sgn * sig[2]])
            for nd, Nv in zip(nids, [N1, N2]):
                k = iface.index(nd)
                acc[k] += t * Nv * ws * (L / 2.0)
    return acc


def project_w(v, ys, w=None):
    P = np.array([trace_basis(y) for y in ys])
    if w is None:
        w = np.ones(len(ys))
    W = np.diag(w)
    return np.linalg.solve(P.T @ W @ P, P.T @ W @ v)


def build_S(side, mode, nx=NXS, ny=NYS, two_sided=False, weighted=False):
    x0, x1 = (0.0, X0) if side == 'L' else (X0, Lx)
    nodes, elems, iface = build(x0, x1, nx, ny)
    K = assemble(nodes, elems)
    ys = np.array([nodes[k, 1] for k in iface])
    at_right = (side == 'L')
    W = None
    if weighted:
        W = np.ones(len(ys)); W[0] = W[-1] = 0.5
    S = np.zeros((2 * NT, 2 * NT))
    for c in range(2):
        for j in range(NT):
            k = c * NT + j

            def gf(y, c=c, j=j):
                p = trace_basis(y)[j]
                return (p, 0.0) if c == 0 else (0.0, p)
            far = None if side == 'L' else (Lx, lambda y: 0.0)
            u = solve_dirichlet(nodes, elems, K, iface, gf, far)
            if mode == 'nodal':
                lam = nodal_forces(nodes, elems, u, iface, at_right, nx, ny)
                # 一致节点力 -> 用一维质量阵还原节点面力值 -> 再投影
                h = Ly / ny
                _N = len(iface)
                Mm = (h / 6.0) * (2 * np.eye(_N) + np.diag(np.ones(_N - 1), 1) + np.diag(np.ones(_N - 1), -1))
                lam = np.linalg.solve(Mm, lam)
            else:
                lam = lam_edge(nodes, elems, u, iface, at_right, nx, ny, two_sided)
            S[:NT, k] = project_w(lam[:, 0], ys, W)
            S[NT:, k] = project_w(lam[:, 1], ys, W)
    return S, nodes, elems, iface, ys


def l2_weights(nt=NT):
    yg, wg = np.polynomial.legendre.leggauss(60)
    yy = 0.5 * (yg + 1.0) * Ly; ww = wg * (Ly / 2.0)
    Pb = np.array([trace_basis(y, nt) for y in yy])
    return np.array([float(np.sum(ww * Pb[:, i] * Pb[:, i])) for i in range(nt)])


def main():
    print("=== DtN 算子自伴性专项审计 ===")
    hv = l2_weights()
    H = np.diag(np.tile(hv, 2))
    cases = [
        ('A 现用(单侧点式 + 等权LS)', dict(two_sided=False, weighted=False), 'edge'),
        ('B 两侧平均点式 + 等权LS', dict(two_sided=True, weighted=False), 'edge'),
        ('C 单侧点式 + 梯形加权LS', dict(two_sided=False, weighted=True), 'edge'),
        ('D 一致节点力 + 加权投影', dict(two_sided=False, weighted=True), 'nodal'),
    ]
    print(f"{'提取方式':<30}{'asym(HS)/|HS|':>16}{'min eig(HS)/|HS|':>20}{'cond':>10}")
    store = {}
    for name, kw, mode in cases:
        mode_ = 'nodal' if mode == 'nodal' else 'edge'
        SL, *_ = build_S('L', mode_, **kw)
        SR, *_ = build_S('R', mode_, **kw)
        HS = H @ SL
        asym = np.abs(HS - HS.T).max() / np.abs(HS).max()
        ev = np.linalg.eigvalsh(0.5 * (HS + HS.T))
        print(f"{name:<30}{asym:>16.3e}{ev.min()/np.abs(ev).max():>20.3e}{np.linalg.cond(SL):>10.1f}")
        store[name] = (SL, SR)
    # 精度对照: (S_L+S_R)g = -q 与整体 FEM 界面位移
    print("\n界面解精度 (与同一网格整体 FEM 比较):")
    # 整体 FEM
    nF, eF = R.build_mesh(2 * NXS, NYS, 0.0, Lx, 0.0, Ly)
    KF = assemble(nF, eF)
    ndof = 2 * len(nF)
    left = np.where(np.abs(nF[:, 0]) < 1e-9)[0]; right = np.where(np.abs(nF[:, 0] - Lx) < 1e-9)[0]
    fix = {}
    for k in left:
        fix[2 * k] = 0.0; fix[2 * k + 1] = 0.0
    for k in right:
        fix[2 * k] = R.right_disp(nF[k, 1])
    fds = np.array(list(fix)); fs = set(fds.tolist())
    free = np.array([d for d in range(ndof) if d not in fs])
    uM = np.zeros(ndof)
    uM[free] = np.linalg.solve(KF[np.ix_(free, free)], -KF[np.ix_(free, fds)] @ np.array([fix[d] for d in fds]))
    for d, v in fix.items():
        uM[d] = v
    ifn = sorted(np.where(np.abs(nF[:, 0] - X0) < 1e-9)[0], key=lambda k: nF[k, 1])
    yi = np.linspace(0, Ly, 41)
    gM = np.column_stack([np.interp(yi, nF[ifn, 1], uM[2 * np.array(ifn)]),
                          np.interp(yi, nF[ifn, 1], uM[2 * np.array(ifn) + 1])])
    for name, (SL, SR) in store.items():
        # q 用与 S 相同的提取方式
        mode_ = 'nodal' if 'nodal' in name or name.startswith('D') else 'edge'
        kw = dict(two_sided=('B' in name), weighted=('C' in name or 'D' in name))
        # 重新按同方式构造 q
        nodes, elems, iface = build(X0, Lx)
        K = assemble(nodes, elems)
        ys = np.array([nodes[k, 1] for k in iface])
        u = solve_dirichlet(nodes, elems, K, iface, lambda y: (0.0, 0.0), (Lx, R.right_disp))
        if mode_ == 'nodal':
            lam = nodal_forces(nodes, elems, u, iface, False)
            h = Ly / NYS
            _N = len(iface)
            Mm = (h / 6.0) * (2 * np.eye(_N) + np.diag(np.ones(_N - 1), 1) + np.diag(np.ones(_N - 1), -1))
            lam = np.linalg.solve(Mm, lam)
        else:
            lam = lam_edge(nodes, elems, u, iface, False, NXS, NYS, kw['two_sided'])
        W = None
        if kw['weighted']:
            W = np.ones(len(ys)); W[0] = W[-1] = 0.5
        q = np.concatenate([project_w(lam[:, 0], ys, W), project_w(lam[:, 1], ys, W)])
        g = np.linalg.solve(SL + SR, -q)
        gT = np.column_stack([np.array([sum(g[j] * trace_basis(y)[j] for j in range(NT)) for y in yi]),
                              np.array([sum(g[NT + j] * trace_basis(y)[j] for j in range(NT)) for y in yi])])
        rel = np.linalg.norm(gT - gM) / np.linalg.norm(gM)
        print(f"  {name:<30} 界面位移 vs 整体FEM = {rel:.4%}")


if __name__ == '__main__':
    main()
