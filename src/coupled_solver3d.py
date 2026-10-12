# -*- coding: utf-8 -*-
"""
三维结构 FEM - 神经网络(PINN/ELM 线性化) 耦合求解器
====================================================
以三维线弹性长方体为例:
  - 左半域 [0, X0] x [0, Ly] x [0, Lz]  用有限元(FEM, 8 节点六面体 H8)
  - 右半域 [X0, Lx] x [0, Ly] x [0, Lz] 用神经网络(多项式特征网络, 等效 PINN 线性化训练)
  - 界面 x = X0: 罚函数强耦合 (弱位移连续)
  - 右端 x = Lx: 给定位移 g(y,z) (按工况不同: 拉伸 / 弯曲 / 扭转 / 组合)
  - 左端 x = 0: 固定 u = 0

同一增广矩阵 Ku=f 一次求解 -> 同时得到 FEM 节点位移与 NN 系数 (同一个求解器)。
实现 4 种工况, 并与纯有限元(全区域细化网格 + Dirichlet 凝聚)参考解对比。

求解器基于 numpy (密集直接求解)。
"""
import numpy as np
import math, time, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ----------------------------------------------------------- 中文字体
_FONT = None
for _cand in ['Microsoft YaHei', 'SimHei', 'STSong', 'Arial Unicode MS']:
    try:
        font_manager.findfont(_cand, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [_cand]
        _FONT = _cand
        break
    except Exception:
        continue
plt.rcParams['axes.unicode_minus'] = False

# ----------------------------------------------------------- 物理参数
E_mat, NU = 2100.0, 0.30
Lx, Ly, Lz, X0 = 2.0, 1.0, 1.0, 1.0
G = E_mat / (2.0 * (1.0 + NU))

def make_D(E, nu):
    fac = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    D = np.zeros((6, 6))
    D[0, 0] = D[1, 1] = D[2, 2] = fac * (1.0 - nu)
    D[0, 1] = D[0, 2] = D[1, 0] = D[1, 2] = D[2, 0] = D[2, 1] = fac * nu
    D[3, 3] = D[4, 4] = D[5, 5] = fac * (1.0 - 2.0 * nu) / 2.0
    return D

D = make_D(E_mat, NU)

def von_mises(sig):
    sx, sy, sz, txy, tyz, tzx = sig
    return math.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
                     + 3.0 * (txy ** 2 + tyz ** 2 + tzx ** 2))

# ----------------------------------------------------------- 工况定义 (右端给定位移)
D0 = 0.01      # 拉伸 / 弯曲 位移幅值
TW = 0.01      # 扭转 位移幅值

def right_disp_3d(y, z, case):
    """右端 x=Lx 给定位移 (随工况变化)。返回 (g, mask)，g=(ux,uy,uz) 为位移值, mask 标记哪些分量被约束。
    仅约束与载荷相关的分量, 其余分量自由(由弹性求解), 避免端部位移约束引入虚假的 Saint-Venant 端部效应。"""
    yc, zc = Ly / 2.0, Lz / 2.0
    if case == 'tension':                 # 均匀拉伸: 仅约束轴向 -> 均匀正应力
        return ((D0, 0.0, 0.0), [True, False, False])
    elif case == 'bending':               # 弯曲: 仅约束轴向(沿 z 线性) -> 纯弯曲正应力
        return ((D0 * (z - zc) / (Lz / 2.0), 0.0, 0.0), [True, False, False])
    elif case == 'torsion':               # 扭转: 约束截面扭转位移 -> 纯剪应力, 轴向自由(由翘曲发展)
        return ((0.0, -TW * (z - zc), TW * (y - yc)), [False, True, True])
    elif case == 'combined':              # 组合: 拉伸+弯曲(轴向) + 扭转(横向) 全部约束
        return ((D0 + D0 * (z - zc) / (Lz / 2.0), -TW * (z - zc), TW * (y - yc)), [True, True, True])
    else:
        raise ValueError(case)

CASE_CN = {'tension': '拉伸', 'bending': '弯曲', 'torsion': '扭转', 'combined': '拉+弯+扭组合'}

# ----------------------------------------------------------- 3D 网格 (H8)
def build_mesh_3d(nx, ny, nz, x0, x1, y0, y1, z0, z1):
    xs = [x0 + (x1 - x0) * i / nx for i in range(nx + 1)]
    ys = [y0 + (y1 - y0) * j / ny for j in range(ny + 1)]
    zs = [z0 + (z1 - z0) * k / nz for k in range(nz + 1)]
    nodes = []
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                nodes.append((xs[i], ys[j], zs[k]))
    nnx, nny, nnz = nx + 1, ny + 1, nz + 1
    elems = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                a = k * nny * nnx + j * nnx + i
                b = k * nny * nnx + j * nnx + (i + 1)
                c = k * nny * nnx + (j + 1) * nnx + (i + 1)
                d = k * nny * nnx + (j + 1) * nnx + i
                e = (k + 1) * nny * nnx + j * nnx + i
                f = (k + 1) * nny * nnx + j * nnx + (i + 1)
                g = (k + 1) * nny * nnx + (j + 1) * nnx + (i + 1)
                h = (k + 1) * nny * nnx + (j + 1) * nnx + i
                elems.append([a, b, c, d, e, f, g, h])
    return np.array(nodes, dtype=float), np.array(elems, dtype=int), nnx, nny, nnz

def h8_shape(xi, eta, ze):
    sgn = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
           (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    N, dN = [], []
    for (sx, sy, sz) in sgn:
        ni = 0.125 * (1 + sx * xi) * (1 + sy * eta) * (1 + sz * ze)
        N.append(ni)
        dN.append([0.125 * sx * (1 + sy * eta) * (1 + sz * ze),
                   0.125 * sy * (1 + sx * xi) * (1 + sz * ze),
                   0.125 * sz * (1 + sx * xi) * (1 + sy * eta)])
    return N, dN

def fem_assemble(nodes, elems):
    nn = len(nodes)
    ndof = 3 * nn
    K = np.zeros((ndof, ndof))
    g = np.polynomial.legendre.leggauss(2)[0]
    for el in elems:
        c0 = nodes[el[0]]
        c6 = nodes[el[6]]
        hx = (c6[0] - c0[0]) / 2.0
        hy = (c6[1] - c0[1]) / 2.0
        hz = (c6[2] - c0[2]) / 2.0
        for xi in g:
            for eta in g:
                for ze in g:
                    N, dN = h8_shape(xi, eta, ze)
                    dNx = np.array([dN[r][0] / hx for r in range(8)])
                    dNy = np.array([dN[r][1] / hy for r in range(8)])
                    dNz = np.array([dN[r][2] / hz for r in range(8)])
                    B = np.zeros((6, 24))
                    for r in range(8):
                        B[0, 3 * r] = dNx[r]
                        B[1, 3 * r + 1] = dNy[r]
                        B[2, 3 * r + 2] = dNz[r]
                        B[3, 3 * r] = dNy[r];     B[3, 3 * r + 1] = dNx[r]
                        B[4, 3 * r + 1] = dNz[r]; B[4, 3 * r + 2] = dNy[r]
                        B[5, 3 * r + 2] = dNx[r]; B[5, 3 * r] = dNz[r]
                    # H8 自然坐标 ξ,η,ζ ∈ [-1,1] -> 物理单元尺寸 (2hx)(2hy)(2hz)
                    # 雅可比 J = (Δx/2)(Δy/2)(Δz/2) = hx·hy·hz (leggauss(2) 权重为 1, 已在循环中隐式求和)
                    detJ = hx * hy * hz
                    Ke = detJ * (B.T @ D @ B)
                    idx = [3 * el[r] + c for r in range(8) for c in range(3)]
                    K[np.ix_(idx, idx)] += Ke
    return K, ndof, nn

# ----------------------------------------------------------- NN 特征 (3D 多项式)
def make_features_3d(max_deg=3):
    feats = []
    for p in range(max_deg + 1):
        for q in range(max_deg + 1):
            for r in range(max_deg + 1):
                feats.append((p, q, r))
    return feats

def feat_val_3d(feat, x, y, z):
    xi = 2.0 * (x - X0) / (Lx - X0) - 1.0
    eta = 2.0 * y / Ly - 1.0
    ze = 2.0 * z / Lz - 1.0
    p, q, r = feat
    return (xi ** p) * (eta ** q) * (ze ** r)

def feat_grad_3d(feat, x, y, z):
    xi = 2.0 * (x - X0) / (Lx - X0) - 1.0
    eta = 2.0 * y / Ly - 1.0
    ze = 2.0 * z / Lz - 1.0
    dxi, det, dze = 2.0 / (Lx - X0), 2.0 / Ly, 2.0 / Lz
    p, q, r = feat
    val = (xi ** p) * (eta ** q) * (ze ** r)
    dx = (p * xi ** (p - 1) * eta ** q * ze ** r) * dxi if p > 0 else 0.0
    dy = (xi ** p * q * eta ** (q - 1) * ze ** r) * det if q > 0 else 0.0
    dz = (xi ** p * eta ** q * r * ze ** (r - 1)) * dze if r > 0 else 0.0
    return val, dx, dy, dz

def strain_vec(comp, dx, dy, dz):
    # 位移第 comp 分量场 u_comp = sum c_a phi_a 对应变向量的贡献 [epsx,epsy,epsz,gxy,gyz,gzx]
    sv = np.zeros(6)
    if comp == 0:
        sv[0] = dx;  sv[3] = dy;  sv[5] = dz
    elif comp == 1:
        sv[1] = dy;  sv[3] = dx;  sv[4] = dz
    else:
        sv[2] = dz;  sv[4] = dy;  sv[5] = dx
    return sv

def nn_assemble(feats, tikhonov=1e-2):
    nf = len(feats)
    n3 = 3 * nf
    KN = np.zeros((n3, n3))
    gx, gw = np.polynomial.legendre.leggauss(5)
    # 在 NN 区域 [X0,Lx]x[0,Ly]x[0,Lz] 上高斯积分 (注意乘以高斯权重 w_i*w_j*w_k)
    for xi, wxi in zip(gx, gw):
        x = 0.5 * ((Lx + X0) + (Lx - X0) * xi)
        for eta, weta in zip(gx, gw):
            y = Ly * (eta + 1.0) / 2.0
            for ze, wze in zip(gx, gw):
                z = Lz * (ze + 1.0) / 2.0
                detJ = ((Lx - X0) / 2.0) * (Ly / 2.0) * (Lz / 2.0)
                wgt = wxi * weta * wze
                M = np.zeros((n3, 6))   # 行: 3a+c, 列: 应变分量
                for a, fa in enumerate(feats):
                    _, dx, dy, dz = feat_grad_3d(fa, x, y, z)
                    for c in range(3):
                        M[3 * a + c, :] = strain_vec(c, dx, dy, dz)
                KN += detJ * wgt * (M @ D @ M.T)
    KN += tikhonov * np.eye(n3)
    return KN

# ----------------------------------------------------------- 耦合求解
def solve_coupled(case, nn_deg=3, rho=1e6, tik=1e-2, fnx=8, fny=4, fnz=4):
    nodes, elems, nnx, nny, nnz = build_mesh_3d(fnx, fny, fnz, 0.0, X0, 0.0, Ly, 0.0, Lz)
    KF, ndofF, nnF = fem_assemble(nodes, elems)
    feats = make_features_3d(nn_deg)
    nf = len(feats)
    KN = nn_assemble(feats, tikhonov=tik)
    n3 = 3 * nf
    ndof = ndofF + n3
    K = np.zeros((ndof, ndof))
    K[:ndofF, :ndofF] = KF
    K[ndofF:, ndofF:] = KN
    f = np.zeros(ndof)

    # 左端 x=0: 罚函数固定 u=0
    left = [k for k in range(nnF) if abs(nodes[k, 0]) < 1e-9]
    for k in left:
        for c in range(3):
            K[3 * k + c, 3 * k + c] += rho

    # 界面 x=X0: 罚函数弱位移连续 (完整 Gram 矩阵)
    interf = [k for k in range(nnF) if abs(nodes[k, 0] - X0) < 1e-9]
    for k in interf:
        xj, yj, zj = nodes[k]
        phis = [feat_val_3d(fe, xj, yj, zj) for fe in feats]
        for c in range(3):
            ix = 3 * k + c
            for a in range(nf):
                ja = ndofF + 3 * a + c
                for b in range(nf):
                    jb = ndofF + 3 * b + c
                    K[ja, jb] += rho * phis[a] * phis[b]
                K[ix, ja] -= rho * phis[a]
                K[ja, ix] -= rho * phis[a]
            K[ix, ix] += rho

    # 右端 x=Lx: 罚函数给定位移 g(y,z)
    gpy, gwy = np.polynomial.legendre.leggauss(8)
    gpz, gwz = np.polynomial.legendre.leggauss(8)
    for ey, wy in zip(gpy, gwy):
        yy = Ly * (ey + 1.0) / 2.0
        for ez, wz in zip(gpz, gwz):
            zz = Lz * (ez + 1.0) / 2.0
            g, mask = right_disp_3d(yy, zz, case)
            w = wy * wz * (Ly / 2.0) * (Lz / 2.0)
            phs = [feat_val_3d(feats[a], Lx, yy, zz) for a in range(nf)]
            for c in range(3):
                if not mask[c]:
                    continue
                gc = g[c]
                for a in range(nf):
                    ja = ndofF + 3 * a + c
                    for b in range(nf):
                        jb = ndofF + 3 * b + c
                        K[ja, jb] += rho * phs[a] * phs[b] * w
                    f[ja] += rho * gc * phs[a] * w

    u = np.linalg.solve(K, f)
    ufem = u[:ndofF]
    coeffs = u[ndofF:].reshape(nf, 3)
    return dict(nodes=nodes, elems=elems, nnx=nnx, nny=nny, nnz=nnz,
                ndofF=ndofF, nnF=nnF, ufem=ufem, coeffs=coeffs, feats=feats,
                interf=interf, left=left, case=case)

# ----------------------------------------------------------- 纯有限元参考解
def solve_reference_3d(case, nx=16, ny=4, nz=4):
    nodes, elems, nnx, nny, nnz = build_mesh_3d(nx, ny, nz, 0.0, Lx, 0.0, Ly, 0.0, Lz)
    K, ndof, nn = fem_assemble(nodes, elems)
    left = set(k for k in range(nn) if abs(nodes[k, 0]) < 1e-9)
    right = set(k for k in range(nn) if abs(nodes[k, 0] - Lx) < 1e-9)
    fixed = set()
    ufix = {}
    for k in range(nn):
        for c in range(3):
            if k in left:
                fixed.add(3 * k + c); ufix[3 * k + c] = 0.0
            if k in right:
                g, mask = right_disp_3d(nodes[k, 1], nodes[k, 2], case)
                if mask[c]:
                    fixed.add(3 * k + c)
                    ufix[3 * k + c] = g[c]
    free = [c for c in range(ndof) if c not in fixed]
    N = len(free)
    Kr = np.zeros((N, N))
    fr = np.zeros(N)
    for i, ci in enumerate(free):
        for j, cj in enumerate(free):
            Kr[i, j] = K[ci, cj]
        for comp in fixed:
            fr[i] -= K[ci, comp] * ufix[comp]
    ur = np.linalg.solve(Kr, fr)
    u = np.zeros(ndof)
    for i, comp in enumerate(free):
        u[comp] = ur[i]
    for comp in fixed:
        u[comp] = ufix[comp]
    return dict(nodes=nodes, elems=elems, nnx=nnx, nny=nny, nnz=nnz,
                ndof=ndof, nn=nn, u=u, case=case)

# ----------------------------------------------------------- 场插值
def fem_interp(sol, x, y, z):
    nodes = sol['nodes']
    nnx, nny, nnz = sol['nnx'], sol['nny'], sol['nnz']
    u = sol['ufem'] if 'ufem' in sol else sol['u']
    hx = (X0 if sol.get('ndofF') else Lx) / (nnx - 1)
    # 判断该 sol 属于左半域(FEM 子域)还是全区域参考
    span_x = X0 if sol.get('ndofF') is not None and 'ufem' in sol else Lx
    # 注意: 参考解 span_x = Lx; 耦合 FEM 子域 span_x = X0
    if sol.get('ufem') is not None and sol.get('ndofF') is not None:
        span_x = X0
        nx = nnx - 1
    else:
        span_x = Lx
        nx = nnx - 1
    hy = Ly / (nny - 1)
    hz = Lz / (nnz - 1)
    i = min(int(x / hx), nx - 1); j = min(int(y / hy), nny - 2); k = min(int(z / hz), nnz - 2)
    xl = (x - i * hx) / hx * 2 - 1
    yl = (y - j * hy) / hy * 2 - 1
    zl = (z - k * hz) / hz * 2 - 1
    N, _ = h8_shape(xl, yl, zl)
    out = [0.0, 0.0, 0.0]
    for r in range(8):
        if r == 0: ni0 = k * nny * nnx + j * nnx + i
        elif r == 1: ni0 = k * nny * nnx + j * nnx + (i + 1)
        elif r == 2: ni0 = k * nny * nnx + (j + 1) * nnx + (i + 1)
        elif r == 3: ni0 = k * nny * nnx + (j + 1) * nnx + i
        elif r == 4: ni0 = (k + 1) * nny * nnx + j * nnx + i
        elif r == 5: ni0 = (k + 1) * nny * nnx + j * nnx + (i + 1)
        elif r == 6: ni0 = (k + 1) * nny * nnx + (j + 1) * nnx + (i + 1)
        else: ni0 = (k + 1) * nny * nnx + (j + 1) * nnx + i
        for c in range(3):
            out[c] += N[r] * u[3 * ni0 + c]
    return tuple(out)

def nn_point_disp_3d(sol, x, y, z):
    coeffs = sol['coeffs']
    feats = sol['feats']
    ux = sum(coeffs[a, 0] * feat_val_3d(feats[a], x, y, z) for a in range(len(feats)))
    uy = sum(coeffs[a, 1] * feat_val_3d(feats[a], x, y, z) for a in range(len(feats)))
    uz = sum(coeffs[a, 2] * feat_val_3d(feats[a], x, y, z) for a in range(len(feats)))
    return (ux, uy, uz)

def disp_at(sol, x, y, z):
    if sol.get('ufem') is not None and sol.get('ndofF') is not None:
        # 耦合解
        if x <= X0:
            return fem_interp(sol, x, y, z)
        else:
            return nn_point_disp_3d(sol, x, y, z)
    else:
        return fem_interp(sol, x, y, z)

def stress_at(sol, x, y, z, h=1e-3):
    def f_(xx, yy, zz):
        return disp_at(sol, xx, yy, zz)
    up = f_(x + h, y, z); um = f_(x - h, y, z)
    vp = f_(x, y + h, z); vm = f_(x, y - h, z)
    wp = f_(x, y, z + h); wm = f_(x, y, z - h)
    exx = (up[0] - um[0]) / (2 * h)
    eyy = (vp[1] - vm[1]) / (2 * h)
    ezz = (wp[2] - wm[2]) / (2 * h)
    gxy = (up[1] - um[1]) / (2 * h) + (vp[0] - vm[0]) / (2 * h)
    gyz = (vp[2] - vm[2]) / (2 * h) + (wp[1] - wm[1]) / (2 * h)
    gzx = (wp[0] - wm[0]) / (2 * h) + (up[2] - um[2]) / (2 * h)
    sig = D @ np.array([exx, eyy, ezz, gxy, gyz, gzx])
    return von_mises(sig)

# ----------------------------------------------------------- 误差指标
def l2_disp_error(csol, rsol):
    nx, ny, nz = 30, 16, 16
    num = 0.0; den = 0.0
    for i in range(nx):
        x = Lx * (i + 0.5) / nx
        for j in range(ny):
            y = Ly * (j + 0.5) / ny
            for k in range(nz):
                z = Lz * (k + 0.5) / nz
                uc = disp_at(csol, x, y, z)
                ur = disp_at(rsol, x, y, z)
                for c in range(3):
                    num += (uc[c] - ur[c]) ** 2
                    den += ur[c] ** 2
    return math.sqrt(num / den) if den > 0 else 0.0

def l2_vm_error(csol, rsol):
    nx, ny, nz = 30, 16, 16
    num = 0.0; den = 0.0
    for i in range(nx):
        x = Lx * (i + 0.5) / nx
        for j in range(ny):
            y = Ly * (j + 0.5) / ny
            for k in range(nz):
                z = Lz * (k + 0.5) / nz
                sc = stress_at(csol, x, y, z)
                sr = stress_at(rsol, x, y, z)
                num += (sc - sr) ** 2
                den += sr ** 2
    return math.sqrt(num / den) if den > 0 else 0.0

# ----------------------------------------------------------- 切片云图 (对比)
def slice_cloud(csol, rsol, case, field='vm', slice_y=None, stress_fn_c=None, stress_fn_r=None):
    if slice_y is None:
        slice_y = Ly / 2.0
    sfc = stress_fn_c if stress_fn_c is not None else (lambda x, y, z: stress_at(csol, x, y, z))
    sfr = stress_fn_r if stress_fn_r is not None else (lambda x, y, z: stress_at(rsol, x, y, z))
    nsx, nsz = 60, 30
    cx = np.zeros((nsz, nsx)); rx = np.zeros((nsz, nsx)); ex = np.zeros((nsz, nsx))
    for j in range(nsz):
        z = Lz * (j + 0.5) / nsz
        for i in range(nsx):
            x = Lx * (i + 0.5) / nsx
            sc = sfc(x, slice_y, z) if field == 'vm' else math.hypot(*disp_at(csol, x, slice_y, z))
            sr = sfr(x, slice_y, z) if field == 'vm' else math.hypot(*disp_at(rsol, x, slice_y, z))
            cx[j, i] = sc; rx[j, i] = sr; ex[j, i] = abs(sc - sr)
    return cx, rx, ex

def plot_compare(csol, rsol, case, field, outdir, stress_fn_c=None, stress_fn_r=None, tag=''):
    cx, rx, ex = slice_cloud(csol, rsol, case, field, stress_fn_c=stress_fn_c, stress_fn_r=stress_fn_r)
    vmax = max(cx.max(), rx.max())
    if vmax <= 1e-12:
        vmax = 1.0
    cmap = 'jet'
    fig, axs = plt.subplots(1, 3, figsize=(15.0, 4.2), dpi=150)
    titles = [f'耦合解 ({CASE_CN[case]})', '纯有限元 (参考)', '误差 |Δ|']
    data = [cx, rx, ex]
    for ax, d, t in zip(axs, data, titles):
        if t.startswith('误差'):
            im = ax.imshow(d, origin='lower', extent=[0, Lx, 0, Lz], aspect='auto',
                           cmap='hot', vmin=0, vmax=ex.max())
        else:
            im = ax.imshow(d, origin='lower', extent=[0, Lx, 0, Lz], aspect='auto',
                           cmap=cmap, vmin=0, vmax=vmax)
        ax.axvline(X0, color='white', ls='--', lw=1.2)
        ax.set_title(t, fontsize=10)
        ax.set_xlabel('x'); ax.set_ylabel('z')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f'{CASE_CN[case]} — {("von Mises 应力" if field=="vm" else "位移幅值")} 中面(y={Ly/2:.2f})切片对比',
                 fontsize=12)
    fname = f'cmp3d_{case}_{field}.png'
    path = os.path.join(outdir, fname)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path

# ----------------------------------------------------------- 主程序
def main():
    outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    os.makedirs(outdir, exist_ok=True)
    cases = ['tension', 'bending', 'torsion', 'combined']
    rows = []
    t0 = time.time()
    for case in cases:
        t1 = time.time()
        csol = solve_coupled(case)
        rsol = solve_reference_3d(case)
        rel = l2_disp_error(csol, rsol)
        rvm = l2_vm_error(csol, rsol)
        # 界面位移残差
        resid = 0.0
        for k in csol['interf']:
            xj, yj, zj = csol['nodes'][k]
            un = nn_point_disp_3d(csol, xj, yj, zj)
            uf = (csol['ufem'][3 * k], csol['ufem'][3 * k + 1], csol['ufem'][3 * k + 2])
            resid = max(resid, math.hypot(un[0] - uf[0], un[1] - uf[1], un[2] - uf[2]))
        plot_compare(csol, rsol, case, 'vm', outdir)
        plot_compare(csol, rsol, case, 'disp', outdir)
        rows.append((case, rel, rvm, resid))
        print(f'[{case}] 用时{time.time()-t1:.1f}s | 位移L2={rel:.4%} 应力L2={rvm:.4%} 界面残差={resid:.3e}')
    # 汇总
    with open(os.path.join(outdir, 'summary3d.txt'), 'w', encoding='utf-8') as fp:
        fp.write('三维 FEM-NN 耦合 vs 纯有限元 多工况对比\n')
        fp.write(f'几何: {Lx}x{Ly}x{Lz}, 分割 x={X0}; FEM子域网格 8x4x4(H8), NN 多项式 deg=3\n')
        fp.write(f'参考: 全区域 20x10x10 均匀 H8 + Dirichlet 凝聚\n')
        fp.write(f'{"工况":<12}{"位移L2误差":>14}{"应力L2误差":>14}{"界面位移残差":>16}\n')
        for case, rel, rvm, resid in rows:
            fp.write(f'{CASE_CN[case]:<12}{rel:>13.4%}{rvm:>13.4%}{resid:>15.3e}\n')
    print(f'总用时 {time.time()-t0:.1f}s, 结果见 {outdir}')

if __name__ == '__main__':
    main()
