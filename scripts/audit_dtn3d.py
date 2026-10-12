# -*- coding: utf-8 -*-
r"""
三维界面 DtN (Steklov-Poincare) 算子: Galerkin 一致节点力  vs  点式配点传递
==========================================================================
把 2D 审计 (audit_dtn_adjoint.py / coupled_operator.py) 的结论推广到 3D,
使论文中 2D / 3D 使用同一套界面算子公式。

核心命题 (2D 已数值验证, 本脚本在 3D 复核)
------------------------------------------
设子域刚度矩阵 K 对称, 界面自由度集合 G, 其余自由自由度 F。定义

    S := K_GG - K_GF K_FF^{-1} K_FG          (Schur 补 = 一致节点反力算子)

则 S 对称半正定 (K 对称的直接后果)。取界面迹空间基 {p_i}, 令

    P[node, i] = p_i(y_node, z_node),   M[i,j] = \int_Gamma p_i p_j dGamma

Galerkin(一致节点力) 传递定义为

    S_tr := M^{-1} P^T S P      =>      M S_tr = P^T S P

右端是 P^T S P, 因 S 对称而 **恒对称** —— 自伴性由 K 的对称性保证, 与基的选取无关。
点式(配点)传递用单侧相邻单元的节点应力构造 R_pw, 背后没有刚度矩阵:

    M S_tr^pw = P^T R_pw        **一般不对称**

界面平衡取"外法线"约定: t_L^out + t_R^out = 0 => (S_tr_L + S_tr_R) c = -(b_L + b_R)。
"""
import numpy as np
import math, os, time
import coupled_solver3d as C
import stress_recovery3d as SR

import _paths  # noqa: F401  (adds src/ to the path)

E, NU = C.E_mat, C.NU
Lx, Ly, Lz, X0 = C.Lx, C.Ly, C.Lz, C.X0
D = C.D


# ------------------------------------------------------------------ 迹空间
def leg(n, e):
    """1D Legendre P_0..P_{n-1} (三项递推)。"""
    P = [1.0, e]
    for j in range(2, n):
        P.append(((2 * j - 1) * e * P[j - 1] - (j - 1) * P[j - 2]) / j)
    return np.array(P[:n], dtype=float)


def tb_list(tdeg):
    return [(a, b) for a in range(tdeg + 1) for b in range(tdeg + 1)]


def trace_basis(tb, y, z, tdeg):
    """张量积 Legendre: p_ab(y,z) = P_a(eta_y) P_b(eta_z)。"""
    ey = 2.0 * y / Ly - 1.0
    ez = 2.0 * z / Lz - 1.0
    Py = leg(tdeg + 1, ey)
    Pz = leg(tdeg + 1, ez)
    return np.array([Py[a] * Pz[b] for (a, b) in tb], dtype=float)


def trace_basis_mono(tb, y, z, tdeg):
    """对照用的 monomial 基 (Vandermonde 型, 用于展示条件数劣化)。"""
    ey = 2.0 * y / Ly - 1.0
    ez = 2.0 * z / Lz - 1.0
    return np.array([ey ** a * ez ** b for (a, b) in tb], dtype=float)


def l2_mass(tb, tdeg, nq=24, basis=trace_basis):
    r"""界面 L2 质量矩阵 M[i,j] = \int\int p_i p_j dy dz (Gauss 精确积分)。
    Legendre 基下应为对角阵。"""
    g, w = np.polynomial.legendre.leggauss(nq)
    yy = 0.5 * (g + 1.0) * Ly
    zz = 0.5 * (g + 1.0) * Lz
    wy = w * (Ly / 2.0)
    wz = w * (Lz / 2.0)
    nt = len(tb)
    V = np.array([[basis(tb, y, z, tdeg) for y in yy] for z in zz])   # (nq,nq,nt)
    M = np.zeros((nt, nt))
    for i in range(nt):
        for j in range(i, nt):
            M[i, j] = M[j, i] = float(np.einsum('a,b,ab->', wy, wz, V[:, :, i] * V[:, :, j]))
    return M


def block(M1, n3):
    """把 nt×nt 的质量阵扩成 3nt×3nt (三个分量同构)。"""
    nt = M1.shape[0]
    out = np.zeros((3 * nt, 3 * nt))
    for c in range(3):
        out[c * nt:(c + 1) * nt, c * nt:(c + 1) * nt] = M1
    return out


# ------------------------------------------------------------------ 子域装配
def build_side(side, case, fnx, fny, fnz):
    if side == 'L':
        nodes, elems, nnx, nny, nnz = C.build_mesh_3d(fnx, fny, fnz, 0.0, X0, 0.0, Ly, 0.0, Lz)
        iface_i, dir_i = fnx, 0
    else:
        nodes, elems, nnx, nny, nnz = C.build_mesh_3d(fnx, fny, fnz, X0, Lx, 0.0, Ly, 0.0, Lz)
        iface_i, dir_i = 0, fnx
    K, ndof, nn = C.fem_assemble(nodes, elems)
    # 界面节点统一按 (k 外, j 内) 排序 -> 两侧界面节点一一对应
    iface_nodes = [k * nny * nnx + j * nnx + iface_i for k in range(nnz) for j in range(nny)]
    # 远场 Dirichlet
    dir_map = {}
    for k in range(nnz):
        for j in range(nny):
            nd = k * nny * nnx + j * nnx + dir_i
            if side == 'L':
                for c in range(3):
                    dir_map[3 * nd + c] = 0.0
            else:
                g, mask = C.right_disp_3d(nodes[nd, 1], nodes[nd, 2], case)
                for c in range(3):
                    if mask[c]:
                        dir_map[3 * nd + c] = g[c]
    gdofs = [3 * nd + c for nd in iface_nodes for c in range(3)]
    gset = set(gdofs)
    fdofs = [d for d in range(ndof) if d not in gset and d not in dir_map]
    return dict(nodes=nodes, elems=elems, K=K, ndof=ndof,
                nnx=nnx, nny=nny, nnz=nnz, fnx=fnx, fny=fny, fnz=fnz,
                iface_i=iface_i, dir_i=dir_i, iface_nodes=iface_nodes,
                gdofs=gdofs, fdofs=fdofs, dir_map=dir_map, side=side)


def schur(Sp):
    """返回 (S_eff, q): S_eff = K_GG - K_GF K_FF^{-1} K_FG (界面一致节点反力算子),
    q = 远场给定位移在界面上产生的一致节点力。"""
    K, fd, gd = Sp['K'], Sp['fdofs'], Sp['gdofs']
    Kff = K[np.ix_(fd, fd)]
    Kfg = K[np.ix_(fd, gd)]
    Kgf = K[np.ix_(gd, fd)]
    Kgg = K[np.ix_(gd, gd)]
    X = np.linalg.solve(Kff, Kfg)                 # (nf, ng) 一次分解多次回代
    Seff = Kgg - Kgf @ X
    if Sp['dir_map']:
        dd = list(Sp['dir_map'].keys())
        ud = np.array([Sp['dir_map'][d] for d in dd])
        uf = np.linalg.solve(Kff, -(K[np.ix_(fd, dd)] @ ud))
        q = Kgf @ uf + K[np.ix_(gd, dd)] @ ud
    else:
        q = np.zeros(len(gd))
    return Seff, q


def trace_P(Sp, tb, tdeg):
    ys = Sp['nodes'][Sp['iface_nodes'], 1]
    zs = Sp['nodes'][Sp['iface_nodes'], 2]
    nif, nt = len(ys), len(tb)
    P = np.zeros((3 * nif, 3 * nt))
    for i in range(nif):
        v = trace_basis(tb, ys[i], zs[i], tdeg)
        for c in range(3):
            P[3 * i + c, c * nt:(c + 1) * nt] = v
    return P


def nodal_area(Sp, nq=8):
    r"""界面节点集总面积 w_i = \int_Gamma N_i dGamma (双线性面形函数)。"""
    nny, fny, fnz = Sp['nny'], Sp['fny'], Sp['fnz']
    nif = len(Sp['iface_nodes'])
    w = np.zeros(nif)
    g, wg = np.polynomial.legendre.leggauss(nq)
    jac = (Ly / fny / 2.0) * (Lz / fnz / 2.0)
    for jj in range(fny):
        for kk in range(fnz):
            loc = [kk * nny + jj, kk * nny + jj + 1,
                   (kk + 1) * nny + jj + 1, (kk + 1) * nny + jj]
            for e, we in zip(g, wg):
                for f, wf in zip(g, wg):
                    Nq = [(1 - e) * (1 - f) / 4, (1 + e) * (1 - f) / 4,
                          (1 + e) * (1 + f) / 4, (1 - e) * (1 + f) / 4]
                    for r in range(4):
                        w[loc[r]] += Nq[r] * we * wf * jac   # 必须含 Gauss 权重
    return w


def pointwise_nodal_traction(Sp, ufull):
    """界面节点上的外法线面力 (nif,3): 用**单侧**相邻单元在节点处求应力。
    这正是 2D pointwise_lam 的 3D 对应物 (配点型提取)。"""
    nodes, elems = Sp['nodes'], Sp['elems']
    nnx, nny, fnx, fny, fnz = Sp['nnx'], Sp['nny'], Sp['fnx'], Sp['fny'], Sp['fnz']
    out = np.zeros((len(Sp['iface_nodes']), 3))
    for idx, nd in enumerate(Sp['iface_nodes']):
        i = nd % nnx
        j = (nd // nnx) % nny
        k = nd // (nnx * nny)
        if Sp['side'] == 'L':
            ie, xi = fnx - 1, 1.0
        else:
            ie, xi = 0, -1.0
        je = min(j, fny - 1)
        ke = min(k, fnz - 1)
        eta = 2.0 * (j - je) - 1.0
        ze = 2.0 * (k - ke) - 1.0
        el = elems[ke * fny * fnx + je * fnx + ie]
        ue = np.array([ufull[3 * el[r] + c] for r in range(8) for c in range(3)])
        _, B = SR._BN(nodes, el, xi, eta, ze)
        sig = D @ (B @ ue)
        sgn = 1.0 if Sp['side'] == 'L' else -1.0
        out[idx] = (sgn * sig[0], sgn * sig[3], sgn * sig[5])   # (sxx, sxy, sxz) · n_out
    return out


def pw_operator(Sp, area):
    """R_pw: 界面位移 -> 节点面力×面积 (ng×ng)。"""
    K, fd, gd = Sp['K'], Sp['fdofs'], Sp['gdofs']
    Kff = K[np.ix_(fd, fd)]
    Kfg = K[np.ix_(fd, gd)]
    Uf = np.linalg.solve(Kff, -Kfg)              # (nf, ng)
    ng = len(gd)
    R = np.zeros((ng, ng))
    for k in range(ng):
        uf = np.zeros(Sp['ndof'])
        uf[gd[k]] = 1.0
        uf[fd] = Uf[:, k]
        t = pointwise_nodal_traction(Sp, uf)
        R[:, k] = (t * area[:, None]).reshape(-1)
    return R


# ------------------------------------------------------------------ 整体 FEM 参考
def whole_fem(case, nx, ny, nz):
    nodes, elems, nnx, nny, nnz = C.build_mesh_3d(nx, ny, nz, 0.0, Lx, 0.0, Ly, 0.0, Lz)
    K, ndof, nn = C.fem_assemble(nodes, elems)
    fix = {}
    for k in range(nn):
        if abs(nodes[k, 0]) < 1e-9:
            for c in range(3):
                fix[3 * k + c] = 0.0
        elif abs(nodes[k, 0] - Lx) < 1e-9:
            g, mask = C.right_disp_3d(nodes[k, 1], nodes[k, 2], case)
            for c in range(3):
                if mask[c]:
                    fix[3 * k + c] = g[c]
    fd = np.array(list(fix.keys()))
    fs = set(fd.tolist())
    fr = np.array([d for d in range(ndof) if d not in fs])
    u = np.zeros(ndof)
    u[fr] = np.linalg.solve(K[np.ix_(fr, fr)], -(K[np.ix_(fr, fd)] @ np.array([fix[d] for d in fd])))
    for d, v in fix.items():
        u[d] = v
    return nodes, elems, nnx, nny, nnz, u


def iface_ref(nodes, u, nnx, nny, nx):
    """整体网格上 x=X0 处界面节点的位移 (按 (j,k) 排序), 返回 (25...,3)。"""
    i = nx // 2                                   # X0 = Lx/2 -> 恰好 i = nx/2
    idx = [k * nny * nnx + j * nnx + i for k in range(nny) for j in range(nny)]
    return np.array([[u[3 * nd], u[3 * nd + 1], u[3 * nd + 2]] for nd in idx])


# ------------------------------------------------------------------ 诊断
def op_metrics(MS):
    """给定 M·S_tr, 返回 (不对称度, 归一化最小特征值)。"""
    asym = np.abs(MS - MS.T).max() / np.abs(MS).max()
    ev = np.linalg.eigvalsh(0.5 * (MS + MS.T))
    return asym, ev.min() / np.abs(ev).max()


def build_ops(case, fnx, fny, fnz, tdeg, basis=trace_basis):
    tb = tb_list(tdeg)
    nt = len(tb)
    M1 = l2_mass(tb, tdeg, basis=basis)
    Mf = block(M1, 3 * nt)
    ops = {}
    for side in ('L', 'R'):
        Sp = build_side(side, case, fnx, fny, fnz)
        Seff, q = schur(Sp)
        P = trace_P(Sp, tb, tdeg)
        area = nodal_area(Sp)
        # --- Galerkin: S_tr = M^{-1} P^T Seff P,   b = M^{-1} P^T q
        A = P.T @ Seff @ P
        Sg = np.linalg.solve(Mf, A)
        bg = np.linalg.solve(Mf, P.T @ q)
        # --- 点式: S_tr^pw = M^{-1} P^T R_pw P  (R_pw: 界面位移 -> 节点面力)
        R = pw_operator(Sp, area)
        Apw = P.T @ R @ P
        Spw = np.linalg.solve(Mf, Apw)
        ops[side] = dict(P=P, Mf=Mf, M1=M1, Sg=Sg, bg=bg, Spw=Spw, A=A, Apw=Apw,
                         Seff=Seff, q=q, area=area, tb=tb, nt=nt)
    return ops


def solve_iface(ops, key):
    SL, SR = ops['L'][key], ops['R'][key]
    if key == 'Sg':
        bL, bR = ops['L']['bg'], ops['R']['bg']
    else:
        bL = np.linalg.solve(ops['L']['Mf'], ops['L']['P'].T @ ops['L']['q'])
        bR = np.linalg.solve(ops['R']['Mf'], ops['R']['P'].T @ ops['R']['q'])
    return np.linalg.solve(SL + SR, -(bL + bR))


def main():
    outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    os.makedirs(outdir, exist_ok=True)
    log = []
    def P(*a):
        s = ' '.join(str(x) for x in a)
        print(s, flush=True)
        log.append(s)

    P('=== 三维界面 DtN 算子审计: Galerkin 一致节点力 vs 点式配点 ===')
    case = 'tension'
    tdeg = 3
    fnx, fny, fnz = 8, 4, 4

    # ---------- 1) 迹基条件数: monomial vs Legendre ----------
    tb = tb_list(tdeg)
    Mo = l2_mass(tb, tdeg, basis=trace_basis_mono)
    Ml = l2_mass(tb, tdeg, basis=trace_basis)
    SpL = build_side('L', case, fnx, fny, fnz)
    ys = SpL['nodes'][SpL['iface_nodes'], 1]
    zs = SpL['nodes'][SpL['iface_nodes'], 2]
    Po = np.array([trace_basis_mono(tb, y, z, tdeg) for y, z in zip(ys, zs)])
    Pl = np.array([trace_basis(tb, y, z, tdeg) for y, z in zip(ys, zs)])
    P('')
    P(f'--- 1) 迹空间基条件数 ({len(SpL["iface_nodes"])} 界面节点, deg={tdeg}, nt={len(tb)}) ---')
    P(f'  nodal 矩阵 cond:  monomial {np.linalg.cond(Po):.3e}   Legendre {np.linalg.cond(Pl):.3e}')
    P(f'  质量阵 cond   :  monomial {np.linalg.cond(Mo):.3e}   Legendre {np.linalg.cond(Ml):.3e}')
    P(f'  质量阵非对角范数/对角: monomial {np.abs(Mo - np.diag(np.diag(Mo))).max()/np.abs(np.diag(Mo)).max():.3e}'
      f'   Legendre {np.abs(Ml - np.diag(np.diag(Ml))).max()/np.abs(np.diag(Ml)).max():.3e}')

    # ---------- 2) 自伴性与正定性 ----------
    t0 = time.time()
    ops = build_ops(case, fnx, fny, fnz, tdeg)
    P('')
    P(f'--- 2) 自伴性 / 正定性 (工况={case}, 网格 {fnx}x{fny}x{fnz}, 用时 {time.time()-t0:.1f}s) ---')
    P(f'  {"传递方式":<12}{"不part":>6}{"max|MS-(MS)^T|/max|MS|":>26}{"min eig / |eig|":>18}')
    for side in ('L', 'R'):
        for key, name in (('A', 'galerkin'), ('Apw', 'pointwise')):
            a, e = op_metrics(ops[side][key])
            P(f'  {name+" "+side:<12}{"":>6}{a:>26.3e}{e:>18.3e}')

    # ---------- 2b) 右子域近零模态识别 + 耦合算子正定性 ----------
    P('')
    P('--- 2b) 半正定/正定性的来源 ---')
    AR = ops['R']['A']
    ev2, V2 = np.linalg.eigh(0.5 * (AR + AR.T))
    nt_ = ops['R']['nt']
    P(f'  galerkin R 最小特征值/|eig|max = {ev2.min()/np.abs(ev2).max():.3e}  (数值零)')
    v0 = V2[:, 0]
    cn = [np.linalg.norm(v0[c * nt_:(c + 1) * nt_]) for c in range(3)]
    P(f'  对应模态的分量能量: (ux,uy,uz) = ({cn[0]:.2e}, {cn[1]:.2e}, {cn[2]:.2e})')
    P(f'  uz 分量中常数项 |c_00| / 整个模态 = '
      f'{abs(v0[2*nt_]) / np.linalg.norm(v0):.4f}')
    P('  -> 远场仅约束被 mask 标记的分量(拉伸: 仅 ux), 未约束方向 (uy,uz) 的刚体平移')
    P('     在界面上的迹 (0, c_y, c_z) 是零能量场, 故单子域算子半正定而非正定。')
    Bg0 = ops['L']['A'] + ops['R']['A']
    evb = np.linalg.eigvalsh(0.5 * (Bg0 + Bg0.T))
    P(f'  但**耦合**算子 B = M(S_tr_L + S_tr_R): min eig/|eig|max = '
      f'{evb.min()/np.abs(evb).max():.3e}  (>0, 正定)')
    P(f'    -> 另一子域 (远场全约束) 提供了缺失方向的刚度, 故耦合界面方程唯一可解。')


    nodes, elems, nnx, nny, nnz, uref = whole_fem(case, 2 * fnx, fny, fnz)
    g_ref = iface_ref(nodes, uref, nnx, nny, 2 * fnx)
    P('')
    P(f'--- 3) 界面位移 vs 整体 FEM (匹配网格 {2*fnx}x{fny}x{fnz}) ---')
    P(f'  {"组合 (a_sgn,q_sgn)":<22}{"ux 误差":>12}{"uy 误差":>12}{"uz 误差":>12}')
    for key, name in (('Sg', 'galerkin'), ('Spw', 'pointwise')):
        SL, SR = ops['L'][key], ops['R'][key]
        bL = np.linalg.solve(ops['L']['Mf'], ops['L']['P'].T @ ops['L']['q'])
        bR = np.linalg.solve(ops['R']['Mf'], ops['R']['P'].T @ ops['R']['q'])
        for a_sgn in (1.0, -1.0):
            for q_sgn in (1.0, -1.0):
                c = np.linalg.solve(SL + a_sgn * SR, q_sgn * (bL + bR))
                gnum = (ops['L']['P'] @ c).reshape(-1, 3)
                ex = [np.linalg.norm(gnum[:, k] - g_ref[:, k]) / np.linalg.norm(g_ref[:, k])
                      for k in range(3)]
                P(f'  {name:<10}a={a_sgn:+.0f} q={q_sgn:+.0f} {ex[0]:>12.3%}{ex[1]:>12.3%}{ex[2]:>12.3%}')
    res = {}
    for key, name in (('Sg', 'galerkin'), ('Spw', 'pointwise')):
        SL, SR = ops['L'][key], ops['R'][key]
        bL = np.linalg.solve(ops['L']['Mf'], ops['L']['P'].T @ ops['L']['q'])
        bR = np.linalg.solve(ops['R']['Mf'], ops['R']['P'].T @ ops['R']['q'])
        c = np.linalg.solve(SL + SR, -(bL + bR))
        gnum = (ops['L']['P'] @ c).reshape(-1, 3)
        res[name] = [np.linalg.norm(gnum[:, k] - g_ref[:, k]) / np.linalg.norm(g_ref[:, k])
                     for k in range(3)]
    P('')
    P('  【正确界面方程 (S_L+S_R)c = -(b_L+b_R) 下的界面位移误差】')
    for name in ('galerkin', 'pointwise'):
        e = res[name]
        P(f'    {name:<10}: ux {e[0]:.4%}   uy {e[1]:.4%}   uz {e[2]:.4%}   '
          f'(合成 {math.sqrt(sum(x*x for x in e)/3):.4%})')

    # ---------- 3b) 机制诊断: 算子不准确 vs 方程组病态 ----------
    P('')
    P('--- 3b) 机制诊断: 传递方式如何导致失败 ---')
    # 把整体 FEM 的真实界面位移最小二乘投影到迹空间
    PL = ops['L']['P']
    gflat = g_ref.reshape(-1)
    c_ex = np.linalg.solve(PL.T @ PL, PL.T @ gflat)
    Bg = ops['L']['A'] + ops['R']['A']              # M (S_tr_L + S_tr_R), galerkin
    Bp = ops['L']['Apw'] + ops['R']['Apw']          # 同上, pointwise
    r_gal = Bg @ c_ex
    disc = np.linalg.norm(Bp @ c_ex - r_gal) / np.linalg.norm(r_gal)
    PLq = ops['L']['P'].T @ ops['L']['q'] + ops['R']['P'].T @ ops['R']['q']
    P(f'  算子差 (在真实界面位移处): ||B_pw c_ex - B_gal c_ex|| / ||B_gal c_ex|| = {disc:.3%}')
    P(f'  cond(B_gal) = {np.linalg.cond(Bg):.3e}')
    P(f'  cond(B_pw)  = {np.linalg.cond(Bp):.3e}')
    for nm, B, key in (('galerkin', Bg, 'Sg'), ('pointwise', Bp, 'Spw')):
        c = np.linalg.solve(B, -(ops['L']['P'].T @ ops['L']['q'] + ops['R']['P'].T @ ops['R']['q']))
        res = np.linalg.norm(B @ c + PLq) / np.linalg.norm(PLq)
        P(f'  {nm:<10}: 解 ||c|| = {np.linalg.norm(c):.3e}, 真实 ||c_ex|| = {np.linalg.norm(c_ex):.3e}, '
          f'残差 = {res:.2e}')

    # ---------- 4) 收敛阶 ----------
    P('')
    P('--- 4) 网格收敛 (界面位移误差 vs 2x 密整体 FEM, deg=2) ---')
    P(f'  {"(fnx,fny,fnz)":<18}{"galerkin 合成误差":>18}{"pointwise 合成误差":>20}{"asym(gal)":>12}')
    hist = []
    for (a_, b_, c_) in [(6, 3, 3), (8, 4, 4), (10, 5, 5)]:
        td = 2
        tb2 = tb_list(td)
        M1 = l2_mass(tb2, td)
        Mf = block(M1, 3 * len(tb2))
        gs = {}
        asym_g = None
        for side in ('L', 'R'):
            Sp = build_side(side, case, a_, b_, c_)
            Seff, q = schur(Sp)
            Pp = trace_P(Sp, tb2, td)
            area = nodal_area(Sp)
            A = Pp.T @ Seff @ Pp
            Sg = np.linalg.solve(Mf, A)
            R = pw_operator(Sp, area)
            Spw = np.linalg.solve(Mf, Pp.T @ R @ Pp)
            bg = np.linalg.solve(Mf, Pp.T @ q)
            gs[side] = (Sg, Spw, bg, Pp)
            if side == 'L':
                asym_g = op_metrics(A)[0]
        nodes2, elems2, n2x, n2y, n2z, uref2 = whole_fem(case, 2 * a_, b_, c_)
        gr = iface_ref(nodes2, uref2, n2x, n2y, 2 * a_)
        out = {}
        for idxk, nm in ((0, 'galerkin'), (1, 'pointwise')):
            cvec = np.linalg.solve(gs['L'][idxk] + gs['R'][idxk],
                                   -(gs['L'][2] + gs['R'][2]))
            gn = (gs['L'][3] @ cvec).reshape(-1, 3)
            ee = [np.linalg.norm(gn[:, k] - gr[:, k]) / np.linalg.norm(gr[:, k]) for k in range(3)]
            out[nm] = math.sqrt(sum(x * x for x in ee) / 3)
        h = 1.0 / a_
        hist.append((a_, h, out['galerkin'], out['pointwise']))
        P(f'  {str((a_,b_,c_)):<18}{out["galerkin"]:>18.4%}{out["pointwise"]:>20.4%}{asym_g:>12.3e}')
    if len(hist) >= 3:
        d1 = abs(hist[1][2] - hist[0][2]); d2 = abs(hist[2][2] - hist[1][2])
        h1, h2, h3 = hist[0][1], hist[1][1], hist[2][1]
        best_p, bf = None, 1e18
        for pp in np.linspace(0.2, 5.0, 481):
            f = (h1 ** pp - h2 ** pp) / (h2 ** pp - h3 ** pp) - d1 / d2
            if abs(f) < bf:
                bf, best_p = abs(f), pp
        P(f'  -> Galerkin 界面位移收敛阶 p = {best_p:.2f}  (理论 O(h^2) -> p=2)')
        d1p = abs(hist[1][3] - hist[0][3]); d2p = abs(hist[2][3] - hist[1][3])
        bp2, bf2 = None, 1e18
        for pp in np.linspace(0.2, 5.0, 481):
            f = (h1 ** pp - h2 ** pp) / (h2 ** pp - h3 ** pp) - d1p / d2p
            if abs(f) < bf2:
                bf2, bp2 = abs(f), pp
        P(f'  -> Pointwise 界面位移收敛阶 p = {bp2:.2f}  (对照)')

    with open(os.path.join(outdir, 'audit_dtn3d.txt'), 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(log) + '\n')
    print(f'[保存] {os.path.join(outdir, "audit_dtn3d.txt")}')


if __name__ == '__main__':
    main()
