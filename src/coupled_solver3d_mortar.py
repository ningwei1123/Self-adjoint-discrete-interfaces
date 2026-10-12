# -*- coding: utf-8 -*-
"""
三维 FEM-NN 耦合求解器 (mortar / 拉格朗日乘子界面耦合)
=====================================================
相比 coupled_solver3d.py 的罚函数法, 本文件用 积分投影拉格朗日乘子 (mortar) 处理界面:
  约束 ∫_Γ (u_F - u_N) · p_j dΓ = 0   (p_j 为 NN 在界面上的迹空间基)
右端给定位移 BC 也用同样的投影 mortar 处理: ∫_A (u_N - g) · q_j dA = 0
左端 u=0 用 Dirichlet 凝聚 (condensation) 精确施加。
无罚参数敏感度 -> 位移/应力精度远高于罚函数法。

其余建模 (H8 单元, 3D 线弹性, 多项式 NN 特征, 工况, 参考解) 与 coupled_solver3d.py 一致。
"""
import numpy as np
import math, time, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import coupled_solver3d as C
import stress_recovery3d as S

D = C.D
Lx, Ly, Lz, X0 = C.Lx, C.Ly, C.Lz, C.X0
CASE_CN = C.CASE_CN

# 中文标签回退: 只在调用方尚未设定出版字体时生效, 不再无条件覆盖 rcParams。
# (曾导致 figs_paper 在 import 本模块后, 图 2-4 全部掺入 Microsoft YaHei)
try:
    from figstyle import use_cjk_fallback
    use_cjk_fallback()
except Exception:                                    # figstyle 不可用时保持默认
    pass


def build_mortar(case, nn_deg=3, trace_deg=3, fnx=8, fny=4, fnz=4, tik=1e-2):
    # ---- FEM 子域 [0,X0]
    nodes, elems, nnx, nny, nnz = C.build_mesh_3d(fnx, fny, fnz, 0.0, X0, 0.0, Ly, 0.0, Lz)
    KF, ndofF, nnF = C.fem_assemble(nodes, elems)
    # ---- NN 子域 特征
    feats = C.make_features_3d(nn_deg)
    nf = len(feats)
    KN = C.nn_assemble(feats, tikhonov=tik)
    n3 = 3 * nf
    # ---- 左端 u=0 凝聚
    left = [k for k in range(nnF) if abs(nodes[k, 0]) < 1e-9]
    left_dofs = set()
    for k in left:
        for c in range(3):
            left_dofs.add(3 * k + c)
    freeF = [d for d in range(ndofF) if d not in left_dofs]
    Nf = len(freeF)
    fmap = {d: i for i, d in enumerate(freeF)}
    KFr = np.zeros((Nf, Nf))
    for i, ci in enumerate(freeF):
        for j, cj in enumerate(freeF):
            KFr[i, j] = KF[ci, cj]

    # ---- 迹空间基 p_j(eta_y,eta_z)=eta_y^a eta_z^b
    tfeats = [(a, b) for a in range(trace_deg + 1) for b in range(trace_deg + 1)]
    nt = len(tfeats)

    def _leg(n, e):
        """1D Legendre P_0..P_{n-1} 在 e 处的值 (三项递推)。"""
        P = [1.0, e]
        for j in range(2, n):
            P.append(((2 * j - 1) * e * P[j - 1] - (j - 1) * P[j - 2]) / j)
        return np.array(P[:n], dtype=float)

    def q_val(a, b, y, z):
        """张量积 Legendre 迹基 p_ab(y,z) = P_a(eta_y) P_b(eta_z)。

        monomial 基 (eta_y^a eta_z^b) 在界面节点上是 Vandermonde 型矩阵, 条件数随阶数
        爆炸; 与 2D (coupled_operator.py) 统一改用正交基后条件数降到 O(10^2)。
        """
        ey = 2.0 * y / Ly - 1.0
        ez = 2.0 * z / Lz - 1.0
        return _leg(trace_deg + 1, ey)[a] * _leg(trace_deg + 1, ez)[b]

    # ---- 界面 x=X0 投影矩阵 P_F(3nt,Nf), P_N(3nt,n3)
    PF = np.zeros((3 * nt, Nf))
    PN = np.zeros((3 * nt, n3))
    g2, w2 = np.polynomial.legendre.leggauss(8)
    hy = Ly / fny
    hz = Lz / fnz
    # FEM 界面节点 (x=X0) 即 i=fnx 层; 面四节点 (fnx,jj,kk) 等
    for jj in range(fny):
        for kk in range(fnz):
            fquad = [(fnx, jj, kk), (fnx, jj + 1, kk), (fnx, jj + 1, kk + 1), (fnx, jj, kk + 1)]
            for ey, wy in zip(g2, w2):
                for ez, wz in zip(g2, w2):
                    y = Ly * (jj + (ey + 1) / 2) / fny
                    z = Lz * (kk + (ez + 1) / 2) / fnz
                    Nfq = [(1 - ey) * (1 - ez) / 4, (1 + ey) * (1 - ez) / 4,
                           (1 + ey) * (1 + ez) / 4, (1 - ey) * (1 + ez) / 4]
                    dA = (hy / 2.0) * (hz / 2.0) * wy * wz
                    for c in range(3):
                        for t, (a, b) in enumerate(tfeats):
                            qv = q_val(a, b, y, z)
                            for r in range(4):
                                # fquad 元组为 (i, j, k); 节点编号 = k*nny*nnx + j*nnx + i
                                ir, jr, kr = fquad[r]
                                gid = kr * nny * nnx + jr * nnx + ir
                                if 3 * gid + c in left_dofs:
                                    continue
                                PF[3 * t + c, fmap[3 * gid + c]] += qv * Nfq[r] * dA
                            for aa in range(nf):
                                pn = C.feat_val_3d(feats[aa], X0, y, z)  # 含 (-1)^p
                                PN[3 * t + c, 3 * aa + c] += qv * pn * dA

    # ---- 右端 x=Lx 投影 mortar BC: 仅对"被约束分量"(mask=True)施加 Q_N c = g_proj
    #      (未约束分量在右端必须自由, 否则会被人为钳制 -> 弯曲工况下扼杀横向挠度)
    _, mask_r = C.right_disp_3d(0.0, 0.0, case)   # mask 与 y,z 无关
    mcs = [c for c in range(3) if mask_r[c]]
    roff = {c: i * nt for i, c in enumerate(mcs)}
    nr = nt * len(mcs)
    QN = np.zeros((nr, n3))
    gg = np.zeros(nr)
    for jj in range(fny):
        for kk in range(fnz):
            for ey, wy in zip(g2, w2):
                for ez, wz in zip(g2, w2):
                    y = Ly * (jj + (ey + 1) / 2) / fny
                    z = Lz * (kk + (ez + 1) / 2) / fnz
                    dA = (hy / 2.0) * (hz / 2.0) * wy * wz
                    g, mask = C.right_disp_3d(y, z, case)
                    for c in mcs:
                        r0 = roff[c]
                        for t, (a, b) in enumerate(tfeats):
                            qv = q_val(a, b, y, z)
                            for aa in range(nf):
                                qn = C.feat_val_3d(feats[aa], Lx, y, z)
                                QN[r0 + t, 3 * aa + c] += qv * qn * dA
                            gg[r0 + t] += qv * g[c] * dA

    # ---- 组装 saddle-point 系统
    nlam = 3 * nt + nr
    N = Nf + n3 + nlam
    M = np.zeros((N, N))
    f = np.zeros(N)
    # 块1: u_F
    M[:Nf, :Nf] = KFr
    # 块2: c
    M[Nf:Nf + n3, Nf:Nf + n3] = KN
    # 块3: lambda_iface (行 Nf+n3 .. Nf+n3+3nt)
    M[:Nf, Nf + n3:Nf + n3 + 3 * nt] = PF.T
    M[Nf:Nf + n3, Nf + n3:Nf + n3 + 3 * nt] = -PN.T
    M[Nf + n3:Nf + n3 + 3 * nt, :Nf] = PF
    M[Nf + n3:Nf + n3 + 3 * nt, Nf:Nf + n3] = -PN
    # 块4: lambda_right (仅被约束分量)
    M[Nf:Nf + n3, Nf + n3 + 3 * nt:] = QN.T
    M[Nf + n3 + 3 * nt:, Nf:Nf + n3] = QN
    f[Nf + n3 + 3 * nt:] = gg

    sol_vec = np.linalg.solve(M, f)
    ufem = np.zeros(ndofF)
    for i, d in enumerate(freeF):
        ufem[d] = sol_vec[i]
    coeffs = sol_vec[Nf:Nf + n3].reshape(nf, 3)
    return dict(nodes=nodes, elems=elems, nnx=nnx, nny=nny, nnz=nnz,
                ndofF=ndofF, nnF=nnF, ufem=ufem, coeffs=coeffs, feats=feats,
                interf=[k for k in range(nnF) if abs(nodes[k, 0] - X0) < 1e-9],
                left=left, case=case, nf=nf, nt=nt)


# ----------------------------------------------------------- 云图 (与 coupled_solver3d 共用)
def plot_compare(sol, rsol, case, field, outdir):
    return C.plot_compare(sol, rsol, case, field, outdir)


def l2_vm_field(fc, fr, n=36):
    num = den = 0.0
    for i in range(n):
        x = Lx * (i + 0.5) / n
        for j in range(14):
            y = Ly * (j + 0.5) / 14
            for k in range(14):
                z = Lz * (k + 0.5) / 14
                a = fc(x, y, z); b = fr(x, y, z)
                num += (a - b) ** 2; den += b * b
    return math.sqrt(num / den) if den > 0 else 0.0


def main():
    outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    os.makedirs(outdir, exist_ok=True)
    cases = ['tension', 'bending', 'torsion', 'combined']
    rows = []
    t0 = time.time()
    for case in cases:
        t1 = time.time()
        csol = build_mortar(case)
        rsol = C.solve_reference_3d(case)
        rel = C.l2_disp_error(csol, rsol)
        rvm_raw = C.l2_vm_error(csol, rsol)              # 中心差分应力
        # SPR 节点应力恢复 + NN 解析应力 (混合场)
        spr_c = S.spr_nodal(csol['nodes'], csol['elems'], csol['ufem'], degree=2)
        spr_r = S.spr_nodal(rsol['nodes'], rsol['elems'], rsol['u'], degree=2)
        fc = S.build_mixed_vm(csol, spr_c)
        fr = S.build_mixed_vm(rsol, spr_r)
        rvm_spr = l2_vm_field(fc, fr)
        # 界面位移残差
        resid = 0.0
        for k in csol['interf']:
            xj, yj, zj = csol['nodes'][k]
            un = C.nn_point_disp_3d(csol, xj, yj, zj)
            uf = (csol['ufem'][3 * k], csol['ufem'][3 * k + 1], csol['ufem'][3 * k + 2])
            resid = max(resid, math.hypot(un[0] - uf[0], un[1] - uf[1], un[2] - uf[2]))
        # 云图: 应力用 SPR 平滑场, 位移用默认
        C.plot_compare(csol, rsol, case, 'vm', outdir, stress_fn_c=fc, stress_fn_r=fr)
        C.plot_compare(csol, rsol, case, 'disp', outdir)
        rows.append((case, rel, rvm_raw, rvm_spr, resid))
        print(f'[{case}] 用时{time.time()-t1:.1f}s | 位移L2={rel:.4%} '
              f'应力L2(差分)={rvm_raw:.4%} 应力L2(SPR)={rvm_spr:.4%} 界面残差={resid:.3e}')
    with open(os.path.join(outdir, 'summary3d_mortar.txt'), 'w', encoding='utf-8') as fp:
        fp.write('三维 FEM-NN 耦合 (mortar + SPR 应力恢复) vs 纯有限元 多工况对比\n')
        fp.write(f'几何: {Lx}x{Ly}x{Lz}, 分割 x={X0}; FEM子域 8x4x4(H8), NN 多项式 deg=3, 迹空间 deg=3\n')
        fp.write(f'参考: 全区域 16x4x4 均匀 H8 + Dirichlet 凝聚 (与 FEM 子域同密度)\n')
        fp.write(f'{"工况":<12}{"位移L2误差":>14}{"应力L2(差分)":>16}{"应力L2(SPR)":>14}{"界面残差":>14}\n')
        for case, rel, rvm_raw, rvm_spr, resid in rows:
            fp.write(f'{CASE_CN[case]:<12}{rel:>13.4%}{rvm_raw:>15.4%}{rvm_spr:>13.4%}{resid:>13.3e}\n')
    print(f'总用时 {time.time()-t0:.1f}s, 结果见 {outdir}')


if __name__ == '__main__':
    main()
