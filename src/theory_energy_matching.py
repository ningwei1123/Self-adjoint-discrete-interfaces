# -*- coding: utf-8 -*-
"""
能量尺度匹配理论 (Energy-Scale Matching Theory) —— 推导 + 数值验证
==================================================================
理论要点 (可发表):
  设两子域的界面 Dirichlet-to-Neumann 算子 (Steklov–Poincaré) 为 S_F, S_N,
  其柔度 (compliance) 记为 M_F = S_F^{-1}, M_N = S_N^{-1} (在公共迹空间上)。
  串联形式的界面耦合满足"柔度可加":
        M = M_F + M_N .
  定义 **能量尺度匹配数**  χ = M_N / M_F  (无量纲柔度比)。
  则界面位移分配比  u_Γ/Δ = M_F/(M_F+M_N) = 1/(1+χ)。
    - χ = 1 (M_F=M_N)  → u_Γ/Δ = 1/2, 两子域"等能量刚度", 耦合误差最小;
    - χ ≠ 1            → 位移偏向柔的一侧, 耦合误差 ∝ |χ-1|/(1+χ)。

数值验证: 用 interface_methods 的凝聚法代码提取 M_F, M_N (逐自由度界面 Schur),
并人工缩放 FEM 刚度 s (s=8 即本轮修好的 "8× 雅可比" 缺陷), 扫描 χ,
比较"理论预测的界面位移分配比 1/(1+χ)"与"实际耦合解测得值"。
"""
import numpy as np
import math
import coupled_solver as CS
import interface_methods as IM

Lx, Ly, X0 = CS.Lx, CS.Ly, CS.X0

# 为了得到干净的"轴向分配"实验, 把右端 BC 临时改成均匀 ux=Δ
_ORIG = CS.right_disp
_UNIFORM = {'on': False}
def _patched(y, D0=0.01):
    return D0 if _UNIFORM['on'] else _ORIG(y, D0)
CS.right_disp = _patched


def operators(KF_scale=1.0, max_deg=5, fnx=12, fny=8):
    """返回界面柔度 m_F, m_N (常值 ux 迹模式) 与匹配数 chi = m_N/m_F。"""
    KF, nodes, elems, ndofF, nnF, left, interf, feats, fnx, fny = IM.build_fem(fnx, fny, max_deg)
    KF = np.array(KF) * KF_scale
    nf = len(feats)
    KN, fN = IM.build_nn(feats, 1e6)

    leftDof = set()
    for k in left:
        leftDof.add(2 * k); leftDof.add(2 * k + 1)
    G = set()
    for k in interf:
        G.add(2 * k); G.add(2 * k + 1)
    I = [c for c in range(ndofF) if c not in leftDof and c not in G]
    Glist = sorted(G)
    freeF = I + Glist
    KFr = KF[np.ix_(freeF, freeF)]
    nI = len(I)
    Kii = KFr[:nI, :nI]; KiG = KFr[:nI, nI:]; KGG = KFr[nI:, nI:]
    KGG_F = KGG - KiG.T @ np.linalg.solve(Kii, KiG)      # FEM 界面 Schur = S_F

    nt = 6
    gp, gw = np.polynomial.legendre.leggauss(80)
    ss = Ly * (gp + 1.0) / 2.0; ws = Ly / 2.0 * gw
    nn_if = len(interf)
    yk = np.array([nodes[k][1] for k in interf]); h = Ly / (nn_if - 1)
    Nm = np.zeros((len(ss), nn_if))
    for k in range(nn_if - 1):
        m = (ss >= yk[k] - 1e-12) & (ss <= yk[k + 1] + 1e-12)
        t = (ss[m] - yk[k]) / h
        Nm[m, k] += 1 - t; Nm[m, k + 1] += t
    P = np.array([((2 * ss / Ly) - 1.0) ** j for j in range(nt)])
    Phis = np.array([[CS.feat_val(fe, X0, s) for fe in feats] for s in ss])
    Bx = np.zeros((nt, len(Glist))); By = np.zeros((nt, len(Glist)))
    Cx = np.zeros((nt, nf))
    for j in range(nt):
        for k in range(nn_if):
            i0 = Glist.index(2 * interf[k]); i1 = Glist.index(2 * interf[k] + 1)
            Bx[j, i0] = np.sum(ws * Nm[:, k] * P[j]); By[j, i1] = np.sum(ws * Nm[:, k] * P[j])
        Cx[j] = np.sum(ws[:, None] * Phis * P[j][:, None], axis=0)
    B = np.vstack([Bx, By])                              # (2nt, nG)
    Cw = np.zeros((2 * nt, 2 * nf)); Cw[:nt, :nf] = Cx; Cw[nt:, nf:] = Cx
    M_F = B @ np.linalg.solve(KGG_F, B.T)                # FEM 界面柔度
    M_N = Cw @ np.linalg.inv(KN) @ Cw.T                  # NN  界面柔度
    a = np.zeros(2 * nt); a[0] = 1.0                     # 常值 ux 迹模式
    m_F = float(a @ M_F @ a); m_N = float(a @ M_N @ a)
    return dict(chi=m_N / m_F, m_F=m_F, m_N=m_N, fnx=fnx, fny=fny, max_deg=max_deg)


def observed_split_scaled(KF_scale=1.0, max_deg=5, Delta=0.01):
    """在带 KF_scale 的系统上, 用均匀 ux=Δ 求解, 返回界面中点 ux/Δ。"""
    KF, nodes, elems, ndofF, nnF, left, interf, feats, fnx, fny = IM.build_fem(12, 8, max_deg)
    KF = np.array(KF) * KF_scale
    nf = len(feats)
    CS.right_disp = lambda y, D0=0.01: D0        # 均匀 BC
    KN, fN = IM.build_nn(feats, 1e6)
    CS.right_disp = _patched
    leftDof = set()
    for k in left:
        leftDof.add(2 * k); leftDof.add(2 * k + 1)
    G = set()
    for k in interf:
        G.add(2 * k); G.add(2 * k + 1)
    I = [c for c in range(ndofF) if c not in leftDof and c not in G]
    Glist = sorted(G)
    freeF = I + Glist
    KFr = KF[np.ix_(freeF, freeF)]
    nI = len(I); nG = len(Glist)
    Kii = KFr[:nI, :nI]; KiG = KFr[:nI, nI:]; KGG = KFr[nI:, nI:]
    KGG_F = KGG - KiG.T @ np.linalg.solve(Kii, KiG)
    KNinv = np.linalg.inv(KN)
    nt = 6
    gp, gw = np.polynomial.legendre.leggauss(80)
    ss = Ly * (gp + 1.0) / 2.0; ws = Ly / 2.0 * gw
    nn_if = len(interf)
    yk = np.array([nodes[k][1] for k in interf]); h = Ly / (nn_if - 1)
    Nm = np.zeros((len(ss), nn_if))
    for k in range(nn_if - 1):
        m = (ss >= yk[k] - 1e-12) & (ss <= yk[k + 1] + 1e-12)
        t = (ss[m] - yk[k]) / h
        Nm[m, k] += 1 - t; Nm[m, k + 1] += t
    P = np.array([((2 * ss / Ly) - 1.0) ** j for j in range(nt)])
    Phis = np.array([[CS.feat_val(fe, X0, s) for fe in feats] for s in ss])
    Bx = np.zeros((nt, nG)); By = np.zeros((nt, nG)); Cx = np.zeros((nt, nf))
    for j in range(nt):
        for k in range(nn_if):
            i0 = Glist.index(2 * interf[k]); i1 = Glist.index(2 * interf[k] + 1)
            Bx[j, i0] = np.sum(ws * Nm[:, k] * P[j]); By[j, i1] = np.sum(ws * Nm[:, k] * P[j])
        Cx[j] = np.sum(ws[:, None] * Phis * P[j][:, None], axis=0)
    B = np.vstack([Bx, By])
    Cw = np.zeros((2 * nt, 2 * nf)); Cw[:nt, :nf] = Cx; Cw[nt:, nf:] = Cx
    T = Cw @ KNinv @ Cw.T
    g = Cw @ KNinv @ fN
    nM = nG + 2 * nt
    M = np.zeros((nM, nM)); rhs = np.zeros(nM)
    M[:nG, :nG] = KGG_F; M[:nG, nG:] = B.T
    M[nG:, :nG] = B;     M[nG:, nG:] = -T
    rhs[nG:] = g
    sv = np.linalg.solve(M, rhs)
    uG = sv[:nG]
    # 界面中点 ux
    ymid = Ly / 2.0
    kk = min(range(nn_if), key=lambda k: abs(nodes[interf[k]][1] - ymid))
    idx = Glist.index(2 * interf[kk])
    return float(uG[idx]) / Delta


if __name__ == '__main__':
    print("=" * 74)
    print(" 能量尺度匹配理论验证: chi = M_N/M_F  ->  u_Γ/Δ 应为 1/(1+chi)")
    print("=" * 74)
    print(f"{'配置':<26}{'chi':>10}{'理论 1/(1+χ)':>14}{'实测 u_Γ/Δ':>12}")
    print("-" * 74)

    configs = [('FEM×1/8 (NN 过刚)', 1 / 8, 5),
               ('FEM×1/4', 1 / 4, 5),
               ('FEM×1/2', 1 / 2, 5),
               ('FEM×1  (匹配)', 1.0, 5),
               ('FEM×2', 2.0, 5),
               ('FEM×4', 4.0, 5),
               ('FEM×8  (8×雅可比缺陷)', 8.0, 5),
               ('FEM×1, NN deg=3', 1.0, 3),
               ('FEM×1, NN deg=7', 1.0, 7)]
    rows = []
    for name, s, deg in configs:
        op = operators(KF_scale=s, max_deg=deg)
        chi = op['chi']
        pred = 1.0 / (1.0 + chi)
        obs = observed_split_scaled(KF_scale=s, max_deg=deg)
        rows.append((name, chi, pred, obs))
        print(f"{name:<26}{chi:>10.4f}{pred:>14.4f}{obs:>12.4f}")

    # 保存
    import os
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'theory_energy_matching.txt'))
    with open(out, 'w', encoding='utf-8') as fp:
        fp.write("能量尺度匹配理论验证 (chi = M_N/M_F, 预测 u_Gamma/Delta = 1/(1+chi))\n")
        fp.write(f"{'配置':<26}{'chi':>10}{'理论':>14}{'实测':>12}{'相对偏差':>12}\n")
        for name, chi, pred, obs in rows:
            fp.write(f"{name:<26}{chi:>10.4f}{pred:>14.4f}{obs:>12.4f}{abs(pred-obs)/max(pred,1e-9):>11.2%}\n")
    print(f"\n[保存] {out}")

    # ---- 图: 实测分配比 vs chi, 与理论曲线 ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for _c in ['Microsoft YaHei', 'SimHei', 'STSong', 'Arial Unicode MS']:
        try:
            font_manager.findfont(_c, fallback_to_default=False)
            plt.rcParams['font.sans-serif'] = [_c]; break
        except Exception:
            continue
    plt.rcParams['axes.unicode_minus'] = False
    chis = np.array([r[1] for r in rows])
    obs = np.array([r[3] for r in rows])
    xx = np.logspace(np.log10(0.08), np.log10(12), 200)
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=150)
    ax.plot(xx, 1.0 / (1.0 + xx), 'k-', lw=2, label='理论 u_Γ/Δ = 1/(1+χ)')
    ax.plot(chis, obs, 'o', ms=9, mfc='#d62728', mec='k', label='实测 (各配置)')
    ax.axvline(1.0, color='#2ca02c', ls='--', lw=1.4)
    ax.text(1.06, 0.86, '匹配点 χ=1\n(两子域等柔度)', color='#2ca02c', fontsize=9)
    ax.annotate('8×雅可比缺陷\n(χ≈8)', xy=(8.06, 0.109), xytext=(2.2, 0.20),
                arrowprops=dict(arrowstyle='->', color='k'), fontsize=9)
    ax.set_xscale('log')
    ax.set_xlabel('能量尺度匹配数  χ = M_N / M_F  (对数轴)')
    ax.set_ylabel('界面位移分配比  u_Γ / Δ')
    ax.set_title('能量尺度匹配理论：界面位移分配比预测与验证', fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    pfig = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'theory_energy_matching.png'))
    fig.savefig(pfig, bbox_inches='tight'); plt.close(fig)
    print(f"[出图] {pfig}")
