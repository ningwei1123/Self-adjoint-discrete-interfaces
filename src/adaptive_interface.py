# -*- coding: utf-8 -*-
"""
自适应界面 (Adaptive Interface) —— 后验误差指示子驱动的 NN 基函数富集
====================================================================
思路:
  1. 用 mortar 界面耦合求一次解;
  2. 以 **强形式残差** r(x,y)=|∇·σ_N(x,y)| (NN 子域的 PDE 残差) 作为后验误差指示子,
     它无需真解、可直接由当前解算出, 是高误差区的可靠标记;
  3. 在指示子最大的位置自适应添加 **径基(RBF)局部富集项** φ=exp(-r²/s²),
     把自由度用到真正需要的边界层/界面附近;
  4. 迭代, 记录 "误差 - 自由度数" 曲线, 与"均匀提高多项式阶数"对比。

预期: 自适应富集以更少自由度达到同等精度 (自适应优于均匀)。
"""
import os, math, time
import numpy as np
import coupled_solver as CS
import interface_methods as IM
import stress_recovery2d as R

Lx, Ly, X0 = CS.Lx, CS.Ly, CS.X0
D00, D01, D22 = CS.D00, CS.D01, CS.D22


def solve_mortar_feats(feats, fnx=12, fny=8, rho=1e6):
    KF, nodes, elems, ndofF, nnF, left, interf, _, fnx, fny = IM.build_fem(fnx, fny, 5)
    KF = np.array(KF)
    nf = len(feats)
    freeF = IM.fem_free_index_map(left, ndofF)
    NF = len(freeF)
    KN, fN = IM.build_nn(feats, rho)
    Bx, By, Cx, Cy = IM.mortar_matrices(nodes, interf, feats, freeF)
    nt = Bx.shape[0]
    nM = NF + 2 * nf + 2 * nt
    M = np.zeros((nM, nM)); rhs = np.zeros(nM)
    M[:NF, :NF] = KF[np.ix_(freeF, freeF)]
    M[NF:NF + nf, NF:NF + nf] = KN[:nf, :nf]
    M[NF + nf:NF + 2 * nf, NF + nf:NF + 2 * nf] = KN[nf:, nf:]
    M[NF:NF + nf, NF + nf:NF + 2 * nf] = KN[:nf, nf:]
    M[NF + nf:NF + 2 * nf, NF:NF + nf] = KN[nf:, :nf]
    rhs[NF:NF + nf] = fN[:nf]; rhs[NF + nf:NF + 2 * nf] = fN[nf:]
    lam0 = NF + 2 * nf
    M[lam0:lam0 + nt, :NF] = Bx;            M[:NF, lam0:lam0 + nt] = Bx.T
    M[lam0 + nt:lam0 + 2 * nt, :NF] = By;   M[:NF, lam0 + nt:lam0 + 2 * nt] = By.T
    M[lam0:lam0 + nt, NF:NF + nf] = -Cx;    M[NF:NF + nf, lam0:lam0 + nt] = -Cx.T
    M[lam0 + nt:lam0 + 2 * nt, NF + nf:NF + 2 * nf] = -Cy
    M[NF + nf:NF + 2 * nf, lam0 + nt:lam0 + 2 * nt] = -Cy.T
    sv = np.linalg.solve(M, rhs)
    ufem = np.zeros(ndofF)
    for idx, comp in enumerate(freeF):
        ufem[comp] = sv[idx]
    return IM.build_sol_dict(nodes, elems, fnx, fny, ufem,
                             sv[NF:NF + nf], sv[NF + nf:NF + 2 * nf], feats, interf, left, ndofF)


def nn_stress(sol, x, y):
    c, d, feats = sol['c'], sol['d'], sol['feats']
    nf = len(feats)
    exx = sum(c[a] * CS.feat_grad(feats[a], x, y)[0] for a in range(nf))
    eyy = sum(d[a] * CS.feat_grad(feats[a], x, y)[1] for a in range(nf))
    exy = 0.5 * (sum(c[a] * CS.feat_grad(feats[a], x, y)[1] for a in range(nf))
                 + sum(d[a] * CS.feat_grad(feats[a], x, y)[0] for a in range(nf)))
    return (D00 * exx + D01 * eyy, D01 * exx + D00 * eyy, D22 * 2 * exy)


def strong_residual(sol, x, y, h=2e-3):
    """r = |∇·σ| 在 NN 子域 (无体力时为 0, 指示离散误差)。"""
    def s(xx, yy):
        return nn_stress(sol, xx, yy)
    dxx_dx = (s(x + h, y)[0] - s(x - h, y)[0]) / (2 * h)
    dxy_dy = (s(x, y + h)[2] - s(x, y - h)[2]) / (2 * h)
    dxy_dx = (s(x + h, y)[2] - s(x - h, y)[2]) / (2 * h)
    dyy_dy = (s(x, y + h)[1] - s(x, y - h)[1]) / (2 * h)
    r1 = dxx_dx + dxy_dy
    r2 = dxy_dx + dyy_dy
    return math.hypot(r1, r2)


def ref_disp(ref, x, y):
    hx = Lx / ref['nx']; hy = Ly / ref['ny']
    i = min(max(int(x / hx), 0), ref['nx'] - 1); j = min(max(int(y / hy), 0), ref['ny'] - 1)
    xi = 2 * (x - i * hx) / hx - 1; eta = 2 * (y - j * hy) / hy - 1
    N, _, _ = R._q4(xi, eta)
    a = j * (ref['nx'] + 1) + i
    idx = [a, a + 1, a + 1 + (ref['nx'] + 1), a + (ref['nx'] + 1)]
    return (float(N @ np.array([ref['u'][2 * k] for k in idx])),
            float(N @ np.array([ref['u'][2 * k + 1] for k in idx])))


def disp_err(sol, ref, nx=60, ny=30):
    num = den = 0.0
    for i in range(nx):
        x = Lx * (i + 0.5) / nx
        for j in range(ny):
            y = Ly * (j + 0.5) / ny
            if x <= X0:
                uc = CS.fem_point_disp(sol, x, y)
            else:
                uc = CS.nn_point_disp(sol, x, y)
            ur = ref_disp(ref, x, y)
            num += (uc[0] - ur[0]) ** 2 + (uc[1] - ur[1]) ** 2
            den += ur[0] ** 2 + ur[1] ** 2
    return math.sqrt(num / max(den, 1e-30))


def main():
    ref = R.solve_reference_np(56, 20)
    print(f"参考 56x20, 节点 {ref['nn']}")

    # ---------- (A) 均匀提高多项式阶数 ----------
    print("\n[A] 均匀提高多项式阶数:")
    uni = []
    for deg in [5, 6, 7, 8, 9, 10]:
        feats = CS.make_features(deg)
        sol = solve_mortar_feats(feats)
        e = disp_err(sol, ref)
        uni.append((len(feats), e))
        print(f"    deg={deg:2d}  特征数={len(feats):3d}  位移L2={e:.4%}")

    # ---------- (B) 自适应 RBF 富集 (残差指示子) ----------
    print("\n[B] 自适应 RBF 富集:")
    rbfs = []
    adap = []
    s_rbf = 0.09
    NSTEP = 12
    ind_map = None
    for step in range(NSTEP):
        feats = CS.make_features(5, rbfs)
        sol = solve_mortar_feats(feats)
        e = disp_err(sol, ref)
        adap.append((len(feats), e))
        print(f"    步{step}  特征数={len(feats):3d}  位移L2={e:.4%}")
        # 指示子网格 (NN 区)
        gx = np.linspace(X0 + 0.03 * (Lx - X0), Lx - 0.02, 34)
        gy = np.linspace(0.02, Ly - 0.02, 30)
        I = np.zeros((len(gy), len(gx)))
        for jj, y in enumerate(gy):
            for ii, x in enumerate(gx):
                I[jj, ii] = strong_residual(sol, x, y)
        ind_map = (gx, gy, I)
        if step == NSTEP - 1:
            break
        # 选最大指示子的点 (最小间距去重)
        cand = [(I[jj, ii], gx[ii], gy[jj]) for jj in range(len(gy)) for ii in range(len(gx))]
        cand.sort(reverse=True)
        added = 0
        for val, cx, cy in cand:
            if all((cx - a) ** 2 + (cy - b) ** 2 > (0.16) ** 2 for (a, b, _) in rbfs):
                rbfs.append((cx, cy, s_rbf)); added += 1
                if added >= 10:
                    break

    # ---------- 图 ----------
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for _c in ['Microsoft YaHei', 'SimHei', 'STSong', 'Arial Unicode MS']:
        try:
            font_manager.findfont(_c, fallback_to_default=False)
            plt.rcParams['font.sans-serif'] = [_c]; break
        except Exception:
            continue
    plt.rcParams['axes.unicode_minus'] = False
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)
    ux = [a[0] for a in uni]; ue = [a[1] * 100 for a in uni]
    ax = [a[0] for a in adap]; ae = [a[1] * 100 for a in adap]
    axs[0].plot(ux, ue, 'o-', color='#1f77b4', label='均匀提高多项式阶数')
    axs[0].plot(ax, ae, 's-', color='#d62728', label='自适应 RBF 富集 (残差指示子)')
    axs[0].set_xlabel('NN 基函数个数 (自由度代理)'); axs[0].set_ylabel('位移 L2 误差 (%)')
    axs[0].set_yscale('log'); axs[0].grid(alpha=0.3, which='both'); axs[0].legend(fontsize=9)
    axs[0].set_title('自适应 vs 均匀: 误差-自由度', fontsize=11)
    gx, gy, I = ind_map
    im = axs[1].imshow(I, origin='lower', extent=[gx[0], gx[-1], gy[0], gy[-1]], aspect='auto', cmap='hot')
    for (a, b, s) in rbfs:
        axs[1].plot(a, b, 'o', ms=5, mfc='none', mec='#17becf', mew=1.4)
    axs[1].axvline(X0, color='white', ls='--', lw=1.0)
    axs[1].set_xlabel('x'); axs[1].set_ylabel('y')
    axs[1].set_title('后验指示子 |∇·σ_N| 与 RBF 富集点(青圈)', fontsize=10)
    fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    fig.suptitle('自适应界面/基函数富集', fontsize=12)
    pfig = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'adaptive_interface.png'))
    fig.savefig(pfig, bbox_inches='tight'); plt.close(fig)
    print(f"\n[出图] {pfig}")

    out = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'adaptive_interface.txt'))
    with open(out, 'w', encoding='utf-8') as fp:
        fp.write("自适应界面 (残差指示子驱动的 RBF 基函数富集) vs 均匀提高阶数\n")
        fp.write(f"{'方案':<14}{'特征数':>8}{'位移L2':>12}\n")
        for (nf, e) in uni:
            fp.write(f"{'均匀deg':<14}{nf:>8}{e:>11.4%}\n")
        for (nf, e) in adap:
            fp.write(f"{'自适应RBF':<14}{nf:>8}{e:>11.4%}\n")
    print(f"[保存] {out}")


if __name__ == '__main__':
    main()
