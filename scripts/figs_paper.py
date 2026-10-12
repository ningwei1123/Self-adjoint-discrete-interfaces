# -*- coding: utf-8 -*-
"""
论文复图生成器 (Fig 1-4) —— 投稿级版
====================================
Fig 1  框架与两个不变量 (概念示意)
Fig 2  能量匹配律 + 罚函数有效 chi 平台 + 3D 精度饱和
Fig 3  自伴性核心图 (2D 不对称度热图 + 有符号特征谱 + 汇总柱)
Fig 4  一致性阶 (2D Richardson) + 3D 迹空间分辨率与退化

规范: Elsevier/CMAME —— 190 mm 设计宽度, 字体 >=7 pt, 线宽 0.25-1.5 pt,
色盲安全色板, constrained layout 防重叠. 数值管线与 v1 完全一致.
"""
import _paths  # noqa: F401  (adds src/ to the path)
import os
import math
import time
import io
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm

import figstyle as FS
import coupled_solver as C2
import interface_methods as IM
import stress_recovery2d as R2
import audit_dtn_adjoint as AD
import audit_dtn3d as A3

FS.apply()

C_GAL, C_COL, C_ACC = FS.C_GAL, FS.C_COL, FS.C_ACC
C_GRY = FS.C_GRY
# the output directory can be relocated with RESULTS_DIR; the default is
# results/ beside this file
OUT = os.path.join(os.environ.get('RESULTS_DIR', os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))), 'results', 'figs')
os.makedirs(OUT, exist_ok=True)
W = FS.W_DOUBLE                       # 190 mm


def save(fig, name):
    FS.save(fig, name, OUT, tiff=False)


# ==================================================================== Fig 1
def fig01():
    fig = plt.figure(figsize=(W, 2.62), layout='constrained')
    gs = fig.add_gridspec(1, 3, width_ratios=[1.42, 1.00, 1.20])

    # ---------- (a) 域分解与 DtN 归约 ----------
    ax = fig.add_subplot(gs[0, 0])
    ax.set_aspect('equal')
    ax.axis('off')
    ax.grid(False)
    ax.add_patch(Rectangle((0, 0), 1, 1, fc='#eaf2fd', ec='none', zorder=0))
    ax.add_patch(Rectangle((1, 0), 1, 1, fc='#fdf3e3', ec='none', zorder=0))
    ax.add_patch(Rectangle((0, 0), 2, 1, fc='none', ec='k', lw=0.9, zorder=4))
    # FE 网格 (左, 4x3 -> 与 "Q4" 示意比例协调)
    for i in range(5):
        ax.plot([i / 4.0, i / 4.0], [0, 1], color='#4c7fbd', lw=0.4, zorder=2)
    for j in range(4):
        ax.plot([0, 1], [j / 3.0, j / 3.0], color='#4c7fbd', lw=0.4, zorder=2)
    # 网络配点 (右)
    rng = np.random.default_rng(3)
    ax.plot(1.08 + 0.84 * rng.random(40), 0.06 + 0.88 * rng.random(40), '.',
            color=C_ACC, ms=2.6, zorder=3)
    ax.plot([1, 1], [0, 1], ls=(0, (3.5, 2)), color=C_ACC, lw=1.4, zorder=5)
    ax.text(0.5, 1.30, r'$\Omega_F$: FE (Q4/H8)', ha='center', va='bottom',
            fontsize=8, color='#2b5c99')
    ax.text(1.5, 1.30, r'$\Omega_N$: network', ha='center', va='bottom',
            fontsize=8, color='#a5700a')
    ax.annotate('', xy=(-0.02, 0.5), xytext=(-0.30, 0.5),
                arrowprops=dict(arrowstyle='-|>', lw=1.0, color='k'))
    ax.text(-0.34, 0.5, r'$u=0$', ha='right', va='center', fontsize=8)
    ax.annotate('', xy=(2.30, 0.5), xytext=(2.02, 0.5),
                arrowprops=dict(arrowstyle='-|>', lw=1.0, color='k'))
    ax.text(2.34, 0.5, r'$u=g$', ha='left', va='center', fontsize=8)
    ax.text(1.0, -0.13, r'$\Gamma$', ha='center', va='top', fontsize=9,
            color=C_ACC)
    ax.text(0.0, -0.34,
            'eliminate interiors:\n'
            r'$(S_F+S_N)\,g=-q$' + '\n'
            r'$3n_t=48$ unknowns  vs  6964 parameters',
            ha='left', va='top', fontsize=7.6, color='#333')
    ax.set_xlim(-0.62, 2.85)
    ax.set_ylim(-1.02, 1.58)
    FS.panel(ax, 'a', x=0.0, y=1.02)

    # ---------- (b) 串联弹簧 + 能量匹配律 ----------
    ax = fig.add_subplot(gs[0, 1])
    ax.set_aspect('equal')
    ax.axis('off')
    ax.grid(False)
    for x0, k, col in [(0.0, 'F', '#2b5c99'), (0.5, 'N', '#a5700a')]:
        v = np.linspace(0, 1, 200)
        ax.plot(x0 + 0.5 * v, 0.11 * np.sin(v * 4 * math.pi), color=col, lw=1.1)
        ax.text(x0 + 0.25, 0.26, r'$k_%s$' % k, ha='center', va='bottom',
                fontsize=8.5, color=col)
    ax.plot([-0.10, 0.0], [0, 0], 'k', lw=2.0)
    ax.plot([1.0, 1.10], [0, 0], 'k', lw=2.0)
    ax.plot([0.5, 0.5], [-0.14, 0.14], ls=(0, (3, 2)), color=C_ACC, lw=1.4)
    ax.text(0.5, -0.30, r'$u_\Gamma$', ha='center', va='top', fontsize=8.5,
            color=C_ACC)
    ax.annotate('', xy=(1.34, -0.05), xytext=(1.05, -0.05),
                arrowprops=dict(arrowstyle='-|>', lw=1.1, color='k'))
    ax.text(1.19, -0.30, r'$\Delta$', ha='center', va='top', fontsize=8.5)
    ax.text(0.62, 0.95, r'$\dfrac{u_\Gamma}{\Delta}=\dfrac{1}{1+\chi}$',
            ha='center', va='center', fontsize=11)
    ax.text(0.62, 0.56, r'$\chi=M_N/M_F$', ha='center', va='center', fontsize=8.5)
    ax.text(0.0, -0.62,
            r'$\chi\approx1$: balanced'+'\n'
            'penalty = extra spring\n'
            r'$\chi$ computable a priori',
            ha='left', va='top', fontsize=7.6, color='#333')
    ax.set_xlim(-0.30, 1.52)
    ax.set_ylim(-1.25, 1.22)
    FS.panel(ax, 'b', x=0.0, y=1.02)

    # ---------- (c) 两种传递配对 ----------
    ax = fig.add_subplot(gs[0, 2])
    ax.axis('off')
    ax.grid(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    rng = np.random.default_rng(7)
    A = np.zeros((6, 6))
    iu = np.triu_indices(6)
    A[iu] = rng.random(len(iu[0]))
    A = A + A.T - np.diag(np.diag(A))
    B = A + 0.9 * (rng.random((6, 6)) - 0.5) * 0.8
    for M_, x0, ttl, ok, sub in [
            (A, 0.02, 'Galerkin', True, r'$MS=P^{\top}SP$'),
            (B, 0.54, 'Collocation', False, r'$MS=P^{\top}\!WT$')]:
        col = C_GAL if ok else C_COL
        ax.imshow(M_, extent=[x0, x0 + 0.40, 0.45, 0.845], cmap=FS.CM_DIV,
                  vmin=-1, vmax=1, aspect='auto', zorder=1)
        ax.add_patch(Rectangle((x0, 0.45), 0.40, 0.395, fill=False, ec=col,
                               lw=1.2, zorder=2))
        ax.text(x0 + 0.20, 0.945, ttl, ha='center', va='bottom',
                fontsize=8.5, color=col, fontweight='bold')
        ax.text(x0 + 0.20, 0.895, sub, ha='center', va='top', fontsize=8.5)
        ax.text(x0 + 0.20, 0.375,
                (r'$\checkmark$ symmetric' if ok else r'$\times$ not symmetric'),
                ha='center', va='top', fontsize=8.5, color=col,
                fontweight='bold')
    # 两张热图共用一条色标: 没有色标的 imshow 无法判读颜色含义
    ax.imshow(np.linspace(-1, 1, 128)[None, :],
              extent=[0.30, 0.70, 0.414, 0.440], cmap=FS.CM_DIV,
              vmin=-1, vmax=1, aspect='auto', zorder=1)
    ax.text(0.285, 0.427, r'$-1$', ha='right', va='center', fontsize=8.5)
    ax.text(0.715, 0.427, r'$+1$', ha='left', va='center', fontsize=8.5)
    ax.text(0.5, 1.00, 'traction transfer pairing', ha='center', va='bottom',
            fontsize=8.5)
    ax.text(0.5, 0.255,
            r'Galerkin:  $r=Ku$  (consistent reaction)'
            '\n'
            r'pairing $\langle t,\phi_i\rangle_\Gamma$  (weak form)',
            ha='center', va='top', fontsize=7.6, color='#2b5c99')
    ax.text(0.5, 0.055,
            'Collocation:  one-sided element sample\n'
            'unweighted least squares',
            ha='center', va='top', fontsize=7.6, color=C_COL)
    FS.panel(ax, 'c', x=0.0, y=1.06)
    save(fig, 'fig01_framework')


# ==================================================================== Fig 2
def _iface_ratio(sol):
    """界面轴向位移比 u_G/Delta (标量归约)。"""
    fnodes = np.array(sol['fnodes'])
    ufem = np.array(sol['ufem'])
    ifd = np.array(sol['interf'])
    ys = fnodes[ifd, 1]
    uG = float(np.mean(ufem[2 * ifd]))
    D = float(np.mean([C2.right_disp(y) for y in ys]))
    return uG / D


def _matching_sweep():
    """The chi sweep and its measured allocation, read from the results file.

    These five pairs were hard-coded in fig02.  results/theory_energy_matching.txt,
    written by theory_energy_matching.py, holds the same numbers, so the figure now
    reads them: a change in the sweep can no longer leave the figure stale without a
    check noticing, which is the failure mode found in the learning curves (Round 4).

    The file is written with GBK column captions, so it is decoded accordingly ---
    reading it as UTF-8 and ignoring errors silently drops the localised labels and
    makes row selection impossible.  The rows kept are the five stiffness-scale
    entries; the later rows are extra configurations (NN degree variants) that panel
    (a) does not plot.
    """
    data_dir = os.environ.get('DATA_DIR', os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))
    path = os.path.join(data_dir, 'theory_energy_matching.txt')
    raw = open(path, 'rb').read()
    text = None
    for enc in ('gbk', 'utf-8'):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise RuntimeError(f'cannot decode {path}')
    keep = ('1/8', '1/2', '1,', '4', '8')
    rows = []
    for line in text.splitlines():
        nums = re.findall(r'\d+\.\d+', line)
        if len(nums) < 4:
            continue
        head = line.split(nums[0])[0]
        if 'deg' in head:
            continue
        # the five sweep rows each carry a bare stiffness scale in the label
        m = re.search(r'FEM[^0-9]*([0-9]+(?:/[0-9]+)?)\s*(?:\(|$)', head)
        if not m:
            m = re.search(r'FEM[^0-9]*([0-9]+(?:/[0-9]+)?)', head)
        if not m:
            continue
        scale = m.group(1)
        if scale not in ('1/8', '1/2', '1', '4', '8'):
            continue
        rows.append((float(nums[0]), float(nums[2])))
    if len(rows) != 5:
        raise RuntimeError(f'expected 5 sweep rows, parsed {len(rows)} from {path}')
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])


def fig02():
    chi, obs = _matching_sweep()

    rhos = np.array([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
    ratios = np.array([_iface_ratio(C2.solve_coupled(rho=float(r))) for r in rhos])
    r_mortar = _iface_ratio(IM.solve_mortar())
    chi_eff = 1.0 / ratios - 1.0
    chi_mat = 1.0 / r_mortar - 1.0

    import coupled_solver3d as C3
    import coupled_solver3d_mortar as M3
    rsol = C3.solve_reference_3d('tension')
    e_mortar = C3.l2_disp_error(M3.build_mortar('tension'), rsol)
    rho3 = np.array([1e4, 1e5, 1e6, 1e7, 1e8, 1e10, 1e12])
    e3 = np.array([C3.l2_disp_error(C3.solve_coupled('tension', rho=float(r)), rsol)
                   for r in rho3])
    print('  2D u_G/Delta = ' + str(np.round(ratios, 4)) + f'  (mortar {r_mortar:.4f})')
    print('  2D chi_eff   = ' + str(np.round(chi_eff, 4)) + f'  (material {chi_mat:.4f})')
    print('  3D penalty L2 = ' + str(np.round(e3, 5)) + f'  (Galerkin {e_mortar:.6f})')

    fig = plt.figure(figsize=(W, 2.42), layout='constrained')
    gs = fig.add_gridspec(1, 3)

    # (a) matching law
    ax = fig.add_subplot(gs[0, 0])
    xx = np.logspace(-1, 1.2, 200)
    ax.plot(xx, 1 / (1 + xx), '-', color=C_GAL, lw=1.6, label=r'$1/(1+\chi)$')
    ax.plot(chi, obs, 'o', ms=5.5, mfc='white', mec=C_COL, mew=1.4,
            label='measured', zorder=3)
    ax.axvline(1.0, color='#999', lw=0.8, ls=':')
    ax.plot([1.014], [0.495], '*', ms=11, color=C_ACC, zorder=4)
    ax.set_xscale('log')
    ax.set_xlabel(r'matching number  $\chi=M_N/M_F$')
    ax.set_ylabel(r'interface share  $u_\Gamma/\Delta$')
    ax.set_title('(a) energy-matching law')
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc='lower left')
    ax.annotate(r'balanced, $\chi\approx1$', xy=(1.014, 0.495), xytext=(1.5, 0.78),
                fontsize=FS.MIN_ANNOT, color=C_ACC,
                arrowprops=dict(arrowstyle='->', lw=0.9, color=C_ACC))

    # (b) 2D allocation vs penalty coefficient (双 y 轴)
    ax = fig.add_subplot(gs[0, 1])
    ax.semilogx(rhos, ratios, 'o-', color=C_COL, lw=1.3, ms=4.5,
                label='penalty (left axis)')
    ax.axhline(r_mortar, color=C_GAL, lw=1.5, ls='--',
               label=r'mortar ($\rho$-free)')
    ax.set_xlabel(r'penalty coefficient  $\rho$')
    ax.set_ylabel(r'$u_\Gamma/\Delta$   (2D)')
    ax.set_title(r'(b) allocation converges in $\rho$')
    ax.set_ylim(0.15, 0.70)
    ax.legend(loc='upper left')
    axt = ax.twinx()
    lo, hi = ax.get_ylim()
    axt.set_ylim(1 / hi - 1, 1 / lo - 1)
    axt.set_ylabel(r'$\chi_{\rm eff}=\Delta/u_\Gamma-1$')
    axt.grid(False)
    axt.axhline(chi_mat, color=C_GAL, lw=0.8, ls=':', alpha=0.8)
    ax.text(0.975, 0.115, r'material $\chi$', transform=ax.transAxes, ha='right',
            va='bottom', fontsize=FS.MIN_ANNOT, color=C_GAL)

    # (c) 3D accuracy vs penalty coefficient
    ax = fig.add_subplot(gs[0, 2])
    ax.loglog(rho3, 100 * e3, 's-', color=C_COL, lw=1.3, ms=4.5,
              label='penalty coupling')
    ax.axhline(100 * e_mortar, color=C_GAL, lw=1.5, ls='--',
               label='Galerkin transfer = %.4f%%' % (100 * e_mortar))
    ax.set_xlabel(r'penalty coefficient  $\rho$')
    ax.set_ylabel(r'3D displacement rel. $L^2$  (\%)')
    ax.set_title(r'(c) accuracy saturates in $\rho$')
    ax.set_ylim(3e-3, 20)
    # 原 loc='center left' 的图例文字向右溢出坐标区, 与 '124x gap' 重叠
    ax.legend(loc='upper right')
    ax.annotate('', xy=(rho3[-1], 100 * e3[-1]),
                xytext=(rho3[-1], 100 * e_mortar),
                arrowprops=dict(arrowstyle='<->', lw=1.0, color='k'))
    ax.annotate('%.0f$\\times$ gap' % (e3[-1] / e_mortar),
                xy=(rho3[-1], math.sqrt(100 * e3[-1] * 100 * e_mortar)),
                xytext=(3e9, 0.05), fontsize=FS.MIN_ANNOT,
                arrowprops=dict(arrowstyle='-', lw=0.8, color='k'))
    save(fig, 'fig02_energy_matching')
    return dict(rhos=rhos, ratios=ratios, chi_eff=chi_eff, chi_mat=chi_mat,
                r_mortar=r_mortar, rho3=rho3, e3=e3, e_mortar=e_mortar)


# ==================================================================== Fig 3
def dtn2d(side, mode, nx, ny, nt=AD.NT):
    """2D interface operator, weighted form A = M S^tr (blockwise component order)."""
    x0, x1 = (0.0, A3.X0) if side == 'L' else (A3.X0, A3.Lx)
    nodes, elems, iface = AD.build(x0, x1, nx, ny)
    K = AD.assemble(nodes, elems)
    ys = np.array([nodes[k, 1] for k in iface])
    m = len(iface)
    P = np.array([AD.trace_basis(y, nt) for y in ys])
    w = np.full(m, A3.Ly / ny)
    w[0] = w[-1] = A3.Ly / (2.0 * ny)
    A = np.zeros((2 * nt, 2 * nt))
    for c in range(2):
        for j in range(nt):
            if c == 0:
                gf = (lambda y, j=j: (AD.trace_basis(y, nt)[j], 0.0))
            else:
                gf = (lambda y, j=j: (0.0, AD.trace_basis(y, nt)[j]))
            far = None if side == 'L' else (A3.Lx, lambda y: 0.0)
            u = AD.solve_dirichlet(nodes, elems, K, iface, gf, far)
            if mode == 'galerkin':
                r = K @ u
                for cp in range(2):
                    rv = np.array([r[2 * k + cp] for k in iface])
                    A[cp * nt:(cp + 1) * nt, c * nt + j] = P.T @ rv
            else:
                at_right = (side == 'L')
                lam = AD.lam_edge(nodes, elems, u, iface, at_right, nx, ny, False)
                for cp in range(2):
                    A[cp * nt:(cp + 1) * nt, c * nt + j] = P.T @ (w * lam[:, cp])
    h = AD.l2_weights(nt)
    M = np.diag(np.tile(h, 2))
    return A, M, P


def iface2d(mode, nx=12, ny=12, nt=AD.NT):
    """2D interface solve with the same convention as the 3D audit."""
    A_L, M, P = dtn2d('L', mode, nx, ny, nt)
    A_R, _, _ = dtn2d('R', mode, nx, ny, nt)
    x0, x1 = A3.X0, A3.Lx
    nodes, elems, iface = AD.build(x0, x1, nx, ny)
    K = AD.assemble(nodes, elems)
    ys = np.array([nodes[k, 1] for k in iface])
    m = len(iface)
    w = np.full(m, A3.Ly / ny)
    w[0] = w[-1] = A3.Ly / (2.0 * ny)
    u = AD.solve_dirichlet(nodes, elems, K, iface,
                           lambda y: (0.0, 0.0), (A3.Lx, C2.right_disp))
    b = np.zeros(2 * nt)
    if mode == 'galerkin':
        r = K @ u
        for cp in range(2):
            b[cp * nt:(cp + 1) * nt] = P.T @ np.array([r[2 * k + cp] for k in iface])
    else:
        lam = AD.lam_edge(nodes, elems, u, iface, False, nx, ny, False)
        for cp in range(2):
            b[cp * nt:(cp + 1) * nt] = P.T @ (w * lam[:, cp])
    nF, eF = R2.build_mesh(2 * nx, ny, 0.0, A3.Lx, 0.0, A3.Ly)
    KF = R2.assemble_np(nF, eF)
    nd = 2 * len(nF)
    lf = np.where(np.abs(nF[:, 0]) < 1e-9)[0]
    rf = np.where(np.abs(nF[:, 0] - A3.Lx) < 1e-9)[0]
    fx = {}
    for k in lf:
        fx[2 * k] = 0.0
        fx[2 * k + 1] = 0.0
    for k in rf:
        fx[2 * k] = C2.right_disp(nF[k, 1])
    fdd = np.array(list(fx))
    fset = set(fdd.tolist())
    fr = np.array([d for d in range(nd) if d not in fset])
    uM = np.zeros(nd)
    uM[fr] = np.linalg.solve(KF[np.ix_(fr, fr)],
                             -KF[np.ix_(fr, fdd)] @ np.array([fx[d] for d in fdd]))
    for d, v in fx.items():
        uM[d] = v
    iM = sorted(np.where(np.abs(nF[:, 0] - A3.X0) < 1e-9)[0], key=lambda k: nF[k, 1])
    yM = nF[iM, 1]
    gref = np.column_stack([np.interp(ys, yM, uM[2 * np.array(iM)]),
                            np.interp(ys, yM, uM[2 * np.array(iM) + 1])])
    best = None
    for qs in (1.0, -1.0):
        c = np.linalg.solve(A_L + A_R, qs * b)
        gn = np.column_stack([P @ c[:nt], P @ c[nt:]])
        e = np.linalg.norm(gn - gref) / np.linalg.norm(gref)
        if best is None or e < best[0]:
            best = (e, qs, gn, c)
    e, qs, gn, c = best
    uG = float(np.mean(gn[:, 0]))
    D = float(np.mean([C2.right_disp(y) for y in ys]))
    Ph = np.zeros((2 * m, 2 * nt))
    for i in range(m):
        v = AD.trace_basis(ys[i], nt)
        for cc in range(2):
            Ph[2 * i + cc, cc * nt:(cc + 1) * nt] = v
    c_ref = np.linalg.lstsq(Ph, gref.reshape(-1), rcond=None)[0]
    return dict(err=e, chi=D / uG - 1.0, A_L=A_L, A_R=A_R, b=b, P=P,
                c=c, c_ref=c_ref, gref=gref, ys=ys)


def fig03():
    nx, ny = 12, 12
    Acol_L, _, _ = dtn2d('L', 'collocation', nx, ny)
    Agal_L, M2, _ = dtn2d('L', 'galerkin', nx, ny)
    Acol_R, _, _ = dtn2d('R', 'collocation', nx, ny)
    Agal_R, _, _ = dtn2d('R', 'galerkin', nx, ny)

    def asym(A_):
        return np.abs(A_ - A_.T).max() / np.abs(A_).max()

    a_col = asym(Acol_L)
    a_gal = asym(Agal_L)
    print(f'  2D asym: collocation {a_col:.3e}  galerkin {a_gal:.3e}')

    ops = A3.build_ops('tension', 8, 4, 4, 3)
    P3 = ops['L']['P']
    A3col, A3gal = ops['L']['Apw'], ops['L']['A']
    a3_col, a3_gal = asym(A3col), asym(A3gal)
    print(f'  3D asym: collocation {a3_col:.3e}  galerkin {a3_gal:.3e}')

    # 3D 界面位移误差 (对应 Table 2 的 3D 行): 现算, 不再硬编码 23.6 / 0.042。
    # 配置与 Table 2 一致: H8 子域 8x4x4, tdeg=3 -> n_t=16, 25 界面节点,
    # 整体参考 16x4x4。此前这里写死常量, 与正文表 2 之间存在静默失同步风险。
    _n2, _e2, _nx2, _ny2, _nz2, _uref = A3.whole_fem('tension', 16, 4, 4)
    _gr = A3.iface_ref(_n2, _uref, _nx2, _ny2, 16)
    e3 = {}
    for _key, _nm in (('Sg', 'galerkin'), ('Spw', 'collocation')):
        _c = A3.solve_iface(ops, _key)
        _gn = (P3 @ _c).reshape(-1, 3)
        _ee = [np.linalg.norm(_gn[:, k] - _gr[:, k]) / np.linalg.norm(_gr[:, k])
               for k in range(3)]
        e3[_nm] = math.sqrt(sum(x * x for x in _ee) / 3)
    print(f'  3D interface err: collocation {e3["collocation"]:.4%}  '
          f'galerkin {e3["galerkin"]:.4%}')

    # 版式: 三行。上排两张热图各半宽; 中排汇总柱通栏; 下排两张有符号谱各半宽。
    # 旧版把 (c)(d)(e) 挤在一行 (1.42 : 0.79 : 0.79), 谱图只拿到约 26% 宽度,
    # 横轴 12/49 个特征值挤在一起且图例压曲线。改为各占半宽后, 谱图宽度翻倍。
    fig = plt.figure(figsize=(W, 5.75), layout='constrained')
    gs = fig.add_gridspec(3, 1, height_ratios=[1.00, 0.82, 0.90], hspace=0.22)
    top = gs[0, 0].subgridspec(1, 2, wspace=0.12)
    mid = gs[1, 0].subgridspec(1, 1)
    bot = gs[2, 0].subgridspec(1, 2, wspace=0.20)

    # ---- 上排: 两个不对称度热图 ----
    Dcol = np.abs(Acol_L - Acol_L.T)
    Dgal = np.abs(Agal_L - Agal_L.T)
    eps = 1e-17
    vmin, vmax = eps, max(Dcol.max(), 1e-3)
    for k, (D, ttl, cc) in enumerate([(Dcol, '(a) collocation transfer', C_COL),
                                      (Dgal, '(b) Galerkin transfer', C_GAL)]):
        ax = fig.add_subplot(top[0, k])
        im = ax.imshow(np.maximum(D, eps), cmap=FS.CM_LOG,
                       norm=LogNorm(vmin=vmin, vmax=vmax), origin='lower')
        ax.set_title(ttl, color=cc)
        ax.set_xlabel('trace mode')
        ax.set_ylabel('trace mode')
        ax.grid(False)
        ax.text(0.04, 0.96,
                r'max$|MS-(MS)^{\top}|=$' + f'{D.max():.2e}',
                transform=ax.transAxes, ha='left', va='top', fontsize=FS.MIN_ANNOT,
                color=cc, fontweight='bold',
                bbox=dict(fc='white', ec=cc, lw=0.6, alpha=0.88,
                          boxstyle='round,pad=0.22'))
        FS.cbar(fig, im, ax, label=r'$|MS-(MS)^{\top}|$')

    # ---- 下排: (d)(e) 两张有符号谱各占半宽 ----
    Dc = iface2d('collocation')
    Dg = iface2d('galerkin')
    e2_col, e2_gal = Dc['err'], Dg['err']
    c_ex = Dg['c_ref']
    disc2 = (np.linalg.norm((Dc['A_L'] + Dc['A_R']) @ c_ex
                            - (Dg['A_L'] + Dg['A_R']) @ c_ex)
             / np.linalg.norm((Dg['A_L'] + Dg['A_R']) @ c_ex))
    print(f'  2D interface err: collocation {e2_col:.4%}  galerkin {e2_gal:.4%}')
    print(f'  2D op. discrepancy at a reference trace: {disc2:.2%}')

    def spec(ax, A1, A2, ttl):
        for A_, nm, col, ls in [(A1, 'collocation', C_COL, '--'),
                                (A2, 'Galerkin', C_GAL, '-')]:
            ev = np.linalg.eigvalsh(0.5 * (A_ + A_.T)) / np.abs(A_).max()
            ev = np.sort(ev)[::-1]
            ax.semilogy(np.abs(ev), ls, color=col, lw=1.3, marker='o', ms=3.0,
                        label=nm)
            neg = ev < 0
            if neg.any():
                ax.semilogy(np.where(neg)[0], np.abs(ev[neg]), 'v', color=col,
                            ms=6.5, mfc='none', mew=1.3)
        ax.axhline(1e-16, color='#999', lw=0.8, ls=':')
        ax.semilogy([], [], 'v', color='#444444', ms=6.5, mfc='none', mew=1.3,
                    label='negative eigenvalue')
        ax.set_xlabel('eigenvalue index')
        ax.set_ylabel(r'$|\lambda|/\|\cdot\|$')
        ax.set_title(ttl)
        ax.set_ylim(1e-17, 4)
        # 图例移到右下角: 左下角与 1e-16 参考线、以及配点曲线的最低段重叠。
        ax.legend(loc='lower right', fontsize=FS.MIN_ANNOT - 0.6,
                  framealpha=0.94, edgecolor='none', borderpad=0.35,
                  labelspacing=0.35, handletextpad=0.6, handlelength=1.6)
        ax.grid(alpha=0.22, axis='y', which='both')

    axc = fig.add_subplot(mid[0, 0])
    names = ['asym 2D', 'asym 3D', 'op.disc 2D', 'err 2D', 'err 3D']
    vc = [a_col, a3_col, disc2, e2_col, e3['collocation']]
    vg = [max(a_gal, 1e-17), max(a3_gal, 1e-17), 1e-16, e2_gal, e3['galerkin']]
    xs = np.arange(len(names))
    bw = 0.30
    axc.bar(xs - bw / 2, vc, bw, color=C_COL, label='collocation')
    axc.bar(xs + bw / 2, vg, bw, color=C_GAL, label='Galerkin')
    axc.set_yscale('log')
    axc.set_ylim(1e-17, 1e4)
    axc.set_xticks(xs)
    # 该面板通栏 (~190 mm), 五个组名横排绰绰有余; 竖排 90 度反而难读
    axc.set_xticklabels(names, fontsize=FS.MIN_ANNOT)
    axc.set_ylabel('relative value')
    axc.set_title('(c) every measure separates the two pairings',
                  fontsize=FS.MIN_ANNOT)
    axc.legend(loc='upper right', fontsize=FS.MIN_ANNOT - 0.5, ncol=2,
               frameon=False)
    axc.grid(alpha=0.22, axis='y', which='both')
    # 柱高 0.1~1 在对数轴上已接近顶边, 数值标签必须放进柱内, 否则会顶到标题。
    for i, (a_, b_) in enumerate(zip(vc, vg)):
        axc.annotate(f'{a_:.3g}' if a_ >= 0.01 else f'{a_:.0e}',
                     xy=(i - bw / 2, max(a_, 1e-17)),
                     xytext=(0, -3), textcoords='offset points',
                     ha='center', va='top', fontsize=FS.MIN_ANNOT - 0.5,
                     color='white', fontweight='bold')
        # 前两根 Galerkin 柱落在舍入底噪上, 不是测量值; 前己方 0 是按定义
        bl = {0: r'$\leq10^{-15}$', 1: r'$\leq10^{-15}$',
              2: r'$0$ (ref.)'}.get(i)
        if bl is None:
            bl = f'{b_:.3g}' if b_ >= 0.01 else f'{b_:.0e}'
        axc.annotate(bl, xy=(i + bw / 2, max(b_, 1e-17)),
                     xytext=(0, 3), textcoords='offset points',
                     ha='center', va='bottom', fontsize=FS.MIN_ANNOT - 0.5,
                     color=C_GAL)

    axd = fig.add_subplot(bot[0, 0])
    spec(axd, Acol_L, Agal_L, '(d) 2D signed spectrum')

    axe = fig.add_subplot(bot[0, 1])
    spec(axe, A3col, A3gal, '(e) 3D signed spectrum')

    save(fig, 'fig03_self_adjoint')
    return dict(a_col=a_col, a_gal=a_gal, a3_col=a3_col, a3_gal=a3_gal)


# ==================================================================== Fig 4
def fig04():
    # (a) 2D 一致性阶 —— 与审计 P5 同口径:
    #     对 DtN 探针解出的界面迹 g 做三点 Richardson。
    #     旧版误用"整体参考解的界面网格差分", 得到 1.97, 与正文/表 2 的 1.99 不符。
    import func_operator as F
    ccoef = np.array([0.30, -0.20, 0.15, -0.25])
    gs_, hs = [], []
    for nx in (8, 12, 16, 24):
        M_, q_ = F.probe_field(ccoef, F.NU, nx, nx)
        gs_.append(np.linalg.solve(M_, -q_))
        hs.append(1.0 / nx)
    d = [np.linalg.norm(gs_[i + 1] - gs_[i]) for i in range(3)]
    hh = hs[:3]
    best, bf = None, 1e18
    for pp in np.linspace(0.2, 5.0, 481):
        f = (hh[0] ** pp - hh[1] ** pp) / (hh[1] ** pp - hh[2] ** pp) - d[0] / d[1]
        if abs(f) < bf:
            bf, best = abs(f), pp
    print(f'  2D Richardson p = {best:.3f}')

    rows = []
    for tdeg in (2, 3, 4):
        tb = A3.tb_list(tdeg)
        nt = len(tb)
        M1 = A3.l2_mass(tb, tdeg)
        Mf = A3.block(M1, 3 * nt)
        for (a_, b_, c_) in [(6, 4, 4), (8, 5, 5), (10, 6, 6)]:
            G = {}
            for side in ('L', 'R'):
                Sp = A3.build_side(side, 'tension', a_, b_, c_)
                Seff, q = A3.schur(Sp)
                Pp = A3.trace_P(Sp, tb, tdeg)
                area = A3.nodal_area(Sp)
                Sg = np.linalg.solve(Mf, Pp.T @ Seff @ Pp)
                Spw = np.linalg.solve(Mf, Pp.T @ A3.pw_operator(Sp, area) @ Pp)
                bg = np.linalg.solve(Mf, Pp.T @ q)
                G[side] = (Sg, Spw, bg, Pp)
            n2, e2, nx2, ny2, nz2, uref = A3.whole_fem('tension', 2 * a_, b_, c_)
            gr = A3.iface_ref(n2, uref, nx2, ny2, 2 * a_)
            nif = len(G['L'][3]) // 3
            out = {}
            for idxk, nm in ((0, 'galerkin'), (1, 'collocation')):
                cv = np.linalg.solve(G['L'][idxk] + G['R'][idxk],
                                     -(G['L'][2] + G['R'][2]))
                gn = (G['L'][3] @ cv).reshape(-1, 3)
                ee = [np.linalg.norm(gn[:, k] - gr[:, k]) / np.linalg.norm(gr[:, k])
                      for k in range(3)]
                out[nm] = math.sqrt(sum(x * x for x in ee) / 3)
            rows.append((tdeg, nt, nif, a_, out['galerkin'], out['collocation']))
            print(f'    tdeg={tdeg} mesh={(a_, b_, c_)} nt={nt} nif={nif} '
                  f'gal={out["galerkin"]:.4%} col={out["collocation"]:.4%}')

    fig = plt.figure(figsize=(W, 2.42), layout='constrained')
    gs = fig.add_gridspec(1, 3)

    ax = fig.add_subplot(gs[0, 0])
    hsf = np.array(hh)
    ax.loglog(hsf, d, 'o-', color=C_GAL, lw=1.3, ms=5, label='DtN interface trace')
    href = np.array([0.055, 0.14])
    ax.loglog(href, d[-1] * (href / hh[-1]) ** 2, 'k--', lw=1.0, label=r'$O(h^2)$')
    ax.set_xlabel('$h$ (interface mesh)')
    ax.set_ylabel(r'$d_k$ (trace difference)')
    ax.set_title(f'(a) 2D: Richardson order $p$ = {best:.2f}')
    ax.legend(loc='lower right')

    ax = fig.add_subplot(gs[0, 1])
    for (a_, col, mk) in [(6, '#4c7fc4', 'o'), (8, '#2b5c99', 's'),
                          (10, '#123a66', '^')]:
        sel = [r for r in rows if r[3] == a_]
        ax.semilogy([r[1] for r in sel], [r[4] for r in sel], mk + '-',
                    color=col, lw=1.2, ms=4.5, label=f'{a_}$\\times$4$\\times$4')
        ax.semilogy([r[1] for r in sel], [r[5] for r in sel], mk + '--',
                    color=C_COL, lw=1.0, ms=4, mfc='none', alpha=0.75)
    ax.set_xlabel(r'trace modes  $n_t$')
    ax.set_ylabel('interface rel. $L^2$')
    ax.set_title('(b) 3D: trace resolution, not $h$')
    ax.set_ylim(1e-14, 5)
    h1, = ax.plot([], [], '-', color='#333', lw=1.2, label='solid: Galerkin')
    h2, = ax.plot([], [], '--', color=C_COL, lw=1.2, label='dashed: collocation')
    ax.legend(loc='lower left')

    ax = fig.add_subplot(gs[0, 2])
    ratio = np.array([r[2] / r[1] for r in rows])
    gal = np.array([r[4] for r in rows])
    ax.semilogy(ratio, gal, 'o', color=C_GAL, ms=5)
    ax.axvline(1.0, color=C_ACC, lw=1.2, ls='--')
    ax.axvspan(1.0, ratio.min() if ratio.min() > 1 else 1.6, color=C_ACC,
               alpha=0.07)
    ax.set_xlabel(r'$n_t/n_\Gamma$')
    ax.set_ylabel('interface rel. $L^2$')
    ax.set_title('(c) limitation is resolution, not coupling')
    ax.set_ylim(1e-14, 5)
    ax.set_xlim(0.90, 5.90)          # 给右侧标注留出净空
    # 原标注 xytext=(0.72, 1e-5) 从 x<1 起排, 于是横穿 x=1 的退化参考线
    # 并溢出右边界; 引线也长得像一条斜线。整体挪到 x>1 的空白区。
    ax.annotate('degenerate:\ntrace space interpolates\nexactly',
                xy=(1.0, gal.min() * 3.0), xytext=(1.16, 1.5e-9),
                fontsize=FS.MIN_ANNOT, color=C_ACC, ha='left', va='bottom',
                arrowprops=dict(arrowstyle='-', lw=0.8, color=C_ACC,
                                shrinkA=3, shrinkB=2))
    save(fig, 'fig04_consistency')
    return rows


if __name__ == '__main__':
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    t0 = time.time()
    if which in ('all', '1'):
        print('--- Fig 1 ---')
        fig01()
    if which in ('all', '2'):
        print('--- Fig 2 ---')
        fig02()
    if which in ('all', '3'):
        print('--- Fig 3 ---')
        fig03()
    if which in ('all', '4'):
        print('--- Fig 4 ---')
        fig04()
    print(f'done in {time.time()-t0:.1f}s -> {OUT}')
