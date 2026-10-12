# -*- coding: utf-8 -*-
"""
论文复图生成器 (Fig 5-9) —— 投稿级版
====================================
Fig 5  2D 四方法 3x4 复合云图 (|u| / sigma_vm / |Delta sigma_vm|)
Fig 6  3D 四工况 3x4 复合 (45deg 应力视图 / 切片应力 / 切片误差)
Fig 7  应力恢复与补片秩亏
Fig 8  参数类型决定代理架构
Fig 9  强形式 Neumann 零空间机理

规范: Elsevier/CMAME —— 190 mm, 字体 >=7 pt, 线宽 0.25-1.5 pt,
色盲安全色板, constrained layout. 数值管线与 v1 完全一致。
"""
import _paths  # noqa: F401  (adds src/ to the path)
import os
import io
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import figstyle as FS
import coupled_solver as C2
import interface_methods as IM
import stress_recovery2d as R2
import coupled_solver3d as C3
import coupled_solver3d_mortar as M3
import stress_recovery3d as S3
import param_operator as PO

FS.apply()

C_GAL, C_COL, C_ACC = FS.C_GAL, FS.C_COL, FS.C_ACC
C_GRN, C_GRY = FS.C_GRN, FS.C_GRY
# the output directory can be relocated with RESULTS_DIR; the default is
# results/ beside this file
OUT = os.path.join(os.environ.get('RESULTS_DIR', os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))), 'results', 'figs')
os.makedirs(OUT, exist_ok=True)
W = FS.W_DOUBLE


def save(fig, name):
    FS.save(fig, name, OUT, tiff=False)


# ==================================================================== Fig 5
Lx2, Ly2, X02 = C2.Lx, C2.Ly, C2.X0


def _ref_disp2d(ref, x, y):
    nx, ny = ref['nx'], ref['ny']
    hx, hy = Lx2 / nx, Ly2 / ny
    i = min(int(x / hx), nx - 1)
    j = min(int(y / hy), ny - 1)
    xi = 2 * (x - i * hx) / hx - 1
    eta = 2 * (y - j * hy) / hy - 1
    N = R2._q4(xi, eta)[0]
    a = j * (nx + 1) + i
    idx = [a, a + 1, a + 1 + (nx + 1), a + (nx + 1)]
    return (float(N @ np.array([ref['u'][2 * k] for k in idx])),
            float(N @ np.array([ref['u'][2 * k + 1] for k in idx])))


def _2d_all(NX=90, NY=46):
    t0 = time.time()
    ref = R2.solve_reference_np(56, 20)
    vm_r = np.array([R2.vm_of(s) for s in
                     R2.spr_nodal2d(ref['nodes'], ref['elems'], ref['u'])])
    xs = np.linspace(1e-9, Lx2 - 1e-9, NX)
    ys = np.linspace(1e-9, Ly2 - 1e-9, NY)
    Dref = np.zeros((NY, NX))
    Vref = np.zeros((NY, NX))
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            ux, uy = _ref_disp2d(ref, x, y)
            Dref[j, i] = math.hypot(ux, uy)
            Vref[j, i] = R2.interp_nodal_vm2d(ref['nodes'], ref['elems'], vm_r,
                                              ref['nx'], ref['ny'], Lx2, x, y)
    out = []
    for name, solver in [('penalty', IM.solve_penalty),
                         ('pointwise\nLagrange', IM.solve_lagrange_pointwise),
                         ('mortar', IM.solve_mortar),
                         ('condensation', IM.solve_condensation)]:
        sol = solver()
        fn = np.array(sol['fnodes'])
        fe = np.array(sol['felems'])
        uf = np.array(sol['ufem'])
        vm_n = np.array([R2.vm_of(s) for s in R2.spr_nodal2d(fn, fe, uf)])
        D = np.zeros((NY, NX))
        V = np.zeros((NY, NX))
        E = np.zeros((NY, NX))
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                if x <= X02:
                    ux, uy = C2.fem_point_disp(sol, x, y)
                    vm = R2.interp_nodal_vm2d(fn, fe, vm_n, sol['fnx'], sol['fny'],
                                              X02, x, y)
                else:
                    ux, uy = C2.nn_point_disp(sol, x, y)
                    vm = C2.nn_point_vm(sol, x, y)
                D[j, i] = math.hypot(ux, uy)
                V[j, i] = vm
                E[j, i] = abs(vm - Vref[j, i])
        out.append((name, D, V, E))
        print(f'    2D {name.splitlines()[0]:<12} done')
    print(f'  Fig5 data in {time.time()-t0:.1f}s')
    return xs, ys, Dref, Vref, out



def _acc_from_results():
    """The three unseen-field accuracies, parsed from results/func_operator.txt.

    Line format: "未见场相对 L2:  线性 10.1123%   无trunk MLP 4.0041%   DeepONet 1.4288%"
    Falls back to an explicit error so a missing file is loud rather than silent.
    """
    import re as _re
    data_dir = os.environ.get('DATA_DIR', os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))
    path = os.path.join(data_dir, 'func_operator.txt')
    with io.open(path, encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            if 'DeepONet' in line and 'MLP' in line and '%' in line:
                nums = _re.findall(r'(\d+\.\d+)%', line)
                if len(nums) >= 3:
                    return [float(x) for x in nums[:3]]
    raise RuntimeError('unseen-field accuracies not found in func_operator.txt')


def fig05(cache=None):
    xs, ys, Dref, Vref, out = cache if cache else _2d_all()
    fig = plt.figure(figsize=(W, 4.30), layout='constrained')
    gs = fig.add_gridspec(3, 4, hspace=0.10, wspace=0.045)

    row_lab = [r'$|u|$', r'$\sigma_{\rm vm}$  (SPR)',
               r'$|\Delta\sigma_{\rm vm}|$']
    cb_lab = ['displacement  $|u|$', r'von Mises  $\sigma_{\rm vm}$',
              r'interface mismatch  $|\Delta\sigma_{\rm vm}|$']
    cms = [FS.CM_SEQ, FS.CM_SEQ, FS.CM_ERR]

    vmin = [0, 0, 0]
    vmax = [max(Dref.max(), max(o[1].max() for o in out)),
            max(Vref.max(), max(o[2].max() for o in out)),
            max(o[3].max() for o in out)]

    for r in range(3):
        if r == 0:
            data = [(o[1], Dref) for o in out]
        elif r == 1:
            data = [(o[2], Vref) for o in out]
        else:
            data = [(o[3], None) for o in out]
        for c in range(4):
            ax = fig.add_subplot(gs[r, c])
            Z = data[c][0]
            im = ax.contourf(xs, ys, Z, 24, cmap=cms[r],
                             vmin=vmin[r], vmax=vmax[r])
            ax.axvline(X02, color='w', ls='--', lw=0.9)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(out[c][0])
            if c == 0:
                ax.set_ylabel(row_lab[r])
            if c == 3:
                tk = MaxNLocator(nbins=5).tick_values(vmin[r], vmax[r])
                cb = FS.cbar(fig, im, ax, label=cb_lab[r], shrink=0.92,
                             size=8.0, ticks=tk)
                if r == 0:
                    # 0.0096 量级写 4 位小数过于拥挤, 改用 10^-3 单位
                    tk0 = np.linspace(0.0, 10.0, 5) * 1e-3
                    cb.set_ticks(tk0)
                    cb.ax.set_yticklabels([f'{t * 1e3:g}' for t in tk0],
                                          fontsize=8.5)
                    cb.set_label(r'displacement  $|u|$  ($\times10^{-3}$)',
                                 fontsize=8.5)

    fig.supxlabel('dashed line: interface $\\Gamma$     |     '
                  'left: FE subdomain     right: network subdomain',
                  fontsize=8.5)
    save(fig, 'fig05_2d_fields')


# ==================================================================== Fig 6
Lx3, Ly3, Lz3, X03 = C3.Lx, C3.Ly, C3.Lz, C3.X0
CASES = ['tension', 'bending', 'torsion', 'combined']


def _face_polys(field, cmap, norm, n=20):
    polys, cols = [], []
    u = np.linspace(0, 1, n + 1)

    def add(pts_fn):
        for i in range(n):
            for j in range(n):
                p00 = pts_fn(u[i], u[j])
                p10 = pts_fn(u[i + 1], u[j])
                p11 = pts_fn(u[i + 1], u[j + 1])
                p01 = pts_fn(u[i], u[j + 1])
                polys.append([p00, p10, p11, p01])
                pc = pts_fn((u[i] + u[i + 1]) / 2, (u[j] + u[j + 1]) / 2)
                cols.append(cmap(norm(field(*pc))))
    add(lambda a, b: (Lx3, Ly3 * a, Lz3 * b))
    add(lambda a, b: (Lx3 * a, Ly3, Lz3 * b))
    add(lambda a, b: (Lx3 * a, Ly3 * b, Lz3))
    return polys, cols


def _surface_range(fn, n=18):
    vals = []
    for a in np.linspace(0.02, 0.98, n):
        for b in np.linspace(0.02, 0.98, n):
            vals += [fn(Lx3, Ly3 * a, Lz3 * b), fn(Lx3 * a, Ly3, Lz3 * b),
                     fn(Lx3 * a, Ly3 * b, Lz3)]
    return min(vals), max(vals)


def _3d_all(NS=76, NZ=38):
    t0 = time.time()
    out = {}
    for case in CASES:
        cs = M3.build_mortar(case)
        rs = C3.solve_reference_3d(case)
        spr_c = S3.spr_nodal(cs['nodes'], cs['elems'], cs['ufem'], degree=2)
        spr_r = S3.spr_nodal(rs['nodes'], rs['elems'], rs['u'], degree=2)
        nv_c = S3.naive_nodal(cs['nodes'], cs['elems'], cs['ufem'])
        fc = S3.build_mixed_vm(cs, spr_c)
        fnv = S3.build_mixed_vm(cs, nv_c)
        fr = S3.build_mixed_vm(rs, spr_r)
        e_spr = M3.l2_vm_field(fc, fr)
        e_nv = M3.l2_vm_field(fnv, fr)
        xs = np.linspace(1e-9, Lx3 - 1e-9, NS)
        zs = np.linspace(1e-9, Lz3 - 1e-9, NZ)
        Vc = np.zeros((NZ, NS))
        Vr = np.zeros((NZ, NS))
        Er = np.zeros((NZ, NS))
        ybar = Ly3 / 2.0
        for j, z in enumerate(zs):
            for i, x in enumerate(xs):
                a = fc(x, ybar, z)
                b = fr(x, ybar, z)
                Vc[j, i] = a
                Vr[j, i] = b
                Er[j, i] = abs(a - b)
        out[case] = dict(fc=fc, fr=fr, xs=xs, zs=zs, Vc=Vc, Vr=Vr, Er=Er,
                         e_spr=e_spr, e_nv=e_nv,
                         rng_c=_surface_range(fc), rng_r=_surface_range(fr),
                         disp=C3.l2_disp_error(cs, rs))
        print(f'    3D {case:<9} spr {e_spr:.4%}  naive {e_nv:.4%}  '
              f'disp {out[case]["disp"]:.4%}')
    print(f'  Fig6/7a data in {time.time()-t0:.1f}s')
    return out


def fig06(cache):
    fig = plt.figure(figsize=(W, 5.75), layout='constrained')
    gs = fig.add_gridspec(3, 4, height_ratios=[1.34, 1.0, 1.0],
                          hspace=0.10, wspace=0.04)

    row_lab = [r'$\sigma_{\rm vm}$  (45$^\circ$ surface)',
               r'$\sigma_{\rm vm}$  (slice $y=L_y/2$)',
               r'$|\Delta\sigma_{\rm vm}|$  (slice)']
    vmax1 = max(max(cache[c]['rng_c'][1], cache[c]['rng_r'][1]) for c in CASES)
    vmin1 = min(min(cache[c]['rng_c'][0], cache[c]['rng_r'][0]) for c in CASES)
    vmax3 = max(cache[c]['Er'].max() for c in CASES)

    row_ax = {0: [], 1: [], 2: []}
    cb_rows = {}
    for r in range(3):
        for c, case in enumerate(CASES):
            d = cache[case]
            ax = fig.add_subplot(gs[r, c],
                                 projection='3d' if r == 0 else None)
            row_ax[r].append(ax)
            if r == 0:
                cmap = plt.get_cmap(FS.CM_SEQ)
                norm = plt.Normalize(vmin1, vmax1)
                polys, cols = _face_polys(d['fc'], cmap, norm)
                ax.add_collection3d(Poly3DCollection(polys, facecolors=cols,
                                                     edgecolors='none'))
                ax.set_xlim(0, Lx3)
                ax.set_ylim(0, Ly3)
                ax.set_zlim(0, Lz3)
                ax.set_box_aspect((Lx3, Ly3, Lz3))
                ax.view_init(elev=24, azim=38)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                ax.grid(False)
                for s in (ax.xaxis, ax.yaxis, ax.zaxis):
                    s.pane.set_alpha(0.0)
                ax.set_title(case)
            else:
                Z = d['Vc'] if r == 1 else d['Er']
                if r == 1:
                    im = ax.contourf(d['xs'], d['zs'], Z, 24, cmap=FS.CM_SEQ,
                                     vmin=vmin1, vmax=vmax1)
                else:
                    im = ax.contourf(d['xs'], d['zs'], Z, 24, cmap=FS.CM_ERR,
                                     vmin=0, vmax=vmax3)
                ax.axvline(X03, color='w', ls='--', lw=0.9)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                for s in ax.spines.values():
                    s.set_visible(False)
                ax.text(0.035, 0.945, f'max {Z.max():.2f}',
                        transform=ax.transAxes, ha='left', va='top',
                        fontsize=FS.MIN_ANNOT, color='w')
            if c == 0:
                if r == 0:
                    ax.text2D(-0.10, 0.5, row_lab[r], transform=ax.transAxes,
                              rotation=90, ha='center', va='center')
                else:
                    ax.set_ylabel(row_lab[r])
            if c == 3 and r > 0:
                lo, hi = (vmin1, vmax1) if r == 1 else (0.0, vmax3)
                cb_rows[r] = FS.cbar(
                    fig, im, ax, shrink=0.92, size=8.0,
                    ticks=MaxNLocator(nbins=5).tick_values(lo, hi),
                    label=(r'von Mises  $\sigma_{\rm vm}$' if r == 1
                           else r'interface mismatch  $|\Delta\sigma_{\rm vm}|$'))

    # 上排 3D 曲面单独一条 colorbar (绑定到已存在的 3D 轴, 不再新建 2D 轴)
    sm = plt.cm.ScalarMappable(cmap=FS.CM_SEQ,
                               norm=plt.Normalize(vmin1, vmax1))
    cb0 = fig.colorbar(sm, ax=row_ax[0], pad=0.01, fraction=0.030,
                       shrink=0.90, location='right',
                       ticks=MaxNLocator(nbins=6).tick_values(vmin1, vmax1))
    cb0.ax.tick_params(labelsize=8.5)
    cb0.set_label(r'von Mises  $\sigma_{\rm vm}$', fontsize=8.5)
    save(fig, 'fig06_3d_multiload')


# ==================================================================== Fig 7
def _patch_svals(X, Xn, degree=2):
    dx = X[:, 0] - Xn[0]
    dy = X[:, 1] - Xn[1]
    A = np.column_stack([np.ones_like(dx), dx, dy, dx * dx, dx * dy, dy * dy])
    return np.linalg.svd(A, compute_uv=False)


def _patch_svals3(X, Xn, degree=2):
    return np.linalg.svd(S3._basis(X, Xn, degree), compute_uv=False)


def _gauss_samples2d(nodes, elems, node):
    """节点相关单元的 2D 高斯点坐标 (采样点集)。"""
    if not hasattr(_gauss_samples2d, 'acc'):
        acc = [[] for _ in range(len(nodes))]
        for e, xy in zip(elems, [np.array(nodes[el]) for el in elems]):
            for (dNdx, dNdy, dJ, px, py, N) in R2._geo(xy):
                for k in e:
                    acc[k].append((px, py))
        _gauss_samples2d.acc = acc
    return np.array(_gauss_samples2d.acc[node])


def fig07(cache):
    fig = plt.figure(figsize=(W, 2.55), layout='constrained')
    # (b) 需要更宽: 五条曲线的图例必须落在左下空白区而不压到下降段
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.32, 0.83])

    # (a) naive vs SPR nodal stress error
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(CASES))
    nv = [100 * cache[c]['e_nv'] for c in CASES]
    sp = [100 * cache[c]['e_spr'] for c in CASES]
    ax.bar(x - 0.2, nv, 0.4, color=C_COL, label='naive nodal averaging')
    ax.bar(x + 0.2, sp, 0.4, color=C_GAL, label='SPR recovery')
    for i in range(len(CASES)):
        for xo, v in ((-0.2, nv[i]), (0.2, sp[i])):
            ax.annotate(f'{v:.2f}', xy=(i + xo, v), xytext=(0, 3),
                        textcoords='offset points', rotation=90, ha='center',
                        va='bottom', fontsize=FS.MIN_ANNOT)
    ax.set_xticks(x)
    # 四个工况名横排会首尾相接(面板宽度不够), 旋转 30° 并右对齐
    ax.set_xticklabels(CASES, rotation=30, ha='right', rotation_mode='anchor')
    ax.set_ylim(0, 36)
    ax.set_ylabel(r'von Mises rel. $L^2$  (\%)')
    ax.set_title('(a) stress recovery is mandatory')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.22, axis='y')

    # (b) patch design-matrix singular values
    ax = fig.add_subplot(gs[0, 1])
    n2, e2 = R2.build_mesh(6, 5, 0.0, 1.0, 0.0, 1.0)
    cases2 = [('interior', 2 * 7 + 3), ('edge', 3 * 7 + 0), ('corner', 0)]
    n3, e3, nnx3, nny3, nnz3 = C3.build_mesh_3d(4, 3, 3, 0.0, 1.0, 0.0, 1.0,
                                                0.0, 1.0)
    nd_int3 = 1 * nny3 * nnx3 + 1 * nnx3 + 2
    sty = [('-', 'o'), ('--', 's'), (':', '^'), ('-.', 'v')]
    for k, (nm, nd) in enumerate(cases2):
        X = _gauss_samples2d(n2, e2, nd)
        sv = _patch_svals(X, n2[nd])
        sv = sv / sv[0]
        ax.semilogy(np.arange(1, len(sv) + 1), np.maximum(sv, 1e-18),
                    sty[k][0], marker=sty[k][1], ms=3.5, color=C_COL, lw=1.1,
                    label=f'2D {nm} ({len(X)} pts)')
    print('    2D patch sample counts: ' +
          ', '.join(f'{nm} {len(_gauss_samples2d(n2, e2, nd))}' for nm, nd in cases2))
    for k, (nm, nd) in enumerate([('interior', nd_int3), ('corner', 0)]):
        X = np.array([p for gi, smp in S3.elem_gauss_stress(n3, e3,
                                                            np.zeros(3 * len(n3)))
                      for (p, _) in smp if nd in e3[gi]])
        if len(X) == 0:
            continue
        print(f'    3D patch sample count: {nm} {len(X)}')
        sv = _patch_svals3(X, n3[nd])
        sv = sv / sv[0]
        ax.semilogy(np.arange(1, len(sv) + 1), np.maximum(sv, 1e-18),
                    sty[k + 2][0], marker=sty[k + 2][1], ms=3.5, color=C_GAL,
                    lw=1.1, label=f'3D {nm} ({len(X)} pts)')
    ax.axhline(1e-8, color='#333', lw=1.0, ls='--')
    ax.annotate('rank threshold', xy=(4.6, 1e-8), xytext=(4.4, 3e-7),
                fontsize=FS.MIN_ANNOT, color='#333',
                arrowprops=dict(arrowstyle='->', lw=0.8, color='#333'))
    ax.set_xlabel('singular value index')
    ax.set_ylabel(r'$\sigma_i/\sigma_1$')
    ax.set_title('(b) patch matrix is rank deficient')
    # 不透明白底: 两条下降曲线横穿左下区域, 无边框图例会被曲线划断
    ax.legend(loc='lower left', fontsize=FS.MIN_ANNOT - 0.4, frameon=True,
              facecolor='white', framealpha=1.0, edgecolor='none')

    # (c) constant stress field must be recovered exactly
    ax = fig.add_subplot(gs[0, 2])
    A1 = 0.012
    nodes, elems = R2.build_mesh(6, 5, 0.0, 1.0, 0.0, 1.0)
    u = np.zeros(2 * len(nodes))
    for k, (xx, yy) in enumerate(nodes):
        u[2 * k] = A1 * xx
        u[2 * k + 1] = -R2.NU * A1 * yy
    sig_c = R2.D2 @ np.array([A1, -R2.NU * A1, 0.0])
    guarded = R2.spr_nodal2d(nodes, elems, u)
    e_g = np.abs(guarded - sig_c).max() / np.abs(sig_c).max()
    acc = [[] for _ in range(len(nodes))]
    geo = [R2._geo(np.array(nodes[el])) for el in elems]
    for gi, e in enumerate(elems):
        for (dNdx, dNdy, dJ, px, py, N) in geo[gi]:
            ue = np.array([u[2 * el] for el in e])
            ve = np.array([u[2 * el + 1] for el in e])
            exx = dNdx @ ue
            eyy = dNdy @ ve
            exy = 0.5 * (dNdy @ ue + dNdx @ ve)
            s6 = R2.D2 @ np.array([exx, eyy, 2 * exy])
            for k in e:
                acc[k].append(((px, py), s6))
    ung = np.zeros((len(nodes), 3))
    for k in range(len(nodes)):
        S = acc[k]
        X = np.array([p for (p, _) in S])
        sv = _patch_svals(X, nodes[k])
        A = np.column_stack([np.ones(len(S)), X[:, 0] - nodes[k][0],
                             X[:, 1] - nodes[k][1],
                             (X[:, 0] - nodes[k][0]) ** 2,
                             (X[:, 0] - nodes[k][0]) * (X[:, 1] - nodes[k][1]),
                             (X[:, 1] - nodes[k][1]) ** 2]) if len(S) >= 6 else \
            np.column_stack([np.ones(len(S)), X[:, 0] - nodes[k][0],
                             X[:, 1] - nodes[k][1]])
        for cp in range(3):
            coef = np.linalg.lstsq(A, np.array([s[cp] for (_, s) in S]),
                                   rcond=None)[0]
            ung[k, cp] = coef[0]
    e_u = np.abs(ung - sig_c).max() / np.abs(sig_c).max()
    ax.bar([0, 1], [max(e_u, 1e-18), max(e_g, 1e-18)], 0.5,
           color=[C_COL, C_GAL])
    ax.set_yscale('log')
    ax.set_ylim(1e-18, 1e-3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['unguarded', 'rank-guarded'])
    ax.set_ylabel('const.-stress recovery error')
    ax.set_title('(c) silent bias at boundary nodes')
    ax.annotate(f'{e_u:.1e}', xy=(0, max(e_u, 1e-18)), xytext=(0, 3),
                textcoords='offset points', ha='center', fontsize=FS.MIN_ANNOT)
    ax.annotate(f'{e_g:.1e}', xy=(1, max(e_g, 1e-18)), xytext=(0, 3),
                textcoords='offset points', ha='center', fontsize=FS.MIN_ANNOT)
    ax.grid(alpha=0.22, axis='y', which='both')
    print(f'  Fig7c: unguarded {e_u:.3e} -> guarded {e_g:.3e}')
    save(fig, 'fig07_recovery')


# ==================================================================== Fig 8
def fig08():
    fig = plt.figure(figsize=(W, 4.55), layout='constrained')
    gs = fig.add_gridspec(2, 2, hspace=0.14, wspace=0.16)

    # (a) scalar parameter: operator parity
    ax = fig.add_subplot(gs[0, 0])
    Tgt, nu = [], []
    for v in PO.NUS_TRAIN:
        M, q = PO.probe(v)
        Tgt.append(np.concatenate([M.reshape(-1), q]))
        nu.append(v)
    poly = PO.fit_poly(PO.NUS_TRAIN, Tgt, deg=4)
    xs_, ys_ = [], []
    for v in PO.NUS_TEST:
        M, q = PO.probe(v)
        xs_.append(np.concatenate([M.reshape(-1), q]))
        ys_.append(PO.eval_poly(poly, v))
    xs_ = np.concatenate(xs_)
    ys_ = np.concatenate(ys_)
    ax.plot(xs_, ys_, '.', ms=3, color=C_GAL)
    lo, hi = min(xs_.min(), ys_.min()), max(xs_.max(), ys_.max())
    ax.plot([lo, hi], [lo, hi], 'k-', lw=0.9)
    ax.set_xlabel('exact probed operator')
    ax.set_ylabel('polynomial surrogate')
    ax.set_title(r'(a) scalar $\nu$: polynomial is optimal')
    ax.text(0.05, 0.94, r'op. error $10^{-6}$' '\n' r'trace error $10^{-8}$',
            transform=ax.transAxes, fontsize=FS.MIN_ANNOT, va='top')

    # (b) learning curves
    ax = fig.add_subplot(gs[0, 1])
    N = np.array([10, 24, 48, 96, 192])
    lin = np.array([79.355, 12.825, 13.200, 11.485, 10.112])
    mlp = np.array([69.788, 41.978, 26.744, 15.198, 4.004])
    dn = np.array([20.248, 14.101, 10.324, 6.750, 2.077])
    ax.semilogx(N, lin, 'o--', color=C_GRY, lw=1.2, ms=4, label='linear / ridge')
    ax.semilogx(N, mlp, 's-.', color=C_ACC, lw=1.2, ms=4, label='MLP (no trunk)')
    ax.semilogx(N, dn, '^-', color=C_GAL, lw=1.4, ms=4.5, label='DeepONet')
    ax.set_xlabel('training fields  $N$')
    ax.set_ylabel('unseen-field rel. $L^2$  (%)')
    ax.set_title('(b) plateau vs monotone descent')
    ax.set_ylim(0, 92)
    ax.legend(loc='upper right')
    # 原文案 (13, 55) 单行落在 x=13..55、y=55, 而 linear/ridge 灰虚线恰好
    # 从 (10,79) 俯冲到 (24,13), 整条穿过文字; 且两者同为灰色。
    # 改为两行、右移到 x>=30 的空白区, 并加白色底衬。
    ax.annotate('plateau =\nhypothesis-class limit', xy=(96, 15.198),
                xytext=(45, 58), fontsize=FS.MIN_ANNOT, color='#333',
                va='top', ha='left',
                bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.6),
                arrowprops=dict(arrowstyle='->', lw=0.8, color='#333'))

    # (c) unseen-field error
    ax = fig.add_subplot(gs[1, 0])
    lbl = ['linear', 'MLP\n(no trunk)', 'DeepONet']
    # Read rather than retype.  The values below are the single-protocol figures
    # published in func_operator.txt; hard-coding them meant a change in the
    # surrogate experiment would silently leave this panel stale (Round 4 found
    # exactly that failure mode in the learning-curve arrays).
    val = _acc_from_results()
    ax.bar(lbl, val, 0.55, color=[C_GRY, C_ACC, C_GAL])
    for i, v in enumerate(val):
        ax.annotate(f'{v:.2f}%', xy=(i, v), xytext=(0, 3),
                    textcoords='offset points', ha='center',
                    fontsize=FS.MIN_ANNOT)
    ax.set_yscale('log')
    ax.set_ylim(1, 22)
    ax.set_ylabel('unseen-field rel. $L^2$  (%)')
    ax.set_title('(c) function parameter: branch is necessary')
    ax.grid(alpha=0.22, axis='y', which='both')

    # (d) accuracy vs cost
    ax = fig.add_subplot(gs[1, 1])
    pts = [('this work', 0.402, 0.3, C_GAL, '*', 230, (8, -2)),
           ('CPINN', 5.391, 873.9, C_GRN, 'o', 55, (-9, 5)),
           ('XPINN', 12.221, 1007.6, C_COL, 's', 55, (9, -6)),
           ('single PINN', 24.814, 347.1, C_GRY, '^', 55, (-9, 5))]
    for nm, e, t, col, mk, ms, off in pts:
        ax.scatter([t], [e], s=ms, color=col, marker=mk, zorder=3, label=nm)
        ax.annotate(nm, (t, e), textcoords='offset points', xytext=off,
                    fontsize=FS.MIN_ANNOT, ha='right' if off[0] < 0 else 'left',
                    va='center')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.1, 2e4)
    ax.set_ylim(0.25, 60)
    ax.set_xlabel('cost  (s)')
    ax.set_ylabel('interface rel. $L^2$  (%)')
    ax.set_title('(d) an order more accurate, three orders cheaper')
    ax.grid(alpha=0.22, which='both')
    save(fig, 'fig08_paramtype')


# ==================================================================== Fig 9
def fig09():
    fig = plt.figure(figsize=(W, 2.62), layout='constrained')
    # (b) 内容最密(三曲线+三行图例+两处标注), 给它更多宽度; (c) 需容纳四个柱标签不粘连
    gs = fig.add_gridspec(1, 3, width_ratios=[0.85, 1.30, 0.95])

    # (a) the uniform-shear null mode
    ax = fig.add_subplot(gs[0, 0])
    xx = np.linspace(0, 1, 200)
    yy = np.linspace(0, 1, 120)
    Xg, Yg = np.meshgrid(xx * C2.Lx, yy * C2.Ly)
    c = 7e-3
    Uy = c * Xg
    cs = ax.contourf(Xg, Yg, 1e3 * Uy, 20, cmap=FS.CM_SEQ)
    ax.contour(Xg, Yg, 1e3 * Uy, 8, colors='w', linewidths=0.35)
    ax.axvline(C2.X0, color=C_ACC, ls='--', lw=1.2)
    ax.plot([C2.Lx], [C2.Ly / 2], 'rx', ms=7, mew=1.5)
    ax.annotate(r'$\sigma_{xy}=0$ required', xy=(C2.Lx, C2.Ly / 2),
                xytext=(C2.Lx * 0.30, C2.Ly * 0.80), fontsize=FS.MIN_ANNOT,
                color='#c1121f', ha='left', va='center',
                arrowprops=dict(arrowstyle='->', lw=0.9, color='#c1121f'))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(r'(a) null mode $u=(0,cx)$')
    ax.grid(False)
    FS.cbar(fig, cs, ax, label=r'$u_y\times10^{3}$', size=8.0,
            ticks=np.linspace(0.0, 1e3 * c * C2.Lx, 5))

    # (b) error vs training, with and without Neumann conditions
    ax = fig.add_subplot(gs[0, 1])
    st = np.array([2000, 4000, 6000, 8000])
    noN = np.array([48.44, 51.70, 52.99, 53.48])
    withN = np.array([22.12, 21.45, 17.24, 15.40])
    cpN = np.array([22.69, 21.92, 21.71, 21.75])
    ax.plot(st, noN, 'o--', color=C_COL, lw=1.3, ms=4.5,
            label='Dirichlet only')
    ax.plot(st, withN, '^-', color=C_GAL, lw=1.5, ms=5,
            label='+ traction-free')
    ax.plot(st, cpN, 's:', color=C_GRN, lw=1.1, ms=4, label='+ cond. (CPINN)')
    ax.annotate('error grows\nwith training', xy=(7300, 52.6),
                xytext=(0.58, 0.72), textcoords='axes fraction',
                va='top', ha='left',
                fontsize=FS.MIN_ANNOT, color=C_COL,
                arrowprops=dict(arrowstyle='->', lw=0.8, color=C_COL))
    # x3.5 = 末步 Dirichlet-only 53.48% 对 +traction-free 15.40% 的比值。
    # 原写法用一条长引线指向蓝色末点, 会被误读成与 CPINN 绿线比较。
    ax.annotate('', xy=(8080, withN[-1]), xytext=(8080, noN[-1]),
                arrowprops=dict(arrowstyle='<->', lw=1.0, color='#333',
                                shrinkA=5, shrinkB=5))
    ax.annotate(r'$\times3.5$', xy=(8080, 0.5 * (noN[-1] + withN[-1])),
                xytext=(-7, 0), textcoords='offset points',
                ha='right', va='center', fontsize=FS.MIN_ANNOT,
                color='#333')
    ax.set_xlabel('Adam steps')
    ax.set_ylabel('field rel. $L^2$  (%)')
    ax.set_title('(b) the missing condition is the cause')
    ax.set_ylim(10, 62)
    ax.set_xlim(1700, 9300)
    ax.legend(loc='center left', fontsize=FS.MIN_ANNOT)

    # (c) final comparison
    # 纵轴是「界面 rel. L2」, 必须用各方法的**界面**误差 (Table 6 界面列 /
    # code/_run_pinn.log)。旧版误用了场误差 26.565/17.124/7.488, 却把本方法的
    # 界面值 0.402 放进同一张图 —— 两个口径混用, 与 Table 6「CPINN 界面 5.39%」冲突。
    ax = fig.add_subplot(gs[0, 2])
    lbl = ['single\nPINN', 'XPINN', 'CPINN', 'this\nwork']
    val = [24.814, 12.221, 5.391, 0.402]
    col = [C_GRY, C_COL, C_GRN, C_GAL]
    ax.bar(lbl, val, 0.6, color=col)
    for i, v in enumerate(val):
        ax.annotate(f'{v:.3f}%' if v > 1 else f'{v:.3f}%', xy=(i, v),
                    xytext=(0, 3), textcoords='offset points', ha='center',
                    fontsize=FS.MIN_ANNOT)
    ax.set_yscale('log')
    ax.set_ylim(0.2, 90)
    ax.set_ylabel('interface rel. $L^2$  (%)')
    ax.set_title('(c) identical budget, same problem')
    ax.grid(alpha=0.22, axis='y', which='both')
    save(fig, 'fig09_neumann')


if __name__ == '__main__':
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    t0 = time.time()
    c2 = c3 = None
    if which in ('all', '5', '6', '7'):
        if which in ('all', '6', '7'):
            print('--- 预计算 3D 四工况 ---')
            c3 = _3d_all()
        if which in ('all', '5'):
            print('--- 预计算 2D 四方法 ---')
            c2 = _2d_all()
    if which in ('all', '5'):
        print('--- Fig 5 ---')
        fig05(c2)
    if which in ('all', '6'):
        print('--- Fig 6 ---')
        fig06(c3)
    if which in ('all', '7'):
        print('--- Fig 7 ---')
        fig07(c3)
    if which in ('all', '8'):
        print('--- Fig 8 ---')
        fig08()
    if which in ('all', '9'):
        print('--- Fig 9 ---')
        fig09()
    print(f'done in {time.time()-t0:.1f}s -> {OUT}')
