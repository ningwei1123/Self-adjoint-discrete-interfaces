# -*- coding: utf-8 -*-
"""
DeepONet 参数化算子 A(ν): 把加速闭环
======================================
思路:
  1. 在若干 ν 上"探测"界面 DtN 算子 (修复②③后的正确形式):
        界面方程  (S_L(ν) + S_R(ν)) g = −q(ν)
     记 M(ν)=S_L(ν)+S_R(ν) (2nt×2nt)、q(ν) (2nt)。每次探测需 2*(2nt) + 1 次子域 FEM 求解。
  2. 训练**参数化算子代理** ν ↦ (M(ν), q(ν))  (多项式基线 + MLP 对照)。
  3. 对新 ν*: 代理一次前向即得 M、q, 直接解 2nt×2nt 小系统 -> **零子域求解**完成耦合。
  4. 与"逐个 ν 重新探测"对比: 精度 + 速度。
"""
import os, math, time
import numpy as np
import stress_recovery2d as R
import interface_methods as IM

E = 2100.0
Lx, Ly, X0 = 2.0, 1.0, 1.0
NX, NY, NT = 12, 12, 6
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
NUS_TRAIN = [0.20, 0.24, 0.28, 0.32, 0.36, 0.40]           # 训练 ν
NUS_TEST = [0.22, 0.26, 0.30, 0.34, 0.38]                   # 未见的插值 ν


def D2_of(nu):
    G = E / (2 * (1 + nu))
    D00 = E / (1 - nu * nu); D01 = nu * D00
    return np.array([[D00, D01, 0.0], [D01, D00, 0.0], [0.0, 0.0, G]])


def assemble(nodes, elems, D2):
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


def trace_basis(y, nt=NT):
    e = 2.0 * y / Ly - 1.0
    P = [1.0, e]
    for j in range(2, nt):
        P.append(((2 * j - 1) * e * P[j - 1] - (j - 1) * P[j - 2]) / j)
    return np.array(P[:nt], dtype=float)


def project(v, ys):
    P = np.array([trace_basis(y) for y in ys])
    return np.linalg.lstsq(P, v, rcond=None)[0]


def _trace_h(nt=NT):
    yg, wg = np.polynomial.legendre.leggauss(60)
    yy = 0.5 * (yg + 1.0) * Ly; ww = wg * (Ly / 2.0)
    P = np.array([trace_basis(y, nt) for y in yy])
    return np.array([float(np.sum(ww * P[:, i] * P[:, i])) for i in range(nt)])


TH = _trace_h()


def transfer_galerkin(r_node, iface, nodes, comp, nt=NT):
    ys = np.array([nodes[k, 1] for k in iface])
    Pn = np.array([trace_basis(y, nt) for y in ys])
    rv = np.array([r_node[2 * k + comp] for k in iface])
    return (Pn.T @ rv) / TH


def build(x0, x1):
    nodes, elems = R.build_mesh(NX, NY, x0, x1, 0.0, Ly)
    iface = sorted([k for k in range(len(nodes)) if abs(nodes[k, 0] - X0) < 1e-9], key=lambda k: nodes[k, 1])
    return nodes, elems, iface


def solve_dirichlet(nodes, elems, K, iface, gfun, far):
    ndof = 2 * len(nodes); fix = {}
    for k in iface:
        fix[2 * k], fix[2 * k + 1] = gfun(nodes[k, 1])
    if far is None:
        for k in [k for k in range(len(nodes)) if abs(nodes[k, 0]) < 1e-9]:
            fix[2 * k] = 0.0; fix[2 * k + 1] = 0.0
    else:
        xv, uxf = far
        for k in [k for k in range(len(nodes)) if abs(nodes[k, 0] - xv) < 1e-9]:
            fix[2 * k] = uxf(nodes[k, 1])
    fds = list(fix); free = [d for d in range(ndof) if d not in set(fds)]
    u = np.zeros(ndof)
    if free:
        u[free] = np.linalg.solve(K[np.ix_(free, free)], -K[np.ix_(free, fds)] @ np.array([fix[d] for d in fds]))
    for d, v in fix.items(): u[d] = v
    return u


def pointwise_lam(nodes, elems, u, iface, at_right, D2):
    m = len(iface); lam = np.zeros((m, 2))
    for j in range(NY):
        col = NX - 1 if at_right else 0
        e = elems[j * NX + col]; xy = nodes[e]
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
                lam[iface.index(node)] = (sgn * sig[0], sgn * sig[2])
    return lam


def probe(nu, count=False):
    """探测 M(ν)=S_L(ν)+S_R(ν) (2NT,2NT) 与 q(ν) (2NT,)。"""
    D2 = D2_of(nu)
    S = {}
    for side in ('L', 'R'):
        x0, x1 = (0.0, X0) if side == 'L' else (X0, Lx)
        nodes, elems, iface = build(x0, x1)
        K = assemble(nodes, elems, D2)
        ys = np.array([nodes[k, 1] for k in iface]); at_right = (side == 'L')
        Sside = np.zeros((2 * NT, 2 * NT))
        for c in range(2):
            for j in range(NT):
                k = c * NT + j
                gf = (lambda y, c=c, j=j: (trace_basis(y)[j], 0.0) if c == 0 else (0.0, trace_basis(y)[j]))
                far = None if side == 'L' else (Lx, lambda y: 0.0)
                u = solve_dirichlet(nodes, elems, K, iface, gf, far)
                r = K @ u
                Sside[:NT, k] = transfer_galerkin(r, iface, nodes, 0)
                Sside[NT:, k] = transfer_galerkin(r, iface, nodes, 1)
        S[side] = Sside
    # q: 右子域远端点部载荷
    nodes, elems, iface = build(X0, Lx)
    K = assemble(nodes, elems, D2)
    ys = np.array([nodes[k, 1] for k in iface])
    u = solve_dirichlet(nodes, elems, K, iface, lambda y: (0.0, 0.0), (Lx, R.right_disp))
    r = K @ u
    q = np.concatenate([transfer_galerkin(r, iface, nodes, 0),
                        transfer_galerkin(r, iface, nodes, 1)])
    return S['L'] + S['R'], q


# ---------- 参数化算子代理 (多项式) ----------
def fit_poly(nus, targets, deg):
    """targets: list of arrays; 返回 系数 (deg+1, ...)。"""
    A = np.array([[nu ** p for p in range(deg + 1)] for nu in nus])   # (N, deg+1)
    T = np.array(targets).reshape(len(nus), -1)                        # (N, D)
    coef, *_ = np.linalg.lstsq(A, T, rcond=None)                       # (deg+1, D)
    return coef


def eval_poly(coef, nu):
    return sum(coef[p] * nu ** p for p in range(coef.shape[0]))


# ---------- 参数化算子代理 (numpy MLP) ----------
def fit_mlp(nus, targets, hid=64, epochs=6000, lr=2e-3, seed=0):
    rng = np.random.default_rng(seed)
    X = (np.array(nus).reshape(-1, 1) - 0.3) / 0.1
    T = np.array(targets).reshape(len(nus), -1)
    mx, sx = T.mean(0), T.std(0) + 1e-12
    Tn = (T - mx) / sx
    W1 = rng.normal(0, 1 / math.sqrt(1), (1, hid)); b1 = np.zeros(hid)
    W2 = rng.normal(0, 1 / math.sqrt(hid), (hid, hid)); b2 = np.zeros(hid)
    W3 = rng.normal(0, 1 / math.sqrt(hid), (hid, T.shape[1])); b3 = np.zeros(T.shape[1])
    params = [W1, b1, W2, b2, W3, b3]
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
    b1a, b2a, eps = 0.9, 0.999, 1e-8
    for t in range(1, epochs + 1):
        h1 = np.tanh(X @ W1 + b1); h2 = np.tanh(h1 @ W2 + b2); out = h2 @ W3 + b3
        e = (out - Tn) * 2.0 / Tn.size
        dW3 = h2.T @ e; db3 = e.sum(0)
        d = (e @ W3.T) * (1 - h2 ** 2); dW2 = h1.T @ d; db2 = d.sum(0)
        d = (d @ W2.T) * (1 - h1 ** 2); dW1 = X.T @ d; db1 = d.sum(0)
        grads = [dW1, db1, dW2, db2, dW3, db3]
        lrt = lr * (0.95 ** (t / 1000))
        for i in range(len(params)):
            m[i] = b1a * m[i] + (1 - b1a) * grads[i]
            v[i] = b2a * v[i] + (1 - b2a) * grads[i] ** 2
            mh = m[i] / (1 - b1a ** t); vh = v[i] / (1 - b2a ** t)
            params[i] -= lrt * mh / (np.sqrt(vh) + eps)
    return params, mx, sx


def eval_mlp(params, mx, sx, nu):
    W1, b1, W2, b2, W3, b3 = params
    X = np.array([[(nu - 0.3) / 0.1]])
    h1 = np.tanh(X @ W1 + b1); h2 = np.tanh(h1 @ W2 + b2); out = (h2 @ W3 + b3)[0]
    return out * sx + mx


def coupl_solve(M, q):
    g = np.linalg.solve(M, -q)
    ychk = np.linspace(0, Ly, 41)
    gx = np.array([sum(g[j] * trace_basis(y)[j] for j in range(NT)) for y in ychk])
    gy = np.array([sum(g[NT + j] * trace_basis(y)[j] for j in range(NT)) for y in ychk])
    return g, gx, gy, ychk


def main():
    print("=== 参数化界面算子 A(ν): 探测 -> 训练 -> 新 ν 零子域求解 ===")
    t0 = time.time()
    Ms, qs = [], []
    for nu in NUS_TRAIN:
        M, q = probe(nu); Ms.append(M); qs.append(q)
    t_train = time.time() - t0
    Tgt = [np.concatenate([M.reshape(-1), q]) for M, q in zip(Ms, qs)]
    n_solves = len(NUS_TRAIN) * (2 * 2 * NT + 1)
    print(f"训练: {len(NUS_TRAIN)} 个 ν × {2*2*NT+1} 次子域求解 = {n_solves} 次, 用时 {t_train:.2f}s")

    poly = fit_poly(NUS_TRAIN, Tgt, deg=4)
    mlp_p, mlp_mx, mlp_sx = fit_mlp(NUS_TRAIN, Tgt)

    print(f"\n{'ν*':>6}{'poly ux':>12}{'MLP ux':>12}{'poly uy':>12}{'MLP uy':>12}{'算子(poly)':>13}")
    rows = []
    for nu in NUS_TEST:
        Mx, qx = probe(nu); _, gxx_ex, gyy_ex, _ = coupl_solve(Mx, qx)
        vx = np.concatenate([Mx.reshape(-1), qx])
        vp = eval_poly(poly, nu); _, gxx_p, gyy_p, _ = coupl_solve(vp[:4*NT*NT].reshape(2*NT, 2*NT), vp[4*NT*NT:])
        vm = eval_mlp(mlp_p, mlp_mx, mlp_sx, nu); _, gxx_m, gyy_m, _ = coupl_solve(vm[:4*NT*NT].reshape(2*NT, 2*NT), vm[4*NT*NT:])
        ex_p = np.linalg.norm(gxx_p - gxx_ex)/np.linalg.norm(gxx_ex); ey_p = np.linalg.norm(gyy_p - gyy_ex)/np.linalg.norm(gyy_ex)
        ex_m = np.linalg.norm(gxx_m - gxx_ex)/np.linalg.norm(gxx_ex); ey_m = np.linalg.norm(gyy_m - gyy_ex)/np.linalg.norm(gyy_ex)
        eop = np.linalg.norm(vp - vx)/np.linalg.norm(vx)
        rows.append((nu, ex_p, ey_p, ex_m, ey_m, eop))
        print(f"{nu:>6.2f}{ex_p:>12.2e}{ex_m:>12.2e}{ey_p:>12.2e}{ey_m:>12.2e}{eop:>13.2e}")

    t0 = time.time(); probe(0.30); t_one = time.time() - t0
    t0 = time.time()
    for _ in range(200):
        vp = eval_poly(poly, 0.31); coupl_solve(vp[:4*NT*NT].reshape(2*NT, 2*NT), vp[4*NT*NT:])
    t_sur = (time.time() - t0)/200
    print(f"\n单次: 精确探测 {t_one*1e3:.1f} ms  vs  代理前向+界面解 {t_sur*1e3:.3f} ms  (加速 {t_one/t_sur:.0f}×)")
    print(f"      训练一次({n_solves} 次子域求解) 后, 任意新 ν 均**零子域求解**。")

    # ---- 图 ----
    import matplotlib; import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for _c in ['Microsoft YaHei','SimHei','STSong','Arial Unicode MS']:
        try:
            font_manager.findfont(_c, fallback_to_default=False); plt.rcParams['font.sans-serif']=[_c]; break
        except Exception: continue
    plt.rcParams['axes.unicode_minus']=False
    Mx, qx = probe(0.30); vx = np.concatenate([Mx.reshape(-1), qx])
    vp = eval_poly(poly, 0.30)
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=150)
    axs[0].plot(vx, vp, '.', ms=4, color='#1f77b4')
    lo, hi = min(vx.min(), vp.min()), max(vx.max(), vp.max())
    axs[0].plot([lo, hi], [lo, hi], 'r-', lw=1.0)
    axs[0].set_xlabel('精确探测的算子/载荷项'); axs[0].set_ylabel('参数化代理预测')
    axs[0].set_title('界面算子 parity (ν*=0.30, 未见)', fontsize=10); axs[0].grid(alpha=0.3)
    degs = [2,3,4,5]; errs=[]
    for dg in degs:
        pl = fit_poly(NUS_TRAIN, Tgt, dg)
        ee=[]
        for nu in NUS_TEST:
            Mq, qq = probe(nu); vq = np.concatenate([Mq.reshape(-1), qq])
            ee.append(np.linalg.norm(eval_poly(pl, nu)-vq)/np.linalg.norm(vq))
        errs.append(np.mean(ee))
    axs[1].semilogy(degs, errs, 'o-', color='#d62728')
    axs[1].set_xlabel('多项式阶数'); axs[1].set_ylabel('算子相对误差 (未见 ν 平均)')
    axs[1].set_title('参数化代理精度 vs 阶数', fontsize=10); axs[1].grid(alpha=0.3, which='both')
    fig.suptitle('参数化界面算子 A(ν) 代理: 训练一次, 新 ν 零子域求解', fontsize=12)
    pf = os.path.join(OUT, 'param_operator.png')
    fig.savefig(pf, bbox_inches='tight'); plt.close(fig)
    print(f"[出图] {pf}")

    out = os.path.join(OUT, 'param_operator.txt')
    with open(out, 'w', encoding='utf-8') as fp:
        fp.write("参数化界面算子 A(ν) 代理 (多项式 deg4 / MLP)\n")
        fp.write(f"训练 ν={NUS_TRAIN} ({n_solves} 次子域求解, {t_train:.2f}s), 测试 ν={NUS_TEST}\n")
        fp.write(f"{'ν*':>6}{'poly ux':>14}{'poly uy':>14}{'MLP ux':>14}{'MLP uy':>14}{'算子误差':>14}\n")
        for nu, exp, eyp, exm, eym, eop in rows:
            fp.write(f"{nu:>6.2f}{exp:>13.3e}{eyp:>13.3e}{exm:>13.3e}{eym:>13.3e}{eop:>13.3e}\n")
        fp.write(f"单次: 探测 {t_one*1e3:.2f} ms vs 代理 {t_sur*1e3:.4f} ms, 加速 {t_one/t_sur:.0f}x\n")
    print(f"[保存] {out}")


if __name__ == '__main__':
    main()
