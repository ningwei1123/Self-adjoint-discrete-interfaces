# -*- coding: utf-8 -*-
"""
参数化 DeepONet 子域代理 (Parametric DeepONet Surrogate)
=======================================================
目标: 把右半 NN 子域从"固定多项式特征"升级为**可复用的算子代理**——
学习子域的 Dirichlet-to-Neumann (DtN) 算子:
        输入函数: 界面位移迹  g(y) = (u_x,u_y)|_Γ
        参数:     端部载荷迹  h(y) = u_x|_{x=Lx}  与  材料泊松比 ν
        输出函数: 界面面力    λ(y) = σ·n |_Γ = (−σ_xx, −σ_xy)|_Γ
一次训练后, 任意新 (g,h,ν) 只需一次前向即可得到界面面力,
无需再组装/求解子域 FEM, 从而把耦合求解约化为界面小系统。

DeepONet 结构 (numpy 手写 MLP + Adam):
    branch: [g 采样(2m), h 采样(m), ν] -> p 个系数 b_k
    trunk : y                              -> 2p 个基值 (λx, λy 各 p)
    output: λ(y) = Σ_k b_k t_k(y)

数据: 随机光滑迹 -> 精细 FEM 求解子域 -> Γ 上面力作为标签。
"""
import os, math, time
import numpy as np
import stress_recovery2d as R

E_mat = 2100.0
Lx, Ly, X0 = 2.0, 1.0, 1.0
NX, NY = 12, 12            # 子域 FEM 网格
M = NY + 1                 # Γ 上传感器数 = 13
NUS_TRAIN = [0.20, 0.24, 0.28, 0.32, 0.36, 0.40]
NUS_TEST = [0.22, 0.26, 0.30, 0.34, 0.38]    # 插值参数, 训练未见
NSAMP_PER_NU = 120
P = 32                     # DeepONet 基维数


def D2_of(nu):
    D00 = E_mat / (1 - nu * nu); D01 = E_mat * nu / (1 - nu * nu)
    D22 = E_mat / (2 * (1 + nu))
    return np.array([[D00, D01, 0.0], [D01, D00, 0.0], [0.0, 0.0, D22]])


def assemble_K(nodes, elems, D2):
    ndof = 2 * len(nodes)
    K = np.zeros((ndof, ndof))
    gp = [-1.0 / math.sqrt(3), 1.0 / math.sqrt(3)]
    for e in elems:
        xy = nodes[e]
        for gx in gp:
            for gy in gp:
                N, dxi, det = R._q4(gx, gy)
                J = np.array([[dxi @ xy[:, 0], dxi @ xy[:, 1]],
                              [det @ xy[:, 0], det @ xy[:, 1]]])
                dJ = np.linalg.det(J); iJ = np.linalg.inv(J)
                dNdx = iJ[0, 0] * dxi + iJ[0, 1] * det
                dNdy = iJ[1, 0] * dxi + iJ[1, 1] * det
                B = np.zeros((3, 8))
                for k in range(4):
                    B[0, 2 * k] = dNdx[k]; B[1, 2 * k + 1] = dNdy[k]
                    B[2, 2 * k] = dNdy[k]; B[2, 2 * k + 1] = dNdx[k]
                idx = [2 * e[k] + c for k in range(4) for c in range(2)]
                K[np.ix_(idx, idx)] += B.T @ D2 @ B * dJ
    return K


def solve_sub(nodes, elems, K, gx, gy_, hx):
    """Γ: u = (gx,gy) (全 Dirichlet); x=Lx: ux=hx, uy 自由。返回 u 向量。
    gx/gy_/hx 按边界节点自下而上的传感器顺序给出。"""
    nn = len(nodes); ndof = 2 * nn
    left = sorted([k for k in range(nn) if abs(nodes[k, 0] - X0) < 1e-9], key=lambda k: nodes[k, 1])
    right = sorted([k for k in range(nn) if abs(nodes[k, 0] - Lx) < 1e-9], key=lambda k: nodes[k, 1])
    fixed = []; ufix = {}
    for si, k in enumerate(left):
        fixed.append(2 * k); ufix[2 * k] = gx[si]
        fixed.append(2 * k + 1); ufix[2 * k + 1] = gy_[si]
    for si, k in enumerate(right):
        fixed.append(2 * k); ufix[2 * k] = hx[si]
    fixed = np.array(fixed)
    free = np.array([d for d in range(ndof) if d not in set(fixed.tolist())])
    Kr = K[np.ix_(free, free)]
    fr = -K[np.ix_(free, fixed)] @ np.array([ufix[d] for d in fixed])
    u = np.zeros(ndof)
    u[free] = np.linalg.solve(Kr, fr)
    for d, v in ufix.items():
        u[d] = v
    return u


def traction_gamma(nodes, elems, u, D2, nx, ny):
    """Γ(x=X0) 上的面力 λ=(−σ_xx,−σ_xy), 按 Γ 节点返回 (M,2)。"""
    lam = {}
    for j in range(ny):
        e = elems[j * nx + 0]
        xy = nodes[e]
        for nat, node in [((-1.0, -1.0), e[0]), ((-1.0, 1.0), e[3])]:
            _, dxi, det = R._q4(*nat)
            J = np.array([[dxi @ xy[:, 0], dxi @ xy[:, 1]],
                          [det @ xy[:, 0], det @ xy[:, 1]]])
            iJ = np.linalg.inv(J)
            dNdx = iJ[0, 0] * dxi + iJ[0, 1] * det
            dNdy = iJ[1, 0] * dxi + iJ[1, 1] * det
            ux = u[2 * e]; uy = u[2 * e + 1]
            exx = dNdx @ ux; eyy = dNdy @ uy; exy = 0.5 * (dNdy @ ux + dNdx @ uy)
            sig = D2 @ np.array([exx, eyy, 2 * exy])
            lam[node] = (-sig[0], -sig[2])
    left = [k for k in range(len(nodes)) if abs(nodes[k, 0] - X0) < 1e-9]
    left.sort(key=lambda k: nodes[k, 1])
    return np.array([lam[k] for k in left])          # (M,2)


def rand_trace(rng, m, amp):
    y = np.linspace(0, Ly, m)
    c = rng.normal(0, 1, 5) * amp
    return c[0] + sum(c[j] * np.cos(j * np.pi * y / Ly) for j in range(1, 5))


# ---------------- DeepONet (numpy MLP + Adam) ----------------
def init_mlp(sizes, rng):
    Ws = []; bs = []
    for i in range(len(sizes) - 1):
        Ws.append(rng.normal(0, math.sqrt(2.0 / sizes[i]), (sizes[i], sizes[i + 1])))
        bs.append(np.zeros(sizes[i + 1]))
    return Ws, bs


def fwd(Ws, bs, X):
    acts = [X]; a = X
    for i in range(len(Ws)):
        z = a @ Ws[i] + bs[i]
        a = np.tanh(z) if i < len(Ws) - 1 else z
        acts.append(a)
    return acts


def bwd(Ws, acts, dOut):
    dWs = [None] * len(Ws); dbs = [None] * len(Ws)
    d = dOut
    for i in range(len(Ws) - 1, -1, -1):
        dWs[i] = acts[i].T @ d
        dbs[i] = d.sum(0)
        if i > 0:
            d = (d @ Ws[i].T) * (1.0 - acts[i] ** 2)
    return dWs, dbs


def main():
    rng = np.random.default_rng(0)
    nodes, elems = R.build_mesh(NX, NY, X0, Lx, 0.0, Ly)
    Kfac = {nu: assemble_K(nodes, elems, D2_of(nu)) for nu in set(NUS_TRAIN + NUS_TEST)}

    def gen(nus, nsamp):
        Xb = []; Yb = []
        for nu in nus:
            K = Kfac[nu]
            yv = np.linspace(0, Ly, M)
            for _ in range(nsamp):
                amp = rng.uniform(0.002, 0.015)
                gx = rand_trace(rng, M, amp)
                gy = rand_trace(rng, M, amp)
                # 端部载荷: 抛物型主项(与实际工况一致) + 小幅随机
                hx = np.array([R.right_disp(y) for y in yv]) + rand_trace(rng, M, 0.003)
                u = solve_sub(nodes, elems, K, gx, gy, hx)
                lam = traction_gamma(nodes, elems, u, D2_of(nu), NX, NY)
                Xb.append(np.concatenate([gx, gy, hx, [nu]]))
                Yb.append(lam)
        return np.array(Xb), np.array(Yb)

    Xtr, Ytr = gen(NUS_TRAIN, NSAMP_PER_NU)
    Xte, Yte = gen(NUS_TEST, 40)
    print(f"训练样本 {Xtr.shape[0]}, 测试样本 {Xte.shape[0]}, 输入维 {Xtr.shape[1]}, 输出 {Ytr.shape[1:]}")

    # 标准化
    mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-9
    my, sy = Ytr.reshape(-1, 2).mean(0), Ytr.reshape(-1, 2).std(0) + 1e-9
    Xtr_n = (Xtr - mx) / sx; Xte_n = (Xte - mx) / sx
    Ytr_n = (Ytr - my) / sy; Yte_n = (Yte - my) / sy

    ys = np.linspace(0, Ly, M)
    tr_in = ((ys - ys.mean()) / (ys.std() + 1e-9)).reshape(-1, 1)

    Wb, bb = init_mlp([Xtr.shape[1], 96, 96, P], rng)
    Wt, bt = init_mlp([1, 96, 96, 2 * P], rng)
    params = [Wb, bb, Wt, bt]

    def forward(Xb_n, tr_in):
        ab = fwd(Wb, bb, Xb_n)
        at = fwd(Wt, bt, tr_in)
        b = ab[-1]                                   # (N,P)
        t = at[-1]                                   # (K,2P)
        lx = b @ t[:, :P].T
        ly = b @ t[:, P:].T
        return lx, ly, ab, at

    # Adam
    m_ = [np.zeros_like(w) if isinstance(w, np.ndarray) else [np.zeros_like(wi) for wi in w] for w in params]
    v_ = [np.zeros_like(w) if isinstance(w, np.ndarray) else [np.zeros_like(wi) for wi in w] for w in params]
    lr0, b1, b2, eps = 3e-3, 0.9, 0.999, 1e-8
    tstep = 0
    best = 1e18; best_loss = 1e18; best_params = None
    t0 = time.time()
    for ep in range(16001):
        lr = lr0 * (0.90 ** (ep / 1000.0))
        lx, ly, ab, at = forward(Xtr_n, tr_in)
        e = np.empty_like(Ytr_n)
        e[..., 0] = lx - Ytr_n[..., 0]
        e[..., 1] = ly - Ytr_n[..., 1]
        loss = np.mean(e ** 2)
        if loss < best_loss:
            best_loss = loss
            best_params = ([w.copy() for w in Wb], [b.copy() for b in bb],
                           [w.copy() for w in Wt], [b.copy() for b in bt])
        N = Xtr_n.shape[0]
        # dLoss/dlx = 2*ex/N, dly
        dlx = (2.0 / (N * M)) * e[..., 0]           # (N,M)
        dly = (2.0 / (N * M)) * e[..., 1]
        # b: lx = b @ t[:, :P].T -> db = dlx @ t[:,:P] ; dl = b.T @ dlx
        t = at[-1]
        db = dlx @ t[:, :P] + dly @ t[:, P:]
        dt = np.empty_like(t)
        dt[:, :P] = dlx.T @ ab[-1]
        dt[:, P:] = dly.T @ ab[-1]
        # backprop branch (last layer is linear -> dZ=db)
        dWb = [None] * (len(Wb)); dbb = [None] * (len(Wb))
        d = db
        for i in range(len(Wb) - 1, -1, -1):
            dWb[i] = ab[i].T @ d; dbb[i] = d.sum(0)
            if i > 0:
                d = (d @ Wb[i].T) * (1.0 - ab[i] ** 2)
        # backprop trunk (last linear -> dZ=dt)
        dWt = [None] * (len(Wt)); dbt = [None] * (len(Wt))
        d = dt
        for i in range(len(Wt) - 1, -1, -1):
            dWt[i] = at[i].T @ d; dbt[i] = d.sum(0)
            if i > 0:
                d = (d @ Wt[i].T) * (1.0 - at[i] ** 2)
        grads = [dWb, dbb, dWt, dbt]
        tstep += 1
        for pi, (wlist, glist) in enumerate(zip(params, grads)):
            for i in range(len(wlist)):
                g = glist[i]
                m_[pi][i] = b1 * m_[pi][i] + (1 - b1) * g
                v_[pi][i] = b2 * v_[pi][i] + (1 - b2) * g * g
                mh = m_[pi][i] / (1 - b1 ** tstep); vh = v_[pi][i] / (1 - b2 ** tstep)
                wlist[i] -= lr * mh / (np.sqrt(vh) + eps)
        if ep % 2000 == 0:
            print(f"  ep{ep:5d}  loss={loss:.3e}  ({time.time()-t0:.1f}s)")

    # 恢复最优参数
    if best_params is not None:
        for w, ws in zip(Wb, best_params[0]): w[...] = ws
        for b, bs in zip(bb, best_params[1]): b[...] = bs
        for w, ws in zip(Wt, best_params[2]): w[...] = ws
        for b, bs in zip(bt, best_params[3]): b[...] = bs
    print(f"  best loss={best_loss:.3e}")

    # 评估
    lx, ly, _, _ = forward(Xtr_n, tr_in)
    lx = lx * sy[0] + my[0]; ly = ly * sy[1] + my[1]
    tr_err = np.sqrt(np.mean((np.stack([lx, ly], -1) - Ytr) ** 2)) / np.sqrt(np.mean(Ytr ** 2))
    lx, ly, _, _ = forward(Xte_n, tr_in)
    lx = lx * sy[0] + my[0]; ly = ly * sy[1] + my[1]
    te_err = np.sqrt(np.mean((np.stack([lx, ly], -1) - Yte) ** 2)) / np.sqrt(np.mean(Yte ** 2))
    print(f"\nDeepONet 相对 L2 误差: 训练 {tr_err:.3%}   测试(未见 ν/迹) {te_err:.3%}")

    # 加速: 子域 FEM 求解 vs 代理前向
    nu = 0.30
    K = Kfac[nu]
    gx = rand_trace(rng, M, 0.01); gy = rand_trace(rng, M, 0.01); hx = rand_trace(rng, M, 0.01)
    t0 = time.time()
    for _ in range(20):
        u = solve_sub(nodes, elems, K, gx, gy, hx); traction_gamma(nodes, elems, u, D2_of(nu), NX, NY)
    dt_fem = (time.time() - t0) / 20
    xin = ((np.concatenate([gx, gy, hx, [nu]]) - mx) / sx).reshape(1, -1)
    t0 = time.time()
    for _ in range(200):
        forward(xin, tr_in)
    dt_sur = (time.time() - t0) / 200
    print(f"单次: 子域FEM {dt_fem*1e3:.2f} ms  vs  代理前向 {dt_sur*1e3:.3f} ms  (加速 {dt_fem/dt_sur:.0f}×)")

    # ---- 图: parity + 例程面力剖面 ----
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
    pred_te = np.stack([lx, ly], -1)
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=150)
    tt = Yte.ravel(); pp = pred_te.ravel()
    axs[0].plot(tt, pp, '.', ms=3, alpha=0.35, color='#1f77b4')
    lo = min(tt.min(), pp.min()); hi = max(tt.max(), pp.max())
    axs[0].plot([lo, hi], [lo, hi], 'r-', lw=1.2)
    axs[0].set_xlabel('子域 FEM 面力 λ'); axs[0].set_ylabel('DeepONet 预测 λ')
    axs[0].set_title(f'测试集 parity (未见 ν 与迹), 相对L2={te_err:.2%}', fontsize=10)
    axs[0].grid(alpha=0.3)
    axs[1].plot(ys, Yte[0, :, 0], 'o-', color='#2ca02c', ms=4, label='FEM  λx')
    axs[1].plot(ys, pred_te[0, :, 0], 's--', color='#d62728', ms=4, label='DeepONet λx')
    axs[1].plot(ys, Yte[0, :, 1], 'o-', color='#17becf', ms=4, label='FEM  λy')
    axs[1].plot(ys, pred_te[0, :, 1], 's--', color='#9467bd', ms=4, label='DeepONet λy')
    axs[1].set_xlabel('y (界面 Γ)'); axs[1].set_ylabel('面力 λ')
    axs[1].set_title('例: 界面面力剖面 (测试样本)', fontsize=10)
    axs[1].legend(fontsize=8); axs[1].grid(alpha=0.3)
    fig.suptitle('参数化 DeepONet 子域代理: 学习 Dirichlet-to-Neumann 算子', fontsize=12)
    pfig = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'deeponet_surrogate.png'))
    fig.savefig(pfig, bbox_inches='tight'); plt.close(fig)
    print(f"[出图] {pfig}")

    # 持久化代理参数 (供耦合求解复用)
    np.savez(os.path.join(os.path.dirname(__file__), '..', 'results', 'deeponet_params.npz'),
             m=M, Lx=Lx, Ly=Ly, X0=X0, NX=NX, NY=NY,
             mx=mx, sx=sx, my=my, sy=sy,
             **{f'Wb{i}': Wb[i] for i in range(len(Wb))},
             **{f'bb{i}': bb[i] for i in range(len(bb))},
             **{f'Wt{i}': Wt[i] for i in range(len(Wt))},
             **{f'bt{i}': bt[i] for i in range(len(bt))})
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'deeponet_surrogate.txt'))
    with open(out, 'w', encoding='utf-8') as fp:
        fp.write("参数化 DeepONet 子域代理 (学习 DtN 算子: g(y),h(y),ν -> λ(y))\n")
        fp.write(f"样本: 训练 {Xtr.shape[0]}(ν={NUS_TRAIN}), 测试 {Xte.shape[0]}(ν={NUS_TEST})\n")
        fp.write(f"DeepONet: branch{list(Wb[0].shape)} trunk{list(Wt[0].shape)} p={P}\n")
        fp.write(f"相对 L2 误差: 训练 {tr_err:.4%}  测试 {te_err:.4%}\n")
        fp.write(f"单次推理: 子域FEM {dt_fem*1e3:.3f} ms vs 代理 {dt_sur*1e3:.4f} ms, 加速 {dt_fem/dt_sur:.0f}×\n")
    print(f"[保存] {out}")


if __name__ == '__main__':
    main()
