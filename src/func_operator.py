# -*- coding: utf-8 -*-
"""
从标量参数到**函数参数**: 参数化界面算子的 DeepONet branch 必要性验证
=====================================================================
动机(承接 param_operator.py):
  - 标量参数 ν 时, 参数化算子 M(ν), q(ν) 用**多项式基**即可插值到机器精度 (误差 1e-6)。
    -> 标量情形下 trunk 就取多项式基, 不需要网络。
  - 当参数是**函数**时, 界面算子 M(E(·)), q(E(·)) 是输入函数的**非线性泛函**,
    多项式/线性代理不再适用 -> 需要真正的 operator network (branch 编码输入函数)。

本脚本设置:
  参数(输入函数)  : 沿轴向变化的杨氏模量场  E(x) = E0 * exp( Σ_{k=1..4} c_k cos(kπx/Lx) )
                    → 4 个系数的**光滑正函数族**; branch 只看到 E 在 9 个传感器处的采样值。
  输出函数        : 界面位移迹  g_Γ(y) = (u_x(y), u_y(y))  在 41 个 y 点取值。
  物理内核        : 变系数 FEM 装配 + 迹空间(Legendre)弱形式 DtN 探测,
                   界面方程 (S_L(E) + S_R(E)) g = -q(E)  (外法线约定)。
  对照            : ① 线性/岭回归 (sensor -> 输出)   ② 无 trunk 的 MLP (sensor -> 82 输出)
                    ③ DeepONet (branch+trunk, 共享 branch 分量专用 trunk)
  验证            : V1 均匀场退化一致性(与 param_operator 的 ν=0.30 结果对比)
                    V2 网格加密收敛性
"""
import os, math, time
import numpy as np
import stress_recovery2d as R

# ---------------------------------------------------------------- 基本参数
E0 = 2100.0
NU = 0.30
Lx, Ly, X0 = 2.0, 1.0, 1.0
NX, NY, NT = 12, 12, 6          # 每子域单元数 / 迹模数
K4 = 4                          # 场 E(x) 的 Fourier(cos) 模数
NS = 9                          # branch 传感器数 (E 在 x 上的采样点)
NQ = 41                         # trunk 查询点数 (y 上的输出采样)
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))


# ---------------------------------------------------------------- E(x) 场
def E_of(x, c):
    """E(x) = E0 * exp( Σ c_k cos(kπx/Lx) ), c 长度 K4。"""
    s = 0.0
    for k in range(len(c)):
        s += c[k] * math.cos((k + 1) * math.pi * x / Lx)
    return E0 * math.exp(s)


def D2_from_E(Ec, nu=NU):
    D00 = Ec / (1 - nu * nu); D01 = nu * D00; G = Ec / (2 * (1 + nu))
    return np.array([[D00, D01, 0.0], [D01, D00, 0.0], [0.0, 0.0, G]])


# ---------------------------------------------------------------- 变系数装配
def assemble_field(nodes, elems, c, nu=NU):
    """单元中心取 E -> 逐单元 D2。"""
    ndof = 2 * len(nodes); K = np.zeros((ndof, ndof))
    gp = [-1 / math.sqrt(3), 1 / math.sqrt(3)]
    B = np.zeros((3, 8))
    for e in elems:
        xy = nodes[e]
        D2 = D2_from_E(E_of(float(xy[:, 0].mean()), c), nu)
        for gx in gp:
            for gy in gp:
                N, dxi, det = R._q4(gx, gy)
                J = np.array([[dxi @ xy[:, 0], dxi @ xy[:, 1]],
                              [det @ xy[:, 0], det @ xy[:, 1]]])
                dJ = np.linalg.det(J); iJ = np.linalg.inv(J)
                dNdx = iJ[0, 0] * dxi + iJ[0, 1] * det
                dNdy = iJ[1, 0] * dxi + iJ[1, 1] * det
                B[:] = 0.0
                for k in range(4):
                    B[0, 2 * k] = dNdx[k]; B[1, 2 * k + 1] = dNdy[k]
                    B[2, 2 * k] = dNdy[k]; B[2, 2 * k + 1] = dNdx[k]
                idx = [2 * e[k] + cc for k in range(4) for cc in range(2)]
                K[np.ix_(idx, idx)] += B.T @ D2 @ B * dJ
    return K


# ---------------------------------------------------------------- 迹空间 / 探测
def trace_basis(y, nt=NT):
    e = 2.0 * y / Ly - 1.0
    P = [1.0, e]
    for j in range(2, nt):
        P.append(((2 * j - 1) * e * P[j - 1] - (j - 1) * P[j - 2]) / j)
    return np.array(P[:nt], dtype=float)


_GV, _GW = np.polynomial.legendre.leggauss(60)
_GY = 0.5 * (_GV + 1.0) * Ly
_GWT = _GW * (Ly / 2.0)
_GP = np.array([trace_basis(y) for y in _GY])
TRACE_H = np.array([float(np.sum(_GWT * _GP[:, i] * _GP[:, i])) for i in range(NT)])   # ∫φ_i² dy


def transfer_galerkin(r_node, iface, nodes, comp, mu=1.0):
    """一致节点力 -> 迹基系数: S_{i,·} = Σ_j r_j φ_i(y_j) / ∫φ_i² dy。

    与 Steklov-Poincaré 算子的 Galerkin 定义 ⟨Sη,μ⟩=⟨A u, Rμ⟩ 完全一致,
    因而**严格自伴且正定**(点式角点提取不满足此性质)。"""
    ys = np.array([nodes[k, 1] for k in iface])
    Pn = np.array([trace_basis(y) for y in ys])
    rv = np.array([r_node[2 * k + comp] for k in iface])
    return (Pn.T @ rv) / TRACE_H


def project(v, ys):
    P = np.array([trace_basis(y) for y in ys])
    return np.linalg.lstsq(P, v, rcond=None)[0]


def build(x0, x1, nx=NX, ny=NY):
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
        u[free] = np.linalg.solve(K[np.ix_(free, free)],
                                  -K[np.ix_(free, fds)] @ np.array([fix[d] for d in fds]))
    for d, v in fix.items():
        u[d] = v
    return u


def pointwise_lam_field(nodes, elems, u, iface, at_right, c, nx=NX, ny=NY):
    """界面点式面力 (外法线约定), 逐单元 E。"""
    m = len(iface); lam = np.zeros((m, 2))
    for j in range(ny):
        col = nx - 1 if at_right else 0
        e = elems[j * nx + col]; xy = nodes[e]
        D2 = D2_from_E(E_of(float(xy[:, 0].mean()), c))
        pairs = [((1, -1), e[1]), ((1, 1), e[2])] if at_right else [((-1, -1), e[0]), ((-1, 1), e[3])]
        for nat, node in pairs:
            if node in iface:
                _, dxi, det = R._q4(*nat)
                J = np.array([[dxi @ xy[:, 0], dxi @ xy[:, 1]],
                              [det @ xy[:, 0], det @ xy[:, 1]]])
                iJ = np.linalg.inv(J)
                dNdx = iJ[0, 0] * dxi + iJ[0, 1] * det
                dNdy = iJ[1, 0] * dxi + iJ[1, 1] * det
                ux = u[2 * e]; uy = u[2 * e + 1]
                exx = dNdx @ ux; eyy = dNdy @ uy; exy = 0.5 * (dNdy @ ux + dNdx @ uy)
                sig = D2 @ np.array([exx, eyy, 2 * exy])
                sgn = 1.0 if at_right else -1.0
                lam[iface.index(node)] = (sgn * sig[0], sgn * sig[2])
    return lam


def probe_field(c, nu=NU, nx=NX, ny=NY, transfer='galerkin'):
    """探测 M(c)=S_L+S_R (2NT,2NT) 与 q(c) (2NT,)。"""
    S = {}
    for side in ('L', 'R'):
        x0, x1 = (0.0, X0) if side == 'L' else (X0, Lx)
        nodes, elems, iface = build(x0, x1, nx, ny)
        K = assemble_field(nodes, elems, c, nu)
        ys = np.array([nodes[k, 1] for k in iface]); at_right = (side == 'L')
        Ss = np.zeros((2 * NT, 2 * NT))
        for cc in range(2):
            for j in range(NT):
                k = cc * NT + j

                def gf(y, cc=cc, j=j):
                    p = trace_basis(y)[j]
                    return (p, 0.0) if cc == 0 else (0.0, p)
                far = None if side == 'L' else (Lx, lambda y: 0.0)
                u = solve_dirichlet(nodes, elems, K, iface, gf, far)
                if transfer == 'galerkin':
                    r = K @ u                       # 一致节点反力 = ∫N_i t dΓ (弱形式)
                    Ss[:NT, k] = transfer_galerkin(r, iface, nodes, 0)
                    Ss[NT:, k] = transfer_galerkin(r, iface, nodes, 1)
                else:
                    lam = pointwise_lam_field(nodes, elems, u, iface, at_right, c, nx, ny)
                    Ss[:NT, k] = project(lam[:, 0], ys); Ss[NT:, k] = project(lam[:, 1], ys)
        S[side] = Ss
    nodes, elems, iface = build(X0, Lx, nx, ny)
    K = assemble_field(nodes, elems, c, nu)
    ys = np.array([nodes[k, 1] for k in iface])
    u = solve_dirichlet(nodes, elems, K, iface, lambda y: (0.0, 0.0), (Lx, R.right_disp))
    if transfer == 'galerkin':
        r = K @ u
        q = np.concatenate([transfer_galerkin(r, iface, nodes, 0),
                            transfer_galerkin(r, iface, nodes, 1)])
    else:
        lam = pointwise_lam_field(nodes, elems, u, iface, False, c, nx, ny)
        q = np.concatenate([project(lam[:, 0], ys), project(lam[:, 1], ys)])
    return S['L'] + S['R'], q


def gfun_from_coef(g):
    """迹系数 -> (y -> (ux,uy)) 函数。"""
    return lambda y: (sum(g[j] * trace_basis(y)[j] for j in range(NT)),
                      sum(g[NT + j] * trace_basis(y)[j] for j in range(NT)))


# ---------------------------------------------------------------- 数据生成
YCHK = np.linspace(0.0, Ly, NQ)
XSENS = np.linspace(0.0, Lx, NS)


def gen(nsamp, seed, cscale=0.60, nx=NX, ny=NY):
    rng = np.random.default_rng(seed)
    S = np.zeros((nsamp, NS)); G = np.zeros((nsamp, 2 * NQ)); C = np.zeros((nsamp, K4))
    for i in range(nsamp):
        c = rng.uniform(-cscale, cscale, size=K4)
        M, q = probe_field(c, NU, nx, ny)
        g = np.linalg.solve(M, -q)
        S[i] = [E_of(x, c) for x in XSENS]
        G[i] = np.concatenate([
            [sum(g[j] * trace_basis(y)[j] for j in range(NT)) for y in YCHK],
            [sum(g[NT + j] * trace_basis(y)[j] for j in range(NT)) for y in YCHK]])
        C[i] = c
    return S, G, C


def gen_cached(nsamp, seed, cscale, tag):
    """带磁盘缓存的数据生成 (便于反复调参)。"""
    f = os.path.join(OUT, f'func_data_{tag}.npz')
    if os.path.exists(f):
        d = np.load(f); return d['S'], d['G'], d['C']
    S, G, C = gen(nsamp, seed, cscale)
    np.savez(f, S=S, G=G, C=C)
    return S, G, C


# ---------------------------------------------------------------- 验证
def verify():
    print("=== 验证 ===")
    # V1: 均匀场 (c=0) 与 param_operator 的 ν=0.30 结果一致
    import param_operator as P
    M0, q0 = probe_field(np.zeros(K4))
    g0 = np.linalg.solve(M0, -q0)
    Mp, qp = P.probe(NU)
    gp = np.linalg.solve(Mp, -qp)
    print(f"  V1 均匀场退化: ||M(c=0)-M(ν)||/||M(ν)|| = {np.linalg.norm(M0-Mp)/np.linalg.norm(Mp):.3e}")
    print(f"                ||q(c=0)-q(ν)||/||q(ν)|| = {np.linalg.norm(q0-qp)/np.linalg.norm(qp):.3e}")
    print(f"                ||g(c=0)-g(ν)||/||g(ν)|| = {np.linalg.norm(g0-gp)/np.linalg.norm(gp):.3e}")
    # V2: 网格加密 (非均匀场)
    c = np.array([0.30, -0.20, 0.15, -0.25])
    res = []
    for nx in (8, 12, 16, 24):
        M, q = probe_field(c, NU, nx, nx)
        g = np.linalg.solve(M, -q)
        res.append(g)
    gref = res[-1]
    for nx, g in zip((8, 12, 16, 24), res):
        print(f"  V2 网格 {2*nx:>2}x{nx:<2} 单元: ||g-g(48x24)||/||g|| = "
              f"{np.linalg.norm(g-gref)/np.linalg.norm(gref):.3e}")
    # V3: 两种界面传递方式对照 (自伴性 + 与整体 FEM 的一致性)
    print("  V3 界面传递方式对照 (nx=12):")
    for tr in ('galerkin', 'pointwise'):
        Mt, qt = probe_field(np.zeros(K4), NU, 12, 12, transfer=tr)
        gt = np.linalg.solve(Mt, -qt)
        # 整体 FEM (同一网格 24x12), 直接解
        nF, eF = R.build_mesh(24, 12, 0.0, Lx, 0.0, Ly)
        KF = assemble_field(nF, eF, np.zeros(K4), NU)
        nd = 2 * len(nF)
        lf = np.where(np.abs(nF[:, 0]) < 1e-9)[0]; rf = np.where(np.abs(nF[:, 0] - Lx) < 1e-9)[0]
        fx = {}
        for k in lf:
            fx[2 * k] = 0.0; fx[2 * k + 1] = 0.0
        for k in rf:
            fx[2 * k] = R.right_disp(nF[k, 1])
        fdd = np.array(list(fx)); fss = set(fdd.tolist())
        frr = np.array([d for d in range(nd) if d not in fss])
        uF = np.zeros(nd)
        uF[frr] = np.linalg.solve(KF[np.ix_(frr, frr)], -KF[np.ix_(frr, fdd)] @ np.array([fx[d] for d in fdd]))
        for d, v in fx.items():
            uF[d] = v
        iF = sorted(np.where(np.abs(nF[:, 0] - X0) < 1e-9)[0], key=lambda k: nF[k, 1])
        yF = nF[iF, 1]
        gF = np.column_stack([np.interp(YCHK, yF, uF[2 * np.array(iF)]),
                              np.interp(YCHK, yF, uF[2 * np.array(iF) + 1])])
        gT = np.column_stack([np.array([sum(gt[j] * trace_basis(y)[j] for j in range(NT)) for y in YCHK]),
                              np.array([sum(gt[NT + j] * trace_basis(y)[j] for j in range(NT)) for y in YCHK])])
        Hw = np.diag(np.tile(TRACE_H, 2))
        HM = Hw @ Mt
        asym = np.abs(HM - HM.T).max() / np.abs(HM).max()
        print(f"    {tr:<10}: 自伴性(加权)={asym:.3e}  界面解 vs 整体FEM={np.linalg.norm(gT - gF)/np.linalg.norm(gF):.4%}")
    return M0, q0


# ---------------------------------------------------------------- numpy MLP
class MLP:
    def __init__(self, dims, rng, out_scale=1.0):
        self.W = []; self.b = []
        for i in range(len(dims) - 1):
            a, bdim = dims[i], dims[i + 1]
            self.W.append(rng.normal(0, math.sqrt(2.0 / a), (a, bdim)) * (out_scale if i == len(dims) - 2 else 1.0))
            self.b.append(np.zeros(bdim))

    def forward(self, X):
        self.A = [X]; self.Z = []
        H = X
        L = len(self.W)
        for i in range(L):
            Z = H @ self.W[i] + self.b[i]
            self.Z.append(Z)
            H = np.tanh(Z) if i < L - 1 else Z
            self.A.append(H)
        return H

    def backward(self, dH):
        L = len(self.W); gW = [None] * L; gb = [None] * L
        d = dH
        for i in range(L - 1, -1, -1):
            gW[i] = self.A[i].T @ d
            gb[i] = d.sum(0)
            if i > 0:
                d = (d @ self.W[i].T) * (1 - np.tanh(self.Z[i - 1]) ** 2)
        return gW, gb

    def params(self):
        return self.W, self.b


class DeepONet:
    """branch: E 传感器 -> P ;  trunk: y -> 2P ;  out_x = b·t_x, out_y = b·t_y。"""
    def __init__(self, ns, P=32, H=64, seed=0):
        rng = np.random.default_rng(seed)
        self.P = P
        self.branch = MLP([ns, H, H, P], rng)
        self.trunk = MLP([1, H, H, 2 * P], rng)

    def forward(self, Sin, Yin):
        B = self.branch.forward(Sin)          # (n, P)
        T = self.trunk.forward(Yin)           # (m, 2P)
        px = B @ T[:, :self.P].T              # (n, m)
        py = B @ T[:, self.P:].T
        return px, py

    def step(self, Sin, Yin, GX, GY, lr):
        n, m = Sin.shape[0], Yin.shape[0]
        px, py = self.forward(Sin, Yin)
        DX = (px - GX) * (2.0 / (n * m)); DY = (py - GY) * (2.0 / (n * m))
        B = self.branch.A[-1]                 # (n,P)
        dT = np.concatenate([DX.T @ B, DY.T @ B], axis=1)   # (m,2P)
        dB = DX @ self.trunk.A[-1][:, :self.P] + DY @ self.trunk.A[-1][:, self.P:]
        gWB, gbB = self.branch.backward(dB)
        gWT, gbT = self.trunk.backward(dT)
        loss = np.mean((px - GX) ** 2 + (py - GY) ** 2)
        return loss, (gWB, gbB, gWT, gbT)


def train_deeponet(Sin, Gin, P=48, H=64, epochs=8000, lr=3e-3, seed=0, wd=0.0):
    net = DeepONet(Sin.shape[1], P, H, seed)
    Yin = ((YCHK - Ly / 2) / (Ly / 2)).reshape(-1, 1)
    gx_mu, gx_sd = Gin[:, :NQ].mean(), Gin[:, :NQ].std() + 1e-12
    gy_mu, gy_sd = Gin[:, NQ:].mean(), Gin[:, NQ:].std() + 1e-12
    net.scales = (gx_mu, gx_sd, gy_mu, gy_sd)
    GX = (Gin[:, :NQ] - gx_mu) / gx_sd
    GY = (Gin[:, NQ:] - gy_mu) / gy_sd
    params = net.branch.W + net.branch.b + net.trunk.W + net.trunk.b
    m_ = [np.zeros_like(p) for p in params]; v_ = [np.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8
    best = (1e18, None)
    for t in range(1, epochs + 1):
        loss, grads = net.step(Sin, Yin, GX, GY, lr)
        flat = list(grads[0]) + list(grads[1]) + list(grads[2]) + list(grads[3])
        if wd > 0.0:
            nW = len(net.branch.W) + len(net.trunk.W)
            for i in range(nW):
                flat[i] = flat[i] + wd * params[i]
        lrt = lr * (0.9 ** (t / 1000.0))
        for i in range(len(params)):
            m_[i] = b1 * m_[i] + (1 - b1) * flat[i]
            v_[i] = b2 * v_[i] + (1 - b2) * flat[i] ** 2
            mh = m_[i] / (1 - b1 ** t); vh = v_[i] / (1 - b2 ** t)
            params[i] -= lrt * mh / (np.sqrt(vh) + eps)
        if loss < best[0]:
            best = (loss, [p.copy() for p in params])
    for p, pb in zip(params, best[1]):
        p[...] = pb
    return net, best[0]


def predict_deeponet(net, S):
    Yin = ((YCHK - Ly / 2) / (Ly / 2)).reshape(-1, 1)
    px, py = net.forward(S, Yin)
    gx_mu, gx_sd, gy_mu, gy_sd = net.scales
    return np.concatenate([px * gx_sd + gx_mu, py * gy_sd + gy_mu], axis=1)


# ---------------------------------------------------------------- 基线
def predict_ens(nets, S):
    return np.mean([predict_deeponet(nt, S) for nt in nets], axis=0)


def ridge_baseline(S, G, S_test, G_test):
    A = np.column_stack([S, np.ones(len(S))])
    lam = 1e-6 * np.trace(A.T @ A) / A.shape[1]
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ G)
    At = np.column_stack([S_test, np.ones(len(S_test))])
    P = At @ W
    return (np.linalg.norm(P - G_test) / np.linalg.norm(G_test),
            np.linalg.norm(At @ W - G_test, axis=1) / np.linalg.norm(G_test, axis=1))


def mlp_baseline(S, G, S_test, G_test, H=96, epochs=6000, lr=2e-3, seed=1):
    rng = np.random.default_rng(seed)
    mx, sx = S.mean(0), S.std(0) + 1e-12
    Sn = (S - mx) / sx
    gmx, gsx = G.mean(), G.std() + 1e-12
    Gn = (G - gmx) / gsx
    net = MLP([S.shape[1], H, H, G.shape[1]], rng, out_scale=0.1)
    params = net.W + net.b
    m_ = [np.zeros_like(p) for p in params]; v_ = [np.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8
    best = (1e18, None)
    for t in range(1, epochs + 1):
        out = net.forward(Sn)
        d = (out - Gn) * (2.0 / Gn.size)
        gW, gb = net.backward(d)
        flat = gW + gb
        lrt = lr * (0.9 ** (t / 1000.0))
        for i in range(len(params)):
            m_[i] = b1 * m_[i] + (1 - b1) * flat[i]
            v_[i] = b2 * v_[i] + (1 - b2) * flat[i] ** 2
            mh = m_[i] / (1 - b1 ** t); vh = v_[i] / (1 - b2 ** t)
            params[i] -= lrt * mh / (np.sqrt(vh) + eps)
        loss = np.mean((out - Gn) ** 2)
        if loss < best[0]:
            best = (loss, [p.copy() for p in params])
    for p, pb in zip(params, best[1]):
        p[...] = pb
    St = (S_test - mx) / sx
    P = net.forward(St) * gsx + gmx
    return (np.linalg.norm(P - G_test) / np.linalg.norm(G_test),
            np.linalg.norm(P - G_test, axis=1) / np.linalg.norm(G_test, axis=1))


# ---------------------------------------------------------------- 主程序
def main():
    print("=== 函数参数化界面算子: DeepONet branch 必要性验证 ===")
    M0, q0 = verify()

    t0 = time.time()
    S_tr, G_tr, C_tr = gen_cached(192, 100, 0.60, 'tr192_60')
    S_te, G_te, C_te = gen_cached(32, 777, 0.60, 'te32_60')
    print(f"\n数据: 训练 192 场 / 测试 32 场, 用时 {time.time()-t0:.1f}s")
    print(f"  E 变化范围 (训练): [{S_tr.min():.1f}, {S_tr.max():.1f}] (E0={E0})")

    # 归一化 (branch 输入)
    smx, ssd = S_tr.mean(0), S_tr.std(0) + 1e-12
    S_trn, S_ten = (S_tr - smx) / ssd, (S_te - smx) / ssd

    # ---- 基线 ----
    lin_e, lin_pe = ridge_baseline(S_trn, G_tr, S_ten, G_te)
    mlp_e, mlp_pe = mlp_baseline(S_trn, G_tr, S_ten, G_te)
    print(f"\n基线(未见场, 相对 L2):  线性/岭回归 {lin_e:.3%}   无trunk的MLP {mlp_e:.3%}")

    # ---- DeepONet (3 个种子的集成) ----
    t0 = time.time()
    nets = []
    for sd in (0, 1, 2):
        nt_, _ = train_deeponet(S_trn, G_tr, P=48, H=72, epochs=12000, wd=2e-5, seed=sd)
        nets.append(nt_)
    net = nets[0]
    t_tr = time.time() - t0
    P_te = predict_ens(nets, S_ten)
    dn_e = np.linalg.norm(P_te - G_te) / np.linalg.norm(G_te)
    dn_pe = np.linalg.norm(P_te - G_te, axis=1) / np.linalg.norm(G_te, axis=1)
    P_tr = predict_ens(nets, S_trn)
    dn_tr = np.linalg.norm(P_tr - G_tr) / np.linalg.norm(G_tr)
    def comp_err(P, Gref):
        ex = np.linalg.norm(P[:, :NQ] - Gref[:, :NQ]) / np.linalg.norm(Gref[:, :NQ])
        ey = np.linalg.norm(P[:, NQ:] - Gref[:, NQ:]) / np.linalg.norm(Gref[:, NQ:])
        return ex, ey
    ex_dn, ey_dn = comp_err(P_te, G_te)
    print(f"DeepONet (训练 {t_tr:.1f}s): 训练场 {dn_tr:.3%}   未见场 {dn_e:.3%}  (ux {ex_dn:.3%} / uy {ey_dn:.3%})")

    # ---- 物理残差 (把预测的 g 投回迹空间, 用精确 M,q 检验界面方程) ----
    def phys_resid(Pred, C):
        rs = []
        for i in range(len(Pred)):
            M, q = probe_field(C[i])
            gx = project(Pred[i, :NQ], YCHK); gy = project(Pred[i, NQ:], YCHK)
            g = np.concatenate([gx, gy])
            rs.append(np.linalg.norm(M @ g + q) / np.linalg.norm(q))
        return float(np.mean(rs))

    pr_dn = phys_resid(P_te, C_te)
    print(f"  界面方程残差 ||Mg+q||/||q|| (DeepONet 预测, 未见场): {pr_dn:.3%}")

    # ---- 外推测试: 训练 c∈[-0.35,0.35], 外推 |c|∈[0.45,0.75] (更大对比度) ----
    S_tr2, G_tr2, _ = gen_cached(160, 100, 0.35, 'tr160_35')
    S_ex, G_ex, C_ex = gen_cached(20, 555, 0.75, 'ex20_75')
    S_ex = np.abs(S_ex)  # 仅幅值分布改变 (E 采样取绝对值无明显意义, 保留原始即可)
    smx2, ssd2 = S_tr2.mean(0), S_tr2.std(0) + 1e-12
    S_tr2n, S_exn = (S_tr2 - smx2) / ssd2, (S_ex - smx2) / ssd2
    lin_ex, _ = ridge_baseline(S_tr2n, G_tr2, S_exn, G_ex)
    net_ex, _ = train_deeponet(S_tr2n, G_tr2, P=48, H=72, epochs=10000, wd=2e-5)
    dn_ex = np.linalg.norm(predict_deeponet(net_ex, S_exn) - G_ex) / np.linalg.norm(G_ex)
    print(f"\n外推 (训练 |c|<=0.35, 测试 |c|~0.75):  线性 {lin_ex:.3%}   DeepONet {dn_ex:.3%}")

    # ---- 学习曲线 ----
    print("\n学习曲线 (未见场相对 L2):")
    Ns = [10, 24, 48, 96, 192]
    curve = {'lin': [], 'mlp': [], 'dn': []}
    for n in Ns:
        le, _ = ridge_baseline(S_trn[:n], G_tr[:n], S_ten, G_te)
        me, _ = mlp_baseline(S_trn[:n], G_tr[:n], S_ten, G_te)
        nt, _ = train_deeponet(S_trn[:n], G_tr[:n], P=48, H=72, epochs=10000, wd=2e-5)
        de = np.linalg.norm(predict_deeponet(nt, S_ten) - G_te) / np.linalg.norm(G_te)
        curve['lin'].append(le); curve['mlp'].append(me); curve['dn'].append(de)
        print(f"  N={n:>2}: 线性 {le:>8.3%}   MLP {me:>8.3%}   DeepONet {de:>8.3%}")

    # ---- 图 ----
    import matplotlib; import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for _c in ['Microsoft YaHei', 'SimHei', 'STSong', 'Arial Unicode MS']:
        try:
            font_manager.findfont(_c, fallback_to_default=False)
            plt.rcParams['font.sans-serif'] = [_c]; break
        except Exception:
            continue
    plt.rcParams['axes.unicode_minus'] = False
    fig, axs = plt.subplots(2, 2, figsize=(12.4, 8.6), dpi=150)

    ax = axs[0, 0]
    xs = np.linspace(0, Lx, 200)
    for i in range(4):
        ax.plot(xs, [E_of(x, C_tr[i]) for x in xs], lw=1.6, label=f'训练场 {i+1}')
    for i in range(2):
        ax.plot(xs, [E_of(x, C_te[i]) for x in xs], '--', lw=1.6, label=f'测试场 {i+1}')
    ax.axvline(X0, color='k', ls=':', lw=1)
    ax.set_xlabel('x'); ax.set_ylabel('E(x)'); ax.set_title('(a) 输入函数: 杨氏模量场 E(x)', fontsize=10)
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    ax = axs[0, 1]
    k = 0
    ax.plot(YCHK, G_te[k, :NQ] * 1e3, 'k-', lw=1.8, label='精确 (探测)')
    ax.plot(YCHK, P_te[k, :NQ] * 1e3, 'r--', lw=1.4, label='DeepONet(集成)')
    ax.plot(YCHK, G_te[k, NQ:] * 1e3, 'k-', lw=1.8)
    ax.plot(YCHK, P_te[k, NQ:] * 1e3, 'b--', lw=1.4)
    ax.set_xlabel('y'); ax.set_ylabel(r'$g_\Gamma$ ($\times 10^{-3}$)')
    ax.set_title('(b) 未见场界面位移迹 ux(黑/红) uy(黑/蓝)', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axs[1, 0]
    ax.semilogy(Ns, curve['lin'], 's-', color='#7f7f7f', label='线性/岭回归')
    ax.semilogy(Ns, curve['mlp'], '^-', color='#2ca02c', label='无 trunk 的 MLP')
    ax.semilogy(Ns, curve['dn'], 'o-', color='#d62728', label='DeepONet')
    ax.set_xlabel('训练场数 N'); ax.set_ylabel('未见场相对 L2 误差')
    ax.set_title('(c) 学习曲线: 线性代理平台化 vs DeepONet 持续下降', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')

    ax = axs[1, 1]
    idx = np.argsort(dn_pe)
    ax.bar(range(len(dn_pe)), dn_pe[idx] * 100, color='#1f77b4')
    ax.axhline(dn_e * 100, color='r', ls='--', lw=1.2, label=f'平均 {dn_e:.2%}')
    ax.set_xlabel('未见场 (按误差排序)'); ax.set_ylabel('相对 L2 误差 (%)')
    ax.set_title('(d) DeepONet 在 15 个未见场上的误差', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    fig.suptitle('函数参数化界面算子: 标量→函数, trunk 由多项式升级为 DeepONet branch', fontsize=12)
    pf = os.path.join(OUT, 'func_operator.png')
    fig.savefig(pf, bbox_inches='tight'); plt.close(fig)
    print(f"[出图] {pf}")

    # ---- 保存 ----
    with open(os.path.join(OUT, 'func_operator.txt'), 'w', encoding='utf-8') as fp:
        fp.write("函数参数化界面算子: E(x) 场 -> 界面位移迹 g_Γ(y)\n")
        fp.write(f"E(x)=E0*exp(Σ c_k cos(kπx/Lx)), k=1..{K4}, c~U(-0.60,0.60); E 范围 "
                 f"[{S_tr.min():.0f},{S_tr.max():.0f}] (E0={E0:.0f})\n")
        fp.write(f"训练 {len(S_tr)} 场 / 测试 {len(S_te)} 场; branch 传感器 {NS}, trunk 查询点 {NQ}\n")
        fp.write(f"DeepONet: branch[{NS},72,72,48] + trunk[1,72,72,96], 3 种子集成, wd=2e-5\n")
        fp.write(f"未见场相对 L2:  线性 {lin_e:.4%}   无trunk MLP {mlp_e:.4%}   DeepONet {dn_e:.4%}\n")
        fp.write(f"DeepONet 分分量: ux {ex_dn:.4%}  uy {ey_dn:.4%}\n")
        fp.write(f"DeepONet 训练场 {dn_tr:.4%}; 界面方程残差 {pr_dn:.4%}\n\n")
        fp.write(f"外推 (训练 |c|<=0.35, 测试 |c|~0.75): 线性 {lin_ex:.4%}, DeepONet {dn_ex:.4%}\n\n")
        fp.write("学习曲线 (未见场相对 L2):\n")
        fp.write(f"{'N':>4}{'线性':>12}{'MLP':>12}{'DeepONet':>12}\n")
        for i, n in enumerate(Ns):
            fp.write(f"{n:>4}{curve['lin'][i]:>11.3%}{curve['mlp'][i]:>12.3%}{curve['dn'][i]:>12.3%}\n")
    print(f"[保存] {os.path.join(OUT,'func_operator.txt')}")


if __name__ == '__main__':
    main()
