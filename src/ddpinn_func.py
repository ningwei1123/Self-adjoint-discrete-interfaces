# -*- coding: utf-8 -*-
"""
函数参数设定下的区域分解 PINN 定量对比
=======================================
对象: 标准 XPINN (Jagtap & Karniadakis, 2020) / CPINN (conservative PINN)
      与 本文方法 (FEM 子域 + 迹空间弱形式 DtN 算子 + 界面小系统)

设定(与 func_operator.py 完全一致):
  参数 = 杨氏模量场 E(x) = E0*exp(Σ_{k=1..4} c_k cos(kπx/Lx))   (函数参数)
  域 [0,2]x[0,1], 界面 x=1;  x=0: u=0;  x=2: ux = D0*(1-t^2), uy 自由
  线弹性 2D 平面应力, ν=0.30

公平性:
  - 三个 PINN 变体用**相同**的 MLP 宽度/深度、配点数、Adam 步数;
  - 损失权重用**梯度范数自适应**(Wang et al. 2021, "understanding and
    mitigating gradient pathologies"), 避免人工调权带来的不公平;
  - 输出非量纲化 u = U0 * net(X), U0=D0;
  - 参考解 = 同一 E(x) 的 FEM 细网格解(48x24, 与审计中 A3 收敛一致)。
"""
import os, math, time
import numpy as np
import torch
import torch.nn as nn

import stress_recovery2d as R
import func_operator as F

torch.set_num_threads(8)
E0 = F.E0; NU = F.NU; Lx, Ly, X0 = F.Lx, F.Ly, F.X0
K4 = F.K4; D0 = 0.01; U0 = D0
# ---- 无量纲尺度: 使残差/面力/位移各项均 O(1) ----
SC_T = E0 * D0 / Lx          # 应力(面力)尺度
SC_R = E0 * D0 / (Lx * Lx)   # 应力梯度尺度
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
DEV = 'cpu'


# ------------------------------------------------------------------ E(x), D(x)
def E_t(x, c):
    s = 0.0
    for k in range(K4):
        s = s + c[k] * torch.cos((k + 1) * math.pi * x / Lx)
    return E0 * torch.exp(s)


def Dcoef_t(x, c):
    Ee = E_t(x, c)
    D00 = Ee / (1 - NU * NU); D01 = NU * D00; G = Ee / (2 * (1 + NU))
    return D00, D01, G


# ------------------------------------------------------------------ FEM 参考解
def solve_reference_field(c, nx=48, ny=24):
    nodes, elems = R.build_mesh(nx, ny, 0.0, Lx, 0.0, Ly)
    K = F.assemble_field(nodes, elems, c, NU)
    nn_ = len(nodes); ndof = 2 * nn_
    left = np.where(np.abs(nodes[:, 0]) < 1e-9)[0]
    right = np.where(np.abs(nodes[:, 0] - Lx) < 1e-9)[0]
    fix = {}; ufix = {}
    for k in left:
        fix[2 * k] = 0.0; fix[2 * k + 1] = 0.0
    for k in right:
        fix[2 * k] = R.right_disp(nodes[k, 1])
    fds = list(fix); fs = set(fds)
    free = [d for d in range(ndof) if d not in fs]
    u = np.zeros(ndof)
    u[free] = np.linalg.solve(K[np.ix_(free, free)],
                              -K[np.ix_(free, fds)] @ np.array([fix[d] for d in fds]))
    for d, v in fix.items():
        u[d] = v
    return nodes, elems, u, nx, ny


def fem_eval_grid(nodes, u, nx, ny, x0, x1, xg, yg):
    hx = (x1 - x0) / nx; hy = Ly / ny
    out = np.zeros((len(yg), len(xg), 2))
    for jj, y in enumerate(yg):
        fy = min(max(y / hy, 0.0), ny - 1e-12); j = int(fy); ty = fy - j
        for ii, x in enumerate(xg):
            fx = min(max((x - x0) / hx, 0.0), nx - 1e-12); i = int(fx); tx = fx - i
            n00 = j * (nx + 1) + i
            n10 = n00 + 1; n01 = n00 + nx + 1; n11 = n01 + 1
            for cc in range(2):
                out[jj, ii, cc] = ((1 - tx) * (1 - ty) * u[2 * n00 + cc] + tx * (1 - ty) * u[2 * n10 + cc]
                                   + (1 - tx) * ty * u[2 * n01 + cc] + tx * ty * u[2 * n11 + cc])
    return out


# ------------------------------------------------------------------ 本文方法
def our_solution(c, nx=F.NX, ny=F.NY):
    """迹空间弱形式 DtN 探测 -> 界面解 -> 回代重构两子域。"""
    t0 = time.time()
    M, q = F.probe_field(c, NU, nx, ny)
    g = np.linalg.solve(M, -q)
    gfun = F.gfun_from_coef(g)
    nL, eL, iL = F.build(0.0, X0, nx, ny)
    KL = F.assemble_field(nL, eL, c, NU)
    uL = F.solve_dirichlet(nL, eL, KL, iL, gfun, None)
    nR, eR, iR = F.build(X0, Lx, nx, ny)
    KR = F.assemble_field(nR, eR, c, NU)
    uR = F.solve_dirichlet(nR, eR, KR, iR, gfun, (Lx, R.right_disp))
    dt = time.time() - t0
    nparam = 2 * (2 * F.NT)          # 界面未知数 2*(2*nt)
    return dict(gfun=gfun, nL=nL, uL=uL, nR=nR, uR=uR, dt=dt, nparam=nparam,
                nx=nx, ny=ny, g=g)


# ------------------------------------------------------------------ MLP
class Net(nn.Module):
    """带**输入归一化层**(可微, 不影响 autograd 的链式求导)的 MLP。"""

    def __init__(self, width=50, depth=3, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        layers = []; d = 2
        for _ in range(depth):
            lin = nn.Linear(d, width)
            with torch.no_grad():
                lin.weight.copy_(torch.randn(width, d, generator=g) * math.sqrt(1.0 / d))
                lin.bias.zero_()
            layers += [lin, nn.Tanh()]
            d = width
        out = nn.Linear(d, 2)
        with torch.no_grad():
            out.weight.copy_(torch.randn(2, d, generator=g) * math.sqrt(1.0 / d) * 0.1)
            out.bias.zero_()
        layers += [out]
        self.body = nn.Sequential(*layers)
        self.register_buffer('ctr', torch.tensor([Lx / 2, Ly / 2.0]))
        self.register_buffer('scl', torch.tensor([Lx / 2, Ly / 2.0]))

    def forward(self, X):
        return self.body((X - self.ctr) / self.scl)


def make_net(width=50, depth=3, seed=0):
    return Net(width, depth, seed)


def nparam(net):
    return sum(p.numel() for p in net.parameters())


def make_ueval(nets, variant):
    def ueval(xg, yg):
        out = np.zeros((len(yg), len(xg), 2))
        for jj, y in enumerate(yg):
            P = torch.tensor(np.column_stack([xg, np.full_like(xg, y)]),
                             dtype=torch.get_default_dtype())
            with torch.no_grad():
                if variant == 'single':
                    V = U0 * nets[0](P)
                else:
                    w = torch.tensor(xg.reshape(-1, 1) < X0)
                    V = torch.where(w, U0 * nets[0](P), U0 * nets[1](P))
            out[jj, :, :] = V.numpy()
        return out
    return ueval


# ------------------------------------------------------------------ PINN 求解器
def pinn_solve(c, variant, steps=8000, ncol=300, width=50, depth=3, seed=0, log=False,
               weighting='fixed', cb=None, cb_every=0, resample=200):
    """
    variant: 'single' | 'cpinn' | 'xpinn'
    返回: 求值函数 u(x,y) (np), 参数量, 训练时间, 损失历史
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    def sample(ax0, ax1, n):
        X = np.zeros((n, 2))
        X[:, 0] = rng.uniform(ax0, ax1, n)
        X[:, 1] = rng.uniform(0.0, Ly, n)
        return X

    ybc = np.linspace(0.0, Ly, 41)
    XbL = np.column_stack([np.zeros_like(ybc), ybc])
    XbR = np.column_stack([np.full_like(ybc, Lx), ybc])
    Xi = np.column_stack([np.full_like(ybc, X0), ybc])
    gR = np.array([R.right_disp(y) for y in ybc])

    def T(A):
        return torch.tensor(A, dtype=torch.get_default_dtype())

    cT = [torch.tensor(float(v)) for v in c]

    if variant == 'single':
        net = make_net(width, depth, seed)
        Xc = T(sample(0.0, Lx, 2 * ncol))
        nets = [net]
        XL = Xc
    else:
        netL = make_net(width, depth, seed)
        netR = make_net(width, depth, seed + 100)
        XL = T(sample(0.0, X0, ncol))
        XR = T(sample(X0, Lx, ncol))
        nets = [netL, netR]

    XbL_t, XbR_t, Xi_t = T(XbL), T(XbR), T(Xi)
    gR_t = T(gR.reshape(-1, 1))
    params = [p for net in nets for p in net.parameters()]
    opt = torch.optim.Adam(params, lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)

    def resid(net, X):
        Xg = X.clone().requires_grad_(True)
        U = U0 * net(Xg)
        ux, uy = U[:, 0:1], U[:, 1:2]
        ones_u = torch.ones_like(ux)
        gx = torch.autograd.grad(ux, Xg, ones_u, create_graph=True)[0]
        gy = torch.autograd.grad(uy, Xg, ones_u, create_graph=True)[0]
        exx = gx[:, 0:1]; eyy = gy[:, 1:2]
        exy = 0.5 * (gx[:, 1:2] + gy[:, 0:1])
        D00, D01, G = Dcoef_t(Xg[:, 0:1], cT)   # 不可 detach! 否则丢失材料梯度项 D'(x)
        sxx = D00 * exx + D01 * eyy
        syy = D01 * exx + D00 * eyy
        sxy = G * exy
        o = torch.ones_like(sxx)
        dxx = torch.autograd.grad(sxx, Xg, o, create_graph=True)[0]
        dxy = torch.autograd.grad(sxy, Xg, o, create_graph=True)[0]
        dyy = torch.autograd.grad(syy, Xg, o, create_graph=True)[0]
        Rx = (dxx[:, 0:1] + dxy[:, 1:2]) / SC_R      # 无量纲残差
        Ry = (dxy[:, 0:1] + dyy[:, 1:2]) / SC_R
        return Rx, Ry, torch.cat([sxx, sxy], dim=1) / SC_T   # 无量纲面力

    gR_nd = gR_t / D0          # 无量纲目标

    # ---- 自由面 Neumann 点集 (FEM 中为自然边界条件, 强形式必须显式施加) ----
    def free_pts(x0, x1, nside):
        xx = np.linspace(x0, x1, nside)
        top = np.column_stack([xx, np.full_like(xx, Ly)])
        bot = np.column_stack([xx, np.zeros_like(xx)])
        return T(top), T(bot)

    XR_end = T(np.column_stack([np.full_like(ybc, Lx), ybc]))     # x=Lx
    if variant == 'single':
        XtopL, XbotL = free_pts(0.0, Lx, 61)
        XtopR = XbotR = None
    else:
        XtopL, XbotL = free_pts(0.0, X0, 31)
        XtopR, XbotR = free_pts(X0, Lx, 31)

    def stress_nd(net, X):
        Xg = X.clone().requires_grad_(True)
        U = U0 * net(Xg)
        ux, uy = U[:, 0:1], U[:, 1:2]; o = torch.ones_like(ux)
        gx = torch.autograd.grad(ux, Xg, o, create_graph=True)[0]
        gy = torch.autograd.grad(uy, Xg, o, create_graph=True)[0]
        exx = gx[:, 0:1]; eyy = gy[:, 1:2]; exy = 0.5 * (gx[:, 1:2] + gy[:, 0:1])
        D00, D01, G = Dcoef_t(Xg[:, 0:1], cT)
        return ((D00 * exx + D01 * eyy) / SC_T, G * exy / SC_T, (D01 * exx + D00 * eyy) / SC_T)

    def neumann(net, xt, xb, xr=None):
        """自由面: y=0,Ly 上 sigma_xy=0, sigma_yy=0; x=Lx 上 sigma_xy=0。"""
        _, sxy_t, syy_t = stress_nd(net, xt)
        _, sxy_b, syy_b = stress_nd(net, xb)
        L = ((sxy_t ** 2).mean() + (syy_t ** 2).mean()
             + (sxy_b ** 2).mean() + (syy_b ** 2).mean()) / 4.0
        if xr is not None:
            _, sxy_r, _ = stress_nd(net, xr)
            L = 0.5 * (L + (sxy_r ** 2).mean())
        return L

    def bc_losses(net_lo, net_hi):
        L_L = (net_lo(XbL_t) ** 2).mean()                       # u/D0 = 0
        L_R = ((net_hi(XbR_t)[:, 0:1] - gR_nd) ** 2).mean()     # ux/D0 = g/D0
        return L_L, L_R

    def interface_losses(netL, netR, use_flux):
        UR_1 = netL(Xi_t); UR_2 = netR(Xi_t)
        L_sol = ((UR_1 - UR_2) ** 2).mean()                     # (u_L-u_R)/D0
        if use_flux:                       # 通量(面力)连续: sigma_xx, sigma_xy 相等
            _, _, t1 = resid(netL, Xi_t)
            _, _, t2 = resid(netR, Xi_t)
            L_if = (((t1 - t2)) ** 2).mean()
        else:                              # XPINN: 残差连续 R_L = R_R
            r1x, r1y, _ = resid(netL, Xi_t)
            r2x, r2y, _ = resid(netR, Xi_t)
            L_if = ((r1x - r2x) ** 2 + (r1y - r2y) ** 2).mean()
        return L_sol, L_if

    w = [1.0] * 10
    hist = []
    t0 = time.time()
    for step in range(1, steps + 1):
        # ---- 配点重采样 (标准做法: 避免过拟合固定配点) ----
        if resample and step % resample == 0:
            if variant == 'single':
                XL = T(sample(0.0, Lx, 2 * ncol))
            else:
                XL = T(sample(0.0, X0, ncol))
                XR = T(sample(X0, Lx, ncol))
        opt.zero_grad(set_to_none=True)
        terms = []
        if variant == 'single':
            n = nets[0]
            rx, ry, _ = resid(n, XL)
            terms.append((rx ** 2 + ry ** 2).mean())
            lL, lR = bc_losses(n, n)
            terms += [lL, lR, neumann(n, XtopL, XbotL, XR_end)]
        else:
            nL, nR = nets
            rxL, ryL, _ = resid(nL, XL)
            rxR, ryR, _ = resid(nR, XR)
            terms.append((rxL ** 2 + ryL ** 2).mean())
            terms.append((rxR ** 2 + ryR ** 2).mean())
            lL, lR = bc_losses(nL, nR)
            terms += [lL, lR]
            lsol, lif = interface_losses(nL, nR, use_flux=(variant == 'cpinn'))
            terms += [lsol, lif,
                      neumann(nL, XtopL, XbotL, None),
                      neumann(nR, XtopR, XbotR, XR_end)]

        # ---- 权重: 无量纲化后各项已 O(1), 默认固定 1.0 ----
        if weighting == 'adaptive' and (step == 1 or (step % 300 == 0)):
            norms = []
            for t in terms:
                gi = torch.autograd.grad(t, params, retain_graph=True, allow_unused=True)
                norms.append(math.sqrt(sum(float((x ** 2).sum()) for x in gi if x is not None)))
            nmax = max(norms) + 1e-30
            w = [nmax / (sd + 1e-30) for sd in norms]
        loss = sum(w[i] * terms[i] for i in range(len(terms)))
        loss.backward()
        opt.step()
        sched.step()
        if cb is not None and cb_every and step % cb_every == 0:
            cb(step, make_ueval(nets, variant))
        if step % 100 == 0:
            hist.append((step, float(loss.detach()), [float(t.detach()) for t in terms]))
        if log and step % 2000 == 0:
            print(f"    [{variant}] step {step}: loss={float(loss.detach()):.3e} terms={[f'{float(t.detach()):.2e}' for t in terms]}")
    dt = time.time() - t0

    ueval = make_ueval(nets, variant)
    return ueval, nparam(nets[0]) * (1 if variant == 'single' else 2), dt, hist


# ------------------------------------------------------------------ 主程序
def main():
    print("=== 函数参数 E(x) 设定: XPINN / CPINN / 本文方法 定量对比 ===")
    FIELDS = [np.array([0.30, -0.20, 0.15, -0.25]),
              np.array([0.55, 0.35, -0.40, 0.25])]
    yg = np.linspace(0.0, Ly, 41); xg = np.linspace(0.0, Lx, 81)
    yi = np.linspace(0.0, Ly, 41)
    rows = []; SV = {}
    t_all = time.time()
    for fi, c in enumerate(FIELDS):
        print(f"\n--- 场 {fi+1}: c={np.round(c,2)} ---")
        nodes, elems, uref, nx, ny = solve_reference_field(c)
        Uref = fem_eval_grid(nodes, uref, nx, ny, 0.0, Lx, xg, yg)
        iref = np.where(np.abs(nodes[:, 0] - X0) < 1e-9)[0]
        iref = sorted(iref, key=lambda k: nodes[k, 1])
        gref_n = np.array([[uref[2 * k], uref[2 * k + 1]] for k in iref])
        gyref = nodes[iref, 1]
        gref = np.column_stack([np.interp(yi, gyref, gref_n[:, 0]),
                                np.interp(yi, gyref, gref_n[:, 1])])
        SV[f'Uref{fi}'] = Uref; SV[f'gref{fi}'] = gref; SV[f'c{fi}'] = c

        # ---- 本文方法 ----
        sol = our_solution(c)
        UL = fem_eval_grid(sol['nL'], sol['uL'], sol['nx'], sol['ny'], 0.0, X0, xg, yg)
        UR = fem_eval_grid(sol['nR'], sol['uR'], sol['nx'], sol['ny'], X0, Lx, xg, yg)
        Uour = np.where((xg.reshape(1, -1, 1) < X0), UL, UR)
        eU = np.linalg.norm(Uour - Uref) / np.linalg.norm(Uref)
        gpred = np.array([sol['gfun'](y) for y in yi])
        eG = np.linalg.norm(gpred - gref) / np.linalg.norm(gref)
        print(f"  [本文 FEM+弱形式DtN] 场L2={eU:.3%}  界面L2={eG:.3%}  "
              f"界面未知数={sol['nparam']}  耗时={sol['dt']:.2f}s")
        SV[f'Uour{fi}'] = Uour; SV[f'gour{fi}'] = gpred
        rows.append((fi, '本文(FEM+弱形式DtN算子)', eU, eG, sol['nparam'], sol['dt']))

        # ---- PINN 变体 ----
        for variant, name in [('single', '单域 PINN'), ('cpinn', 'CPINN(通量连续)'), ('xpinn', 'XPINN(残差连续)')]:
            ue, npar, dt, hist = pinn_solve(c, variant, steps=8000, ncol=300, log=True)
            Up = ue(xg, yg)
            eUp = np.linalg.norm(Up - Uref) / np.linalg.norm(Uref)
            gp = ue(yi * 0 + X0, yi)[:, 0, :]
            eGp = np.linalg.norm(gp - gref) / np.linalg.norm(gref)
            print(f"  [{name}] 场L2={eUp:.3%}  界面L2={eGp:.3%}  参数={npar}  训练={dt:.1f}s")
            SV[f'U_{variant}{fi}'] = Up; SV[f'g_{variant}{fi}'] = gp
            SV[f'hist_{variant}{fi}'] = np.array([[h[0], h[1]] for h in hist])
            rows.append((fi, name, eUp, eGp, npar, dt))

    # ---- 汇总 ----
    names = ['本文(FEM+弱形式DtN算子)', '单域 PINN', 'CPINN(通量连续)', 'XPINN(残差连续)']
    print("\n=== 汇总 (两场平均) ===")
    print(f"{'方法':<26}{'场L2':>10}{'界面L2':>10}{'参数/未知数':>12}{'耗时(s)':>10}")
    summ = []
    for name in names:
        sel = [r for r in rows if r[1] == name]
        eU = float(np.mean([r[2] for r in sel])); eG = float(np.mean([r[3] for r in sel]))
        npp = sel[0][4]; dt = float(np.mean([r[5] for r in sel]))
        summ.append((name, eU, eG, npp, dt))
        print(f"{name:<26}{eU:>9.3%}{eG:>10.3%}{npp:>12}{dt:>10.2f}")

    with open(os.path.join(OUT, 'ddpinn_func.txt'), 'w', encoding='utf-8') as fp:
        fp.write("函数参数 E(x) 设定: 区域分解 PINN 定量对比\n")
        fp.write("域 [0,2]x[0,1], 界面 x=1; E(x)=E0*exp(Σ_{k=1..4} c_k cos(kπx/Lx))\n")
        fp.write("PINN: MLP[2,32,32,32,2] tanh, 配点 300/子域, Adam 8000 步, 梯度范数自适应权重\n")
        fp.write("参考: 变系数 FEM 48x24\n\n")
        for fi, c in enumerate(FIELDS):
            fp.write(f"场{fi+1}: c={np.round(c,2).tolist()}\n")
            fp.write(f"  {'方法':<26}{'场L2':>10}{'界面L2':>10}{'参数':>10}{'耗时(s)':>10}\n")
            for r in [r for r in rows if r[0] == fi]:
                fp.write(f"  {r[1]:<26}{r[2]:>9.3%}{r[3]:>10.3%}{r[4]:>10}{r[5]:>10.2f}\n")
        fp.write("\n两场平均:\n")
        fp.write(f"  {'方法':<26}{'场L2':>10}{'界面L2':>10}{'参数':>10}{'耗时(s)':>10}\n")
        for name, eU, eG, npp, dt in summ:
            fp.write(f"  {name:<26}{eU:>9.3%}{eG:>10.3%}{npp:>10}{dt:>10.2f}\n")
    print(f"[保存] {os.path.join(OUT,'ddpinn_func.txt')}")

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
    fig, axs = plt.subplots(2, 3, figsize=(15.5, 8.8), dpi=150)
    sty = {'gour': ('-', 'r', 2.0, '本文 FEM+弱形式DtN'),
           'g_single': ('--', '#2ca02c', 1.4, '单域 PINN'),
           'g_cpinn': (':', '#1f77b4', 1.8, 'CPINN(通量连续)'),
           'g_xpinn': ('-.', '#d62728', 1.3, 'XPINN(残差连续)')}
    for fi in range(2):
        ax = axs[fi, 0]
        ax.plot(yi, SV[f'gref{fi}'][:, 0] * 1e3, 'k-', lw=2.2, label='参考(FEM 48x24)')
        for k, (mk, col, lw, lab) in sty.items():
            ax.plot(yi, SV[f'{k}{fi}'][:, 0] * 1e3, mk, color=col, lw=lw, label=lab)
        ax.set_xlabel('y'); ax.set_ylabel(r'$u_x|_\Gamma$ ($\times10^{-3}$)')
        ax.set_title(f'({chr(97+fi*3)}) 场{fi+1} 界面 $u_x$ 剖面', fontsize=10)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = axs[fi, 1]
        ax.plot(yi, SV[f'gref{fi}'][:, 1] * 1e3, 'k-', lw=2.2, label='参考')
        for k, (mk, col, lw, lab) in sty.items():
            ax.plot(yi, SV[f'{k}{fi}'][:, 1] * 1e3, mk, color=col, lw=lw, label=lab)
        ax.set_xlabel('y'); ax.set_ylabel(r'$u_y|_\Gamma$ ($\times10^{-3}$)')
        ax.set_title(f'({chr(98+fi*3)}) 场{fi+1} 界面 $u_y$ 剖面', fontsize=10)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = axs[fi, 2]
        err = np.linalg.norm(SV[f'U_xpinn{fi}'] - SV[f'Uref{fi}'], axis=2)
        ref = np.linalg.norm(SV[f'Uref{fi}'], axis=2) + 1e-30
        im = ax.contourf(xg, yg, 100 * err / np.linalg.norm(SV[f'Uref{fi}']) * math.sqrt(err.size) / math.sqrt(err.size),
                         20, cmap='hot')
        ax.axvline(X0, color='w', ls='--', lw=1.2)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title(f'({chr(99+fi*3)}) 场{fi+1} XPINN 全场误差', fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.85, label='|Δu| / ‖u‖ (#)')
    # 汇总柱状
    fig2, ax2 = plt.subplots(1, 1, figsize=(7.4, 4.4), dpi=150)
    lbl = [t[0] for t in summ]
    xx = np.arange(len(lbl)); bw = 0.36
    ax2.bar(xx - bw / 2, [100 * t[1] for t in summ], bw, label='全场 L2 (%)', color='#4c72b0')
    ax2.bar(xx + bw / 2, [100 * t[2] for t in summ], bw, label='界面 L2 (%)', color='#c44e52')
    for i, t in enumerate(summ):
        ax2.text(i - bw / 2, 100 * t[1], f'{100*t[1]:.2f}', ha='center', va='bottom', fontsize=8)
        ax2.text(i + bw / 2, 100 * t[2], f'{100*t[2]:.2f}', ha='center', va='bottom', fontsize=8)
    ax2.set_xticks(xx); ax2.set_xticklabels(lbl, fontsize=8, rotation=12)
    ax2.set_ylabel('相对 L2 误差 (%)'); ax2.set_yscale('log')
    ax2.set_title('函数参数 E(x) 下各方法误差 (两场平均)', fontsize=11)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3, axis='y', which='both')
    # 合并
    fig.suptitle('函数参数设定下的区域分解 PINN 定量对比', fontsize=12)
    pf = os.path.join(OUT, 'ddpinn_func.png')
    fig.savefig(pf, bbox_inches='tight'); plt.close(fig)
    pf2 = os.path.join(OUT, 'ddpinn_func_bar.png')
    fig2.savefig(pf2, bbox_inches='tight'); plt.close(fig2)
    np.savez(os.path.join(OUT, 'ddpinn_func_data.npz'), xg=xg, yg=yg, yi=yi, **SV)
    print(f"[出图] {pf} ; {pf2}")
    print(f"总用时 {time.time()-t_all:.0f}s")


if __name__ == '__main__':
    main()
