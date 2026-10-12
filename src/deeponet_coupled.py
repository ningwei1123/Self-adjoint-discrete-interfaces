# -*- coding: utf-8 -*-
"""
DeepONet 子域代理 接入耦合求解 (端到端加速验证)
================================================
把训练好的 DeepONet 子域代理当作 **右子域的 Dirichlet-to-Neumann 算子**:
    t_N(g,h) = A g + λ0          (g=界面位移迹, h=端部载荷迹, A/λ0 由代理探测得到)
左子域 FEM 的界面 DtN 由一次矩阵分解 + 26 次回代探测得到:
    t_L(g)   = C_L g
界面平衡:  t_L(g) = t_N(g,h)  ->  (C_L - A) g = λ0  (26x26 小系统)
对比: 精确耦合解 (mortar) 的界面位移; 并计时 (子域 FEM 求解 vs 代理评估)。
"""
import os, math, time
import numpy as np
import stress_recovery2d as R
import interface_methods as IM
import coupled_solver as CS

Lx, Ly, X0 = R.Lx, R.Ly, R.X0
E, nu = 2100.0, 0.30
G = E / (2 * (1 + nu))
D00 = E / (1 - nu * nu); D01 = nu * D00
D2 = np.array([[D00, D01, 0.0], [D01, D00, 0.0], [0.0, 0.0, G]])
NX, NY = 12, 12
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))


# ---------------- 载入代理 ----------------
class Surrogate:
    def __init__(self, path):
        z = np.load(path)
        self.m = int(z['m'])
        self.mx, self.sx = z['mx'], z['sx']
        self.my, self.sy = z['my'], z['sy']
        self.Wb = [z[f'Wb{i}'] for i in range(20) if f'Wb{i}' in z]
        self.bb = [z[f'bb{i}'] for i in range(20) if f'bb{i}' in z]
        self.Wt = [z[f'Wt{i}'] for i in range(20) if f'Wt{i}' in z]
        self.bt = [z[f'bt{i}'] for i in range(20) if f'bt{i}' in z]
        self.ys = np.linspace(0, Ly, self.m)
        self.tr = ((self.ys - self.ys.mean()) / (self.ys.std() + 1e-9)).reshape(-1, 1)

    def _mlp(self, W, b, X):
        a = X
        for i in range(len(W) - 1):
            a = np.tanh(a @ W[i] + b[i])
        return a @ W[-1] + b[-1]

    def __call__(self, Xraw):
        """Xraw (N, 3m+1) -> λ (N, m, 2)。"""
        Xn = (Xraw - self.mx) / self.sx
        b = self._mlp(self.Wb, self.bb, Xn)               # (N,p)
        t = self._mlp(self.Wt, self.bt, self.tr)          # (K,2p)
        p = b.shape[1]
        lx = b @ t[:, :p].T; ly = b @ t[:, p:].T
        lx = lx * self.sy[0] + self.my[0]; ly = ly * self.sy[1] + self.my[1]
        return np.stack([lx, ly], -1)


# ---------------- 左子域 FEM + 界面 DtN ----------------
def assemble_K(nodes, elems):
    ndof = 2 * len(nodes); K = np.zeros((ndof, ndof))
    gp = [-1.0 / math.sqrt(3), 1.0 / math.sqrt(3)]
    for e in elems:
        xy = nodes[e]
        for gx in gp:
            for gy in gp:
                N, dxi, det = R._q4(gx, gy)
                J = np.array([[dxi @ xy[:, 0], dxi @ xy[:, 1]], [det @ xy[:, 0], det @ xy[:, 1]]])
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


def traction_interface(nodes, elems, u, iface_nodes, nx, ny, nrm_sign):
    """界面 x=X0 上面力 λ = nrm_sign*(σxx, σxy), 按 iface_nodes 顺序返回 (M,2)。"""
    lam = {}
    for j in range(ny):
        e = elems[j * nx + (nx - 1)]        # 最右一列单元
        xy = nodes[e]
        # 该单元右边界节点 (i=nx) 为 e[1](ξ=1,η=-1) 与 e[2](ξ=1,η=+1)
        for nat, node in [((1.0, -1.0), e[1]), ((1.0, 1.0), e[2])]:
            _, dxi, det = R._q4(*nat)
            J = np.array([[dxi @ xy[:, 0], dxi @ xy[:, 1]], [det @ xy[:, 0], det @ xy[:, 1]]])
            iJ = np.linalg.inv(J)
            dNdx = iJ[0, 0] * dxi + iJ[0, 1] * det
            dNdy = iJ[1, 0] * dxi + iJ[1, 1] * det
            ux = u[2 * e]; uy = u[2 * e + 1]
            exx = dNdx @ ux; eyy = dNdy @ uy; exy = 0.5 * (dNdy @ ux + dNdx @ uy)
            sig = D2 @ np.array([exx, eyy, 2 * exy])
            lam[node] = (nrm_sign * sig[0], nrm_sign * sig[2])
    out = []
    for node in iface_nodes:
        out.append(lam.get(node, (0.0, 0.0)))
    return np.array(out)


def left_dtn(nodes, elems, K, iface_nodes, nx, ny):
    ndof = 2 * len(nodes)
    leftn = [k for k in range(len(nodes)) if abs(nodes[k, 0]) < 1e-9]
    fix = set(2 * k + c for k in leftn for c in range(2))
    ifd = [2 * k + c for k in iface_nodes for c in range(2)]
    free = [d for d in range(ndof) if d not in fix and d not in ifd]
    Kff = K[np.ix_(free, free)]
    Kfi = K[np.ix_(free, ifd)]
    m2 = len(ifd)
    C = np.zeros((m2, m2))
    for k in range(m2):
        gblk = np.zeros(m2); gblk[k] = 1.0          # 块序 [ux(m), uy(m)]
        uvec = np.zeros(m2)
        for i in range(m2 // 2):
            uvec[2 * i] = gblk[i]                    # ux
            uvec[2 * i + 1] = gblk[m2 // 2 + i]      # uy
        u = np.zeros(ndof)
        u[ifd] = uvec
        u[free] = np.linalg.solve(Kff, -Kfi @ uvec)
        lam = traction_interface(nodes, elems, u, iface_nodes, nx, ny, -1.0)   # (m,2)
        C[:, k] = np.concatenate([lam[:, 0], lam[:, 1]])
    return C, iface_nodes


def main():
    sur = Surrogate(os.path.join(OUT, 'deeponet_params.npz'))
    m = sur.m
    ys = sur.ys
    # 端部载荷迹 h (x=Lx 的 ux)
    hx = np.array([R.right_disp(y) for y in ys])

    # ---- 代理探测: λ(g,h) = A g + λ0 ----
    def sur_lam(gx, gy):
        Xin = np.concatenate([gx, gy, hx, [nu]]).reshape(1, -1)
        L = sur(Xin)[0]                          # (m,2)
        return np.concatenate([L[:, 0], L[:, 1]])   # 块序 [λx(m), λy(m)]
    g0 = np.zeros(m); lam0 = sur_lam(g0, g0)
    A = np.zeros((2 * m, 2 * m))
    eps = 0.01                       # 扰动幅值取在代理训练量级内 (线性算子 -> 除 eps 得 DtN)
    for k in range(2 * m):
        gx = np.zeros(m); gy = np.zeros(m)
        if k < m: gx[k] = eps
        else: gy[k - m] = eps
        A[:, k] = (sur_lam(gx, gy) - lam0) / eps

    # ---- 左子域 FEM 界面 DtN ----
    nodes, elems = R.build_mesh(NX, NY, 0.0, X0, 0.0, Ly)
    K = assemble_K(nodes, elems)
    iface_nodes = sorted([k for k in range(len(nodes)) if abs(nodes[k, 0] - X0) < 1e-9],
                         key=lambda k: nodes[k, 1])
    C_L, _ = left_dtn(nodes, elems, K, iface_nodes, NX, NY)

    # ---- 界面平衡 (C_L - A) g = λ0  (符号按物理校准) ----
    best = None
    for sgn in (+1.0, -1.0):
        try:
            g = np.linalg.solve(C_L + sgn * A, sgn * lam0)
        except np.linalg.LinAlgError:
            continue
        resid = np.linalg.norm((C_L + sgn * A) @ g - sgn * lam0)
        if best is None or resid < best[0]:
            best = (resid, sgn, g)
    resid, sgn, g = best
    g_iface = np.stack([g[:m], g[m:]], -1)   # 块序 -> (m,2)

    # ---- 精确耦合解 (mortar) 的界面位移 (插值到传感器 y) ----
    sm = IM.solve_mortar(rho=1e6)
    mif = sorted(sm['interf'], key=lambda k: sm['fnodes'][k][1])
    gy_m = np.array([sm['fnodes'][k][1] for k in mif])
    gxx = np.array([sm['ufem'][2 * k] for k in mif])
    gyy = np.array([sm['ufem'][2 * k + 1] for k in mif])
    gex = np.stack([np.interp(ys, gy_m, gxx), np.interp(ys, gy_m, gyy)], -1)   # (m,2)
    err = np.hypot((g_iface - gex)[:, 0], (g_iface - gex)[:, 1]) / (np.linalg.norm(gex, axis=1) + 1e-30)
    rel_iface = float(np.sqrt(np.mean((g_iface - gex) ** 2)) / np.sqrt(np.mean(gex ** 2)))
    print(f"界面离散度 m={m}, 平衡残差={resid:.2e} (符号 sgn={sgn:+.0f})")
    print(f"界面位移 (代理耦合 vs 精确 mortar): 相对 L2 = {rel_iface:.3%}  最大逐点 = {err.max():.3%}")

    gflat_ex = np.concatenate([gex[:, 0], gex[:, 1]])
    tL = C_L @ gflat_ex; tN = A @ gflat_ex + lam0
    print(f"算子一致性 @ 精确解: ||t_L - t_N||/||t_N|| = {np.linalg.norm(tL-tN)/np.linalg.norm(tN):.3%}")
    print(f"  ||C_L||={np.linalg.norm(C_L):.1f}  ||A||={np.linalg.norm(A):.1f}  ||λ0||={np.linalg.norm(lam0):.1f}")

    # ---- 计时 ----
    t0 = time.time()
    for _ in range(20):
        sm2 = IM.solve_mortar(rho=1e6)
    dt_exact = (time.time() - t0) / 20
    Xin = np.concatenate([gex[:, 0], gex[:, 1], hx, [nu]]).reshape(1, -1)
    t0 = time.time()
    for _ in range(500):
        sur(Xin)
    dt_sur = (time.time() - t0) / 500
    t0 = time.time()
    for _ in range(500):
        np.linalg.solve(C_L + sgn * A, sgn * lam0)
    dt_solve = (time.time() - t0) / 500
    dt_proxy = dt_sur + dt_solve
    print(f"端到端单次: 精确耦合(含子域组装/求解) {dt_exact*1e3:.2f} ms  vs  "
          f"代理(评估+26x26解) {dt_proxy*1e3:.3f} ms  (加速 {dt_exact/dt_proxy:.0f}×)")
    print(f"  (其中: 代理前向 {dt_sur*1e3:.3f} ms + 界面小系统解 {dt_solve*1e3:.3f} ms)")

    with open(os.path.join(OUT, 'deeponet_coupled.txt'), 'w', encoding='utf-8') as fp:
        fp.write("DeepONet 子域代理接入耦合求解\n")
        fp.write(f"界面传感器 m={m}, 平衡残差={resid:.3e}\n")
        fp.write(f"界面位移相对 L2 (代理 vs 精确 mortar) = {rel_iface:.4%}, 最大逐点 = {err.max():.4%}\n")
        fp.write(f"端到端: 精确 {dt_exact*1e3:.3f} ms vs 代理 {dt_proxy*1e3:.4f} ms, 加速 {dt_exact/dt_proxy:.0f}x\n")
    print(f"[保存] {os.path.join(OUT, 'deeponet_coupled.txt')}")


if __name__ == '__main__':
    main()
