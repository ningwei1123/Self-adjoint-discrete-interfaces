# -*- coding: utf-8 -*-
"""
函数参数 E(x) 设定下的对比: 误差 vs 训练预算
============================================
对单域 PINN / CPINN / XPINN 用**同一** MLP 结构、同一配点数、同一 Adam 预算,
在训练过程中记录「场位移 L2 误差 / 界面位移 L2 误差」, 得到误差-预算曲线;
与本文方法 (FEM 子域 + 迹空间弱形式 DtN 算子) 的水平线对比。
"""
import os, time, math
import numpy as np
import ddpinn_func as D

OUT = D.OUT
STEPS = 16000
CB = 1500
NCOL = 200
WIDTH = 40


def main():
    print("=== 函数参数 E(x) 下的误差-训练预算对比 ===", flush=True)
    FIELDS = [np.array([0.30, -0.20, 0.15, -0.25])]
    yg = np.linspace(0.0, D.Ly, 41); xg = np.linspace(0.0, D.Lx, 81)
    yi = np.linspace(0.0, D.Ly, 41)
    allrec = {}
    summary = []

    for fi, c in enumerate(FIELDS):
        print(f"\n--- 场 {fi+1}: c={np.round(c,2)} ---", flush=True)
        nodes, elems, uref, nR, nRy = D.solve_reference_field(c)
        Uref = D.fem_eval_grid(nodes, uref, nR, nRy, 0.0, D.Lx, xg, yg)
        iref = np.where(np.abs(nodes[:, 0] - D.X0) < 1e-9)[0]
        iref = sorted(iref, key=lambda k: nodes[k, 1])
        gn = np.array([[uref[2 * k], uref[2 * k + 1]] for k in iref])
        gref = np.column_stack([np.interp(yi, nodes[iref, 1], gn[:, 0]),
                                np.interp(yi, nodes[iref, 1], gn[:, 1])])

        # ---- 本文方法 ----
        t0 = time.time()
        sol = D.our_solution(c)
        UL = D.fem_eval_grid(sol['nL'], sol['uL'], sol['nx'], sol['ny'], 0.0, D.X0, xg, yg)
        UR = D.fem_eval_grid(sol['nR'], sol['uR'], sol['nx'], sol['ny'], D.X0, D.Lx, xg, yg)
        Uour = np.where(xg.reshape(1, -1, 1) < D.X0, UL, UR)
        eU_our = float(np.linalg.norm(Uour - Uref) / np.linalg.norm(Uref))
        gour = np.array([sol['gfun'](y) for y in yi])
        eG_our = float(np.linalg.norm(gour - gref) / np.linalg.norm(gref))
        t_our = time.time() - t0
        print(f"  [本文] 场L2={eU_our:.3%}  界面L2={eG_our:.3%}  用时={t_our:.2f}s", flush=True)
        summary.append((fi, '本文 FEM+弱形式DtN', eU_our, eG_our, sol['nparam'], t_our))

        for variant, name in [('single', '单域 PINN'), ('cpinn', 'CPINN'), ('xpinn', 'XPINN')]:
            rec = []

            def cb(step, ue, _rec=rec):
                Up = ue(xg, yg)
                eU = float(np.linalg.norm(Up - Uref) / np.linalg.norm(Uref))
                gp = ue(np.full_like(yi, D.X0), yi)[:, 0, :]
                eG = float(np.linalg.norm(gp - gref) / np.linalg.norm(gref))
                _rec.append((step, eU, eG))
                print(f"    [{name}] step={step:>6} 场L2={eU:>8.3%} 界面L2={eG:>8.3%}", flush=True)

            ue, npar, dt, hist = D.pinn_solve(c, variant, steps=STEPS, ncol=NCOL,
                                              width=WIDTH, cb=cb, cb_every=CB)
            allrec[(fi, variant)] = np.array(rec)
            summary.append((fi, name, rec[-1][1], rec[-1][2], npar, dt))
            print(f"  [{name}] 终值 场L2={rec[-1][1]:.3%} 界面L2={rec[-1][2]:.3%} "
                  f"参数={npar} 训练={dt:.0f}s", flush=True)

    # ---------------- 保存 ----------------
    with open(os.path.join(OUT, 'ddpinn_budget.txt'), 'w', encoding='utf-8') as fp:
        fp.write("函数参数 E(x): 误差 vs 训练预算\n")
        fp.write(f"PINN: MLP[2,{WIDTH},{WIDTH},{WIDTH},2] tanh+输入归一化, 配点 {NCOL}/子域,\n")
        fp.write(f"      Adam {STEPS} 步 + cosine 退火, 无量纲损失(权重=1), 每 {CB} 步记录\n")
        fp.write("参考: 变系数 FEM 48x24\n\n")
        for variant, name in [('single', '单域 PINN'), ('cpinn', 'CPINN'), ('xpinn', 'XPINN')]:
            for fi in range(len(FIELDS)):
                if (fi, variant) not in allrec:
                    continue
                fp.write(f"场{fi+1} {name}:\n")
                fp.write(f"  {'步数':>8}{'场L2':>12}{'界面L2':>12}\n")
                for st, eU, eG in allrec[(fi, variant)]:
                    fp.write(f"  {int(st):>8}{eU:>11.3%}{eG:>12.3%}\n")
        fp.write("\n终值汇总:\n")
        fp.write(f"  {'场':>3}{'方法':<22}{'场L2':>10}{'界面L2':>10}{'参数':>8}{'耗时(s)':>10}\n")
        for fi, name, eU, eG, np_, dt in summary:
            fp.write(f"  {fi+1:>3}{name:<22}{eU:>9.3%}{eG:>10.3%}{np_:>8}{dt:>10.1f}\n")
    print(f"[保存] {os.path.join(OUT,'ddpinn_budget.txt')}", flush=True)

    np.savez(os.path.join(OUT, 'ddpinn_budget_data.npz'),
             **{f'rec_{fi}_{v}': allrec[(fi, v)] for (fi, v) in allrec},
             xg=xg, yg=yg, yi=yi, Uref0=Uref)
    print("[保存] ddpinn_budget_data.npz", flush=True)
    print("完成.", flush=True)


if __name__ == '__main__':
    main()
