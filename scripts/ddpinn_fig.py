# -*- coding: utf-8 -*-
"""函数参数下 XPINN/CPINN 对比: 出图"""
import _paths  # noqa: F401  (adds src/ to the path)
import os, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

# the output directory can be relocated with RESULTS_DIR; the default is
# results/ beside this file
OUT = os.path.join(os.environ.get('RESULTS_DIR', os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))), 'results')
for _c in ['Microsoft YaHei', 'SimHei', 'STSong', 'Arial Unicode MS']:
    try:
        font_manager.findfont(_c, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [_c]; break
    except Exception:
        continue
plt.rcParams['axes.unicode_minus'] = False

data_dir = os.environ.get('DATA_DIR', os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))
d = np.load(os.path.join(data_dir, 'ddpinn_budget_data.npz'))
xg, yg, yi = d['xg'], d['yg'], d['yi']
# 本文方法水平线 (来自 _run_pinn.log)
OUR_U, OUR_G, OUR_T = 0.00641, 0.00402, 0.36
NAME = {'single': '单域 PINN', 'cpinn': 'CPINN(通量连续)', 'xpinn': 'XPINN(残差连续)'}
COL = {'single': '#2ca02c', 'cpinn': '#1f77b4', 'xpinn': '#d62728'}
MK = {'single': 'o-', 'cpinn': 's-', 'xpinn': '^-'}

fig, axs = plt.subplots(2, 2, figsize=(12.6, 8.6), dpi=150)

ax = axs[0, 0]
for v in ['single', 'cpinn', 'xpinn']:
    k = f'rec_0_{v}'
    if k in d.files:
        r = d[k]
        ax.semilogy(r[:, 0], 100 * r[:, 1], MK[v], color=COL[v], label=NAME[v], ms=5)
ax.axhline(100 * OUR_U, color='k', ls='--', lw=2.0, label=f'本文 (0.64%, {OUR_T:.1f}s)')
ax.set_xlabel('Adam 训练步数'); ax.set_ylabel('全场位移相对 L2 误差 (%)')
ax.set_title('(a) 全场误差 vs 训练预算', fontsize=11)
ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')

ax = axs[0, 1]
for v in ['single', 'cpinn', 'xpinn']:
    k = f'rec_0_{v}'
    if k in d.files:
        r = d[k]
        ax.semilogy(r[:, 0], 100 * r[:, 2], MK[v], color=COL[v], label=NAME[v], ms=5)
ax.axhline(100 * OUR_G, color='k', ls='--', lw=2.0, label='本文 (0.40%)')
ax.set_xlabel('Adam 训练步数'); ax.set_ylabel('界面位移相对 L2 误差 (%)')
ax.set_title('(b) 界面位移误差 vs 训练预算', fontsize=11)
ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')

ax = axs[1, 0]
lab, eu, eg = ['本文\n(FEM+弱形式DtN)'], [100 * OUR_U], [100 * OUR_G]
for v in ['single', 'cpinn', 'xpinn']:
    k = f'rec_0_{v}'
    if k in d.files:
        r = d[k]; lab.append(NAME[v]); eu.append(100 * r[-1, 1]); eg.append(100 * r[-1, 2])
xx = np.arange(len(lab)); bw = 0.36
ax.bar(xx - bw / 2, eu, bw, label='全场 L2', color='#4c72b0')
ax.bar(xx + bw / 2, eg, bw, label='界面 L2', color='#c44e52')
for i in range(len(lab)):
    ax.text(xx[i] - bw / 2, eu[i], f'{eu[i]:.2f}', ha='center', va='bottom', fontsize=8)
    ax.text(xx[i] + bw / 2, eg[i], f'{eg[i]:.2f}', ha='center', va='bottom', fontsize=8)
ax.set_xticks(xx); ax.set_xticklabels(lab, fontsize=8)
ax.set_ylabel('相对 L2 误差 (%)'); ax.set_yscale('log')
ax.set_title('(c) 同一训练预算 (16000 步) 下的终值', fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y', which='both')

ax = axs[1, 1]
TIMES = {'single': 347.1, 'cpinn': 873.9, 'xpinn': 1007.6}
pts = [(OUR_T, 100 * OUR_U, 'k', 's', '本文 FEM+弱形式DtN')]
for v in ['single', 'cpinn', 'xpinn']:
    k = f'rec_0_{v}'
    if k in d.files:
        pts.append((TIMES[v], 100 * d[k][-1, 1], COL[v], 'o', NAME[v]))
for t_, e_, c_, m_, l_ in pts:
    ax.scatter([t_], [e_], s=130, color=c_, marker=m_, label=l_, zorder=3)
    ax.annotate(f'{e_:.2f}%', (t_, e_), textcoords='offset points', xytext=(6, 6), fontsize=8)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('训练/求解耗时 (s)'); ax.set_ylabel('全场 L2 误差 (%)')
ax.set_title('(d) 精度-成本: 本文在左下角', fontsize=11)
ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')

fig.suptitle('函数参数 E(x) 设定: 区域分解 PINN 与本文方法定量对比', fontsize=12)
pf = os.path.join(OUT, 'ddpinn_budget.png')
fig.savefig(pf, bbox_inches='tight'); plt.close(fig)
print('[出图]', pf)
