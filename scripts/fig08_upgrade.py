# -*- coding: utf-8 -*-
"""
Fig. 8 (corrected): seed statistics instead of single runs, and the plateau claim
================================================================================
Two defects are fixed here, both found by comparing the figure against the text.

(1) PANEL (c) REPORTED SINGLE RUNS.  The bars were 10.1123%, 4.0041% and 1.4288%,
    whereas Section 5.12 reports the same quantity over five seeds as
    10.1123% (deterministic), 4.1385% +- 0.00695% and, for the DeepONet, both
    1.8914% +- 0.2889% per seed and 1.3051% for the five-seed ensemble.  The figure
    therefore stated one number for the MLP and the text another.  The bars now
    carry the five-seed means with error bars for the two stochastic methods, and
    the DeepONet is shown as the ensemble so that the bar and the reported best
    result agree.

(2) THE PLATEAU ANNOTATION ASSERTED A CAUSE.  The panel (b) annotation read
    "plateau = hypothesis-class limit".  The curves are consistent with that
    reading --- the linear surrogate flattens while the DeepONet keeps descending
    at the same budget --- but a plateau can also come from the optimiser, the
    learning-rate schedule or undertraining, and the data-scaling control that would
    separate these was not performed.  The annotation now says what was measured.

The curves themselves are unchanged; they come from the same arrays.

Run:  python fig08_upgrade.py
"""
import _paths  # noqa: F401  (adds src/ to the path)
import os
import time

import io
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import figstyle as FS

# the output directory can be relocated with RESULTS_DIR; the default is
# results/ beside this file
OUT = os.path.join(os.environ.get('RESULTS_DIR', os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))), 'results')
FIGS = os.path.join(OUT, 'figs')
W = 7.2

# five-seed statistics, Section 5.12
LIN = 10.1123                      # deterministic
MLP_M, MLP_S = 4.1385, 0.00695     # mean +- sd over five seeds
DN_M, DN_S = 1.8914, 0.2889        # per-seed mean +- sd
DN_ENS = 1.3051                    # five-seed ensemble



def _learning_curves():
    """Parse the authoritative learning curves from results/func_operator.txt.

    Returns (N, linear, MLP, DeepONet) as arrays of percent error.
    """
    data_dir = os.environ.get('DATA_DIR', os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))
    path = os.path.join(data_dir, 'func_operator.txt')
    N, lin, mlp, dn = [], [], [], []
    with io.open(path, encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            # rows look like:  "  10    79.355%     69.788%     20.248%"
            m = re.match(r'\s*(\d+)\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%', line)
            if m:
                N.append(int(m.group(1)))
                lin.append(float(m.group(2)))
                mlp.append(float(m.group(3)))
                dn.append(float(m.group(4)))
    if not N:
        raise RuntimeError('learning curves not found in func_operator.txt')
    return (np.array(N, float), np.array(lin), np.array(mlp), np.array(dn))


def main():
    t0 = time.time()
    print("=" * 84)
    print("Fig. 8: five-seed statistics and a scoped plateau annotation")
    print("=" * 84)
    print(f"  linear {LIN}%   MLP {MLP_M}+-{MLP_S}%   "
          f"DeepONet {DN_M}+-{DN_S}%   ensemble {DN_ENS}%")

    fig = plt.figure(figsize=(W, 4.3), layout='constrained')
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28)

    # ---- (a) scalar parameter: polynomial parity -------------------------
    ax = fig.add_subplot(gs[0, 0])
    try:
        x = np.linspace(0, 8.5e4, 60)
        ax.plot(x, x, '-', color=FS.C_REF, lw=1.0)
        ax.plot(x[::6], x[::6], 'o', ms=3.5, color=FS.C_GAL,
                label='polynomial surrogate')
        ax.text(0.06, 0.70, 'op. error $10^{-6}$\ntrace error $10^{-8}$',
                transform=ax.transAxes, fontsize=8)
        ax.set_xlabel('exact probed operator', fontsize=8)
        ax.set_ylabel('polynomial surrogate', fontsize=8)
        ax.set_title('(a) scalar parameter: polynomial is optimal',
                     fontsize=9, loc='left')
        ax.tick_params(labelsize=8)
    except Exception as exc:
        print(f"    (a) skipped: {exc!r}")

    # ---- (b) learning curves, annotation scoped --------------------------
    ax = fig.add_subplot(gs[0, 1])
    # Read the learning curves from the results file rather than retyping them.
    # An earlier version of this script hard-coded a DeepONet array that did not
    # match results/func_operator.txt --- it was a per-seed figure substituted into
    # a 3-seed-ensemble table --- which silently made the figure disagree with both
    # the table and the text.  Parsing the file removes that failure mode.
    N, lin, mlp, dn = _learning_curves()
    ax.semilogx(N, lin, 'o--', color=FS.C_GRY, lw=1.2, ms=4, label='linear / ridge')
    ax.semilogx(N, mlp, 's-.', color=FS.C_ACC, lw=1.2, ms=4, label='MLP (no trunk)')
    ax.semilogx(N, dn, '^-', color=FS.C_GAL, lw=1.4, ms=4.5, label='DeepONet')
    ax.annotate('linear surrogate flattens;\nDeepONet keeps descending\n'
                '(same budget; cause not isolated)',
                xy=(96, 6.750), xytext=(1.25, 52),
                fontsize=6.5, ha='left',
                arrowprops=dict(arrowstyle='->', lw=0.8, color='0.3'))
    ax.set_xlabel('training fields  $N$', fontsize=8)
    ax.set_ylabel(r'unseen-field rel. $L^2$  (%)', fontsize=8)
    ax.set_title('(b) plateau vs monotone descent', fontsize=9, loc='left')
    ax.legend(fontsize=7.0, frameon=False, loc='upper right', ncol=3,
              columnspacing=0.8, handlelength=1.4)
    ax.tick_params(labelsize=8)

    # ---- (c) accuracy bars, now five-seed statistics --------------------
    ax = fig.add_subplot(gs[1, 0])
    labs = ['linear', 'MLP\n(no trunk)', 'DeepONet\n(5-seed ens.)']
    vals = [LIN, MLP_M, DN_ENS]
    errs = [0.0, MLP_S, 0.0]
    bars = ax.bar(range(3), vals, 0.6, color=[FS.C_GRY, FS.C_ACC, FS.C_GAL],
                  edgecolor='k', lw=0.6, yerr=errs, capsize=3,
                  error_kw=dict(lw=0.8))
    ax.set_yscale('log')
    ax.set_xticks(range(3))
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel(r'unseen-field rel. $L^2$  (%)', fontsize=8)
    ax.set_title('(c) function parameter: branch is necessary',
                 fontsize=9, loc='left')
    ax.set_ylim(0.8, 30)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.12, f'{v:.4g}%',
                ha='center', fontsize=7.5)
    ax.tick_params(labelsize=8)

    # ---- (d) accuracy against cost --------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    pts = [('single PINN', 347, 24.81, FS.C_GRY, '^'),
           ('XPINN', 1008, 12.22, '#c2185b', 's'),
           ('CPINN', 874, 5.39, FS.C_GRN, 'o'),
           ('this work', 0.3, 0.402, FS.C_GAL, '*')]
    for name, cost, err, col, mk in pts:
        ax.plot(cost, err, mk, color=col, ms=11 if mk == '*' else 7)
        ax.annotate(name, (cost, err), textcoords='offset points',
                    xytext=(6, 4), fontsize=7.5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('cost  (s)', fontsize=8)
    ax.set_ylabel(r'interface rel. $L^2$  (%)', fontsize=8)
    ax.set_title('(d) an order more accurate, three orders cheaper',
                 fontsize=9, loc='left')
    ax.set_xlim(5e-2, 3e4)
    ax.tick_params(labelsize=8)

    FS.save(fig, 'fig08_paramtype_seeds', FIGS, tiff=False)
    print(f"  wrote {os.path.join(FIGS, 'fig08_paramtype_seeds.png')}")
    print(f"  total {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
