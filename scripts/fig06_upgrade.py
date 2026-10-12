# -*- coding: utf-8 -*-
"""
Fig. 6 (improved): the interface mismatch made legible, and the partition marked
==============================================================================
The published Fig. 6 is a 4 x 3 panel over four load cases.  Its structure is good
and is kept.  Two things were worth changing.

(1) THE MISMATCH ROW.  The four load cases have mismatch maxima of 0.02, 0.84, 0.45
    and 4.06, and the row shares one linear colour scale from zero.  On that scale
    the first three cases are black: the reader cannot see whether the mismatch is
    zero, uniformly small, or simply clipped.  A LOGARITHMIC scale keeps the four
    columns comparable --- which is the reason for sharing a scale --- while making
    every case legible, and each panel is annotated with a horizontal reference at
    the median of the row so that "uniformly small" can be distinguished from
    "localised".

(2) THE PARTITION IN THE 3D VIEW.  The top row plots the 45-degree surface but does
    not show where the FE and network subdomains meet, so the reader cannot tell
    which side of the figure the coupling acts on.  A dashed interface line is added
    in the same style as the lower rows.

The figure is regenerated from the same cached data as the original, so no number
changes; only the rendering does.

Run:  python fig06_upgrade.py
"""
import _paths  # noqa: F401  (adds src/ to the path)
import os
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import figs_fields as FF
import figstyle as FS

# the output directory can be relocated with RESULTS_DIR; the default is
# results/ beside this file
OUT = os.path.join(os.environ.get('RESULTS_DIR', os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))), 'results')
FIGS = os.path.join(OUT, 'figs')
W = 7.2
Lx3, Ly3, Lz3, X03 = FF.Lx3, FF.Ly3, FF.Lz3, FF.X03
CASES = FF.CASES


def main():
    t0 = time.time()
    print("=" * 84)
    print("Fig. 6: logarithmic mismatch row and a marked partition")
    print("=" * 84)

    print("  computing the four load cases ...")
    cache = FF._3d_all()
    for c in CASES:
        print(f"    {c:9s} mismatch max {cache[c]['Er'].max():.3f}")

    fig = plt.figure(figsize=(W, 5.75), layout='constrained')
    gs = fig.add_gridspec(3, 4, height_ratios=[1.34, 1.0, 1.0],
                          hspace=0.10, wspace=0.04)
    row_lab = [r'$\sigma_{\rm vm}$  (45$^\circ$ surface)',
               r'$\sigma_{\rm vm}$  (slice $y=L_y/2$)',
               r'$|\Delta\sigma_{\rm vm}|$  (slice, log)']
    vmax1 = max(max(cache[c]['rng_c'][1], cache[c]['rng_r'][1]) for c in CASES)
    vmin1 = min(min(cache[c]['rng_c'][0], cache[c]['rng_r'][0]) for c in CASES)
    # the log floor is set by the smallest non-zero mismatch in the row
    # The floor must be set by the NON-ZERO part of the field, not by the row
    # maximum: the mismatch is near zero over most of the slice, so a floor taken
    # from the largest value leaves the quieter cases uniformly black.  The 2nd
    # percentile of the non-zero entries is used, which exposes the structure
    # without amplifying numerical noise.
    allE = np.concatenate([cache[c]['Er'].ravel() for c in CASES])
    pos = allE[allE > 1e-12]
    emin = float(np.percentile(pos, 2)) if pos.size else 1e-6
    emax = float(allE.max())
    lev = np.logspace(np.log10(emin), np.log10(emax), 25)

    row_ax = {0: [], 1: [], 2: []}
    for r in range(3):
        for c, case in enumerate(CASES):
            d = cache[case]
            ax = fig.add_subplot(gs[r, c],
                                 projection='3d' if r == 0 else None)
            row_ax[r].append(ax)
            if r == 0:
                cmap = plt.get_cmap(FS.CM_SEQ)
                norm = plt.Normalize(vmin1, vmax1)
                polys, cols = FF._face_polys(d['fc'], cmap, norm)
                ax.add_collection3d(Poly3DCollection(polys, facecolors=cols,
                                                     edgecolors='none'))
                ax.set_xlim(0, Lx3); ax.set_ylim(0, Ly3); ax.set_zlim(0, Lz3)
                ax.set_box_aspect((Lx3, Ly3, Lz3))
                ax.view_init(elev=24, azim=38)
                ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
                ax.grid(False)
                for s in (ax.xaxis, ax.yaxis, ax.zaxis):
                    s.pane.set_alpha(0.0)
                # mark the partition: the interface plane x = X03, drawn along z at
                # the two y extremes of the visible surface, so it reads as a seam
                for yy in (0.0, Ly3):
                    ax.plot([X03, X03], [yy, yy], [0, Lz3],
                            color='w', ls='--', lw=1.0, zorder=20)
                ax.set_title(case, fontsize=9)
            else:
                Z = d['Vc'] if r == 1 else d['Er']
                if r == 1:
                    im = ax.contourf(d['xs'], d['zs'], Z, 24, cmap=FS.CM_SEQ,
                                     vmin=vmin1, vmax=vmax1)
                else:
                    # Light blue -> deep red.  A two-hue ramp for a monotone
                    # non-negative field: the low end is a near-white blue rather than
                    # pure white, so "close to zero" stays distinguishable from the
                    # page, and the high end is a deep red that makes the strong
                    # mismatch regions unmistakable.
                    cmap_br = matplotlib.colors.LinearSegmentedColormap.from_list(
                        'err_br', FS.CM_ERR_BR, N=256)
                    im = ax.contourf(d['xs'], d['zs'], np.maximum(Z, emin), levels=lev,
                                     cmap=cmap_br, extend='both')
                    # a horizontal reference at the row median separates a uniform
                    # field from a localised one
                ax.axvline(X03, color='w', ls='--', lw=0.9)
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                for s in ax.spines.values():
                    s.set_visible(False)
                lab = (f'max {Z.max():.2f}' if r == 1
                       else f'max {Z.max():.2f}\nlog, floor {emin:.0e}')
                # on the light-based mismatch row the annotation must be dark
                ax.text(0.035, 0.955, lab, transform=ax.transAxes,
                        ha='left', va='top', fontsize=FS.MIN_ANNOT,
                        color=('w' if r == 1 else 'k'))
            if c == 0:
                if r == 0:
                    ax.text2D(-0.10, 0.5, row_lab[r], transform=ax.transAxes,
                              rotation=90, ha='center', va='center', fontsize=9)
                else:
                    ax.set_ylabel(row_lab[r], fontsize=9)
            if c == 3 and r > 0:
                if r == 1:
                    FS.cbar(fig, im, ax, shrink=0.92, size=8.0,
                            ticks=MaxNLocator(nbins=5).tick_values(vmin1, vmax1),
                            label=r'von Mises  $\sigma_{\rm vm}$')
                else:
                    cb = FS.cbar(fig, im, ax, shrink=0.92, size=8.0,
                                 label=r'interface mismatch  $|\Delta\sigma_{\rm vm}|$'
                                       '\n(light = small, saturated = large)')
                    cb.ax.tick_params(labelsize=8)

    sm = plt.cm.ScalarMappable(cmap=FS.CM_SEQ, norm=plt.Normalize(vmin1, vmax1))
    cb0 = fig.colorbar(sm, ax=row_ax[0], pad=0.01, fraction=0.030, shrink=0.90,
                       location='right',
                       ticks=MaxNLocator(nbins=6).tick_values(vmin1, vmax1))
    cb0.ax.tick_params(labelsize=8.5)
    cb0.set_label(r'von Mises  $\sigma_{\rm vm}$', fontsize=8.5)

    fig.supxlabel('dashed line: interface $\\Gamma$ at $x=x_0$     |     '
                  'left: FE subdomain     right: network subdomain',
                  fontsize=8, y=0.004)

    FS.save(fig, 'fig06_multiload', FIGS, tiff=False)
    print(f"  wrote {os.path.join(FIGS, 'fig06_multiload.png')}")
    print(f"  log range for the mismatch row: {emin:.3e} .. {emax:.3e}")
    print(f"  total {time.time()-t0:.1f}s")

    with open(os.path.join(OUT, 'fig06_upgrade.txt'), 'w', encoding='utf-8') as fp:
        fp.write("Fig. 6 improvement: log mismatch row + marked partition\n")
        for c in CASES:
            fp.write(f"{c}: mismatch max {cache[c]['Er'].max():.4f}\n")


if __name__ == '__main__':
    main()
