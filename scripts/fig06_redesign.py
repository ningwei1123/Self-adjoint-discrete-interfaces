# -*- coding: utf-8 -*-
"""
Fig. 6 (professional redesign): one colour logic for the whole figure
=====================================================================
The previous versions failed for a reason that was diagnosed only after three
attempts: the figure mixed two incompatible colour conventions.  The two von Mises
rows used viridis, whose ZERO is dark purple, and the mismatch row used a light
ramp.  Side by side, the dark magnitude panels and the pale error panel clash, and
no amount of tuning the error ramp can fix a clash that lives in the pairing.

Two further choices were wrong on convention grounds:

  * A diverging ramp (light blue to deep red) for the mismatch.  The mismatch is
    MONOTONE NON-NEGATIVE, so blue/red reads as a sign that does not exist; readers
    of this literature expect a diverging map only for signed fields.
  * A custom hand-tuned ramp.  Published figures in this area use the standard
    perceptually uniform or ColorBrewer maps; a hand-made one does not look like
    the literature even when it is legible.

The redesign therefore fixes the LOGIC rather than the hue:

  * every panel is light-backed, so zero is pale and intensity grows with the value
    --- the figure now reads consistently top to bottom;
  * the two magnitude rows share one cold sequential map (YlGnBu);
  * the error row uses the warm counterpart of the same family (YlOrRd), which is
    the usual convention for an error field: cold for the quantity, warm for its
    error, same lightness logic;
  * the error row keeps its LOGARITHMIC scale, because its four cases span two
    decades and a shared linear scale makes three of them blank.

Typography and spacing are also tightened: larger panels, fewer overlapping
annotations, a single labelled colourbar per row, and the interface drawn with a
thin contrasting halo so it survives a light background.

Run:  python fig06_redesign.py
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
W = 7.4
Lx3, Ly3, Lz3, X03 = FF.Lx3, FF.Ly3, FF.Lz3, FF.X03
CASES = FF.CASES

CM_MAG = 'YlGnBu'    # magnitude fields: cold, light at zero
CM_ERR = 'YlOrRd'    # error field: warm counterpart, light at zero


def interface_line(ax, colour='#222222'):
    """The interface, drawn dark with a thin white halo so it reads on pale maps."""
    for lw, c, z in ((3.0, 'white', 5), (1.6, colour, 6)):
        ax.axvline(X03, color=c, ls='--', lw=lw, zorder=z)


def main():
    t0 = time.time()
    print("=" * 84)
    print("Fig. 6 redesign: one colour logic, light-backed throughout")
    print("=" * 84)
    cache = FF._3d_all()

    vmax1 = max(max(cache[c]['rng_c'][1], cache[c]['rng_r'][1]) for c in CASES)
    vmin1 = 0.0
    allE = np.concatenate([cache[c]['Er'].ravel() for c in CASES])
    pos = allE[allE > 1e-12]
    emin = float(np.percentile(pos, 2)) if pos.size else 1e-6
    emax = float(allE.max())
    lev = np.logspace(np.log10(emin), np.log10(emax), 22)
    print(f"  magnitude range 0 .. {vmax1:.1f}   mismatch {emin:.2e} .. {emax:.2e}")

    fig = plt.figure(figsize=(W, 6.0), layout='constrained')
    gs = fig.add_gridspec(3, 4, height_ratios=[1.28, 1.0, 1.0],
                          hspace=0.13, wspace=0.05)
    row_lab = [r'$\sigma_{\rm vm}$   ($45^\circ$ surface)',
               r'$\sigma_{\rm vm}$   (slice $y=L_y/2$)',
               r'$|\Delta\sigma_{\rm vm}|$   (slice, log)']

    row_ax = {0: [], 1: [], 2: []}
    for r in range(3):
        for c, case in enumerate(CASES):
            d = cache[case]
            ax = fig.add_subplot(gs[r, c], projection='3d' if r == 0 else None)
            row_ax[r].append(ax)

            if r == 0:
                cmap = plt.get_cmap(CM_MAG)
                polys, cols = FF._face_polys(
                    d['fc'], cmap, plt.Normalize(vmin1, vmax1))
                ax.add_collection3d(Poly3DCollection(
                    polys, facecolors=cols, edgecolors='none', rasterized=True))
                ax.set_xlim(0, Lx3); ax.set_ylim(0, Ly3); ax.set_zlim(0, Lz3)
                ax.set_box_aspect((Lx3, Ly3, Lz3))
                ax.view_init(elev=22, azim=35)
                ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
                ax.grid(False)
                for s in (ax.xaxis, ax.yaxis, ax.zaxis):
                    s.pane.set_alpha(0.0)
                # the partition, on the visible 45-degree surface
                for yy in (0.0, Ly3):
                    ax.plot([X03, X03], [yy, yy], [0, Lz3],
                            color='k', ls='--', lw=1.5, zorder=30)
                    ax.plot([X03, X03], [yy, yy], [0, Lz3],
                            color='w', ls='--', lw=0.7, zorder=31)
                ax.set_title(case, fontsize=10, pad=2)
            else:
                Z = d['Vc'] if r == 1 else d['Er']
                if r == 1:
                    # Denser levels: the shared scale reaches 39 while tension and
                    # torsion occupy only 5-11, so a coarse ramp leaves those panels
                    # featureless.  More contour bands restore the structure without
                    # changing the scale or the colormap.
                    lv1 = np.linspace(vmin1, vmax1, 30)
                    im = ax.contourf(d['xs'], d['zs'], Z, levels=lv1,
                                     cmap=CM_MAG, extend='max')
                    ax.contour(d['xs'], d['zs'], Z, levels=lv1[::3],
                               colors='k', linewidths=0.18, alpha=0.30)
                    interface_line(ax)
                    note, ncol = f'{Z.max():.1f}', '#111111'
                else:
                    im = ax.contourf(d['xs'], d['zs'], np.maximum(Z, emin),
                                     levels=lev, cmap=CM_ERR, extend='both')
                    interface_line(ax)
                    note, ncol = f'max {Z.max():.2f}', '#111111'
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                for s in ax.spines.values():
                    s.set_visible(False)
                ax.text(0.030, 0.960, note, transform=ax.transAxes,
                        ha='left', va='top', fontsize=9, color=ncol)

            if c == 0:
                if r == 0:
                    ax.text2D(-0.12, 0.5, row_lab[r], transform=ax.transAxes,
                              rotation=90, ha='center', va='center', fontsize=9)
                else:
                    ax.set_ylabel(row_lab[r], fontsize=9)
            if c == 3 and r > 0:
                if r == 1:
                    FS.cbar(fig, im, ax, shrink=0.94, size=8.5,
                            ticks=MaxNLocator(nbins=5).tick_values(vmin1, vmax1),
                            label=r'von Mises  $\sigma_{\rm vm}$')
                else:
                    cb = FS.cbar(fig, im, ax, shrink=0.94, size=8.5,
                                 label=r'interface mismatch  $|\Delta\sigma_{\rm vm}|$')
                    cb.ax.tick_params(labelsize=8.5)

    sm = plt.cm.ScalarMappable(cmap=CM_MAG, norm=plt.Normalize(vmin1, vmax1))
    cb0 = fig.colorbar(sm, ax=row_ax[0], pad=0.012, fraction=0.032,
                       shrink=0.92, location='right',
                       ticks=MaxNLocator(nbins=6).tick_values(vmin1, vmax1))
    cb0.ax.tick_params(labelsize=8.5)
    cb0.set_label(r'von Mises  $\sigma_{\rm vm}$', fontsize=9)

    fig.supxlabel('interface $\\Gamma$ at $x=x_0$   (dashed)      |      '
                  'left: FE subdomain      right: network subdomain',
                  fontsize=9, y=0.004)

    FS.save(fig, 'fig06_multiload', FIGS, tiff=False)
    print(f"  wrote {os.path.join(FIGS, 'fig06_multiload.png')}")
    print(f"  total {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
