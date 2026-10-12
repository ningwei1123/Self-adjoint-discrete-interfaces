#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 6 of the manuscript: three-dimensional verification, four load cases.
============================================================================

Entry point for Figure 6.  The drawing code lives in `scripts/figs_paper.py` and
`scripts/figs_fields.py` so that all nine figures share one style; this file fixes which
figure is produced and where it is written.

Run from the repository root:

    python scripts/figure_6.py

Output: results/figs/  (PNG at 300 dpi, plus PDF and TIFF masters)
"""
import _paths  # noqa: F401  (adds src/ to the path)
import os
import sys
import warnings

warnings.filterwarnings('ignore')

# importable whether called from the repository root or from scripts/
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from figs_fields import _3d_all, fig06   # noqa: E402


def main():
    """Draw Figure 6."""
    cache = _3d_all()
    fig06(cache)


if __name__ == '__main__':
    main()
