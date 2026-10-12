# -*- coding: utf-8 -*-
"""Put the repository root and the solver directory on the import path.

Imported for its side effect.  The figure library imports the solvers by their bare module
names, so src/ must be importable however the script is invoked.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

for _p in (_ROOT, _HERE, os.path.join(_ROOT, 'src')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
