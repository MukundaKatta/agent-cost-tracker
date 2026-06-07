"""Test package for agent_cost_tracker.

This project uses a ``src/`` layout, so the package is not importable from a
bare checkout unless it is installed (``pip install -e .``) or ``src`` is on the
import path. To keep the suite runnable with the standard-library runner alone::

    python3 -m unittest discover -s tests

we prepend the sibling ``src`` directory to ``sys.path`` here. When the package
is already installed, the installed copy still wins because it is found first on
the path; this fallback only matters for an uninstalled checkout.
"""

from __future__ import annotations

import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)
