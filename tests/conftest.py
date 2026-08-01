"""Make the repository root importable from the tests.

The tests import first-party packages (``analysis``, ``runner``, ``data``).
``python -m pytest`` puts the working directory on ``sys.path`` and they resolve;
a bare ``pytest`` does not, and every test errors at collection. Inserting the
root here makes both invocations behave the same.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
