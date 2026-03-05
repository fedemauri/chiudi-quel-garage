"""Root entry point for Cloud Functions.

Cloud Functions expects main.py at the source root.
Adds src/ to Python path so garage_monitor package is importable.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from garage_monitor.main import check_garage  # noqa: F401
