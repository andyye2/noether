#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import sys
from pathlib import Path

# Make tests/test_training_pipeline/dummy_project importable as top-level `dummy_project`:
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
