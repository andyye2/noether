#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AERO_CFD_SRC = _REPO_ROOT / "recipes" / "aero_cfd" / "src"

if str(_AERO_CFD_SRC) not in sys.path:
    sys.path.insert(0, str(_AERO_CFD_SRC))
