#!/usr/bin/env python3
"""Compatibility entry point for scripts/maintenance/audit_repository_hygiene.py."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.maintenance.audit_repository_hygiene import *
from scripts.maintenance.audit_repository_hygiene import main


if __name__ == "__main__":
    main()
