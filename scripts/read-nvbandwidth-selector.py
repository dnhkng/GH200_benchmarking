#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        return 2
    path = Path(sys.argv[1])
    key = sys.argv[2]
    if not path.exists():
        return 1
    data = json.loads(path.read_text())
    value = data.get("resolved", {}).get(key)
    if not value:
        return 1
    print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
