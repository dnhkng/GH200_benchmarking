#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REQUIRED = [
    "host_to_device_memcpy_ce",
    "device_to_host_memcpy_ce",
    "host_to_device_memcpy_sm",
    "device_to_host_memcpy_sm",
    "device_to_device_memcpy_read_ce",
    "device_to_device_memcpy_write_ce",
    "device_local_copy",
]


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: resolve-nvbandwidth-tests.py <nvbandwidth_list.txt> <out.json>", file=sys.stderr)
        return 2
    list_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    text = list_path.read_text(errors="ignore") if list_path.exists() else ""

    candidates: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        tokens = re.findall(r"[A-Za-z0-9_./:-]+", stripped)
        for token in tokens:
            norm = normalize(token)
            if any(req in norm or norm in req for req in REQUIRED):
                candidates[norm] = token
        m = re.match(r"^\s*(\d+)\s*[:.)-]?\s*(.+)$", line)
        if m:
            idx, desc = m.groups()
            candidates[normalize(desc)] = idx

    resolved: dict[str, str | None] = {}
    missing: list[str] = []
    for req in REQUIRED:
        req_norm = normalize(req)
        selector = None
        if req_norm in candidates:
            selector = candidates[req_norm]
        else:
            for cand_norm, cand_selector in candidates.items():
                if req_norm in cand_norm:
                    selector = cand_selector
                    break
        resolved[req] = selector
        if selector is None:
            missing.append(req)

    out_path.write_text(json.dumps({"resolved": resolved, "missing": missing}, indent=2))
    if missing:
        print("missing nvbandwidth tests: " + ", ".join(missing), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
