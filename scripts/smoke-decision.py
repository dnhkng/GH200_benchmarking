#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path


def triad(path: Path) -> float | None:
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    m = re.search(r"^Triad:\s+([\d.]+)", text, re.MULTILINE)
    if m:
        return float(m.group(1)) / 1000.0
    m = re.search(r"^Triad\s+([\d.]+)", text, re.MULTILINE | re.IGNORECASE)
    if m:
        value = float(m.group(1))
        return value / 1000.0 if value > 1e5 else value
    return None


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: smoke-decision.py <run_dir> <node0> <node1>", file=sys.stderr)
        return 2
    run_dir = Path(sys.argv[1])
    node0 = sys.argv[2]
    node1 = sys.argv[3]
    base = run_dir / "smoke-results"
    gpu0 = triad(base / "babelstream_gpu0_quick.txt")
    gpu1 = triad(base / "babelstream_gpu1_quick.txt")
    n0 = triad(base / f"stream_node{node0}_quick.txt")
    n1 = triad(base / f"stream_node{node1}_quick.txt")
    reasons: list[str] = []

    print(f"GPU0 HBM TRIAD: {gpu0} GB/s")
    print(f"GPU1 HBM TRIAD: {gpu1} GB/s")
    print(f"Node{node0} DDR TRIAD: {n0} GB/s")
    print(f"Node{node1} DDR TRIAD: {n1} GB/s")
    print()

    if gpu0 is not None and gpu0 < 3000:
        reasons.append(f"HBM TRIAD too low on GPU0: {gpu0}")
    if gpu1 is not None and gpu1 < 3000:
        reasons.append(f"HBM TRIAD too low on GPU1: {gpu1}")
    if n0 is not None and (n0 < 380 or n0 > 600):
        reasons.append(f"Grace TRIAD outside expected range on node {node0}: {n0}")
    if n1 is not None and (n1 < 380 or n1 > 600):
        reasons.append(f"Grace TRIAD outside expected range on node {node1}: {n1}")

    if reasons:
        print("FIX FIRST:")
        for reason in reasons:
            print(f"  - {reason}")
    elif all(v is None for v in (gpu0, gpu1, n0, n1)):
        print("FIX FIRST:")
        print("  - No smoke measurements parsed; tools may be missing or failed.")
    else:
        print("PROCEED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
