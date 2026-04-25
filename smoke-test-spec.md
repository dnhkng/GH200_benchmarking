# Dual GH200 Benchmark — Smoke Test

## Purpose

Run this **before** the full benchmark sweep. It validates that:

1. All tools are installed correctly.
2. NUMA bindings work as expected.
3. GPU clocks/persistence are configured.
4. The output directory structure is correct.
5. The numbers are in the right ballpark before committing to a 2–3 hour run.

Target wall time: **~20 minutes**. If it takes longer than 30, something is wrong.

## Prerequisites — verify these first

If any of these fail, fix before proceeding:

```bash
# 1. Both GPUs visible
nvidia-smi -L | wc -l    # expect: 2

# 2. Topology shows SYS link between GPUs (not NVLink)
nvidia-smi topo -m       # expect: GPU0<->GPU1 = SYS

# 3. NUMA shows expected layout
numactl -H               # expect: 4 nodes (2 Grace DDR + 2 HBM exposed by driver)

# 4. CUDA toolkit functional
nvcc --version           # expect: cuda 13.0

# 5. No competing GPU work
nvidia-smi               # expect: 0 processes, 0 MiB used per GPU
```

If any check fails, **stop and fix** — running benchmarks on a misconfigured system wastes 3 hours and produces useless numbers.

## Output

```
~/gh200-bench/smoke-YYYYMMDD-HHMMSS/
├── system-info.txt
├── smoke-results/
│   ├── nvbandwidth_quick.txt
│   ├── stream_node0_quick.txt
│   ├── stream_node1_quick.txt
│   ├── babelstream_gpu0_quick.txt
│   ├── babelstream_gpu1_quick.txt
│   └── crosscheck.txt
├── smoke-summary.md
└── decision.txt          # "PROCEED" or "FIX FIRST: <reasons>"
```

The `decision.txt` is the most important output — it should say one word, plus any reasons.

## Pre-flight: capture state

```bash
RUN_DIR=~/gh200-bench/smoke-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RUN_DIR/smoke-results"
cd "$RUN_DIR"

{
  echo "=== Date ==="; date
  echo "=== uname ==="; uname -a
  echo "=== NUMA ==="; numactl -H
  echo "=== nvidia-smi -L ==="; nvidia-smi -L
  echo "=== nvidia-smi topo -m ==="; nvidia-smi topo -m
  echo "=== GPU clocks ==="; nvidia-smi -q -d CLOCK | head -50
  echo "=== persistence ==="; nvidia-smi -q | grep -i persistence | head -5
  echo "=== CUDA ==="; nvcc --version
  echo "=== driver ==="; cat /proc/driver/nvidia/version
  echo "=== hugepages ==="; grep -i huge /proc/meminfo
  echo "=== THP ==="; cat /sys/kernel/mm/transparent_hugepage/enabled
} > system-info.txt 2>&1
```

## 1. NVBandwidth — minimal test (3 min)

Just two specific tests, not the full suite. Confirms tool works and gives ballpark numbers.

```bash
cd ~/nvbandwidth   # adjust path if installed elsewhere

# Test 0: host_to_device_memcpy_ce — sanity check that C2C works
./nvbandwidth -t 0 > "$RUN_DIR/smoke-results/nvbandwidth_quick.txt" 2>&1

# Test 6: device_to_device_memcpy_read_ce — the SYS link headline number
./nvbandwidth -t 6 >> "$RUN_DIR/smoke-results/nvbandwidth_quick.txt" 2>&1

cd -
```

**Expected ranges** (sanity check, not strict pass/fail):

| Path | Expected | Red flag if |
|------|----------|-------------|
| Host→GPU0 (local C2C) | 300–400 GB/s | < 200 GB/s |
| Host→GPU1 (local C2C) | 300–400 GB/s | < 200 GB/s |
| GPU0→GPU1 read | 50–250 GB/s | < 30 GB/s or > 400 GB/s |
| GPU1→GPU0 read | 50–250 GB/s | < 30 GB/s or > 400 GB/s |

The cross-GPU range is wide because no one has published this exact configuration's numbers. If you see anything outside the red-flag bounds, document it loudly — that's a finding either way.

## 2. STREAM — single-config per node (5 min)

Just one thread count per socket, the recommended one from NVIDIA's tuning guide.

```bash
cd ~/STREAM   # adjust path

sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true

# Node 0 with 72 threads (NVIDIA's recommended config)
OMP_NUM_THREADS=72 OMP_PROC_BIND=spread \
numactl --cpunodebind=0 --membind=0 \
./stream_c.exe > "$RUN_DIR/smoke-results/stream_node0_quick.txt" 2>&1

sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true

# Node 1 with 72 threads
OMP_NUM_THREADS=72 OMP_PROC_BIND=spread \
numactl --cpunodebind=1 --membind=1 \
./stream_c.exe > "$RUN_DIR/smoke-results/stream_node1_quick.txt" 2>&1

cd -
```

**Expected**: TRIAD between **410–486 GB/s** per node, per NVIDIA's Grace tuning guide (480GB Grace config). If either node falls outside this, something is wrong (NUMA misbinding, hugepages disabled, CPU governor issue).

## 3. BabelStream on HBM — one config per GPU (3 min)

```bash
cd ~/BabelStream/build

CUDA_VISIBLE_DEVICES=0 ./cuda-stream -s 268435456 -n 10 \
  > "$RUN_DIR/smoke-results/babelstream_gpu0_quick.txt" 2>&1

CUDA_VISIBLE_DEVICES=1 ./cuda-stream -s 268435456 -n 10 \
  > "$RUN_DIR/smoke-results/babelstream_gpu1_quick.txt" 2>&1

cd -
```

**Expected**: TRIAD between **3300–3600 GB/s** per GPU. The published research number on this hardware is ~3.4 TB/s (85% of 4 TB/s theoretical). If either GPU lands below 3.0 TB/s, GPU clocks may not be locked or the kernel is wrong.

## 4. Cross-check — sanity sums (2 min)

A quick consistency check: NVBandwidth's host-to-device number for a given path should be in the same ballpark as the C2C theoretical (450 GB/s/dir). STREAM's TRIAD on local DDR should be ~50% of NVBandwidth's host-to-device on the same node (TRIAD does 3× the traffic of memcpy, so the ratio is expected).

```bash
{
  echo "=== Cross-check ==="
  echo
  echo "GPU0 HBM TRIAD (BabelStream):"
  grep -i "triad" "$RUN_DIR/smoke-results/babelstream_gpu0_quick.txt" | head -1
  echo
  echo "GPU1 HBM TRIAD (BabelStream):"
  grep -i "triad" "$RUN_DIR/smoke-results/babelstream_gpu1_quick.txt" | head -1
  echo
  echo "Node 0 DDR TRIAD (STREAM):"
  grep -A1 "^Triad" "$RUN_DIR/smoke-results/stream_node0_quick.txt" | head -2
  echo
  echo "Node 1 DDR TRIAD (STREAM):"
  grep -A1 "^Triad" "$RUN_DIR/smoke-results/stream_node1_quick.txt" | head -2
  echo
  echo "NVBandwidth headline numbers:"
  grep -A 30 "memcpy CE" "$RUN_DIR/smoke-results/nvbandwidth_quick.txt" | head -40
} > "$RUN_DIR/smoke-results/crosscheck.txt"
```

## 5. Generate decision.txt

Apply rules:

- HBM TRIAD < 3000 GB/s on either GPU → **FIX FIRST** (clocks not locked, or kernel issue)
- Grace TRIAD < 380 GB/s on either node → **FIX FIRST** (NUMA misbinding, hugepages, or governor)
- Grace TRIAD > 600 GB/s on either node → **FIX FIRST** (suspect: numbers too good, NUMA leak)
- Cross-GPU NVBandwidth < 30 GB/s → document loudly but proceed (this might be reality)
- Otherwise → **PROCEED**

```bash
python3 - <<'PY' > "$RUN_DIR/decision.txt"
import re, pathlib, sys

run_dir = pathlib.Path("$RUN_DIR".replace("$RUN_DIR", "$RUN_DIR"))
# (Claude Code: substitute the actual RUN_DIR path here, or use os.environ)

reasons = []

def get_triad(path):
    text = pathlib.Path(path).read_text(errors="ignore")
    # STREAM format: "Triad:    XXXXX.X    ..."
    m = re.search(r"^Triad:\s+([\d.]+)", text, re.MULTILINE)
    if m: return float(m.group(1))
    # BabelStream format: row with "Triad" then GB/s in column
    m = re.search(r"Triad\s+([\d.]+)", text)
    if m: return float(m.group(1))
    return None

base = run_dir / "smoke-results"
gpu0 = get_triad(base / "babelstream_gpu0_quick.txt")
gpu1 = get_triad(base / "babelstream_gpu1_quick.txt")
n0 = get_triad(base / "stream_node0_quick.txt")
n1 = get_triad(base / "stream_node1_quick.txt")

print(f"GPU0 HBM TRIAD: {gpu0} GB/s")
print(f"GPU1 HBM TRIAD: {gpu1} GB/s")
print(f"Node0 DDR TRIAD: {n0} GB/s")
print(f"Node1 DDR TRIAD: {n1} GB/s")
print()

if gpu0 is None or gpu1 is None:
    reasons.append("Could not parse BabelStream output")
elif gpu0 < 3000 or gpu1 < 3000:
    reasons.append(f"HBM TRIAD too low (GPU0={gpu0}, GPU1={gpu1}); check GPU clocks")

if n0 is None or n1 is None:
    reasons.append("Could not parse STREAM output")
elif n0 < 380 or n1 < 380:
    reasons.append(f"Grace TRIAD too low (n0={n0}, n1={n1}); check NUMA binding, THP, governor")
elif n0 > 600 or n1 > 600:
    reasons.append(f"Grace TRIAD suspiciously high (n0={n0}, n1={n1}); check NUMA leak")

if reasons:
    print("FIX FIRST:")
    for r in reasons:
        print(f"  - {r}")
else:
    print("PROCEED")
PY
```

(Note for Claude Code: the `$RUN_DIR` substitution above is a sketch — use proper Python with `os.environ['RUN_DIR']` or pass as argv.)

## 6. Smoke summary

Write a short markdown:

```bash
{
  echo "# Smoke Test Summary"
  echo
  echo "Run: $(date)"
  echo
  echo "## Decision"
  cat "$RUN_DIR/decision.txt"
  echo
  echo "## Files"
  ls -la "$RUN_DIR/smoke-results/"
  echo
  echo "## Cross-check excerpt"
  cat "$RUN_DIR/smoke-results/crosscheck.txt"
} > "$RUN_DIR/smoke-summary.md"

cat "$RUN_DIR/smoke-summary.md"
```

## What to do based on decision

- **PROCEED**: Everything is in expected ranges. Run the full benchmark spec.
- **FIX FIRST**: Read the reasons. Common fixes:
  - HBM low → `sudo nvidia-smi -pm 1; sudo nvidia-smi -lgc <max-clock>`
  - Grace low → check `cat /sys/kernel/mm/transparent_hugepage/enabled` is `[always]` or `[madvise]`; check `cpupower frequency-info` shows performance governor; verify `numactl -H` shows expected nodes.
  - Numbers parse-fail → look at the actual file output, formats may have changed between tool versions.

After fixing, re-run this smoke test (it's quick). Only when it says PROCEED should the full benchmark run.

## What this does NOT cover

This smoke test deliberately skips:
- The custom CUDA kernels (`memory_modes.cu`, `latency_probe.cu`, `sustained.cu`)
- The full NVBandwidth matrix
- Thread-count sweeps for STREAM
- BabelStream array-size sweeps
- mixbench (roofline)
- Plot generation

Those are all in the full spec, where they belong. The smoke test is just "does the rig produce sane numbers from the tools at all."
