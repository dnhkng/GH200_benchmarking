# Dual GH200 Memory Bandwidth Benchmark Plan (v2)

## Context for Claude Code

This is a benchmarking spec for a personal dual-GH200 workstation. Goal is a complete, reproducible characterization of every memory path on the system, suitable for publishing as a blog post on dnhkng.github.io.

The system is **two GH200 Grace Hopper Superchips**, each consisting of:

- 1× Hopper H100 GPU (96 GB HBM3, ~4 TB/s peak)
- 1× Grace CPU (72× Arm Neoverse V2, 480 GB LPDDR5X, ~500 GB/s peak per socket)
- NVLink-C2C between Grace and its local Hopper (900 GB/s aggregate, 450 GB/s per direction)

The two superchips are connected via `SYS` topology (no NVLink between GPUs, no GPU P2P fast path — verified via `nvidia-smi topo -m`). Inter-GPU traffic crosses CPU/NUMA fabric.

NUMA layout (verify this with `numactl -H` before starting; node IDs may vary):
- NUMA 0: Grace CPU 0 (cores 0–71) + 480 GB LPDDR5X (let's call it `cpu_node_0`)
- NUMA 1: Grace CPU 1 (cores 72–143) + 480 GB LPDDR5X (`cpu_node_1`)
- HBM is exposed as additional NUMA nodes by the driver. On many GH200 configs these appear as nodes 2 and 3, but the driver-side numbering is not stable across reboots/driver versions. **Always re-detect.**

Software baseline:
- CUDA 13.0
- Driver 580.105.08
- Ubuntu (Arm64)

## Run directory setup

Set this once before discovery, install logging, or benchmark execution:

```bash
RUN_DIR="${RUN_DIR:-$HOME/gh200-bench/run-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$RUN_DIR/results" "$RUN_DIR/plots"
touch "$RUN_DIR/install-failures.txt"
echo "RUN_DIR=$RUN_DIR"
```

## NUMA discovery — do this once and record it

Before any benchmark, detect and record the actual NUMA-to-device mapping:

```bash
# Capture full NUMA layout
numactl -H > "$RUN_DIR/numa-topology.txt"

# For each GPU, find which CPU NUMA node its PCI function is attached to.
# nvidia-smi usually returns a full BDF like 00000009:01:00.0. Linux sysfs
# typically exposes the same address as 0009:01:00.0. Keep the bus/device/
# function intact; only normalize the domain width.
gpu_sysfs_path() {
  local gpu="$1"
  local bdf domain rest sysfs
  bdf=$(nvidia-smi -i "$gpu" --query-gpu=pci.bus_id --format=csv,noheader | tr -d ' ' | tr 'A-Z' 'a-z')
  domain=${bdf%%:*}
  rest=${bdf#*:}
  sysfs="/sys/bus/pci/devices/$(printf '%04x' "0x$domain"):$rest"
  if [ ! -e "$sysfs/numa_node" ]; then
    echo "ERROR: cannot find sysfs NUMA node for GPU $gpu at $sysfs" >&2
    return 1
  fi
  printf '%s\n' "$sysfs"
}

for i in 0 1; do
  sysfs=$(gpu_sysfs_path "$i") || node="unknown"
  if [ "${sysfs:-}" ]; then
    node=$(cat "$sysfs/numa_node")
  fi
  echo "GPU $i sysfs=${sysfs:-unknown} attached_cpu_numa_node=$node" | tee -a "$RUN_DIR/gpu-numa-map.txt"
done

# nvidia-smi topo also reports NUMA Affinity per GPU
nvidia-smi topo -mp >> "$RUN_DIR/gpu-numa-map.txt"
```

Record the mapping in `gpu-numa-map.txt` and **use those exact node numbers throughout**. The rest of this spec assumes:
- `GPU_0_NODE` = CPU NUMA node attached to GPU 0 (commonly 0)
- `GPU_1_NODE` = CPU NUMA node attached to GPU 1 (commonly 1)

Set these as shell variables at the top of the run script:

```bash
gpu_cpu_numa_node() {
  local gpu="$1"
  local bdf domain rest sysfs node
  bdf=$(nvidia-smi -i "$gpu" --query-gpu=pci.bus_id --format=csv,noheader | tr -d ' ' | tr 'A-Z' 'a-z')
  domain=${bdf%%:*}
  rest=${bdf#*:}
  sysfs="/sys/bus/pci/devices/$(printf '%04x' "0x$domain"):$rest"
  if [ ! -r "$sysfs/numa_node" ]; then
    echo "ERROR: cannot read NUMA node for GPU $gpu from $sysfs" >&2
    return 1
  fi
  node=$(cat "$sysfs/numa_node")
  if ! numactl -H | grep -q "^node $node "; then
    echo "ERROR: GPU $gpu reported CPU NUMA node $node, but numactl -H does not list it" >&2
    return 1
  fi
  printf '%s\n' "$node"
}

GPU_0_NODE=$(gpu_cpu_numa_node 0)
GPU_1_NODE=$(gpu_cpu_numa_node 1)
echo "GPU_0_NODE=$GPU_0_NODE GPU_1_NODE=$GPU_1_NODE"
```

If the sysfs lookup is fragile, fall back to manual entry — but record the source of the mapping in the output.

## Deliverables

The benchmark run should produce:

1. A `results/` directory with raw output from every tool, named `<tool>_<config>.txt`.
2. A `system-info.txt` capturing the full hardware/software state at run time.
3. A `numa-topology.txt` and `gpu-numa-map.txt` capturing NUMA discovery.
4. A consolidated `summary.md` with a results table per memory path, theoretical vs measured, and percentage of peak.
5. Plot scripts (matplotlib, save as PNG) for: STREAM TRIAD by thread count per node, NVBandwidth matrix as heatmap, latency vs transfer size per path, allocation-mode bandwidth comparison.
6. A `methodology.md` describing each test, what it measures, NUMA placement assumptions, and known caveats.

Everything goes in `~/gh200-bench/` with a timestamped subdir per run: `~/gh200-bench/run-YYYYMMDD-HHMMSS/`.

## Memory paths to characterize (full matrix, both directions)

Symmetric path table — both directions explicit, both GPUs explicit, both Grace sockets explicit. `Hi` = HBM on GPU `i`. `Li` = LPDDR on Grace `i`.

| ID | Source | Destination | Path through | Theoretical |
|----|--------|-------------|--------------|-------------|
| `H0->H0` | GPU0 HBM | GPU0 SMs | local HBM | ~4000 GB/s |
| `H1->H1` | GPU1 HBM | GPU1 SMs | local HBM | ~4000 GB/s |
| `L0->L0` | Grace0 LPDDR | Grace0 cores | local DDR | ~500 GB/s |
| `L1->L1` | Grace1 LPDDR | Grace1 cores | local DDR | ~500 GB/s |
| `L0->H0` | Grace0 LPDDR | GPU0 SMs | local C2C | 450 GB/s/dir |
| `H0->L0` | GPU0 HBM | Grace0 cores | local C2C | 450 GB/s/dir |
| `L1->H1` | Grace1 LPDDR | GPU1 SMs | local C2C | 450 GB/s/dir |
| `H1->L1` | GPU1 HBM | Grace1 cores | local C2C | 450 GB/s/dir |
| `L0->L1` | Grace0 LPDDR | Grace1 cores | inter-Grace fabric | varies |
| `L1->L0` | Grace1 LPDDR | Grace0 cores | inter-Grace fabric | varies |
| `L0->H1` | Grace0 LPDDR | GPU1 SMs | C2C0 + fabric | varies |
| `L1->H0` | Grace1 LPDDR | GPU0 SMs | C2C1 + fabric | varies |
| `H0->L1` | GPU0 HBM | Grace1 cores | C2C0 + fabric | varies |
| `H1->L0` | GPU1 HBM | Grace0 cores | C2C1 + fabric | varies |
| `H0->H1` | GPU0 HBM | GPU1 HBM | the SYS link, full path | varies |
| `H1->H0` | GPU1 HBM | GPU0 HBM | the SYS link, full path | varies |

The cross-socket numbers and direction asymmetry are the publishable findings — every entry above must be measured separately, both directions, and reported even if some values land in the same range.

## Tools to install

All open source. Install order matters because some need others' headers.

```bash
# 1. NVBandwidth — NVIDIA's official tool, the gold standard for inter-device bandwidth
git clone https://github.com/NVIDIA/nvbandwidth.git
cd nvbandwidth && ./debian_install.sh && cd ..

# 2. STREAM (CPU memory bandwidth — McCalpin's classic)
# Build from source with -O3 -mcpu=neoverse-v2 -fopenmp
git clone https://github.com/jeffhammond/STREAM.git
cd STREAM
make stream_c.exe CC=gcc CFLAGS="-O3 -mcpu=neoverse-v2 -fopenmp -DSTREAM_ARRAY_SIZE=400000000 -DNTIMES=20"
cd ..

# 3. BabelStream — successor to STREAM, supports CUDA targets
git clone https://github.com/UoB-HPC/BabelStream.git
cd BabelStream && cmake -Bbuild -H. -DMODEL=cuda -DCMAKE_CUDA_ARCHITECTURES=90 && cmake --build build
cd ..

# 4. mixbench — GPU compute/bandwidth roofline
git clone https://github.com/ekondis/mixbench.git
cd mixbench/mixbench-cuda && make && cd ../..

# 5. likwid (optional, for hardware counters)
sudo apt install likwid
```

If any install fails, log the failure to `install-failures.txt` and continue with what builds. Do not silently skip.

## Pre-flight: capture system state

Save to `system-info.txt`. This is reference material, not a benchmark.

```bash
{
  echo "=== Date ==="; date
  echo "=== uname ==="; uname -a
  echo "=== /proc/cpuinfo summary ==="; lscpu
  echo "=== NUMA topology ==="; numactl -H
  echo "=== nvidia-smi ==="; nvidia-smi
  echo "=== nvidia-smi topo -m ==="; nvidia-smi topo -m
  echo "=== nvidia-smi topo -mp (NUMA affinity) ==="; nvidia-smi topo -mp
  echo "=== GPU clocks ==="; nvidia-smi -q -d CLOCK
  echo "=== driver and CUDA ==="
  cat /proc/driver/nvidia/version
  nvcc --version
  echo "=== meminfo ==="; cat /proc/meminfo | head -30
  echo "=== HugePages ==="; grep -i huge /proc/meminfo
  echo "=== THP ==="; cat /sys/kernel/mm/transparent_hugepage/enabled
  echo "=== CPU governor ==="; cpupower frequency-info 2>/dev/null || echo "cpupower unavailable"
  echo "=== persistence mode ==="; nvidia-smi -q | grep -i persistence
} > "$RUN_DIR/system-info.txt" 2>&1
```

Pre-benchmark hardware setup:

- **GPU persistence mode on** (`sudo nvidia-smi -pm 1`)
- **GPU clocks locked to max** (`sudo nvidia-smi -lgc <max>`) — record the value used. Unlock at the end with `sudo nvidia-smi -rgc`.
- **CPU governor** (`sudo cpupower frequency-set -g performance` if available; record the previous governor and restore at end).
- **No competing workloads** — `nvidia-smi` should show 0 MiB used and no processes before each benchmark.
- **THP enabled** (`echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled` if not already; record original).

If any setup step fails, record the failure but continue. The point is documentation, not perfection.

## Benchmark suite

Run each in order. Each section produces files in `results/`. **Sleep 5 seconds between major sections** to ensure no concurrent GPU work.

### 1. NVBandwidth — full matrix with NUMA-bound host

NVBandwidth's own internal tests run within a single CUDA context with default host allocation. To distinguish local-C2C from cross-socket-host paths, **wrap the binary in numactl with explicit binding** for each host-memory test, and run it twice (once bound to each Grace socket).

```bash
cd nvbandwidth

# List available tests for documentation (this is the authoritative list)
./nvbandwidth -l > "$RUN_DIR/results/nvbandwidth_list.txt"

# Run the full default suite once (NVBandwidth's defaults)
./nvbandwidth -t all > "$RUN_DIR/results/nvbandwidth_all_default.txt" 2>&1

# Now run host-touching tests with explicit NUMA binding to each Grace socket.
# This produces "host on Grace0" and "host on Grace1" variants of every host
# test, which is what we need to label local-C2C vs cross-socket properly.

# Tests of interest — resolve from nvbandwidth_list.txt at run time. Do not
# assume numeric IDs are stable. Common semantic names as of recent versions:
#   host_to_device_memcpy_ce
#   device_to_host_memcpy_ce
#   host_to_device_memcpy_sm
#   device_to_host_memcpy_sm
#   device_to_device_memcpy_read_ce
#   device_to_device_memcpy_write_ce
#   device_local_copy
#
# CLAUDE CODE: implement a resolver before running this section:
#   1. Read nvbandwidth_list.txt.
#   2. Map each required semantic test name above to the installed test selector
#      accepted by `./nvbandwidth -t` (name or numeric ID, depending on version).
#   3. Write the resolved map to results/nvbandwidth_test_map.json.
#   4. If a required test is missing, append it to install-failures.txt and skip
#      only that semantic test. Do not create fake benchmark result files.

declare -A NVBW_TEST
# Example after resolver runs:
# NVBW_TEST[host_to_device_memcpy_ce]="host_to_device_memcpy_ce"
# NVBW_TEST[device_to_host_memcpy_ce]="device_to_host_memcpy_ce"

HOST_TESTS=(host_to_device_memcpy_ce device_to_host_memcpy_ce host_to_device_memcpy_sm device_to_host_memcpy_sm)
GPU_TESTS=(device_to_device_memcpy_read_ce device_to_device_memcpy_write_ce device_local_copy)

# Host tests — bound to Grace0
for t in "${HOST_TESTS[@]}"; do
  [ -n "${NVBW_TEST[$t]:-}" ] || continue
  numactl --cpunodebind=$GPU_0_NODE --membind=$GPU_0_NODE \
    ./nvbandwidth -t "${NVBW_TEST[$t]}" \
    > "$RUN_DIR/results/nvbandwidth_${t}_host_on_node${GPU_0_NODE}.txt" 2>&1
done

# Host tests — bound to Grace1
for t in "${HOST_TESTS[@]}"; do
  [ -n "${NVBW_TEST[$t]:-}" ] || continue
  numactl --cpunodebind=$GPU_1_NODE --membind=$GPU_1_NODE \
    ./nvbandwidth -t "${NVBW_TEST[$t]}" \
    > "$RUN_DIR/results/nvbandwidth_${t}_host_on_node${GPU_1_NODE}.txt" 2>&1
done

# GPU-to-GPU tests — host binding doesn't matter, run once
for t in "${GPU_TESTS[@]}"; do
  [ -n "${NVBW_TEST[$t]:-}" ] || continue
  ./nvbandwidth -t "${NVBW_TEST[$t]}" \
    > "$RUN_DIR/results/nvbandwidth_${t}.txt" 2>&1
done

cd -
```

Output filenames use the semantic test names, not unstable numeric IDs, so the analysis script doesn't need to guess. Both host-binding variants exist for every host-touching test that the installed NVBandwidth supports.

When parsing, the rule is:
- "host on node N" + "GPU N" cell = local C2C (e.g., `nvbandwidth_host_to_device_memcpy_ce_host_on_node0.txt` row 0, col 0)
- "host on node N" + "GPU M" cell (N ≠ M) = cross-socket via C2C+fabric

Analyzer contract: parse `nvbandwidth_all_default.txt` plus each semantic `nvbandwidth_<test>*.txt` file. Do not rely on a single `nvbandwidth_all.txt` file.

### 1A. NVBandwidth follow-up tests — updated v0.9 coverage

The installed NVBandwidth should be checked against upstream before the follow-up section. Record the commit/tag in `RESULTS.md`. As of this plan revision, upstream `main` is `v0.9` (`4a49bda`, 2026-04-08), so no source update is currently available beyond the built binary.

Run these extra tests after the primary matrix. They are not replacements for the main benchmark; they fill gaps that are useful on a dual-GH200 system:

```bash
cd nvbandwidth
mkdir -p "$RUN_DIR/results/additional"

# Machine-readable baseline for reproducibility.
./nvbandwidth -j -t all > "$RUN_DIR/results/additional/nvbandwidth_all_default.json" 2>&1

# Host/device bidirectional pressure, bound to each Grace socket.
EXTRA_HOST_TESTS=(
  host_to_device_bidirectional_memcpy_ce
  device_to_host_bidirectional_memcpy_ce
  host_to_device_bidirectional_memcpy_sm
  device_to_host_bidirectional_memcpy_sm
  all_to_host_memcpy_ce
  all_to_host_bidirectional_memcpy_ce
  host_to_all_memcpy_ce
  host_to_all_bidirectional_memcpy_ce
  all_to_host_memcpy_sm
  all_to_host_bidirectional_memcpy_sm
  host_to_all_memcpy_sm
  host_to_all_bidirectional_memcpy_sm
  host_device_latency_sm
)

for t in "${EXTRA_HOST_TESTS[@]}"; do
  [ -n "${NVBW_TEST[$t]:-}" ] || continue
  for node in "$GPU_0_NODE" "$GPU_1_NODE"; do
    numactl --cpunodebind=$node --membind=$node \
      ./nvbandwidth -t "${NVBW_TEST[$t]}" \
      > "$RUN_DIR/results/additional/nvbandwidth_${t}_host_on_node${node}.txt" 2>&1
  done
done

# Device-local and peer latency/traffic probes. Peer tests may be waived on SYS topology.
EXTRA_GPU_TESTS=(
  device_to_device_latency_sm
  all_to_one_write_ce
  all_to_one_read_ce
  one_to_all_write_ce
  one_to_all_read_ce
  all_to_one_write_sm
  all_to_one_read_sm
  one_to_all_write_sm
  one_to_all_read_sm
)

for t in "${EXTRA_GPU_TESTS[@]}"; do
  [ -n "${NVBW_TEST[$t]:-}" ] || continue
  ./nvbandwidth -t "${NVBW_TEST[$t]}" \
    > "$RUN_DIR/results/additional/nvbandwidth_${t}.txt" 2>&1
done

# Buffer-size sensitivity for host/device copies.
for size_mib in 16 64 512 2048; do
  for t in host_to_device_memcpy_ce device_to_host_memcpy_ce; do
    [ -n "${NVBW_TEST[$t]:-}" ] || continue
    for node in "$GPU_0_NODE" "$GPU_1_NODE"; do
      numactl --cpunodebind=$node --membind=$node \
        ./nvbandwidth -b "$size_mib" -t "${NVBW_TEST[$t]}" \
        > "$RUN_DIR/results/additional/nvbandwidth_${t}_b${size_mib}MiB_host_on_node${node}.txt" 2>&1
    done
  done
done

# Huge-page variant for host allocations. If it fails because huge pages are
# unavailable or require system configuration, record that in TODO.md.
for t in host_to_device_memcpy_ce device_to_host_memcpy_ce; do
  [ -n "${NVBW_TEST[$t]:-}" ] || continue
  for node in "$GPU_0_NODE" "$GPU_1_NODE"; do
    numactl --cpunodebind=$node --membind=$node \
      ./nvbandwidth -H -t "${NVBW_TEST[$t]}" \
      > "$RUN_DIR/results/additional/nvbandwidth_${t}_hugepages_host_on_node${node}.txt" 2>&1
  done
done

cd -
```

Also capture context around these runs:

```bash
nvidia-smi topo -m > "$RUN_DIR/results/additional/nvidia-smi-topo-m.txt" 2>&1
nvidia-smi topo -p2p n > "$RUN_DIR/results/additional/nvidia-smi-topo-p2p-n.txt" 2>&1 || true
nvidia-smi -q -d TEMPERATURE,POWER,CLOCK > "$RUN_DIR/results/additional/nvidia-smi-thermal-power-clock.txt" 2>&1
```

Summarize completion, waivers, and sudo-only blockers in `RESULTS.md` and `TODO.md`.

### 2. STREAM on Grace CPUs

NVIDIA's tuning guide expects 410–486 GB/s TRIAD per Grace socket (480 GB config) with 72 threads. Run a thread sweep per socket and a cross-socket worst-case measurement.

`drop_caches` removed from this section: STREAM allocates its own arrays and initializes them inside the timed program; the page cache is not the relevant confounder. The proper controls are NUMA binding, governor, THP, warmup runs (NTIMES=20 in the build, first iteration discarded by STREAM internally), and repeated measurements.

```bash
cd STREAM

# Per-socket thread sweep
for node in $GPU_0_NODE $GPU_1_NODE; do
  for threads in 8 16 32 48 64 72; do
    for run in 1 2 3; do
      OMP_NUM_THREADS=$threads OMP_PROC_BIND=spread \
      numactl --cpunodebind=$node --membind=$node \
        ./stream_c.exe > "$RUN_DIR/results/stream_node${node}_t${threads}_run${run}.txt" 2>&1
      sleep 1
    done
  done
done

# Cross-socket: cores on one node, memory on the other (both directions)
for run in 1 2 3; do
  OMP_NUM_THREADS=72 OMP_PROC_BIND=spread \
  numactl --cpunodebind=$GPU_0_NODE --membind=$GPU_1_NODE \
    ./stream_c.exe > "$RUN_DIR/results/stream_cross_n${GPU_0_NODE}cores_n${GPU_1_NODE}mem_run${run}.txt" 2>&1

  OMP_NUM_THREADS=72 OMP_PROC_BIND=spread \
  numactl --cpunodebind=$GPU_1_NODE --membind=$GPU_0_NODE \
    ./stream_c.exe > "$RUN_DIR/results/stream_cross_n${GPU_1_NODE}cores_n${GPU_0_NODE}mem_run${run}.txt" 2>&1
done

# Both sockets together (interleaved memory) — for reference, not the main measurement
for run in 1 2 3; do
  OMP_NUM_THREADS=144 OMP_PROC_BIND=spread \
  numactl --interleave=$GPU_0_NODE,$GPU_1_NODE \
    ./stream_c.exe > "$RUN_DIR/results/stream_both_interleave_run${run}.txt" 2>&1
done

cd -
```

Run each STREAM invocation 3 times and keep all three outputs (suffixed `_run1.txt`, `_run2.txt`, `_run3.txt`) — the analysis script reports median across runs to handle outlier variance.

Analyzer contract: STREAM parsers must match `stream_node<N>_t<T>_run<R>.txt`, `stream_cross_n<X>cores_n<Y>mem_run<R>.txt`, and `stream_both_interleave_run<R>.txt`. Do not use unsuffixed STREAM outputs because they are easy to overwrite.

### 3. BabelStream on HBM

```bash
cd BabelStream/build

for gpu in 0 1; do
  for run in 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu ./cuda-stream -s 268435456 -n 20 \
      > "$RUN_DIR/results/babelstream_gpu${gpu}_run${run}.txt" 2>&1
    sleep 1
  done
done

# Sweep array size to find where bandwidth saturates
for size in 16777216 67108864 268435456 1073741824; do
  CUDA_VISIBLE_DEVICES=0 ./cuda-stream -s $size -n 10 \
    > "$RUN_DIR/results/babelstream_gpu0_size${size}.txt" 2>&1
done

cd -
```

Expected: ~3.3–3.5 TB/s achieved (out of 4 TB/s theoretical) on each GPU.

Analyzer contract: report the median of `babelstream_gpu<N>_run<R>.txt` for each GPU. The size sweep is diagnostic only unless explicitly included in the final plots.

### 4. CUDA allocation-mode comparison (custom kernel) — NUMA-controlled

This is one of the centerpiece measurements. The original v1 plan listed `malloc`, `cudaMallocManaged`, `cudaMallocHost`, `cudaMalloc` as a clean comparison, but didn't control where the host-side allocation physically lives. On GH200 that's the central question — pinned/managed/pageable host memory must be allocated and first-touched on a known Grace node, otherwise the C2C path being measured is ambiguous.

The kernel below explicitly:
- Uses `numa_alloc_onnode()` from libnuma for `malloc`-equivalent host pages on a specific node.
- Uses `mmap()`/`numa_alloc_onnode()` + `cudaHostRegister()` for the NUMA-controlled pinned-host measurement. This is preferable to `cudaMallocHost()` because CUDA's pinned allocator does not expose a CPU NUMA-node selector, and already-pinned pages may not be movable with `mbind()`.
- Includes a separate `cudaMallocHost` diagnostic mode only if placement can be verified after allocation; otherwise records it as "placement_unverified" and excludes it from NUMA-labeled headline claims.
- For `cudaMallocManaged` on CUDA 13, uses `cudaMemLocationTypeHostNuma` with the Linux NUMA node ID, then first-touches and verifies/reports observed placement. If porting to older CUDA versions, fall back to CPU preference plus observed-placement reporting instead of claiming exact NUMA placement.
- Performs explicit first-touch on the chosen node before any GPU access.

```cuda
// memory_modes.cu
//
// Compile: nvcc -O3 -arch=sm_90 memory_modes.cu -o memory_modes
//
// Usage:
//   ./memory_modes <gpu_id> <host_node> <buffer_bytes> <iterations>
// Example:
//   ./memory_modes 0 0 4294967296 10
//
// For each allocation mode, the program:
//   1. Allocates a buffer of <buffer_bytes> bytes
//   2. Binds host pages to NUMA node <host_node> (where applicable)
//   3. First-touches the buffer on the appropriate side (CPU thread on host_node,
//      or GPU kernel for device-resident)
//   4. Runs a memory-bound kernel: data[i] = data[i] * 1.01f + 0.01f (read+write)
//   5. Times the kernel using cudaEvent (kernel-only time, not transfer time)
//   6. Records achieved GB/s = (2 * buffer_bytes) / time_seconds
//   7. Repeats <iterations> times; reports min, median, max
//
// Modes tested:
//   - cudaMalloc                 (device-resident, baseline)
//   - cudaHostRegister_onnode    (NUMA-allocated host memory, then pinned)
//   - cudaMallocHost             (diagnostic only, placement verified or excluded)
//   - cudaMallocManaged          (managed, CPU-preferred + first-touch on host_node)
//   - numa_alloc_onnode          (system-allocated, equivalent to malloc on a
//                                 NUMA-pinned page set; tests automatic page
//                                 migration / direct C2C access)
//
// For cudaMallocManaged and numa_alloc_onnode also test:
//   - First-touch on CPU then access from GPU (the common case for inference)
//   - First-touch on GPU then access from CPU
//   - Alternating CPU/GPU access on every iteration (worst case — page thrashing)
//
// Output: writes JSON to stdout with this schema:
//   {
//     "config": {"gpu_id": 0, "host_node": 0, "buffer_bytes": 4294967296},
//     "modes": {
//       "cudaMalloc": {"min_gbs": ..., "median_gbs": ..., "max_gbs": ...},
//       "cudaHostRegister_onnode0": {...},
//       "cudaMallocHost_diagnostic": {"median_gbs": ..., "notes": "placement_verified|placement_unverified"},
//       "cudaMallocManaged_node0_first_touch_cpu": {...},
//       "cudaMallocManaged_node0_first_touch_gpu": {...},
//       "cudaMallocManaged_node0_alternating": {...},
//       "numa_alloc_node0_first_touch_cpu": {...},
//       ...
//     }
//   }
```

CLAUDE CODE: implement this kernel. Key implementation notes:
- Use `numa_run_on_node()` to bind the touching thread before first-touch.
- For the pinned-host headline mode, allocate pageable host memory on `host_node` first (`numa_alloc_onnode()` or `mmap()` followed by `mbind()` and CPU first-touch), verify placement with `move_pages()`, then call `cudaHostRegister()`. If registration changes placement or verification fails, mark the result as invalid for NUMA-labeled conclusions.
- For `cudaMallocHost`, do not rely on post-allocation `mbind()` as the main method. It may fail or silently leave pages in place after pinning. Keep this mode as a diagnostic and include a `notes` field with the observed placement.
- For `cudaMallocManaged`, call `cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, cudaMemLocation{cudaMemLocationTypeHostNuma, host_node})` on CUDA 13 for CPU-resident variants, then use CPU thread binding and first-touch. Every managed result must include the observed placement or `placement_unverified`.
- The memory-bound kernel needs enough blocks/threads to saturate; aim for grid size = SM_count × 4, block size 256.
- Use `cudaEventElapsedTime` for kernel time; do not include allocation time in the measurement.
- Do 3 warmup iterations before the timed iterations, discard those.

Run the program for both GPUs and both host nodes:

```bash
mkdir -p "$RUN_DIR/results/memory_modes"
for gpu in 0 1; do
  for host_node in $GPU_0_NODE $GPU_1_NODE; do
    ./memory_modes $gpu $host_node 4294967296 10 \
      > "$RUN_DIR/results/memory_modes/gpu${gpu}_host${host_node}.json" 2>&1
  done
done
```

This produces 4 JSON files (2 GPUs × 2 host nodes). The analysis script reads all four and produces a comparison table that explicitly labels local-C2C vs cross-socket entries.

Analyzer contract: parse every `results/memory_modes/gpu<N>_host<M>.json` file. Do not expect a single top-level `memory_modes.json`.

### 5. Latency probe (custom kernel) — methodology fixed

**Important methodology note**: CUDA events measure GPU-side time, not CPU-observed latency. For host-visible work (host-to-device, host-on-remote-node, device-to-host) the relevant latency from an inference-engine perspective is what the CPU thread observes after `cudaStreamSynchronize` returns, not the in-kernel duration.

Two distinct measurements are needed:

**(a) One-way bandwidth at small sizes** — `cudaMemcpyAsync` followed by stream sync, timed with `clock_gettime(CLOCK_MONOTONIC)` on the CPU. This is what an inference engine actually observes when staging weights or KV.

**(b) CPU-orchestrated round-trip latency** — copy src→dst, synchronize, then copy dst→src, synchronize. This measures the latency an inference runtime sees when a CPU thread stages work through CUDA copy engines. It is not a GPU-initiated cache-line ping-pong and must not be described as one.

Host-side endpoints should use **NUMA-controlled pinned host buffers only**: allocate on the requested node, first-touch, verify placement with `move_pages()`, then call `cudaHostRegister()`. Pageable host memory has additional overhead from staging and DMA setup that confounds the cross-socket signal.

```cuda
// latency_probe.cu
//
// Compile: nvcc -O3 -arch=sm_90 latency_probe.cu -o latency_probe
//
// Usage:
//   ./latency_probe <mode> <src_node> <dst_node>
// where:
//   mode = oneway_bw | roundtrip_lat
//   src_node, dst_node = device specifiers
//     "h0", "h1" = HBM on GPU 0/1
//     "l0", "l1" = LPDDR on Grace 0/1 (pinned host memory, NUMA-bound)
//
// For each transfer size in [8, 64, 512, 4K, 32K, 256K, 2M, 16M] bytes:
//   - Allocate src buffer on src_node (NUMA-bound for host memory)
//   - Allocate dst buffer on dst_node
//   - Warmup: 100 iterations
//   - Measure: 1000 iterations
//   - For oneway_bw: cudaMemcpyAsync src->dst, cudaStreamSynchronize,
//                    measure with clock_gettime around the sync.
//                    Report achieved GB/s = size / time.
//   - For roundtrip_lat: orchestrating CPU thread sends src->dst, syncs,
//                        sends dst->src, syncs. Time around both sends.
//                        Report median + p99 microseconds.
//
// Output (CSV):
//   mode,src,dst,size_bytes,iterations,median_us,p99_us,gbs
//
// All host buffers must use the same NUMA-controlled pinned path as
// memory_modes: allocate on the requested host node, first-touch, verify with
// move_pages(), then cudaHostRegister().
```

Run all combinations:

```bash
mkdir -p "$RUN_DIR/results/latency"
for mode in oneway_bw roundtrip_lat; do
  for src in h0 h1 l0 l1; do
    for dst in h0 h1 l0 l1; do
      [ "$src" = "$dst" ] && continue
      ./latency_probe $mode $src $dst \
        > "$RUN_DIR/results/latency/${mode}_${src}_to_${dst}.csv" 2>&1
    done
  done
done
```

Analyzer contract: parse all CSV files under `results/latency/`. The CSV header is exactly `mode,src,dst,size_bytes,iterations,median_us,p99_us,gbs`.

### 6. mixbench — compute vs bandwidth roofline

```bash
cd mixbench/mixbench-cuda
CUDA_VISIBLE_DEVICES=0 ./mixbench-cuda > "$RUN_DIR/results/mixbench_gpu0.txt" 2>&1
CUDA_VISIBLE_DEVICES=1 ./mixbench-cuda > "$RUN_DIR/results/mixbench_gpu1.txt" 2>&1
cd -
```

For LLM decode at batch 1, the workload should be deep on the bandwidth-bound side of the roofline.

### 7. Sustained throughput vs burst (custom kernel)

Real LLM workloads do many small transfers, not single large ones. Test sustained throughput:

```cuda
// sustained.cu
//
// Compile: nvcc -O3 -arch=sm_90 sustained.cu -o sustained
//
// Usage: ./sustained <src_node> <dst_node> <chunk_bytes> <num_chunks>
//
// Submits <num_chunks> async cudaMemcpy operations of <chunk_bytes> each
// onto a single CUDA stream, then waits. Reports aggregate throughput.
//
// Run for the same node combinations as the latency probe, with
// 1MB and 16MB chunks. Keep total bytes per path bounded so this measures
// sustained throughput rather than multi-terabyte driver queue behavior.
```

```bash
mkdir -p "$RUN_DIR/results/sustained"
for src in h0 h1 l0 l1; do
  for dst in h0 h1 l0 l1; do
    [ "$src" = "$dst" ] && continue
    for chunk in 1048576 16777216; do
      # 64 GiB total per path/chunk. This is enough to smooth launch overhead
      # without turning the full matrix into a multi-hour transfer-only run.
      chunks=$((64 * 1024 * 1024 * 1024 / chunk))
      ./sustained $src $dst $chunk $chunks \
        > "$RUN_DIR/results/sustained/${src}_to_${dst}_chunk${chunk}.txt" 2>&1
    done
  done
done
```

Compare the sustained throughput against the equivalent NVBandwidth single-shot result. Significant gap = per-transfer overhead worth quantifying.

## Output structure

```
~/gh200-bench/run-YYYYMMDD-HHMMSS/
├── system-info.txt
├── numa-topology.txt
├── gpu-numa-map.txt
├── methodology.md
├── summary.md
├── results.json
├── results/
│   ├── nvbandwidth_list.txt
│   ├── nvbandwidth_all_default.txt
│   ├── nvbandwidth_test_map.json
│   ├── nvbandwidth_<test>_host_on_node<N>.txt   # for each host test, each socket
│   ├── nvbandwidth_<test>.txt                    # for GPU-to-GPU tests
│   ├── stream_node<N>_t<T>_run<R>.txt            # thread sweep per node, 3 runs
│   ├── stream_cross_n<X>cores_n<Y>mem_run<R>.txt # cross-socket, both directions
│   ├── stream_both_interleave_run<R>.txt
│   ├── babelstream_gpu<N>_run<R>.txt             # 3 runs per GPU
│   ├── babelstream_gpu0_size<S>.txt              # array-size sweep
│   ├── memory_modes/gpu<N>_host<M>.json          # 4 files
│   ├── latency/<mode>_<src>_to_<dst>.csv         # full direction matrix
│   ├── sustained/<src>_to_<dst>_chunk<C>.txt     # full direction matrix × 2 chunks
│   ├── mixbench_gpu<N>.txt
│   └── ...
├── plots/
│   ├── stream_triad_by_threads.png
│   ├── nvbandwidth_<label>.png   # one per matrix
│   ├── latency_vs_size.png
│   ├── memory_modes_comparison.png
│   └── roofline_combined.png
└── install-failures.txt (may be empty)
```

## Analysis script contract

The analysis step must match the output names in this plan. In particular:

- NVBandwidth: read `nvbandwidth_all_default.txt`, `nvbandwidth_test_map.json`, and all semantic `nvbandwidth_<test>*.txt` files. Do not expect `nvbandwidth_all.txt`.
- STREAM: read suffixed repeated-run files and report medians across `_run1`, `_run2`, `_run3`.
- BabelStream: read `babelstream_gpu<N>_run<R>.txt` and report medians across runs.
- Memory modes: read every JSON file under `results/memory_modes/`, preserving `gpu` and `host_node` labels from the filename and JSON `config`.
- Latency: read every CSV file under `results/latency/`.
- Sustained throughput: read every file under `results/sustained/` and compare against the closest NVBandwidth path.

If using the current `analyze-results.py`, update it before the full run; its older filename assumptions are not authoritative.

## summary.md content

Generate a single table with all measured paths in the full direction matrix:

| Path | Description | Theoretical | Measured | % peak | Tool | Notes |
|------|-------------|------------:|---------:|-------:|------|-------|
| `H0->H0` | GPU0 HBM local | 4000 | … | … | BabelStream | |
| `H1->H1` | GPU1 HBM local | 4000 | … | … | BabelStream | |
| `L0->L0` | Grace0 LPDDR local (72 thr) | 500 | … | … | STREAM | |
| `L1->L1` | Grace1 LPDDR local (72 thr) | 500 | … | … | STREAM | |
| `L0->H0` | C2C local (host memory on Grace0) | 450 | … | … | nvbandwidth | from h2d_ce, host_on_node0 |
| `H0->L0` | C2C local reverse | 450 | … | … | nvbandwidth | from d2h_ce, host_on_node0 |
| `L1->H1` | C2C local (host memory on Grace1) | 450 | … | … | nvbandwidth | from h2d_ce, host_on_node1 |
| `H1->L1` | C2C local reverse | 450 | … | … | nvbandwidth | from d2h_ce, host_on_node1 |
| `L0->H1` | Cross-socket C2C+fabric | — | … | — | nvbandwidth | h2d_ce host on Grace0 to GPU1 |
| `H1->L0` | Cross-socket reverse | — | … | — | nvbandwidth | d2h_ce host on Grace0 from GPU1 |
| `L1->H0` | Cross-socket C2C+fabric | — | … | — | nvbandwidth | h2d_ce host on Grace1 to GPU0 |
| `H0->L1` | Cross-socket reverse | — | … | — | nvbandwidth | d2h_ce host on Grace1 from GPU0 |
| `L0->L1` | Inter-Grace fabric | — | … | — | STREAM | cross-socket cores/mem |
| `L1->L0` | Inter-Grace fabric reverse | — | … | — | STREAM | cross-socket cores/mem |
| `H0->H1` | The SYS link, read | — | … | — | nvbandwidth | d2d_read_ce |
| `H1->H0` | The SYS link, read reverse | — | … | — | nvbandwidth | d2d_read_ce |

Plus prose paragraphs for: HBM characterization, DDR characterization, C2C local characterization, **cross-socket asymmetry findings** (the publishable headline), the SYS-link characterization, and concrete LLM-serving implications.

## methodology.md content

For each tool, explain:
- What it measures
- Units, theoretical peak, datasheet citation
- **NUMA placement assumptions for this run** (which node host memory was on, which side did first-touch)
- Known caveats

## What this enables — concrete numbers, not heuristics

The headline question for this benchmark — does MoE expert offload to Grace LPDDR work for V4-Flash on this box? — needs concrete bytes-per-token math, not a hand-wavy bandwidth ratio.

For DeepSeek V4-Flash at Q4_K_M (~4.5 bpw average, varies by tensor), with 13B active parameters per token of which roughly 11B are MoE expert weights:

- Per-token MoE expert read = ~11B × 4.5 bits / 8 = ~6.2 GB
- Other active weights (attention, shared, embeddings) = ~2B × ~5 bpw / 8 = ~1.3 GB
- Total active read per token ≈ 7.5 GB

If experts live in Grace LPDDR and are read over local C2C at measured bandwidth `BW_C2C` GB/s, expert-read time per token = 6.2 / BW_C2C seconds. At 400 GB/s C2C this is ~15.5 ms; at 300 GB/s it's ~21 ms. Single-stream tok/s ceiling for that path = 1 / (expert_time + everything_else_time).

The headline conclusions to write up after the run:
- **Best-case HBM bandwidth** (per GPU) — sets the floor on what's achievable when everything fits
- **Best-case C2C bandwidth** (host memory on local Grace, transferred to local GPU) — sets the ceiling for Grace-offloaded experts
- **Cross-socket C2C bandwidth** (host memory on remote Grace, transferred to local GPU) — needed for the "use both Grace memories" picture
- **Worst-case cross-GPU bandwidth** (`H0->H1`) — determines whether tensor parallelism is viable at all
- **Read/write asymmetry** on every direction — the publishable finding
- **Sustained vs burst gap** — how much per-transfer overhead matters in practice

Each of these is a concrete, measured number, not a heuristic. The bytes-per-token math above lets you turn any of them into a tok/s ceiling estimate that's directly comparable to what llama.cpp actually achieves once V4-Flash GGUFs ship.

## Things to be careful about

1. **Run benchmarks one at a time.** Concurrent GPU work skews everything. `sleep 5` between major sections.
2. **Don't trust first runs.** First execution after driver load has cold caches and unstable clocks. The custom kernels include warmup iterations that are discarded; STREAM does this internally; BabelStream does it with `-n` iterations.
3. **NUMA binding is not optional.** A `numactl` mistake produces wrong numbers that look plausible. Always specify both `--cpunodebind` and `--membind` for CPU work; for the custom kernels, verify NUMA placement with `move_pages()` after allocation.
4. **Watch for thermal throttling.** Run `nvidia-smi -q -d TEMPERATURE,POWER,CLOCK` periodically during long sections and log to `thermal-log.txt`.
5. **Hugepages.** STREAM and BabelStream both benefit from 2MB or 1GB hugepages. THP enabled is the easy default.
6. **Document every NUMA binding.** Every output file should have its NUMA configuration recorded either in the filename or in a header comment.
7. **Triple-run the headline measurements.** STREAM and BabelStream get 3 runs each; report median across the three to handle outlier variance.

## Cleanup

At end of run:
```bash
sudo nvidia-smi -rgc           # restore GPU clocks
sudo nvidia-smi -pm 0          # disable persistence (optional)
sudo cpupower frequency-set -g <original>  # restore governor
echo <original> | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
```

Record the original values at the start of the run so this is mechanical.
