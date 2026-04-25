#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_LOG="$REPO_DIR/RESULTS.md"

if ! command -v nvcc >/dev/null 2>&1; then
  for cuda_dir in /usr/local/cuda-13.0 /usr/local/cuda /usr/local/cuda-12.8; do
    if [[ -x "$cuda_dir/bin/nvcc" ]]; then
      export PATH="$cuda_dir/bin:$PATH"
      export LD_LIBRARY_PATH="$cuda_dir/lib64:${LD_LIBRARY_PATH:-}"
      break
    fi
  done
fi

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

append_progress() {
  local message="$1"
  local line="- $(timestamp): $message"
  printf '%s\n' "$line" | tee -a "$RESULTS_LOG"
  if [[ -n "${RUN_DIR:-}" ]]; then
    printf '%s\n' "$line" >> "$RUN_DIR/progress.log"
  fi
}

run_logged() {
  local label="$1"
  local outfile="$2"
  shift 2
  append_progress "START $label"
  (
    printf '=== %s ===\n' "$label"
    printf 'Command:'
    printf ' %q' "$@"
    printf '\nStarted: %s\n\n' "$(timestamp)"
    set +e
    "$@"
    status=$?
    set -e
    printf '\nFinished: %s\nExit: %s\n' "$(timestamp)" "$status"
    exit "$status"
  ) > "$outfile" 2>&1
  append_progress "DONE $label -> $outfile"
}

run_logged_allow_fail() {
  local label="$1"
  local outfile="$2"
  shift 2
  if run_logged "$label" "$outfile" "$@"; then
    return 0
  fi
  local status=$?
  append_progress "FAIL $label exit=$status -> $outfile"
  if [[ -n "${RUN_DIR:-}" ]]; then
    printf '%s\t%s\t%s\n' "$(timestamp)" "$label" "$outfile" >> "$RUN_DIR/install-failures.txt"
  fi
  return 0
}

ensure_run_dir() {
  local kind="${1:-run}"
  RUN_DIR="${RUN_DIR:-$HOME/gh200-bench/${kind}-$(date +%Y%m%d-%H%M%S)}"
  export RUN_DIR
  mkdir -p "$RUN_DIR/results" "$RUN_DIR/plots"
  touch "$RUN_DIR/install-failures.txt" "$RUN_DIR/progress.log"
  append_progress "Using RUN_DIR=$RUN_DIR"
}

gpu_sysfs_path() {
  local gpu="$1"
  local bdf domain rest sysfs
  bdf=$(nvidia-smi -i "$gpu" --query-gpu=pci.bus_id --format=csv,noheader | tr -d ' ' | tr 'A-Z' 'a-z')
  domain=${bdf%%:*}
  rest=${bdf#*:}
  sysfs="/sys/bus/pci/devices/$(printf '%04x' "0x$domain"):$rest"
  if [[ ! -r "$sysfs/numa_node" ]]; then
    printf 'Cannot read NUMA node for GPU %s at %s\n' "$gpu" "$sysfs" >&2
    return 1
  fi
  printf '%s\n' "$sysfs"
}

gpu_cpu_numa_node() {
  local gpu="$1"
  local sysfs node
  sysfs=$(gpu_sysfs_path "$gpu")
  node=$(cat "$sysfs/numa_node")
  if ! numactl -H | grep -q "^node $node "; then
    printf 'GPU %s reported CPU NUMA node %s, but numactl -H does not list it\n' "$gpu" "$node" >&2
    return 1
  fi
  printf '%s\n' "$node"
}

detect_gpu_nodes() {
  GPU_0_NODE=$(gpu_cpu_numa_node 0)
  GPU_1_NODE=$(gpu_cpu_numa_node 1)
  export GPU_0_NODE GPU_1_NODE
  {
    numactl -H > "$RUN_DIR/numa-topology.txt"
    for gpu in 0 1; do
      local sysfs node
      sysfs=$(gpu_sysfs_path "$gpu")
      node=$(cat "$sysfs/numa_node")
      printf 'GPU %s sysfs=%s attached_cpu_numa_node=%s\n' "$gpu" "$sysfs" "$node"
    done
    nvidia-smi topo -mp
  } > "$RUN_DIR/gpu-numa-map.txt" 2>&1
  append_progress "Detected GPU NUMA mapping: GPU_0_NODE=$GPU_0_NODE GPU_1_NODE=$GPU_1_NODE"
}

capture_system_info() {
  {
    echo "=== Date ==="; date
    echo "=== uname ==="; uname -a
    echo "=== lscpu ==="; lscpu
    echo "=== NUMA topology ==="; numactl -H
    echo "=== nvidia-smi ==="; nvidia-smi
    echo "=== nvidia-smi topo -m ==="; nvidia-smi topo -m
    echo "=== nvidia-smi topo -mp ==="; nvidia-smi topo -mp
    echo "=== GPU clocks ==="; nvidia-smi -q -d CLOCK
    echo "=== driver ==="; cat /proc/driver/nvidia/version 2>/dev/null || true
    echo "=== CUDA ==="; nvcc --version 2>/dev/null || echo "nvcc unavailable"
    echo "=== meminfo ==="; head -30 /proc/meminfo
    echo "=== HugePages ==="; grep -i huge /proc/meminfo
    echo "=== THP ==="; cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true
    echo "=== CPU governor ==="; cpupower frequency-info 2>/dev/null || echo "cpupower unavailable"
    echo "=== persistence mode ==="; nvidia-smi -q | grep -i persistence || true
  } > "$RUN_DIR/system-info.txt" 2>&1
  append_progress "Captured system info"
}

find_tool_dir() {
  local name="$1"
  local env_name="$2"
  local value="${!env_name:-}"
  if [[ -n "$value" && -d "$value" ]]; then
    printf '%s\n' "$value"
    return 0
  fi
  for candidate in "$REPO_DIR/$name" "$HOME/$name" "$HOME/gh200-bench/tools/$name"; do
    if [[ -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}
