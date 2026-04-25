#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_DIR/scripts/common.sh"

TOOLS_ROOT="${TOOLS_ROOT:-$HOME/gh200-bench/tools}"

find_or_note() {
  local var="$1"
  local name="$2"
  local value
  value=$(find_tool_dir "$name" "$var" || true)
  printf -v "$var" '%s' "$value"
  export "$var"
}

main() {
  ensure_run_dir smoke
  mkdir -p "$RUN_DIR/smoke-results"
  append_progress "Smoke test started"
  capture_system_info
  detect_gpu_nodes

  find_or_note NVBANDWIDTH_DIR nvbandwidth
  find_or_note STREAM_DIR STREAM
  find_or_note BABELSTREAM_DIR BabelStream

  if [[ -n "${NVBANDWIDTH_DIR:-}" && -x "$NVBANDWIDTH_DIR/nvbandwidth" ]]; then
    nvbw_bin="$NVBANDWIDTH_DIR/nvbandwidth"
  elif [[ -n "${NVBANDWIDTH_DIR:-}" && -x "$NVBANDWIDTH_DIR/build/nvbandwidth" ]]; then
    nvbw_bin="$NVBANDWIDTH_DIR/build/nvbandwidth"
  else
    nvbw_bin=""
  fi

  if [[ -n "$nvbw_bin" ]]; then
    "$nvbw_bin" -l > "$RUN_DIR/smoke-results/nvbandwidth_list.txt" 2>&1 || true
    python3 "$REPO_DIR/scripts/resolve-nvbandwidth-tests.py" \
      "$RUN_DIR/smoke-results/nvbandwidth_list.txt" \
      "$RUN_DIR/smoke-results/nvbandwidth_test_map.json" || true
    for test in host_to_device_memcpy_ce device_to_device_memcpy_read_ce; do
      selector=$(python3 "$REPO_DIR/scripts/read-nvbandwidth-selector.py" "$RUN_DIR/smoke-results/nvbandwidth_test_map.json" "$test" || true)
      if [[ -n "$selector" ]]; then
        "$nvbw_bin" -t "$selector" >> "$RUN_DIR/smoke-results/nvbandwidth_quick.txt" 2>&1 || true
        append_progress "Smoke NVBandwidth $test complete"
      else
        append_progress "Smoke NVBandwidth $test skipped: selector unavailable"
      fi
    done
  else
    append_progress "Smoke NVBandwidth skipped: binary unavailable"
  fi

  if [[ -n "${STREAM_DIR:-}" && -x "$STREAM_DIR/stream_c.exe" ]]; then
    OMP_NUM_THREADS=72 OMP_PROC_BIND=spread numactl --cpunodebind="$GPU_0_NODE" --membind="$GPU_0_NODE" \
      "$STREAM_DIR/stream_c.exe" > "$RUN_DIR/smoke-results/stream_node${GPU_0_NODE}_quick.txt" 2>&1 || true
    append_progress "Smoke STREAM node $GPU_0_NODE complete"
    OMP_NUM_THREADS=72 OMP_PROC_BIND=spread numactl --cpunodebind="$GPU_1_NODE" --membind="$GPU_1_NODE" \
      "$STREAM_DIR/stream_c.exe" > "$RUN_DIR/smoke-results/stream_node${GPU_1_NODE}_quick.txt" 2>&1 || true
    append_progress "Smoke STREAM node $GPU_1_NODE complete"
  else
    append_progress "Smoke STREAM skipped: binary unavailable"
  fi

  if [[ -n "${BABELSTREAM_DIR:-}" && -x "$BABELSTREAM_DIR/build/cuda-stream" ]]; then
    CUDA_VISIBLE_DEVICES=0 "$BABELSTREAM_DIR/build/cuda-stream" -s 268435456 -n 10 \
      > "$RUN_DIR/smoke-results/babelstream_gpu0_quick.txt" 2>&1 || true
    append_progress "Smoke BabelStream GPU0 complete"
    CUDA_VISIBLE_DEVICES=1 "$BABELSTREAM_DIR/build/cuda-stream" -s 268435456 -n 10 \
      > "$RUN_DIR/smoke-results/babelstream_gpu1_quick.txt" 2>&1 || true
    append_progress "Smoke BabelStream GPU1 complete"
  else
    append_progress "Smoke BabelStream skipped: binary unavailable"
  fi

  {
    echo "# Smoke Test Summary"
    echo
    echo "Run directory: \`$RUN_DIR\`"
    echo
    echo "## Progress"
    cat "$RUN_DIR/progress.log"
    echo
    echo "## Files"
    find "$RUN_DIR/smoke-results" -maxdepth 1 -type f -printf '%f\n' | sort
  } > "$RUN_DIR/smoke-summary.md"

  python3 "$REPO_DIR/scripts/smoke-decision.py" "$RUN_DIR" "$GPU_0_NODE" "$GPU_1_NODE" > "$RUN_DIR/decision.txt"
  append_progress "Smoke decision written to $RUN_DIR/decision.txt"
  append_progress "Smoke test finished"
}

main "$@"
