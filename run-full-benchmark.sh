#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_DIR/scripts/common.sh"

FORCE="${FORCE:-0}"
TOOLS_ROOT="${TOOLS_ROOT:-$HOME/gh200-bench/tools}"

usage() {
  cat <<'USAGE'
Usage: ./run-full-benchmark.sh [--force]

Environment:
  RUN_DIR          Existing run directory to resume, or unset to create one.
  TOOLS_ROOT       Tool checkout root. Default: ~/gh200-bench/tools
  NVBANDWIDTH_DIR  Override nvbandwidth checkout.
  STREAM_DIR       Override STREAM checkout.
  BABELSTREAM_DIR  Override BabelStream checkout.
  MIXBENCH_DIR     Override mixbench checkout.
  FORCE=1          Rerun completed sections.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

mark_done() {
  touch "$RUN_DIR/.done-$1"
}

is_done() {
  [[ "$FORCE" != "1" && -e "$RUN_DIR/.done-$1" ]]
}

section() {
  local name="$1"
  shift
  if is_done "$name"; then
    append_progress "SKIP $name already completed"
    return 0
  fi
  append_progress "SECTION $name"
  "$@"
  mark_done "$name"
  append_progress "SECTION DONE $name"
}

clone_if_missing() {
  local name="$1"
  local url="$2"
  local dir="$TOOLS_ROOT/$name"
  mkdir -p "$TOOLS_ROOT"
  if [[ ! -d "$dir/.git" ]]; then
    run_logged_allow_fail "clone $name" "$RUN_DIR/results/install_${name}.txt" git clone "$url" "$dir"
  fi
}

build_boost_if_missing() {
  local prefix="$TOOLS_ROOT/boost-1.83.0"
  if [[ -f "$prefix/lib/cmake/Boost-1.83.0/BoostConfig.cmake" ]]; then
    return 0
  fi
  mkdir -p "$TOOLS_ROOT"
  run_logged_allow_fail "build local Boost program_options" "$RUN_DIR/results/install_boost_program_options.txt" \
    bash -lc "cd '$TOOLS_ROOT' && if [ ! -d boost_1_83_0 ]; then curl -L -o boost_1_83_0.tar.gz https://archives.boost.io/release/1.83.0/source/boost_1_83_0.tar.gz && tar -xzf boost_1_83_0.tar.gz; fi && cd boost_1_83_0 && ./bootstrap.sh --prefix='$prefix' --with-libraries=program_options && ./b2 -j\$(nproc) --with-program_options link=static runtime-link=shared install"
}

install_tools() {
  clone_if_missing nvbandwidth https://github.com/NVIDIA/nvbandwidth.git
  clone_if_missing STREAM https://github.com/jeffhammond/STREAM.git
  clone_if_missing BabelStream https://github.com/UoB-HPC/BabelStream.git
  clone_if_missing mixbench https://github.com/ekondis/mixbench.git

  NVBANDWIDTH_DIR="${NVBANDWIDTH_DIR:-$(find_tool_dir nvbandwidth NVBANDWIDTH_DIR || true)}"
  STREAM_DIR="${STREAM_DIR:-$(find_tool_dir STREAM STREAM_DIR || true)}"
  BABELSTREAM_DIR="${BABELSTREAM_DIR:-$(find_tool_dir BabelStream BABELSTREAM_DIR || true)}"
  MIXBENCH_DIR="${MIXBENCH_DIR:-$(find_tool_dir mixbench MIXBENCH_DIR || true)}"
  export NVBANDWIDTH_DIR STREAM_DIR BABELSTREAM_DIR MIXBENCH_DIR

  build_boost_if_missing

  if [[ -n "${NVBANDWIDTH_DIR:-}" && -x "$NVBANDWIDTH_DIR/debian_install.sh" && ! -x "$NVBANDWIDTH_DIR/nvbandwidth" ]]; then
    run_logged_allow_fail "build nvbandwidth" "$RUN_DIR/results/install_nvbandwidth_build.txt" \
      bash -lc "source '$REPO_DIR/scripts/common.sh' && cmake -S '$NVBANDWIDTH_DIR' -B '$NVBANDWIDTH_DIR/build' -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=90 -DBOOST_ROOT='$TOOLS_ROOT/boost-1.83.0' -DBoost_NO_SYSTEM_PATHS=ON && cmake --build '$NVBANDWIDTH_DIR/build' -j\$(nproc)"
  fi

  if [[ -n "${STREAM_DIR:-}" ]]; then
    run_logged_allow_fail "build STREAM" "$RUN_DIR/results/install_stream_build.txt" \
      make -C "$STREAM_DIR" stream_c.exe CC=gcc CFLAGS="-O3 -mcpu=neoverse-v2 -mcmodel=large -fno-pic -fno-pie -fopenmp -DSTREAM_ARRAY_SIZE=400000000 -DNTIMES=20" LDFLAGS="-no-pie"
  fi

  if [[ -n "${BABELSTREAM_DIR:-}" ]]; then
    run_logged_allow_fail "configure BabelStream" "$RUN_DIR/results/install_babelstream_configure.txt" \
      cmake -S "$BABELSTREAM_DIR" -B "$BABELSTREAM_DIR/build" -DMODEL=cuda -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc -DCUDA_ARCH=sm_90
    run_logged_allow_fail "build BabelStream" "$RUN_DIR/results/install_babelstream_build.txt" \
      cmake --build "$BABELSTREAM_DIR/build"
  fi

  if [[ -n "${MIXBENCH_DIR:-}" && -d "$MIXBENCH_DIR/mixbench-cuda" ]]; then
    run_logged_allow_fail "build mixbench" "$RUN_DIR/results/install_mixbench_build.txt" \
      bash -lc "source '$REPO_DIR/scripts/common.sh' && cmake -S '$MIXBENCH_DIR/mixbench-cuda' -B '$MIXBENCH_DIR/mixbench-cuda/build' -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=90 && cmake --build '$MIXBENCH_DIR/mixbench-cuda/build' -j\$(nproc)"
  fi
}

build_custom_kernels() {
  mkdir -p "$RUN_DIR/bin"
  if ! command -v nvcc >/dev/null 2>&1; then
    append_progress "SKIP custom CUDA build: nvcc unavailable"
    printf '%s\tcustom CUDA build\tnvcc unavailable\n' "$(timestamp)" >> "$RUN_DIR/install-failures.txt"
    return 0
  fi
  for src in memory_modes latency_probe sustained; do
    run_logged_allow_fail "build $src" "$RUN_DIR/results/build_${src}.txt" \
      nvcc -O3 -std=c++17 -arch=sm_90 -I"$REPO_DIR/src" "$REPO_DIR/src/${src}.cu" -o "$RUN_DIR/bin/$src"
  done
}

resolve_nvbandwidth_tests() {
  local list_file="$RUN_DIR/results/nvbandwidth_list.txt"
  local map_file="$RUN_DIR/results/nvbandwidth_test_map.json"
  python3 "$REPO_DIR/scripts/resolve-nvbandwidth-tests.py" "$list_file" "$map_file"
}

run_nvbandwidth() {
  local nvbw_bin="${NVBANDWIDTH_BIN:-${NVBANDWIDTH_DIR:-}/build/nvbandwidth}"
  if [[ -z "${NVBANDWIDTH_DIR:-}" || ! -x "$nvbw_bin" ]]; then
    append_progress "SKIP NVBandwidth: binary unavailable"
    return 0
  fi
  "$nvbw_bin" -l > "$RUN_DIR/results/nvbandwidth_list.txt" 2>&1 || true
  resolve_nvbandwidth_tests || append_progress "NVBandwidth resolver reported missing tests"
  "$nvbw_bin" -t all > "$RUN_DIR/results/nvbandwidth_all_default.txt" 2>&1 || true

  local selector
  for t in host_to_device_memcpy_ce device_to_host_memcpy_ce host_to_device_memcpy_sm device_to_host_memcpy_sm; do
    selector=$(python3 "$REPO_DIR/scripts/read-nvbandwidth-selector.py" "$RUN_DIR/results/nvbandwidth_test_map.json" "$t" || true)
    [[ -n "$selector" ]] || continue
    numactl --cpunodebind="$GPU_0_NODE" --membind="$GPU_0_NODE" "$nvbw_bin" -t "$selector" \
      > "$RUN_DIR/results/nvbandwidth_${t}_host_on_node${GPU_0_NODE}.txt" 2>&1 || true
    append_progress "NVBandwidth $t host_on_node$GPU_0_NODE complete"
    numactl --cpunodebind="$GPU_1_NODE" --membind="$GPU_1_NODE" "$nvbw_bin" -t "$selector" \
      > "$RUN_DIR/results/nvbandwidth_${t}_host_on_node${GPU_1_NODE}.txt" 2>&1 || true
    append_progress "NVBandwidth $t host_on_node$GPU_1_NODE complete"
  done

  for t in device_to_device_memcpy_read_ce device_to_device_memcpy_write_ce device_local_copy; do
    selector=$(python3 "$REPO_DIR/scripts/read-nvbandwidth-selector.py" "$RUN_DIR/results/nvbandwidth_test_map.json" "$t" || true)
    [[ -n "$selector" ]] || continue
    "$nvbw_bin" -t "$selector" > "$RUN_DIR/results/nvbandwidth_${t}.txt" 2>&1 || true
    append_progress "NVBandwidth $t complete"
  done
}

run_stream() {
  if [[ -z "${STREAM_DIR:-}" || ! -x "$STREAM_DIR/stream_c.exe" ]]; then
    append_progress "SKIP STREAM: binary unavailable"
    return 0
  fi
  for node in "$GPU_0_NODE" "$GPU_1_NODE"; do
    for threads in 8 16 32 48 64 72; do
      for run in 1 2 3; do
        OMP_NUM_THREADS="$threads" OMP_PROC_BIND=spread \
        numactl --cpunodebind="$node" --membind="$node" "$STREAM_DIR/stream_c.exe" \
          > "$RUN_DIR/results/stream_node${node}_t${threads}_run${run}.txt" 2>&1 || true
        append_progress "STREAM node=$node threads=$threads run=$run complete"
        sleep 1
      done
    done
  done
  for run in 1 2 3; do
    OMP_NUM_THREADS=72 OMP_PROC_BIND=spread \
    numactl --cpunodebind="$GPU_0_NODE" --membind="$GPU_1_NODE" "$STREAM_DIR/stream_c.exe" \
      > "$RUN_DIR/results/stream_cross_n${GPU_0_NODE}cores_n${GPU_1_NODE}mem_run${run}.txt" 2>&1 || true
    OMP_NUM_THREADS=72 OMP_PROC_BIND=spread \
    numactl --cpunodebind="$GPU_1_NODE" --membind="$GPU_0_NODE" "$STREAM_DIR/stream_c.exe" \
      > "$RUN_DIR/results/stream_cross_n${GPU_1_NODE}cores_n${GPU_0_NODE}mem_run${run}.txt" 2>&1 || true
    OMP_NUM_THREADS=144 OMP_PROC_BIND=spread \
    numactl --interleave="$GPU_0_NODE,$GPU_1_NODE" "$STREAM_DIR/stream_c.exe" \
      > "$RUN_DIR/results/stream_both_interleave_run${run}.txt" 2>&1 || true
    append_progress "STREAM cross/interleave run=$run complete"
  done
}

run_babelstream() {
  local dir="${BABELSTREAM_DIR:-}"
  local binary="$dir/build/cuda-stream"
  if [[ -z "$dir" || ! -x "$binary" ]]; then
    append_progress "SKIP BabelStream: binary unavailable"
    return 0
  fi
  for gpu in 0 1; do
    for run in 1 2 3; do
      CUDA_VISIBLE_DEVICES="$gpu" "$binary" -s 268435456 -n 20 \
        > "$RUN_DIR/results/babelstream_gpu${gpu}_run${run}.txt" 2>&1 || true
      append_progress "BabelStream gpu=$gpu run=$run complete"
      sleep 1
    done
  done
  for size in 16777216 67108864 268435456 1073741824; do
    CUDA_VISIBLE_DEVICES=0 "$binary" -s "$size" -n 10 \
      > "$RUN_DIR/results/babelstream_gpu0_size${size}.txt" 2>&1 || true
    append_progress "BabelStream size sweep size=$size complete"
  done
}

run_memory_modes() {
  [[ -x "$RUN_DIR/bin/memory_modes" ]] || { append_progress "SKIP memory_modes: binary unavailable"; return 0; }
  mkdir -p "$RUN_DIR/results/memory_modes"
  for gpu in 0 1; do
    for host_node in "$GPU_0_NODE" "$GPU_1_NODE"; do
      "$RUN_DIR/bin/memory_modes" "$gpu" "$host_node" 4294967296 10 \
        > "$RUN_DIR/results/memory_modes/gpu${gpu}_host${host_node}.json" 2>&1 || true
      append_progress "memory_modes gpu=$gpu host_node=$host_node complete"
    done
  done
}

run_latency() {
  [[ -x "$RUN_DIR/bin/latency_probe" ]] || { append_progress "SKIP latency_probe: binary unavailable"; return 0; }
  mkdir -p "$RUN_DIR/results/latency"
  for mode in oneway_bw roundtrip_lat; do
    for src in h0 h1 l0 l1; do
      for dst in h0 h1 l0 l1; do
        [[ "$src" == "$dst" ]] && continue
        "$RUN_DIR/bin/latency_probe" "$mode" "$src" "$dst" \
          > "$RUN_DIR/results/latency/${mode}_${src}_to_${dst}.csv" 2>&1 || true
        append_progress "latency mode=$mode $src->$dst complete"
      done
    done
  done
}

run_mixbench() {
  local dir="${MIXBENCH_DIR:-}"
  local binary="$dir/mixbench-cuda/build/mixbench-cuda"
  if [[ -z "$dir" || ! -x "$binary" ]]; then
    append_progress "SKIP mixbench: binary unavailable"
    return 0
  fi
  CUDA_VISIBLE_DEVICES=0 "$binary" > "$RUN_DIR/results/mixbench_gpu0.txt" 2>&1 || true
  append_progress "mixbench gpu=0 complete"
  CUDA_VISIBLE_DEVICES=1 "$binary" > "$RUN_DIR/results/mixbench_gpu1.txt" 2>&1 || true
  append_progress "mixbench gpu=1 complete"
}

run_sustained() {
  [[ -x "$RUN_DIR/bin/sustained" ]] || { append_progress "SKIP sustained: binary unavailable"; return 0; }
  mkdir -p "$RUN_DIR/results/sustained"
  for src in h0 h1 l0 l1; do
    for dst in h0 h1 l0 l1; do
      [[ "$src" == "$dst" ]] && continue
      for chunk in 1048576 16777216; do
        chunks=$((64 * 1024 * 1024 * 1024 / chunk))
        "$RUN_DIR/bin/sustained" "$src" "$dst" "$chunk" "$chunks" \
          > "$RUN_DIR/results/sustained/${src}_to_${dst}_chunk${chunk}.txt" 2>&1 || true
        append_progress "sustained $src->$dst chunk=$chunk chunks=$chunks complete"
      done
    done
  done
}

analyze() {
  if command -v uv >/dev/null 2>&1; then
    uv run --with numpy --with matplotlib "$REPO_DIR/analyze-results.py" "$RUN_DIR" > "$RUN_DIR/analyze.log" 2>&1 || true
  else
    python3 "$REPO_DIR/analyze-results.py" "$RUN_DIR" > "$RUN_DIR/analyze.log" 2>&1 || true
  fi
  append_progress "Analysis complete; see $RUN_DIR/analyze.log"
}

main() {
  ensure_run_dir run
  append_progress "Full benchmark runner started"
  detect_gpu_nodes
  capture_system_info
  section install_tools install_tools
  section build_custom_kernels build_custom_kernels
  section nvbandwidth run_nvbandwidth
  sleep 5
  section stream run_stream
  sleep 5
  section babelstream run_babelstream
  sleep 5
  section memory_modes run_memory_modes
  sleep 5
  section latency run_latency
  sleep 5
  section mixbench run_mixbench
  sleep 5
  section sustained run_sustained
  section analyze analyze
  append_progress "Full benchmark runner finished"
}

main "$@"
