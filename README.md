# Dual GH200 Memory Benchmark

Benchmark suite for characterizing memory paths on a dual NVIDIA Grace Hopper workstation.

The goal is to measure the paths that matter for LLM inference:

- Local HBM bandwidth on each Hopper GPU
- Local Grace LPDDR bandwidth on each CPU socket
- Local Grace to Hopper NVLink C2C bandwidth
- Remote socket Grace to Hopper bandwidth
- Cross GPU staged copy bandwidth on `SYS` topology
- Allocation mode behavior for CUDA, pinned host memory, managed memory, and NUMA placed system memory

## Layout

- `BENCHMARK_SPEC.md`: detailed benchmark plan and methodology
- `RESULTS.md`: durable run log and headline results
- `2026-04-25-gh200-benchmarking.md`: current blog post draft
- `run-full-benchmark.sh`: full benchmark runner
- `run-smoke-test.sh`: shorter preflight benchmark
- `analyze-results.py`: parser and plot generator
- `src/`: custom CUDA benchmark sources
- `scripts/`: helper scripts used by the runners
- `smoke-test-spec.md`: smoke test spec
- `TODO.md`: remaining machine tuning notes

## Running

The scripts write raw output outside the repo by default, under:

```text
/home/grace/gh200-bench
```

Smoke test:

```bash
./run-smoke-test.sh
```

Full benchmark:

```bash
./run-full-benchmark.sh
```

Analyze an existing run:

```bash
uv run --with numpy --with matplotlib analyze-results.py /path/to/run-dir
```

## Current Reference Run

The current reference run is:

```text
/home/grace/gh200-bench/run-20260425-134620
```

The most important summary is in `RESULTS.md`. The detailed methodology is in `BENCHMARK_SPEC.md`.
