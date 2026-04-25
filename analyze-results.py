#!/usr/bin/env python3
"""Consolidate dual-GH200 benchmark outputs into summary tables and plots."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class PathResult:
    path_id: str
    description: str
    theoretical_gbs: Optional[float]
    measured_gbs: Optional[float]
    tool: str
    notes: str = ""

    @property
    def pct_of_peak(self) -> Optional[float]:
        if self.theoretical_gbs and self.measured_gbs:
            return 100.0 * self.measured_gbs / self.theoretical_gbs
        return None


@dataclass
class Results:
    paths: list[PathResult] = field(default_factory=list)
    stream_by_threads: dict[str, dict[int, float]] = field(default_factory=dict)
    nvbw_matrices: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    latency_by_path: dict[str, dict[int, float]] = field(default_factory=dict)
    memory_modes: dict[str, dict[str, float]] = field(default_factory=dict)
    sustained: dict[str, float] = field(default_factory=dict)
    mixbench: dict[str, dict[str, float]] = field(default_factory=dict)
    parse_failures: list[str] = field(default_factory=list)
    raw_files_seen: list[str] = field(default_factory=list)


def safe_read(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except Exception:
        return ""


def median(values: list[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def parse_stream(text: str) -> Optional[float]:
    m = re.search(r"^Triad:\s+([\d.]+)", text, re.MULTILINE)
    return float(m.group(1)) / 1000.0 if m else None


def parse_babelstream(text: str) -> Optional[float]:
    m = re.search(r"^Triad\s+([\d.]+)", text, re.MULTILINE | re.IGNORECASE)
    if not m:
        return None
    value = float(m.group(1))
    return value / 1000.0 if value > 1e5 else value


def parse_nvbandwidth(text: str) -> dict[str, dict[str, dict[str, float]]]:
    matrices: dict[str, dict[str, dict[str, float]]] = {}
    if not text:
        return matrices
    sections = re.split(r"\n(?=Running\s|memcpy\s|memory\s|.*bandwidth)", text)
    for section in sections:
        header = re.search(r"([^\n]*bandwidth[^\n]*)", section, re.IGNORECASE)
        if not header:
            continue
        label = header.group(1).strip()
        rows: dict[str, dict[str, float]] = {}
        cols: list[str] = []
        for line in section.splitlines():
            tokens = line.strip().split()
            if not tokens:
                continue
            if not cols and len(tokens) >= 2 and all(re.fullmatch(r"\d+", t) for t in tokens):
                cols = tokens
                continue
            if cols and re.fullmatch(r"\d+", tokens[0]):
                row = tokens[0]
                rows[row] = {}
                for col, value in zip(cols, tokens[1:]):
                    if value.upper() in {"N/A", "NA", "-"}:
                        continue
                    try:
                        rows[row][col] = float(value)
                    except ValueError:
                        pass
        if rows:
            matrices[label] = rows
    return matrices


def parse_memory_modes(path: Path) -> Optional[dict[str, float]]:
    try:
        data = json.loads(safe_read(path))
    except json.JSONDecodeError:
        return None
    modes = data.get("modes", data)
    out: dict[str, float] = {}
    for mode, stats in modes.items():
        if isinstance(stats, dict) and "median_gbs" in stats:
            out[mode] = float(stats["median_gbs"])
    return out or None


def parse_latency_csv(path: Path) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    text = safe_read(path)
    if not text:
        return out
    reader = csv.DictReader(text.splitlines())
    for row in reader:
        try:
            label = f"{row['mode']}_{row['src']}_to_{row['dst']}"
            out.setdefault(label, {})[int(row["size_bytes"])] = float(row["median_us"])
        except Exception:
            continue
    return out


def parse_sustained(path: Path) -> Optional[float]:
    m = re.search(r"^gbps=([\d.]+)", safe_read(path), re.MULTILINE)
    return float(m.group(1)) if m else None


def parse_mixbench(path: Path) -> Optional[dict[str, float]]:
    text = safe_read(path)
    if not text:
        return None
    sp_gflops: list[float] = []
    sp_gbs: list[float] = []
    for line in text.splitlines():
        if not re.match(r"\s*\d+,", line):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            sp_gflops.append(float(parts[3]))
            sp_gbs.append(float(parts[4]))
        except ValueError:
            continue
    if not sp_gflops:
        return None
    return {"max_sp_gflops": max(sp_gflops), "max_sp_gbs": max(sp_gbs)}


def add_path(results: Results, path_id: str, description: str, theoretical: Optional[float],
             measured: Optional[float], tool: str, notes: str = "") -> None:
    results.paths.append(PathResult(path_id, description, theoretical, measured, tool, notes))


def matrix_value(matrix: dict[str, dict[str, float]], row: str, col: str) -> Optional[float]:
    return matrix.get(row, {}).get(col)


def analyse(run_dir: Path) -> Results:
    res_dir = run_dir / "results"
    if not res_dir.is_dir():
        raise SystemExit(f"No results/ directory found under {run_dir}")
    results = Results()

    for path in sorted(res_dir.glob("stream_node*_t*_run*.txt")):
        results.raw_files_seen.append(str(path.relative_to(res_dir)))
        m = re.match(r"stream_node(\d+)_t(\d+)_run(\d+)\.txt", path.name)
        value = parse_stream(safe_read(path))
        if not m or value is None:
            results.parse_failures.append(str(path.relative_to(res_dir)))
            continue
        node, threads = m.group(1), int(m.group(2))
        results.stream_by_threads.setdefault(node, {}).setdefault(threads, [])
        results.stream_by_threads[node][threads].append(value)  # type: ignore[arg-type]

    for node, by_thread in list(results.stream_by_threads.items()):
        for threads, vals in list(by_thread.items()):
            by_thread[threads] = float(median(vals))  # type: ignore[arg-type]
        if 72 in by_thread:
            add_path(results, f"L{node}->L{node}", f"Grace{node} LPDDR local (STREAM TRIAD, 72 threads)",
                     500.0, by_thread[72], "STREAM", "median of 3 runs")

    cross: dict[tuple[str, str], list[float]] = {}
    for path in sorted(res_dir.glob("stream_cross_n*cores_n*mem_run*.txt")):
        results.raw_files_seen.append(str(path.relative_to(res_dir)))
        m = re.match(r"stream_cross_n(\d+)cores_n(\d+)mem_run(\d+)\.txt", path.name)
        value = parse_stream(safe_read(path))
        if not m or value is None:
            results.parse_failures.append(str(path.relative_to(res_dir)))
            continue
        cross.setdefault((m.group(1), m.group(2)), []).append(value)
    for cores, mem in sorted(cross):
        measured = median(cross[(cores, mem)])
        add_path(results, f"L{mem}->L{cores}", f"Grace{cores} cores reading Grace{mem} LPDDR",
                 None, measured, "STREAM", "cross-socket median")

    for gpu in (0, 1):
        values: list[float] = []
        for path in sorted(res_dir.glob(f"babelstream_gpu{gpu}_run*.txt")):
            results.raw_files_seen.append(str(path.relative_to(res_dir)))
            value = parse_babelstream(safe_read(path))
            if value is None:
                results.parse_failures.append(str(path.relative_to(res_dir)))
            else:
                values.append(value)
        if values:
            add_path(results, f"H{gpu}->H{gpu}", f"GPU{gpu} HBM local (BabelStream Triad)",
                     4000.0, median(values), "BabelStream", "median of 3 runs")

    nvbw_files = [res_dir / "nvbandwidth_all_default.txt"]
    nvbw_files += sorted(res_dir.glob("nvbandwidth_*_host_on_node*.txt"))
    nvbw_files += sorted(p for p in res_dir.glob("nvbandwidth_*.txt")
                         if "host_on_node" not in p.name and p.name not in {"nvbandwidth_list.txt", "nvbandwidth_all_default.txt"})
    for path in nvbw_files:
        if not path.exists():
            continue
        results.raw_files_seen.append(str(path.relative_to(res_dir)))
        matrices = parse_nvbandwidth(safe_read(path))
        if not matrices:
            continue
        for label, matrix in matrices.items():
            key = f"{path.stem}:{label}"
            results.nvbw_matrices[key] = matrix
            if "host_to_device" in path.name:
                host_node = re.search(r"host_on_node(\d+)", path.name)
                if host_node:
                    h = host_node.group(1)
                    for gpu, val in matrix.get("0", matrix.get(h, {})).items():
                        local = h == gpu
                        add_path(results, f"L{h}->H{gpu}", f"Host node {h} to GPU{gpu}",
                                 450.0 if local else None, val, "nvbandwidth", "local C2C" if local else "cross-socket")
            if "device_to_host" in path.name:
                host_node = re.search(r"host_on_node(\d+)", path.name)
                if host_node:
                    h = host_node.group(1)
                    row = matrix.get("0", matrix.get(h, {}))
                    for gpu, val in row.items():
                        local = h == gpu
                        add_path(results, f"H{gpu}->L{h}", f"GPU{gpu} to host node {h}",
                                 450.0 if local else None, val, "nvbandwidth", "local C2C" if local else "cross-socket")
            if "device_to_device_memcpy_read_ce" in path.name:
                val01 = matrix_value(matrix, "0", "1")
                val10 = matrix_value(matrix, "1", "0")
                if val01 is not None:
                    add_path(results, "H0->H1", "GPU0 HBM to GPU1 HBM", None, val01, "nvbandwidth", "SYS read")
                if val10 is not None:
                    add_path(results, "H1->H0", "GPU1 HBM to GPU0 HBM", None, val10, "nvbandwidth", "SYS read")

    for path in sorted((res_dir / "memory_modes").glob("*.json")) if (res_dir / "memory_modes").is_dir() else []:
        results.raw_files_seen.append(str(path.relative_to(res_dir)))
        parsed = parse_memory_modes(path)
        if parsed:
            results.memory_modes[path.stem] = parsed
        else:
            results.parse_failures.append(str(path.relative_to(res_dir)))

    latency_dir = res_dir / "latency"
    if latency_dir.is_dir():
        for path in sorted(latency_dir.glob("*.csv")):
            results.raw_files_seen.append(str(path.relative_to(res_dir)))
            parsed = parse_latency_csv(path)
            for label, by_size in parsed.items():
                results.latency_by_path.setdefault(label, {}).update(by_size)

    sustained_dir = res_dir / "sustained"
    if sustained_dir.is_dir():
        for path in sorted(sustained_dir.glob("*.txt")):
            results.raw_files_seen.append(str(path.relative_to(res_dir)))
            value = parse_sustained(path)
            if value is not None:
                results.sustained[path.stem] = value

    for path in sorted(res_dir.glob("mixbench_gpu*.txt")):
        results.raw_files_seen.append(str(path.relative_to(res_dir)))
        parsed = parse_mixbench(path)
        if parsed:
            results.mixbench[path.stem] = parsed

    existing_paths = {p.path_id for p in results.paths}
    if "H0->H1" not in existing_paths and "h0_to_h1_chunk16777216" in results.sustained:
        add_path(results, "H0->H1", "GPU0 HBM to GPU1 HBM", None,
                 results.sustained["h0_to_h1_chunk16777216"], "sustained", "16 MiB chunks; NVBandwidth D2D waived")
    if "H1->H0" not in existing_paths and "h1_to_h0_chunk16777216" in results.sustained:
        add_path(results, "H1->H0", "GPU1 HBM to GPU0 HBM", None,
                 results.sustained["h1_to_h0_chunk16777216"], "sustained", "16 MiB chunks; NVBandwidth D2D waived")

    return results


def plot_stream(results: Results, out_dir: Path) -> Optional[Path]:
    if not results.stream_by_threads:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    for node, by_thread in sorted(results.stream_by_threads.items()):
        xs = sorted(by_thread)
        ys = [by_thread[x] for x in xs]
        ax.plot(xs, ys, marker="o", label=f"Node {node}")
    ax.set_xlabel("OpenMP threads")
    ax.set_ylabel("STREAM Triad (GB/s)")
    ax.set_title("Grace LPDDR5X bandwidth")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out = out_dir / "stream_triad_by_threads.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_latency(results: Results, out_dir: Path) -> Optional[Path]:
    if not results.latency_by_path:
        return None
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, by_size in sorted(results.latency_by_path.items()):
        xs = sorted(by_size)
        ys = [by_size[x] for x in xs]
        ax.loglog(xs, ys, marker=".", label=label)
    ax.set_xlabel("Transfer size (bytes)")
    ax.set_ylabel("Median latency (us)")
    ax.set_title("Latency vs transfer size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    out = out_dir / "latency_vs_size.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_memory_modes(results: Results, out_dir: Path) -> Optional[Path]:
    if not results.memory_modes:
        return None
    labels: list[str] = []
    values: list[float] = []
    for config, modes in sorted(results.memory_modes.items()):
        for mode, gbs in sorted(modes.items()):
            labels.append(f"{config}:{mode}")
            values.append(gbs)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(values)), values)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("GB/s")
    ax.set_title("CUDA allocation mode bandwidth")
    ax.grid(True, axis="y", alpha=0.3)
    out = out_dir / "memory_modes_comparison.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_nvbw(results: Results, out_dir: Path) -> list[Path]:
    written: list[Path] = []
    for label, matrix in sorted(results.nvbw_matrices.items()):
        rows = sorted(matrix.keys(), key=int)
        cols = sorted({c for row in matrix.values() for c in row}, key=int)
        if not rows or not cols:
            continue
        data = np.full((len(rows), len(cols)), np.nan)
        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                if col in matrix.get(row, {}):
                    data[i, j] = matrix[row][col]
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(data, cmap="viridis")
        ax.set_xticks(range(len(cols)), cols)
        ax.set_yticks(range(len(rows)), rows)
        ax.set_xlabel("destination")
        ax.set_ylabel("source")
        ax.set_title(label[:90], fontsize=8)
        for i in range(len(rows)):
            for j in range(len(cols)):
                if not np.isnan(data[i, j]):
                    ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="GB/s")
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)[:80]
        out = out_dir / f"nvbandwidth_{safe}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
    return written


def plot_roofline(results: Results, out_dir: Path, run_dir: Path) -> Optional[Path]:
    res_dir = run_dir / "results"
    fig, ax = plt.subplots(figsize=(8, 5))
    wrote = False
    for path in sorted(res_dir.glob("mixbench_gpu*.txt")):
        text = safe_read(path)
        xs: list[float] = []
        ys: list[float] = []
        for line in text.splitlines():
            if not re.match(r"\s*\d+,", line):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                xs.append(float(parts[1]))
                ys.append(float(parts[3]))
            except ValueError:
                continue
        if xs and ys:
            ax.plot(xs, ys, marker=".", label=path.stem)
            wrote = True
    if not wrote:
        plt.close(fig)
        return None
    ax.set_xscale("log")
    ax.set_xlabel("Operational intensity (FLOP/byte)")
    ax.set_ylabel("SP GFLOP/s")
    ax.set_title("mixbench roofline-style curve")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    out = out_dir / "roofline_combined.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def write_outputs(run_dir: Path, results: Results, plots: list[Path]) -> None:
    payload = {
        "paths": [asdict(p) for p in results.paths],
        "stream_by_threads": results.stream_by_threads,
        "nvbw_matrices": results.nvbw_matrices,
        "latency_by_path": results.latency_by_path,
        "memory_modes": results.memory_modes,
        "sustained": results.sustained,
        "mixbench": results.mixbench,
        "parse_failures": results.parse_failures,
        "raw_files_seen": sorted(results.raw_files_seen),
    }
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2))

    lines = ["# Dual GH200 Memory Bandwidth Results", ""]
    lines.append(f"Run directory: `{run_dir}`")
    lines.append("")
    lines.append("## Full Path Table")
    lines.append("")
    lines.append("| Path | Description | Theoretical GB/s | Measured GB/s | % peak | Tool | Notes |")
    lines.append("|------|-------------|-----------------:|--------------:|-------:|------|-------|")
    for p in results.paths:
        theoretical = f"{p.theoretical_gbs:.0f}" if p.theoretical_gbs else "-"
        measured = f"{p.measured_gbs:.1f}" if p.measured_gbs is not None else "n/a"
        pct = f"{p.pct_of_peak:.0f}%" if p.pct_of_peak is not None else "-"
        lines.append(f"| `{p.path_id}` | {p.description} | {theoretical} | {measured} | {pct} | {p.tool} | {p.notes} |")
    if not results.paths:
        lines.append("| n/a | No path measurements parsed yet | - | n/a | - | - | Check raw outputs |")
    lines.append("")

    if results.memory_modes:
        lines.append("## Memory Modes")
        lines.append("")
        lines.append("| Config | Mode | Median GB/s |")
        lines.append("|--------|------|------------:|")
        for config, modes in sorted(results.memory_modes.items()):
            for mode, value in sorted(modes.items()):
                lines.append(f"| `{config}` | `{mode}` | {value:.1f} |")
        lines.append("")

    if results.sustained:
        lines.append("## Sustained Throughput")
        lines.append("")
        lines.append("| Path | GB/s |")
        lines.append("|------|-----:|")
        for path, value in sorted(results.sustained.items()):
            lines.append(f"| `{path}` | {value:.1f} |")
        lines.append("")

    if plots:
        lines.append("## Plots")
        lines.append("")

    if results.mixbench:
        lines.append("## Roofline")
        lines.append("")
        lines.append("| GPU | Max SP GFLOP/s | Max observed GB/s |")
        lines.append("|-----|---------------:|------------------:|")
        for gpu, vals in sorted(results.mixbench.items()):
            lines.append(f"| `{gpu}` | {vals['max_sp_gflops']:.1f} | {vals['max_sp_gbs']:.1f} |")
        lines.append("")
        for plot in plots:
            lines.append(f"![{plot.stem}]({plot.relative_to(run_dir)})")
            lines.append("")

    if results.parse_failures:
        lines.append("## Parse Failures")
        lines.append("")
        for failure in results.parse_failures:
            lines.append(f"- `{failure}`")
        lines.append("")

    lines.append(f"<details><summary>Raw files parsed ({len(results.raw_files_seen)})</summary>")
    lines.append("")
    for path in sorted(results.raw_files_seen):
        lines.append(f"- `{path}`")
    lines.append("")
    lines.append("</details>")
    lines.append("")
    (run_dir / "summary.md").write_text("\n".join(lines))

    methodology = [
        "# Dual GH200 Benchmark Methodology",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## System And NUMA Discovery",
        "",
        "The run captured `system-info.txt`, `numa-topology.txt`, and `gpu-numa-map.txt` before benchmarking. GPU0 was treated as attached to CPU NUMA node 0 and GPU1 as attached to CPU NUMA node 1 based on PCI sysfs `numa_node` plus `nvidia-smi topo -mp`.",
        "",
        "## NVBandwidth",
        "",
        "NVBandwidth v0.9 measured CUDA copy-engine and SM host/device paths. Host-touching tests were run twice under `numactl`, once bound to each Grace CPU NUMA node, so local C2C and cross-socket C2C+fabric paths can be separated. GPU-to-GPU NVBandwidth D2D tests were attempted but waived by NVBandwidth on this SYS topology; sustained custom-copy results are used for the cross-GPU headline instead.",
        "",
        "## STREAM",
        "",
        "STREAM measured Grace LPDDR bandwidth using OpenMP thread sweeps on each CPU NUMA node and cross-socket core/memory binding. Values in the summary are medians across three runs. Caveat: this run used the system's current CPU governor and THP setting because sudo was unavailable.",
        "",
        "## BabelStream",
        "",
        "BabelStream CUDA measured local HBM bandwidth on each GPU. The summary reports median Triad bandwidth across three runs per GPU. A size sweep on GPU0 was also recorded for saturation diagnostics.",
        "",
        "## CUDA Allocation Modes",
        "",
        "The custom `memory_modes` benchmark runs a read+write GPU kernel over device memory, NUMA-allocated pinned host memory, CUDA managed memory, CUDA pinned-host diagnostic memory, alternating managed CPU/GPU access, and unregistered NUMA-allocated system memory when CUDA pageable-memory access is supported. Host pages are first-touched on the requested NUMA node and sampled with `move_pages` where possible.",
        "",
        "## Latency Probe",
        "",
        "The custom latency probe measures CPU-observed copy latency using `cudaMemcpyAsync` plus stream synchronization for all `h0/h1/l0/l1` pairs and transfer sizes. Round-trip mode is CPU-orchestrated copy-engine latency, not a GPU-initiated cache-line ping-pong.",
        "",
        "## Sustained Throughput",
        "",
        "The sustained benchmark submits repeated async copies on one stream for 1 MiB and 16 MiB chunks, capped at 64 GiB total per path/chunk. These results quantify burst-vs-sustained behavior and provide cross-GPU measurements where NVBandwidth waived D2D tests.",
        "",
        "## mixbench",
        "",
        "mixbench measured compute-vs-bandwidth tradeoffs on both GPUs. `roofline_combined.png` plots the single-precision roofline-style curve from the raw CSV section in each mixbench output.",
        "",
        "## Known Caveats",
        "",
        "- CPU governor was `ondemand` and THP was `madvise`; passwordless sudo was unavailable to change them.",
        "- NVBandwidth D2D GPU-to-GPU tests were waived on SYS topology.",
        "- CUDA managed-memory placement is reported with observed page samples where available; managed memory may migrate during GPU access.",
    ]
    (run_dir / "methodology.md").write_text("\n".join(methodology) + "\n")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"Not a directory: {run_dir}", file=sys.stderr)
        return 1
    try:
        results = analyse(run_dir)
        plots: list[Path] = []
        if not args.no_plots:
            plots_dir = run_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            for fn in (plot_stream, plot_latency, plot_memory_modes):
                try:
                    plot = fn(results, plots_dir)
                    if plot:
                        plots.append(plot)
                except Exception:
                    traceback.print_exc()
            try:
                plots.extend(plot_nvbw(results, plots_dir))
            except Exception:
                traceback.print_exc()
            try:
                plot = plot_roofline(results, plots_dir, run_dir)
                if plot:
                    plots.append(plot)
            except Exception:
                traceback.print_exc()
        write_outputs(run_dir, results, plots)
    except Exception:
        traceback.print_exc()
        return 1
    print(f"wrote {run_dir / 'summary.md'}")
    print(f"wrote {run_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
