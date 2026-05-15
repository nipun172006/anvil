#!/usr/bin/env python3
"""Sweep clean-query detection thresholds for adaptive_topk_spread."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
RESULTS_PATH = ROOT / "results" / "adaptive_threshold_sweep.md"

DISTANCE_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00]
COSINE_THRESHOLDS = [0.90, 0.94, 0.96, 0.98, 0.99]


@dataclass(frozen=True)
class ThresholdResult:
    mode: str
    distance_threshold: float | None
    cosine_threshold: float | None
    mean_delta: float
    min_delta: float
    mean_spread: float
    min_spread: float
    ok: bool
    note: str


def parse_metric(pattern: str, text: str) -> float:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"missing metric: {pattern}")
    return float(match.group(1))


def run_config(
    detect_mode: str,
    distance_threshold: float | None = None,
    cosine_threshold: float | None = None,
) -> ThresholdResult:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPYCACHEPREFIX": "/private/tmp/anvil_pycache",
            "AGENT_MODE": "adaptive_topk_spread",
            "CLEAN_DETECT_MODE": detect_mode,
        }
    )
    if distance_threshold is not None:
        env["CLEAN_DISTANCE_THRESHOLD"] = str(distance_threshold)
    if cosine_threshold is not None:
        env["CLEAN_COSINE_THRESHOLD"] = str(cosine_threshold)

    cmd = [
        PYTHON,
        "self_check.py",
        "--adapter",
        "adapters.myteam:Engine",
        "--quick",
    ]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = proc.stdout
    try:
        mean_delta = parse_metric(r"mean Δ accuracy \(over seeds\)\s+([+-]?\d+\.\d+)", output)
        min_delta = parse_metric(r"min\s+Δ accuracy \(worst seed\)\s+([+-]?\d+\.\d+)", output)
        mean_spread = parse_metric(r"mean spread reduction\s+(\d+\.\d+)×", output)
        min_spread = parse_metric(r"min\s+spread reduction\s+(\d+\.\d+)×", output)
        return ThresholdResult(
            mode=detect_mode,
            distance_threshold=distance_threshold,
            cosine_threshold=cosine_threshold,
            mean_delta=mean_delta,
            min_delta=min_delta,
            mean_spread=mean_spread,
            min_spread=min_spread,
            ok=proc.returncode == 0,
            note="" if proc.returncode == 0 else f"returncode={proc.returncode}",
        )
    except Exception as exc:
        return ThresholdResult(
            mode=detect_mode,
            distance_threshold=distance_threshold,
            cosine_threshold=cosine_threshold,
            mean_delta=float("nan"),
            min_delta=float("nan"),
            mean_spread=float("nan"),
            min_spread=float("nan"),
            ok=False,
            note=str(exc),
        )


def format_table(results: list[ThresholdResult]) -> str:
    lines = [
        "| detect mode | distance threshold | cosine threshold | mean delta | min delta | mean spread | min spread | ok | note |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for result in results:
        dist = "" if result.distance_threshold is None else f"{result.distance_threshold:g}"
        cos = "" if result.cosine_threshold is None else f"{result.cosine_threshold:g}"
        lines.append(
            f"| {result.mode} | {dist} | {cos} | "
            f"{result.mean_delta:+.3f} | {result.min_delta:+.3f} | "
            f"{result.mean_spread:.2f}x | {result.min_spread:.2f}x | "
            f"{'yes' if result.ok else 'no'} | {result.note} |"
        )
    return "\n".join(lines)


def best_result(results: list[ThresholdResult]) -> ThresholdResult | None:
    valid = [r for r in results if r.ok]
    if not valid:
        return None
    return max(valid, key=lambda r: (r.mean_delta, r.min_delta, r.mean_spread))


def main() -> int:
    results: list[ThresholdResult] = []
    total = len(DISTANCE_THRESHOLDS) + len(COSINE_THRESHOLDS)
    index = 0
    for threshold in DISTANCE_THRESHOLDS:
        index += 1
        print(
            f"[{index:02d}/{total:02d}] mode=distance "
            f"distance_threshold={threshold:g}",
            flush=True,
        )
        result = run_config("distance", distance_threshold=threshold)
        results.append(result)
        print(
            f"    mean_delta={result.mean_delta:+.3f} "
            f"min_delta={result.min_delta:+.3f} "
            f"mean_spread={result.mean_spread:.2f}x "
            f"min_spread={result.min_spread:.2f}x",
            flush=True,
        )

    for threshold in COSINE_THRESHOLDS:
        index += 1
        print(
            f"[{index:02d}/{total:02d}] mode=cosine "
            f"cosine_threshold={threshold:g}",
            flush=True,
        )
        result = run_config("cosine", cosine_threshold=threshold)
        results.append(result)
        print(
            f"    mean_delta={result.mean_delta:+.3f} "
            f"min_delta={result.min_delta:+.3f} "
            f"mean_spread={result.mean_spread:.2f}x "
            f"min_spread={result.min_spread:.2f}x",
            flush=True,
        )

    table = format_table(results)
    best = best_result(results)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w") as f:
        f.write("# Adaptive Threshold Sweep\n\n")
        f.write("Mode: `adaptive_topk_spread`\n\n")
        f.write(table)
        f.write("\n")
        if best is not None:
            f.write("\n## Best By Mean Delta\n\n")
            f.write(
                f"- detect mode: `{best.mode}`\n"
                f"- distance threshold: `{best.distance_threshold}`\n"
                f"- cosine threshold: `{best.cosine_threshold}`\n"
                f"- mean delta: `{best.mean_delta:+.3f}`\n"
                f"- min delta: `{best.min_delta:+.3f}`\n"
                f"- mean spread reduction: `{best.mean_spread:.2f}x`\n"
                f"- min spread reduction: `{best.min_spread:.2f}x`\n"
            )

    print()
    print(table)
    print()
    print(f"wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
