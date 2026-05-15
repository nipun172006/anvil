#!/usr/bin/env python3
"""Sweep top-k/Hessian blend weights on the official quick check."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
RESULTS_PATH = ROOT / "results" / "hessian_geometry_sweep.md"
GEOM_WEIGHTS = [0.05, 0.10, 0.15, 0.25, 0.40, 0.60]
MODES = ["topk_hdiag", "topk_ruiz"]


@dataclass(frozen=True)
class SweepResult:
    mode: str
    topk_weight: float
    geom_weight: float
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


def run_config(mode: str, geom_weight: float) -> SweepResult:
    topk_weight = 1.0 - geom_weight
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPYCACHEPREFIX": "/private/tmp/anvil_pycache",
            "AGENT_MODE": mode,
            "TOPK_WEIGHT": str(topk_weight),
            "GEOM_WEIGHT": str(geom_weight),
        }
    )
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
        return SweepResult(
            mode=mode,
            topk_weight=topk_weight,
            geom_weight=geom_weight,
            mean_delta=mean_delta,
            min_delta=min_delta,
            mean_spread=mean_spread,
            min_spread=min_spread,
            ok=proc.returncode == 0,
            note="" if proc.returncode == 0 else f"returncode={proc.returncode}",
        )
    except Exception as exc:
        return SweepResult(
            mode=mode,
            topk_weight=topk_weight,
            geom_weight=geom_weight,
            mean_delta=float("nan"),
            min_delta=float("nan"),
            mean_spread=float("nan"),
            min_spread=float("nan"),
            ok=False,
            note=str(exc),
        )


def format_table(results: list[SweepResult]) -> str:
    lines = [
        "| mode | topk weight | geom weight | mean delta | min delta | mean spread | min spread | ok | note |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for result in results:
        lines.append(
            f"| {result.mode} | {result.topk_weight:.2f} | {result.geom_weight:.2f} | "
            f"{result.mean_delta:+.3f} | {result.min_delta:+.3f} | "
            f"{result.mean_spread:.2f}x | {result.min_spread:.2f}x | "
            f"{'yes' if result.ok else 'no'} | {result.note} |"
        )
    return "\n".join(lines)


def best_for_mode(results: list[SweepResult], mode: str) -> SweepResult | None:
    valid = [r for r in results if r.ok and r.mode == mode]
    if not valid:
        return None
    return max(valid, key=lambda r: (r.mean_delta, r.min_delta, r.mean_spread))


def main() -> int:
    results: list[SweepResult] = []
    total = len(MODES) * len(GEOM_WEIGHTS)
    index = 0
    for mode in MODES:
        for geom_weight in GEOM_WEIGHTS:
            index += 1
            print(
                f"[{index:02d}/{total:02d}] mode={mode} "
                f"topk={1.0 - geom_weight:.2f} geom={geom_weight:.2f}",
                flush=True,
            )
            result = run_config(mode, geom_weight)
            results.append(result)
            print(
                f"    mean_delta={result.mean_delta:+.3f} "
                f"min_delta={result.min_delta:+.3f} "
                f"mean_spread={result.mean_spread:.2f}x "
                f"min_spread={result.min_spread:.2f}x",
                flush=True,
            )

    table = format_table(results)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w") as f:
        f.write("# Hessian Geometry Blend Sweep\n\n")
        f.write("Modes: `topk_hdiag`, `topk_ruiz`\n\n")
        f.write(table)
        f.write("\n")
        for mode in MODES:
            best = best_for_mode(results, mode)
            if best is None:
                continue
            f.write(f"\n## Best `{mode}` By Mean Delta\n\n")
            f.write(
                f"- topk weight: `{best.topk_weight:.2f}`\n"
                f"- geom weight: `{best.geom_weight:.2f}`\n"
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
