#!/usr/bin/env python3
"""Sweep spread optimizer budget and scale on official quick mode."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
RESULTS_PATH = ROOT / "results" / "spread_optimizer_sweep.md"

BUDGETS = [20, 50, 100, 200]
SCALES = [0.2, 0.4, 0.8]


@dataclass(frozen=True)
class OptimizerResult:
    budget: int
    scale: float
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


def run_config(budget: int, scale: float) -> OptimizerResult:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPYCACHEPREFIX": "/private/tmp/anvil_pycache",
            "AGENT_MODE": "adaptive_topk_spread",
            "SPREAD_OPT_BUDGET": str(budget),
            "SPREAD_OPT_SCALE": str(scale),
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
        return OptimizerResult(
            budget=budget,
            scale=scale,
            mean_delta=mean_delta,
            min_delta=min_delta,
            mean_spread=mean_spread,
            min_spread=min_spread,
            ok=proc.returncode == 0,
            note="" if proc.returncode == 0 else f"returncode={proc.returncode}",
        )
    except Exception as exc:
        return OptimizerResult(
            budget=budget,
            scale=scale,
            mean_delta=float("nan"),
            min_delta=float("nan"),
            mean_spread=float("nan"),
            min_spread=float("nan"),
            ok=False,
            note=str(exc),
        )


def format_table(results: list[OptimizerResult]) -> str:
    lines = [
        "| budget | scale | mean delta | min delta | mean spread | min spread | ok | note |",
        "|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for result in results:
        lines.append(
            f"| {result.budget} | {result.scale:g} | "
            f"{result.mean_delta:+.3f} | {result.min_delta:+.3f} | "
            f"{result.mean_spread:.2f}x | {result.min_spread:.2f}x | "
            f"{'yes' if result.ok else 'no'} | {result.note} |"
        )
    return "\n".join(lines)


def best_result(results: list[OptimizerResult]) -> OptimizerResult | None:
    valid = [r for r in results if r.ok]
    if not valid:
        return None
    return max(valid, key=lambda r: (r.mean_delta, r.min_delta, r.mean_spread))


def main() -> int:
    results: list[OptimizerResult] = []
    total = len(BUDGETS) * len(SCALES)
    index = 0
    for budget in BUDGETS:
        for scale in SCALES:
            index += 1
            print(
                f"[{index:02d}/{total:02d}] budget={budget} scale={scale:g}",
                flush=True,
            )
            result = run_config(budget, scale)
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
        f.write("# Spread Optimizer Budget Sweep\n\n")
        f.write("Mode: `adaptive_topk_spread`\n\n")
        f.write(table)
        f.write("\n")
        if best is not None:
            f.write("\n## Best By Mean Delta\n\n")
            f.write(
                f"- budget: `{best.budget}`\n"
                f"- scale: `{best.scale:g}`\n"
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
