#!/usr/bin/env python3
"""Run a controlled official quick sweep for TOPK_* settings."""

from __future__ import annotations

import itertools
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
RESULTS_PATH = ROOT / "results" / "topk_sweep.md"

K_VALUES = [2, 3, 4, 5, 8, 12]
TEMP_VALUES = [0.25, 0.5, 1.0, 2.0]
CONSENSUS_VALUES = [0.0, 0.1, 0.2, 0.4]
DISTANCE_VALUES = ["euclidean", "cosine"]

DEFAULT_K = 5
DEFAULT_TEMP = 1.0
DEFAULT_CONSENSUS = 0.45
DEFAULT_DISTANCE = "euclidean"


@dataclass(frozen=True)
class SweepConfig:
    k: int
    temp: float
    consensus_weight: float
    distance: str


@dataclass(frozen=True)
class SweepResult:
    config: SweepConfig
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


def run_config(config: SweepConfig) -> SweepResult:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPYCACHEPREFIX": "/private/tmp/anvil_pycache",
            "AGENT_MODE": "topk",
            "TOPK_K": str(config.k),
            "TOPK_TEMP": str(config.temp),
            "TOPK_CONSENSUS_WEIGHT": str(config.consensus_weight),
            "TOPK_DISTANCE": config.distance,
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
            config=config,
            mean_delta=mean_delta,
            min_delta=min_delta,
            mean_spread=mean_spread,
            min_spread=min_spread,
            ok=proc.returncode == 0,
            note="" if proc.returncode == 0 else f"returncode={proc.returncode}",
        )
    except Exception as exc:
        return SweepResult(
            config=config,
            mean_delta=float("nan"),
            min_delta=float("nan"),
            mean_spread=float("nan"),
            min_spread=float("nan"),
            ok=False,
            note=str(exc),
        )


def one_factor_configs() -> list[SweepConfig]:
    configs: list[SweepConfig] = []
    for k in K_VALUES:
        configs.append(SweepConfig(k, DEFAULT_TEMP, DEFAULT_CONSENSUS, DEFAULT_DISTANCE))
    for temp in TEMP_VALUES:
        configs.append(SweepConfig(DEFAULT_K, temp, DEFAULT_CONSENSUS, DEFAULT_DISTANCE))
    for consensus in CONSENSUS_VALUES:
        configs.append(SweepConfig(DEFAULT_K, DEFAULT_TEMP, consensus, DEFAULT_DISTANCE))
    for distance in DISTANCE_VALUES:
        configs.append(SweepConfig(DEFAULT_K, DEFAULT_TEMP, DEFAULT_CONSENSUS, distance))

    deduped: list[SweepConfig] = []
    seen: set[SweepConfig] = set()
    for config in configs:
        if config not in seen:
            deduped.append(config)
            seen.add(config)
    return deduped


def full_grid_configs() -> list[SweepConfig]:
    return [
        SweepConfig(k, temp, consensus, distance)
        for k, temp, consensus, distance in itertools.product(
            K_VALUES, TEMP_VALUES, CONSENSUS_VALUES, DISTANCE_VALUES
        )
    ]


def format_table(results: list[SweepResult]) -> str:
    lines = [
        "| k | temp | consensus | distance | mean delta | min delta | mean spread | min spread | ok | note |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for result in results:
        c = result.config
        lines.append(
            f"| {c.k} | {c.temp:g} | {c.consensus_weight:g} | {c.distance} | "
            f"{result.mean_delta:+.3f} | {result.min_delta:+.3f} | "
            f"{result.mean_spread:.2f}x | {result.min_spread:.2f}x | "
            f"{'yes' if result.ok else 'no'} | {result.note} |"
        )
    return "\n".join(lines)


def best_result(results: list[SweepResult]) -> SweepResult | None:
    valid = [r for r in results if r.ok]
    if not valid:
        return None
    return max(valid, key=lambda r: (r.mean_delta, r.min_delta, r.mean_spread))


def main() -> int:
    full_grid = "--full-grid" in sys.argv[1:]
    configs = full_grid_configs() if full_grid else one_factor_configs()
    results: list[SweepResult] = []
    for index, config in enumerate(configs, start=1):
        print(
            f"[{index:02d}/{len(configs):02d}] "
            f"k={config.k} temp={config.temp:g} "
            f"consensus={config.consensus_weight:g} distance={config.distance}",
            flush=True,
        )
        result = run_config(config)
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
        f.write("# Top-k Official Quick Sweep\n\n")
        f.write("Mode: `topk`\n\n")
        f.write("Scope: ")
        f.write("full grid" if full_grid else "one-factor sweep around current defaults")
        f.write("\n\n")
        f.write(table)
        f.write("\n")
        if best is not None:
            c = best.config
            f.write("\n## Best By Mean Delta\n\n")
            f.write(
                f"- k: `{c.k}`\n"
                f"- temp: `{c.temp:g}`\n"
                f"- consensus weight: `{c.consensus_weight:g}`\n"
                f"- distance: `{c.distance}`\n"
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
