#!/usr/bin/env python3
"""One-factor robustness sweep around adaptive_topk_spread top-k defaults."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
RESULTS_PATH = ROOT / "results" / "adaptive_topk_robust_sweep.md"
TMP_DIR = ROOT / "results" / "tmp_topk_robust"

SEEDS = [7, 13, 31, 42, 97, 101, 211, 503, 1009, 2027, 3141, 9999]
K_VALUES = [8, 10, 12, 14, 16]
TEMP_VALUES = [0.8, 1.0, 1.2]
CONSENSUS_VALUES = [0.35, 0.45, 0.55]
DEFAULT_K = 12
DEFAULT_TEMP = 1.0
DEFAULT_CONSENSUS = 0.45

# This is intentionally a smaller robustness sweep than full judging mode. The
# 20-seed full run is the gate; this sweep only looks for safe top-k improvements.
N_PER_LEVEL = int(os.environ.get("ROBUST_SWEEP_N_PER_LEVEL", "100"))
N_ANISOTROPY = int(os.environ.get("ROBUST_SWEEP_N_ANISOTROPY", "16"))


@dataclass(frozen=True)
class TopKConfig:
    k: int
    temp: float
    consensus: float


@dataclass(frozen=True)
class SweepResult:
    config: TopKConfig
    mean_delta: float
    min_delta: float
    mean_spread: float
    min_spread: float
    negative_delta_count: int
    spread_leq_one_count: int
    runtime_s: float
    ok: bool
    note: str


def configs() -> list[TopKConfig]:
    raw: list[TopKConfig] = []
    for k in K_VALUES:
        raw.append(TopKConfig(k, DEFAULT_TEMP, DEFAULT_CONSENSUS))
    for temp in TEMP_VALUES:
        raw.append(TopKConfig(DEFAULT_K, temp, DEFAULT_CONSENSUS))
    for consensus in CONSENSUS_VALUES:
        raw.append(TopKConfig(DEFAULT_K, DEFAULT_TEMP, consensus))

    seen: set[TopKConfig] = set()
    deduped: list[TopKConfig] = []
    for cfg in raw:
        if cfg not in seen:
            seen.add(cfg)
            deduped.append(cfg)
    return deduped


def summarize_json(path: Path, runtime_s: float, cfg: TopKConfig, ok: bool, note: str) -> SweepResult:
    if not ok:
        return SweepResult(cfg, float("nan"), float("nan"), float("nan"), float("nan"), 0, 0, runtime_s, False, note)
    with path.open() as f:
        report = json.load(f)
    per_seed = report["per_seed"]
    agg = report["aggregated"]
    neg = sum(1 for row in per_seed if float(row["delta"]) < 0.0)
    spread_bad = sum(1 for row in per_seed if float(row["spread_reduction"]) <= 1.0)
    return SweepResult(
        config=cfg,
        mean_delta=float(agg["mean_delta"]),
        min_delta=float(agg["min_delta"]),
        mean_spread=float(agg["mean_spread"]),
        min_spread=float(agg["min_spread"]),
        negative_delta_count=neg,
        spread_leq_one_count=spread_bad,
        runtime_s=runtime_s,
        ok=True,
        note="",
    )


def run_config(cfg: TopKConfig, index: int) -> SweepResult:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TMP_DIR / f"topk_k{cfg.k}_t{cfg.temp:g}_c{cfg.consensus:g}.json"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPYCACHEPREFIX": "/private/tmp/anvil_pycache",
            "AGENT_MODE": "adaptive_topk_spread",
            "TOPK_K": str(cfg.k),
            "TOPK_TEMP": str(cfg.temp),
            "TOPK_CONSENSUS_WEIGHT": str(cfg.consensus),
            "TOPK_DISTANCE": "euclidean",
        }
    )
    cmd = [
        PYTHON,
        "run.py",
        "--adapter",
        "adapters.myteam:Engine",
        "--seeds",
        *[str(seed) for seed in SEEDS],
        "--n-per-level",
        str(N_PER_LEVEL),
        "--n-anisotropy",
        str(N_ANISOTROPY),
        "--out",
        str(out_path),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    runtime_s = time.perf_counter() - t0
    ok = proc.returncode == 0 and out_path.exists()
    note = "" if ok else f"returncode={proc.returncode}: {proc.stdout[-300:]}"
    return summarize_json(out_path, runtime_s, cfg, ok, note)


def format_table(results: list[SweepResult]) -> str:
    lines = [
        "| k | temp | consensus | mean delta | min delta | mean spread | min spread | neg delta | spread <= 1 | runtime s | ok | note |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.config.k} | {r.config.temp:g} | {r.config.consensus:g} | "
            f"{r.mean_delta:+.3f} | {r.min_delta:+.3f} | "
            f"{r.mean_spread:.2f}x | {r.min_spread:.2f}x | "
            f"{r.negative_delta_count} | {r.spread_leq_one_count} | "
            f"{r.runtime_s:.1f} | {'yes' if r.ok else 'no'} | {r.note} |"
        )
    return "\n".join(lines)


def best_candidate(results: list[SweepResult]) -> SweepResult | None:
    valid = [
        r
        for r in results
        if r.ok and r.negative_delta_count == 0 and r.spread_leq_one_count == 0
    ]
    if not valid:
        return None
    return max(valid, key=lambda r: (r.min_delta, r.mean_delta, r.min_spread, r.mean_spread))


def main() -> int:
    all_configs = configs()
    results: list[SweepResult] = []
    for index, cfg in enumerate(all_configs, start=1):
        print(
            f"[{index:02d}/{len(all_configs):02d}] "
            f"k={cfg.k} temp={cfg.temp:g} consensus={cfg.consensus:g}",
            flush=True,
        )
        result = run_config(cfg, index)
        results.append(result)
        print(
            f"    mean_delta={result.mean_delta:+.3f} "
            f"min_delta={result.min_delta:+.3f} "
            f"mean_spread={result.mean_spread:.2f}x "
            f"min_spread={result.min_spread:.2f}x "
            f"neg={result.negative_delta_count} "
            f"spread_bad={result.spread_leq_one_count} "
            f"runtime={result.runtime_s:.1f}s",
            flush=True,
        )

    table = format_table(results)
    best = best_candidate(results)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w") as f:
        f.write("# Adaptive Top-k Robustness Sweep\n\n")
        f.write("Mode: `adaptive_topk_spread`\n\n")
        f.write(f"Seeds: `{SEEDS}`\n\n")
        f.write(f"`n_per_level={N_PER_LEVEL}`, `n_anisotropy={N_ANISOTROPY}`\n\n")
        f.write("This is a one-factor sweep around the current default, not a full Cartesian grid.\n\n")
        f.write(table)
        f.write("\n")
        if best is not None:
            c = best.config
            f.write("\n## Best Gate-Passing Candidate\n\n")
            f.write(
                f"- k: `{c.k}`\n"
                f"- temp: `{c.temp:g}`\n"
                f"- consensus weight: `{c.consensus:g}`\n"
                f"- mean delta: `{best.mean_delta:+.3f}`\n"
                f"- min delta: `{best.min_delta:+.3f}`\n"
                f"- mean spread: `{best.mean_spread:.2f}x`\n"
                f"- min spread: `{best.min_spread:.2f}x`\n"
                f"- runtime: `{best.runtime_s:.1f}s`\n"
            )

    print()
    print(table)
    print()
    print(f"wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
