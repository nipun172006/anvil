#!/usr/bin/env python3
"""Summarize an official P-04 JSON report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_report(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def build_summary(report: dict[str, Any]) -> str:
    rows = report.get("per_seed", [])
    agg = report.get("aggregated", {})
    score = report.get("score", {})

    deltas = [float(r["delta"]) for r in rows]
    spreads = [float(r["spread_reduction"]) for r in rows]
    negative_delta = sum(1 for d in deltas if d < 0.0)
    spread_leq_one = sum(1 for s in spreads if s <= 1.0)

    lines: list[str] = []
    lines.append("# Adaptive Many-Seed Summary")
    lines.append("")
    lines.append("| Seed | Baseline Acc | Agent Acc | Delta | Base Spread | Agent Spread | Spread Ratio |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {int(r['seed'])} | "
            f"{float(r['baseline_accuracy']):.3f} | "
            f"{float(r['agent_accuracy']):.3f} | "
            f"{float(r['delta']):+.3f} | "
            f"{float(r['spread_baseline']):.2f} | "
            f"{float(r['spread_agent']):.2f} | "
            f"{float(r['spread_reduction']):.2f}x |"
        )

    lines.append("")
    lines.append("## Aggregated")
    lines.append("")
    lines.append(f"- Mean delta: `{float(agg.get('mean_delta', np.mean(deltas))):+.3f}`")
    lines.append(f"- Min delta: `{float(agg.get('min_delta', np.min(deltas))):+.3f}`")
    lines.append(f"- Mean spread ratio: `{float(agg.get('mean_spread', np.mean(spreads))):.2f}x`")
    lines.append(f"- Min spread ratio: `{float(agg.get('min_spread', np.min(spreads))):.2f}x`")
    lines.append(f"- Negative-delta seeds: `{negative_delta}`")
    lines.append(f"- Spread `<= 1.0x` seeds: `{spread_leq_one}`")
    if score:
        lines.append(f"- Retrieval points: `{score.get('retrieval_pts')}`")
        lines.append(f"- Anisotropy points: `{score.get('anisotropy_pts')}`")
        lines.append(f"- Total automated: `{score.get('total_automated')} / {score.get('max_automated')}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize P-04 report JSON")
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    summary = build_summary(load_report(args.report))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(summary)
    print(summary, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
