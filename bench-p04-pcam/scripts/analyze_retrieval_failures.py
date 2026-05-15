"""Query-level retrieval error analysis for the official P-04 harness."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.dummy import DummyAgent
from adapters.myteam import Engine
from data import make_patterns, make_test_queries
from harness import pack_params
from pcam_model import PCAMModel, build_default_R
from src.utils.distances import euclidean_distances


DEFAULT_SEEDS = [42, 101]


def _policy_label(distances: np.ndarray) -> str:
    sorted_distances = np.sort(np.asarray(distances, dtype=float))
    d1 = float(sorted_distances[0])
    d2 = float(sorted_distances[1]) if sorted_distances.size > 1 else d1
    margin = d2 - d1
    ratio = d2 / (d1 + 1e-9)
    policy = os.environ.get("RETRIEVAL_POLICY", "fixed").strip().lower()
    if policy != "adaptive_margin":
        return "fixed:k=12,temp=1.0,consensus=0.45"

    margin_threshold = float(os.environ.get("MARGIN_THRESHOLD", 0.015))
    ratio_threshold = float(os.environ.get("RATIO_THRESHOLD", 1.10))
    distance_threshold = float(os.environ.get("DISTANCE_THRESHOLD_RETRIEVAL", 0.95))
    if margin <= margin_threshold or ratio <= ratio_threshold:
        k = os.environ.get("ADAPTIVE_K_AMBIG", "16")
        temp = os.environ.get("ADAPTIVE_TEMP_AMBIG", "1.2")
        return f"adaptive:ambiguous:k={k},temp={temp}"
    if d1 >= distance_threshold:
        k = os.environ.get("ADAPTIVE_K_NOISY", "14")
        temp = os.environ.get("ADAPTIVE_TEMP_NOISY", os.environ.get("ADAPTIVE_TEMP_AMBIG", "1.2"))
        return f"adaptive:noisy:k={k},temp={temp}"
    return "adaptive:stable:k=12,temp=1.0"


def _init_bucket() -> dict[str, float]:
    return {
        "n": 0.0,
        "base_correct": 0.0,
        "agent_correct": 0.0,
        "helped": 0.0,
        "hurt": 0.0,
        "same_wrong": 0.0,
        "same_right": 0.0,
        "mean_d1": 0.0,
        "mean_margin": 0.0,
        "mean_ratio": 0.0,
    }


def _accumulate(bucket: dict[str, float], row: dict[str, Any]) -> None:
    bucket["n"] += 1.0
    bucket["base_correct"] += float(row["base_correct"])
    bucket["agent_correct"] += float(row["agent_correct"])
    bucket["helped"] += float(row["helped"])
    bucket["hurt"] += float(row["hurt"])
    bucket["same_wrong"] += float((not row["base_correct"]) and (not row["agent_correct"]))
    bucket["same_right"] += float(row["base_correct"] and row["agent_correct"])
    bucket["mean_d1"] += float(row["d1"])
    bucket["mean_margin"] += float(row["margin"])
    bucket["mean_ratio"] += float(row["ratio"])


def _finalize(bucket: dict[str, float]) -> dict[str, float]:
    n = max(bucket["n"], 1.0)
    return {
        "n": bucket["n"],
        "base_acc": bucket["base_correct"] / n,
        "agent_acc": bucket["agent_correct"] / n,
        "delta": (bucket["agent_correct"] - bucket["base_correct"]) / n,
        "helped": bucket["helped"],
        "hurt": bucket["hurt"],
        "same_wrong": bucket["same_wrong"],
        "same_right": bucket["same_right"],
        "mean_d1": bucket["mean_d1"] / n,
        "mean_margin": bucket["mean_margin"] / n,
        "mean_ratio": bucket["mean_ratio"] / n,
    }


def _format_summary_table(title: str, buckets: dict[Any, dict[str, float]]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Group | N | Base Acc | Agent Acc | Delta | Helped | Hurt | Same Wrong | Mean d1 | Mean Margin | Mean Ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, bucket in buckets.items():
        stats = _finalize(bucket)
        lines.append(
            f"| {key} | {int(stats['n'])} | {stats['base_acc']:.3f} | "
            f"{stats['agent_acc']:.3f} | {stats['delta']:+.3f} | "
            f"{int(stats['helped'])} | {int(stats['hurt'])} | {int(stats['same_wrong'])} | "
            f"{stats['mean_d1']:.4f} | {stats['mean_margin']:.4f} | {stats['mean_ratio']:.3f} |"
        )
    return lines


def analyze(args: argparse.Namespace) -> str:
    t0 = time.monotonic()
    rows: list[dict[str, Any]] = []
    by_seed: dict[int, dict[str, float]] = defaultdict(_init_bucket)
    by_noise: dict[float, dict[str, float]] = defaultdict(_init_bucket)
    by_margin: dict[str, dict[str, float]] = defaultdict(_init_bucket)
    by_distance: dict[str, dict[str, float]] = defaultdict(_init_bucket)

    for seed in args.seeds:
        X = make_patterns(K=args.K, N=args.N, seed=seed)
        R = build_default_R(N=args.N, seed=seed)
        model = PCAMModel(X, R)
        params = pack_params(model)
        agent = Engine(X, params)
        baseline = DummyAgent(X, params)
        queries, truths, levels = make_test_queries(
            X,
            args.noise_levels,
            args.n_per_level,
            seed=seed,
        )

        for q_idx, (q, truth, level) in enumerate(zip(queries, truths, levels)):
            distances = euclidean_distances(X, q)
            order = np.argsort(distances)
            d1 = float(distances[order[0]])
            d2 = float(distances[order[1]]) if len(order) > 1 else d1
            margin = d2 - d1
            ratio = d2 / (d1 + 1e-9)

            base_pi = baseline.predict_precision(q)
            agent_pi = agent.predict_precision(q)
            base_pred = model.classify(model.run(q, base_pi, u_const=q))
            agent_pred = model.classify(model.run(q, agent_pi, u_const=q))
            base_correct = base_pred == int(truth)
            agent_correct = agent_pred == int(truth)
            row = {
                "seed": int(seed),
                "query": int(q_idx),
                "noise": float(level),
                "truth": int(truth),
                "base_pred": int(base_pred),
                "agent_pred": int(agent_pred),
                "base_correct": bool(base_correct),
                "agent_correct": bool(agent_correct),
                "helped": (not base_correct) and agent_correct,
                "hurt": base_correct and (not agent_correct),
                "d1": d1,
                "d2": d2,
                "margin": margin,
                "ratio": ratio,
                "policy": _policy_label(distances),
            }
            rows.append(row)
            _accumulate(by_seed[int(seed)], row)
            _accumulate(by_noise[float(level)], row)
            margin_key = "<=0.01" if margin <= 0.01 else "<=0.03" if margin <= 0.03 else ">0.03"
            distance_key = ">=0.95" if d1 >= 0.95 else ">=0.80" if d1 >= 0.80 else "<0.80"
            _accumulate(by_margin[margin_key], row)
            _accumulate(by_distance[distance_key], row)

    weakest = sorted(
        ((seed, _finalize(bucket)) for seed, bucket in by_seed.items()),
        key=lambda item: item[1]["delta"],
    )
    hurts = [row for row in rows if row["hurt"]]
    helps = [row for row in rows if row["helped"]]

    lines = [
        "# Retrieval Error Analysis",
        "",
        f"Seeds: `{args.seeds}`",
        f"Noise levels: `{args.noise_levels}`",
        f"Queries per level: `{args.n_per_level}`",
        f"Runtime: `{time.monotonic() - t0:.1f}s`",
        f"Policy: `{os.environ.get('RETRIEVAL_POLICY', 'fixed')}`",
        "",
        "## Overall",
        "",
    ]
    overall = _init_bucket()
    for row in rows:
        _accumulate(overall, row)
    stats = _finalize(overall)
    lines += [
        f"- Baseline accuracy: `{stats['base_acc']:.3f}`",
        f"- Agent accuracy: `{stats['agent_acc']:.3f}`",
        f"- Delta: `{stats['delta']:+.3f}`",
        f"- Helped queries: `{int(stats['helped'])}`",
        f"- Hurt queries: `{int(stats['hurt'])}`",
        f"- Same-wrong queries: `{int(stats['same_wrong'])}`",
        "",
        "## Weakest Seeds",
        "",
    ]
    for seed, seed_stats in weakest[:8]:
        lines.append(
            f"- Seed `{seed}`: delta `{seed_stats['delta']:+.3f}`, "
            f"agent `{seed_stats['agent_acc']:.3f}`, baseline `{seed_stats['base_acc']:.3f}`, "
            f"helped `{int(seed_stats['helped'])}`, hurt `{int(seed_stats['hurt'])}`"
        )
    lines.append("")
    lines += _format_summary_table("By Seed", dict(sorted(by_seed.items())))
    lines.append("")
    lines += _format_summary_table("By Noise Level", dict(sorted(by_noise.items())))
    lines.append("")
    lines += _format_summary_table("By Nearest-Neighbor Margin", dict(sorted(by_margin.items())))
    lines.append("")
    lines += _format_summary_table("By Nearest Distance", dict(sorted(by_distance.items())))
    lines += [
        "",
        "## Pattern Notes",
        "",
        "- Small nearest-neighbor margin is a useful ambiguity signal when its hurt/help balance differs from the overall average.",
        "- Large nearest distance is a useful noise signal when high-distance buckets have weaker delta.",
        f"- Hurt examples inspected: `{len(hurts)}`; helped examples inspected: `{len(helps)}`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze query-level retrieval failures.")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.5, 0.7, 0.8])
    parser.add_argument("--n-per-level", type=int, default=250)
    parser.add_argument("--out", default="results/retrieval_error_analysis.md")
    args = parser.parse_args()
    text = analyze(args)
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
