"""12-seed guarded adaptive-retrieval sweep for P-04."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.myteam import Engine
from data import make_patterns, make_test_queries
from harness import run_multi
from src.utils.distances import euclidean_distances


SEEDS_12 = [7, 13, 31, 42, 97, 101, 211, 503, 1009, 2027, 3141, 9999]
NOISE_LEVELS = [0.5, 0.7, 0.8]
N_PER_LEVEL = 250
K = 16
N = 64
N_ANISO = 16

ENV_KEYS = {
    "RETRIEVAL_POLICY",
    "ADAPTIVE_K_AMBIG",
    "ADAPTIVE_K_NOISY",
    "ADAPTIVE_TEMP_AMBIG",
    "ADAPTIVE_TEMP_NOISY",
    "CLEAN_DETECT_MODE",
    "CLEAN_DISTANCE_THRESHOLD",
    "CLEAN_COSINE_THRESHOLD",
    "RETRIEVAL_GUARD_DISTANCE",
    "RETRIEVAL_GUARD_COSINE",
}


def _base_configs() -> list[dict[str, Any]]:
    aggressive = {
        "RETRIEVAL_POLICY": "adaptive_margin",
        "ADAPTIVE_K_AMBIG": "16",
        "ADAPTIVE_K_NOISY": "16",
        "ADAPTIVE_TEMP_AMBIG": "1.2",
    }
    conservative = {
        "RETRIEVAL_POLICY": "adaptive_margin",
        "ADAPTIVE_K_AMBIG": "14",
        "ADAPTIVE_K_NOISY": "12",
        "ADAPTIVE_TEMP_AMBIG": "1.1",
    }
    configs: list[dict[str, Any]] = [
        {
            "name": "stable",
            "env": {},
        },
    ]
    for threshold in ("0.20", "0.30", "0.40", "0.60", "1.00"):
        configs.append(
            {
                "name": f"aggressive_distance_{threshold}",
                "env": {
                    **aggressive,
                    "CLEAN_DETECT_MODE": "distance",
                    "CLEAN_DISTANCE_THRESHOLD": threshold,
                },
            }
        )
    for threshold in ("0.20", "0.30", "0.40", "0.60"):
        configs.append(
            {
                "name": f"aggressive_either_{threshold}_cos090",
                "env": {
                    **aggressive,
                    "CLEAN_DETECT_MODE": "either",
                    "CLEAN_DISTANCE_THRESHOLD": threshold,
                    "CLEAN_COSINE_THRESHOLD": "0.90",
                },
            }
        )
    for threshold in ("0.20", "0.30", "0.40", "0.60", "1.00"):
        configs.append(
            {
                "name": f"conservative_distance_{threshold}",
                "env": {
                    **conservative,
                    "CLEAN_DETECT_MODE": "distance",
                    "CLEAN_DISTANCE_THRESHOLD": threshold,
                },
            }
        )
    for threshold in ("0.20", "0.30", "0.40", "0.60"):
        configs.append(
            {
                "name": f"conservative_either_{threshold}_cos090",
                "env": {
                    **conservative,
                    "CLEAN_DETECT_MODE": "either",
                    "CLEAN_DISTANCE_THRESHOLD": threshold,
                    "CLEAN_COSINE_THRESHOLD": "0.90",
                },
            }
        )
    return configs


def _apply_env(overrides: dict[str, str]) -> dict[str, str | None]:
    old = {key: os.environ.get(key) for key in ENV_KEYS}
    for key in ENV_KEYS:
        os.environ.pop(key, None)
    for key, value in overrides.items():
        os.environ[key] = str(value)
    return old


def _restore_env(old: dict[str, str | None]) -> None:
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _factory(X: np.ndarray, params: dict[str, Any]) -> Engine:
    return Engine(X, params)


def _clean_route(sq_dist: float, cosine: float) -> bool:
    mode = os.environ.get("CLEAN_DETECT_MODE", "distance").strip().lower()
    if mode not in {"distance", "cosine", "either"}:
        mode = "distance"
    distance_threshold = float(os.environ.get("CLEAN_DISTANCE_THRESHOLD", 0.20))
    cosine_threshold = float(os.environ.get("CLEAN_COSINE_THRESHOLD", 0.98))
    distance_clean = sq_dist <= distance_threshold
    cosine_clean = cosine >= cosine_threshold
    if mode == "cosine":
        clean = cosine_clean
    elif mode == "either":
        clean = distance_clean or cosine_clean
    else:
        clean = distance_clean

    guard_distance = os.environ.get("RETRIEVAL_GUARD_DISTANCE")
    guard_cosine = os.environ.get("RETRIEVAL_GUARD_COSINE")
    if guard_distance not in {None, ""} and sq_dist <= float(guard_distance):
        clean = True
    if guard_cosine not in {None, ""} and cosine >= float(guard_cosine):
        clean = True
    return clean


def _nearest_stats(X: np.ndarray, q: np.ndarray) -> tuple[float, float]:
    distances = euclidean_distances(X, q)
    idx = int(np.argmin(distances))
    sq_dist = float(distances[idx])
    q_norm = float(np.linalg.norm(q))
    x_norm = float(np.linalg.norm(X[idx]))
    cosine = 0.0
    if q_norm > 1e-12 and x_norm > 1e-12:
        cosine = float((X[idx] @ q) / (x_norm * q_norm))
    return sq_dist, cosine


def _route_counts(seeds: list[int]) -> dict[str, int]:
    counts = {
        "retrieval_spread": 0,
        "retrieval_topk": 0,
        "aniso_spread": 0,
        "aniso_topk": 0,
    }
    for seed in seeds:
        X = make_patterns(K=K, N=N, seed=seed)
        queries, _, _ = make_test_queries(X, NOISE_LEVELS, N_PER_LEVEL, seed=seed)
        for q in queries:
            sq_dist, cosine = _nearest_stats(X, q)
            if _clean_route(sq_dist, cosine):
                counts["retrieval_spread"] += 1
            else:
                counts["retrieval_topk"] += 1

        rng_indices = np.random.default_rng(seed)
        indices = rng_indices.choice(K, size=min(N_ANISO, K), replace=False).tolist()
        rng_probe = np.random.default_rng(seed)
        for idx in indices:
            probe = X[idx] + rng_probe.standard_normal(N) * 0.05
            norm = np.linalg.norm(probe)
            if norm > 1e-12:
                probe = probe / norm
            sq_dist, cosine = _nearest_stats(X, probe)
            if _clean_route(sq_dist, cosine):
                counts["aniso_spread"] += 1
            else:
                counts["aniso_topk"] += 1
    return counts


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    per_seed = report["per_seed"]
    agg = report["aggregated"]
    score = report["score"]
    return {
        "mean_delta": float(agg["mean_delta"]),
        "min_delta": float(agg["min_delta"]),
        "mean_spread": float(agg["mean_spread"]),
        "min_spread": float(agg["min_spread"]),
        "neg_count": sum(1 for row in per_seed if float(row["delta"]) < 0.0),
        "spread_failures": sum(1 for row in per_seed if float(row["spread_reduction"]) <= 1.0),
        "score": float(score["total_automated"]),
    }


def _env_label(env: dict[str, str]) -> str:
    return " ".join(f"{key}={value}" for key, value in env.items()) or "none"


def main() -> int:
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    raw_reports: dict[str, dict[str, Any]] = {}

    for config in _base_configs():
        name = config["name"]
        env = config["env"]
        print(f"running {name}", flush=True)
        old = _apply_env(env)
        try:
            t0 = time.monotonic()
            route_counts = _route_counts(SEEDS_12)
            report = run_multi(
                agent_factory=_factory,
                seeds=SEEDS_12,
                K=K,
                N=N,
                noise_levels=NOISE_LEVELS,
                n_per_level=N_PER_LEVEL,
                n_aniso=N_ANISO,
            )
            runtime = time.monotonic() - t0
        finally:
            _restore_env(old)

        raw_reports[name] = report
        summary = _summary(report)
        row = {
            "name": name,
            "env": _env_label(env),
            "runtime": runtime,
            **summary,
            **route_counts,
        }
        rows.append(row)
        print(
            f"done {name}: mean_delta={row['mean_delta']:+.3f} "
            f"min_delta={row['min_delta']:+.3f} mean_spread={row['mean_spread']:.2f} "
            f"min_spread={row['min_spread']:.2f} spread_failures={row['spread_failures']} "
            f"score={row['score']:.2f} runtime={runtime:.1f}s",
            flush=True,
        )

    lines = [
        "# Adaptive Retrieval Guarded 12-Seed Sweep",
        "",
        "Stable 12-seed reference: mean delta `+0.033`, min delta `+0.005`, spread failures `0`.",
        "",
        "| Config | Env | Runtime | Mean Delta | Min Delta | Mean Spread | Min Spread | Neg Seeds | Spread Failures | Score | Retrieval Spread/TopK | Aniso Spread/TopK |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | `{row['env']}` | {row['runtime']:.1f}s | "
            f"{row['mean_delta']:+.3f} | {row['min_delta']:+.3f} | "
            f"{row['mean_spread']:.2f}x | {row['min_spread']:.2f}x | "
            f"{row['neg_count']} | {row['spread_failures']} | {row['score']:.2f} | "
            f"{row['retrieval_spread']}/{row['retrieval_topk']} | "
            f"{row['aniso_spread']}/{row['aniso_topk']} |"
        )

    passing = [
        row
        for row in rows
        if row["neg_count"] == 0
        and row["spread_failures"] == 0
        and row["mean_delta"] > 0.033
        and row["min_delta"] >= 0.005
    ]
    if passing:
        best = max(passing, key=lambda row: (row["mean_delta"], row["min_delta"], row["score"]))
        lines += [
            "",
            "## Best Passing Config",
            "",
            f"`{best['name']}` passes the 12-seed gate with env `{best['env']}`.",
        ]
    else:
        lines += [
            "",
            "## Decision",
            "",
            "No guarded adaptive retrieval config passed the 12-seed gate.",
        ]

    out = results_dir / "adaptive_retrieval_guarded_12_seed.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
