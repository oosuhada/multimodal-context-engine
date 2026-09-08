"""Deterministic retrieval experiment for modality-count imbalance.

The experiment isolates the fusion layer. It intentionally does not load model
weights: synthetic unit vectors let us ask whether duplicated noisy items from
one modality can dominate a context representation.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from context_engine.index import ContextIndex, MediaItem


@dataclass(frozen=True)
class SeedResult:
    seed: int
    recall_at_1: float
    mrr_at_5: float
    ndcg_at_5: float


def _normalize(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _rank_metrics(index: ContextIndex, queries: list[tuple[str, np.ndarray]]) -> tuple[float, float, float]:
    hit = 0.0
    reciprocal = 0.0
    ndcg = 0.0
    for expected_group, query in queries:
        results = index.search(query, top_k=5)
        ids = [result.group_id for result in results]
        if ids and ids[0] == expected_group:
            hit += 1.0
        if expected_group in ids:
            rank = ids.index(expected_group) + 1
            reciprocal += 1.0 / rank
            ndcg += 1.0 / np.log2(rank + 1)
    count = max(len(queries), 1)
    return hit / count, reciprocal / count, ndcg / count


def build_fixture(seed: int, *, groups: int = 48, dimensions: int = 64, image_duplicates: int = 8):
    random = np.random.default_rng(seed)
    truth = np.stack([_normalize(random.normal(size=dimensions)) for _ in range(groups)])
    items: list[MediaItem] = []
    embeddings: list[np.ndarray] = []
    queries: list[tuple[str, np.ndarray]] = []

    for index in range(groups):
        group_id = f"group-{index:03d}"
        true_vector = truth[index]
        nuisance = truth[(index + 1) % groups]
        queries.append((group_id, _normalize(true_vector + random.normal(scale=0.025, size=dimensions))))

        # Image extraction is intentionally biased toward a neighboring context.
        # Duplicating that modality models a common video-frame imbalance.
        for duplicate in range(image_duplicates):
            items.append(MediaItem(id=f"{group_id}-image-{duplicate}", group_id=group_id, modality="image", path="fixture.jpg"))
            embeddings.append(_normalize(0.22 * true_vector + 0.78 * nuisance + random.normal(scale=0.035, size=dimensions)))

        for modality in ("audio", "video"):
            items.append(MediaItem(id=f"{group_id}-{modality}", group_id=group_id, modality=modality, path=f"fixture.{modality}"))
            embeddings.append(_normalize(true_vector + random.normal(scale=0.035, size=dimensions)))

    return items, np.asarray(embeddings, dtype=np.float32), queries


def run_seed(seed: int, fusion: str, **fixture_kwargs: int) -> SeedResult:
    items, embeddings, queries = build_fixture(seed, **fixture_kwargs)
    index = ContextIndex(items=items, embeddings=embeddings, fusion=fusion)
    recall, mrr, ndcg = _rank_metrics(index, queries)
    return SeedResult(seed=seed, recall_at_1=recall, mrr_at_5=mrr, ndcg_at_5=ndcg)


def _aggregate(results: list[SeedResult]) -> dict[str, object]:
    payload: dict[str, object] = {"runs": [asdict(result) for result in results]}
    for field in ("recall_at_1", "mrr_at_5", "ndcg_at_5"):
        values = np.asarray([getattr(result, field) for result in results], dtype=float)
        payload[field] = {
            "mean": round(float(values.mean()), 6),
            "stddev": round(float(values.std(ddof=0)), 6),
            "min": round(float(values.min()), 6),
            "max": round(float(values.max()), 6),
        }
    return payload


def run_benchmark() -> dict[str, object]:
    seeds = list(range(10))
    fixture = {"groups": 48, "dimensions": 64, "image_duplicates": 8}
    baseline = [run_seed(seed, "mean", **fixture) for seed in seeds]
    candidate = [run_seed(seed, "balanced", **fixture) for seed in seeds]
    baseline_summary = _aggregate(baseline)
    candidate_summary = _aggregate(candidate)
    return {
        "experiment": "multimodal-fusion-imbalance-v1",
        "question": "Does modality-balanced fusion resist duplicated noisy frame embeddings better than item-wise mean fusion?",
        "hypothesis": "Balanced fusion will improve retrieval when one noisy modality contributes many more items than the others.",
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "dataset": {"kind": "synthetic controlled embedding fixture", **fixture, "seeds": seeds},
        "baseline": {"fusion": "mean", **baseline_summary},
        "candidate": {"fusion": "balanced", **candidate_summary},
        "limitations": [
            "Synthetic embeddings isolate fusion behavior; they do not measure LanguageBind encoder quality on real media.",
            "The corruption pattern intentionally stresses modality-count imbalance and is not a claim about production data frequency.",
            "A real-media held-out set is still required before claiming end-to-end multimodal retrieval accuracy.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/fusion-retrieval-v1.json"))
    args = parser.parse_args()
    result = run_benchmark()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
