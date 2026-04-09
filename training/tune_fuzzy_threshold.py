"""Подбор порога fuzzy-сравнения и минимальной длины фразы для стадии 1.

Этим перебором и был найден порог 86: на 80 многословные фразы совпадали по
общему префиксу («демонтаж старой ...»), давая 476 лишних срабатываний.

    python -m training.tune_fuzzy_threshold --dataset data/synthetic_dataset.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from service_classifier.config import DEFAULT_DATASET_PATH, DEFAULT_KEY_PHRASES_PATH
from service_classifier.datasets import load_dataset
from service_classifier.detection import KeywordDetector
from service_classifier.evaluation import micro_f1

THRESHOLDS = range(70, 96, 2)
MIN_PHRASE_LENGTHS = (3, 4, 5)


@dataclass
class GridPoint:
    threshold: int
    min_phrase_len: int
    precision: float
    recall: float
    f1: float


def run_grid(dataset_path: Path, keyphrases_path: Path) -> list[GridPoint]:
    dataset = load_dataset(dataset_path)
    truth = [set(ad.target_detected_mc_ids) for ad in dataset]

    points: list[GridPoint] = []
    for threshold, min_len in product(THRESHOLDS, MIN_PHRASE_LENGTHS):
        detector = KeywordDetector(
            keyphrases_path, fuzzy_threshold=threshold, min_phrase_len=min_len
        )
        predicted = [
            set(detector.detect(ad.description, ad.source_mc_id)) for ad in dataset
        ]
        score = micro_f1(truth, predicted)
        points.append(
            GridPoint(threshold, min_len, score.precision, score.recall, score.f1)
        )
        print(
            f"  threshold={threshold:2d}  min_len={min_len}  "
            f"p={score.precision:.4f}  r={score.recall:.4f}  f1={score.f1:.4f}"
        )
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--keyphrases", type=Path, default=DEFAULT_KEY_PHRASES_PATH)
    args = parser.parse_args()

    points = run_grid(args.dataset, args.keyphrases)
    best = max(points, key=lambda point: point.f1)
    print(
        f"\nЛучшая точка: threshold={best.threshold} min_len={best.min_phrase_len} "
        f"f1={best.f1:.4f} (p={best.precision:.4f}, r={best.recall:.4f})"
    )


if __name__ == "__main__":
    main()
