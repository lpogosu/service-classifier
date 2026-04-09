"""Подбор PER_CATEGORY_MIN_MATCHES — сколько совпадений требовать от категории.

Единый порог для всех категорий работает плохо: у 103 «Электрика» при mm=1 было
242 ложных срабатывания, а у 104 «Натяжные потолки» переход на mm=2 обрушивал
recall с 0.833 до 0.613. Скрипт сравнивает единый порог с покатегорийным и
показывает, какие категории выигрывают от ужесточения.

    python -m training.tune_min_matches --dataset data/synthetic_dataset.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from service_classifier.config import (
    CATEGORY_TITLES,
    DEFAULT_DATASET_PATH,
    DEFAULT_KEY_PHRASES_PATH,
    TARGET_MC_IDS,
)
from service_classifier.datasets import LabeledAd, load_dataset
from service_classifier.detection import KeywordDetector
from service_classifier.evaluation import micro_f1, per_category_f1

UNIFORM_CANDIDATES = (1, 2, 3)


def evaluate_config(
    detector: KeywordDetector,
    dataset: list[LabeledAd],
    min_matches: dict[int, int],
) -> tuple[list[set[int]], list[set[int]]]:
    detector.min_matches = min_matches
    truth = [set(ad.target_detected_mc_ids) for ad in dataset]
    predicted = [set(detector.detect(ad.description, ad.source_mc_id)) for ad in dataset]
    return truth, predicted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--keyphrases", type=Path, default=DEFAULT_KEY_PHRASES_PATH)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    detector = KeywordDetector(args.keyphrases)
    tuned = dict(KeywordDetector.PER_CATEGORY_MIN_MATCHES)

    print("Единый порог для всех категорий:")
    for value in UNIFORM_CANDIDATES:
        truth, predicted = evaluate_config(
            detector, dataset, dict.fromkeys(TARGET_MC_IDS, value)
        )
        score = micro_f1(truth, predicted)
        print(
            f"  min_matches={value}  p={score.precision:.4f}  "
            f"r={score.recall:.4f}  f1={score.f1:.4f}"
        )

    truth, predicted = evaluate_config(detector, dataset, tuned)
    tuned_score = micro_f1(truth, predicted)
    print(
        f"\nПокатегорийная настройка: p={tuned_score.precision:.4f}  "
        f"r={tuned_score.recall:.4f}  f1={tuned_score.f1:.4f}"
    )

    print("\nПокатегорийно (текущая конфигурация):")
    for mc_id, score in sorted(per_category_f1(truth, predicted, list(TARGET_MC_IDS)).items()):
        print(
            f"  [{mc_id}] {CATEGORY_TITLES[mc_id]:<28} mm={tuned.get(mc_id, 1)}  "
            f"f1={score.f1:.3f}  p={score.precision:.3f}  r={score.recall:.3f}  n={score.support}"
        )


if __name__ == "__main__":
    main()
