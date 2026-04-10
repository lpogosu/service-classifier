"""Что установил эксперимент: почти все ложные срабатывания идут от фраз-омонимов.

Скрипт раскладывает ошибки стадии 1 по категориям и показывает, какие именно
фразы дали ложное срабатывание и какие тексты остались нераспознанными. Из этого
разбора выросли три решения: покатегорийный min_matches, список якорных слов и
контекстное правило для демонтажа.

    python -m training.experiments.detection_error_analysis --top 8
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from service_classifier.config import (
    CATEGORY_TITLES,
    DEFAULT_DATASET_PATH,
    DEFAULT_KEY_PHRASES_PATH,
    TARGET_MC_IDS,
)
from service_classifier.datasets import load_dataset
from service_classifier.detection import KeywordDetector
from service_classifier.evaluation import micro_f1, per_category_f1

SNIPPET_CHARS = 140


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--keyphrases", type=Path, default=DEFAULT_KEY_PHRASES_PATH)
    parser.add_argument("--top", type=int, default=8, help="Сколько фраз показывать")
    parser.add_argument("--examples", type=int, default=2, help="Примеров пропусков")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    detector = KeywordDetector(args.keyphrases)

    truth: list[set[int]] = []
    predicted: list[set[int]] = []
    false_positive_phrases: dict[int, Counter[str]] = {
        mc_id: Counter() for mc_id in TARGET_MC_IDS
    }
    false_negative_examples: dict[int, list[str]] = {mc_id: [] for mc_id in TARGET_MC_IDS}

    for ad in dataset:
        details = detector.detect_with_details(ad.description, ad.source_mc_id)
        expected = set(ad.target_detected_mc_ids)
        actual = set(details)
        truth.append(expected)
        predicted.append(actual)

        for mc_id in actual - expected:
            false_positive_phrases[mc_id].update(details[mc_id])
        for mc_id in expected - actual:
            if len(false_negative_examples[mc_id]) < args.examples:
                false_negative_examples[mc_id].append(
                    " ".join(ad.description.split())[:SNIPPET_CHARS]
                )

    overall = micro_f1(truth, predicted)
    print(
        f"Итого: f1={overall.f1:.4f}  p={overall.precision:.4f}  "
        f"r={overall.recall:.4f}  fp={overall.fp}  fn={overall.fn}\n"
    )

    scores = per_category_f1(truth, predicted, list(TARGET_MC_IDS))
    for mc_id in TARGET_MC_IDS:
        score = scores[mc_id]
        print(
            f"[{mc_id}] {CATEGORY_TITLES[mc_id]}  f1={score.f1:.3f}  "
            f"fp={score.fp}  fn={score.fn}  n={score.support}"
        )
        for phrase, count in false_positive_phrases[mc_id].most_common(args.top):
            print(f"    FP-фраза  {count:>4}x  {phrase}")
        for snippet in false_negative_examples[mc_id]:
            print(f"    пропуск   {snippet}")
        print()


if __name__ == "__main__":
    main()
