"""Что установил эксперимент: LLM не спасает пограничные случаи GBM.

Идея выглядела разумной: отдать LLM только объявления, где GBM не уверен
(вероятность рядом с порогом), и оставить уверенные за моделью. Метрики честные —
GBM обучается на четырёх фолдах и никогда не видит тестовый. На исходном наборе
кейса LLM исправила 66 решений и испортила 88, суммарно ухудшив accuracy.
Именно поэтому в проде осталась чистая GBM без LLM-надстройки.

Нужен OPENROUTER_API_KEY.

    python -m training.experiments.hybrid_gbm_llm --band 0.15
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

from service_classifier.config import DEFAULT_DATASET_PATH, DEFAULT_KEY_PHRASES_PATH, Settings
from service_classifier.datasets import load_dataset
from service_classifier.detection import KeywordDetector
from service_classifier.evaluation import binary_accuracy
from service_classifier.llm import LLMClient, LLMError, build_client
from service_classifier.split import DEFAULT_SPLIT_THRESHOLD, SPLIT_SYSTEM_PROMPT
from training.train_split_classifier import RANDOM_STATE, build_matrix

CV_FOLDS = 5
CONCURRENCY = 8
TEXT_LIMIT = 500
DEFAULT_BAND = 0.15


async def ask_llm(
    client: LLMClient,
    model: str,
    description: str,
    detected: list[int],
    semaphore: asyncio.Semaphore,
    fallback: bool,
) -> bool:
    async with semaphore:
        try:
            content = await client.complete(
                messages=[
                    {"role": "system", "content": SPLIT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Обнаружены микрокатегории: {detected}\n"
                            f'Текст: "{description[:TEXT_LIMIT]}"'
                        ),
                    },
                ],
                model=model,
                temperature=0.6,
                max_tokens=64,
            )
            return bool(json.loads(content).get("should_split", False))
        except (LLMError, json.JSONDecodeError, ValueError):
            return fallback


async def run(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    client = build_client(settings)
    if client is None:
        raise SystemExit("Нужен OPENROUTER_API_KEY")

    dataset = load_dataset(args.dataset)
    detector = KeywordDetector(args.keyphrases)
    descriptions = [ad.description for ad in dataset]
    detected = [detector.detect(ad.description, ad.source_mc_id) for ad in dataset]
    y = np.array([int(ad.should_split) for ad in dataset])

    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=3)
    matrix, _ = build_matrix(descriptions, detected, tfidf, fit=True)

    # Out-of-fold вероятности: тестовый фолд модель не видела.
    proba = np.zeros(len(y), dtype=float)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for train_index, test_index in cv.split(matrix, y):
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, random_state=RANDOM_STATE
        )
        model.fit(matrix[train_index], y[train_index])
        proba[test_index] = model.predict_proba(matrix[test_index])[:, 1]

    gbm_pred = proba >= DEFAULT_SPLIT_THRESHOLD
    borderline = np.where(np.abs(proba - DEFAULT_SPLIT_THRESHOLD) <= args.band)[0]
    print(
        f"Объявлений: {len(y)}, пограничных в полосе ±{args.band}: {len(borderline)}"
    )

    semaphore = asyncio.Semaphore(CONCURRENCY)
    try:
        llm_answers = await asyncio.gather(
            *(
                ask_llm(
                    client,
                    args.model,
                    descriptions[index],
                    detected[index],
                    semaphore,
                    bool(gbm_pred[index]),
                )
                for index in borderline
            )
        )
    finally:
        await client.close()

    hybrid_pred = gbm_pred.copy()
    improved = degraded = 0
    for index, answer in zip(borderline, llm_answers, strict=True):
        hybrid_pred[index] = answer
        if bool(gbm_pred[index]) == bool(y[index]) and answer != bool(y[index]):
            degraded += 1
        elif bool(gbm_pred[index]) != bool(y[index]) and answer == bool(y[index]):
            improved += 1

    truth = [bool(value) for value in y]
    gbm_score = binary_accuracy(truth, [bool(value) for value in gbm_pred])
    hybrid_score = binary_accuracy(truth, [bool(value) for value in hybrid_pred])

    print(f"\n  GBM      acc={gbm_score.accuracy:.4f}")
    print(f"  GBM+LLM  acc={hybrid_score.accuracy:.4f}")
    print(f"  LLM исправила {improved}, испортила {degraded}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--keyphrases", type=Path, default=DEFAULT_KEY_PHRASES_PATH)
    parser.add_argument("--model", default="qwen/qwen3-235b-a22b-2507")
    parser.add_argument("--band", type=float, default=DEFAULT_BAND)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
