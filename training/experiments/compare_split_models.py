"""Что установил эксперимент: ни одна LLM не обошла GBM на решении о разделении.

Скрипт прогоняет одинаковый промпт через список моделей на одной и той же
стратифицированной выборке и печатает accuracy, F1 и время. На исходном наборе
кейса лучший результат — 0.747 у Qwen3-235B, тогда как GBM на 44 признаках
давал 0.848 по кросс-валидации. Отсюда и вывод: граница «специалист против
комплексного ремонта» определяется распределением разметки, а не рассуждением.

Нужен OPENROUTER_API_KEY.

    python -m training.experiments.compare_split_models --sample 150
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

from service_classifier.config import DEFAULT_DATASET_PATH, DEFAULT_KEY_PHRASES_PATH, Settings
from service_classifier.datasets import LabeledAd, load_dataset
from service_classifier.detection import KeywordDetector
from service_classifier.evaluation import binary_accuracy
from service_classifier.llm import LLMClient, LLMError, build_client
from service_classifier.split import SPLIT_SYSTEM_PROMPT

DEFAULT_MODELS = (
    "qwen/qwen3-235b-a22b-2507",
    "google/gemini-2.0-flash-001",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mistral-large",
    "deepseek/deepseek-chat",
    "qwen/qwen-2.5-72b-instruct",
    "microsoft/phi-4",
)
DEFAULT_SAMPLE = 150
CONCURRENCY = 8
TEXT_LIMIT = 500
RANDOM_SEED = 42


def stratified_sample(dataset: list[LabeledAd], size: int) -> list[LabeledAd]:
    """Равные доли положительных и отрицательных примеров."""
    rng = random.Random(RANDOM_SEED)
    positives = [ad for ad in dataset if ad.should_split]
    negatives = [ad for ad in dataset if not ad.should_split]
    half = size // 2
    return rng.sample(positives, min(half, len(positives))) + rng.sample(
        negatives, min(size - half, len(negatives))
    )


async def predict_one(
    client: LLMClient,
    model: str,
    ad: LabeledAd,
    detected: list[int],
    semaphore: asyncio.Semaphore,
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
                            f'Текст: "{ad.description[:TEXT_LIMIT]}"'
                        ),
                    },
                ],
                model=model,
                temperature=0.6,
                max_tokens=64,
            )
            return bool(json.loads(content).get("should_split", False))
        except (LLMError, json.JSONDecodeError, ValueError):
            return False


async def run(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    client = build_client(settings)
    if client is None:
        raise SystemExit("Нужен OPENROUTER_API_KEY")

    dataset = stratified_sample(load_dataset(args.dataset), args.sample)
    detector = KeywordDetector(args.keyphrases)
    detected = [detector.detect(ad.description, ad.source_mc_id) for ad in dataset]
    truth = [ad.should_split for ad in dataset]
    semaphore = asyncio.Semaphore(CONCURRENCY)

    print(f"Выборка: {len(dataset)} объявлений, положительных {sum(truth)}\n")
    try:
        for model in args.models:
            started = time.perf_counter()
            predictions = await asyncio.gather(
                *(
                    predict_one(client, model, ad, ids, semaphore)
                    for ad, ids in zip(dataset, detected, strict=True)
                )
            )
            score = binary_accuracy(truth, list(predictions))
            elapsed = time.perf_counter() - started
            print(
                f"  {model:<40} acc={score.accuracy:.3f}  "
                f"tp={score.tp} tn={score.tn} fp={score.fp} fn={score.fn}  "
                f"{elapsed:.0f}с"
            )
    finally:
        await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--keyphrases", type=Path, default=DEFAULT_KEY_PHRASES_PATH)
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
