"""Что установил эксперимент: промпт не закрывает разрыв с GBM.

Пять вариантов инструкции — от короткой до контрастной few-shot — прогоняются
на одной выборке одной моделью. На исходном наборе кейса разброс уложился в
несколько процентных пунктов, а контрастный вариант оказался худшим (0.584):
модель уверенно повторяет заданную ей логику, но эта логика не совпадает с
логикой разметчика. Дальнейшая работа над промптом была свёрнута в пользу GBM.

Нужен OPENROUTER_API_KEY.

    python -m training.experiments.compare_split_prompts --sample 150
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from service_classifier.config import DEFAULT_DATASET_PATH, DEFAULT_KEY_PHRASES_PATH, Settings
from service_classifier.datasets import LabeledAd, load_dataset
from service_classifier.detection import KeywordDetector
from service_classifier.evaluation import binary_accuracy
from service_classifier.llm import LLMClient, LLMError, build_client
from service_classifier.split import SPLIT_SYSTEM_PROMPT
from training.experiments.compare_split_models import stratified_sample

CONCURRENCY = 8
TEXT_LIMIT = 500
DEFAULT_SAMPLE = 150

PROMPTS: dict[str, str] = {
    "baseline": SPLIT_SYSTEM_PROMPT,
    "rule_based": (
        "Ты классифицируешь объявления об услугах ремонта.\n"
        "should_split=true, если мастер ПЕРЕЧИСЛЯЕТ конкретные виды работ как свои услуги.\n"
        "should_split=false, если работы описаны как этапы комплексного ремонта.\n"
        'Ответь строго JSON: {"should_split": true/false}'
    ),
    "with_meta": (
        "Ты классифицируешь объявления об услугах ремонта.\n"
        "Опирайся на структуру текста: короткий перечень работ — специалист (true), "
        "длинный маркетинговый текст про бригаду и договор — комплексный ремонт (false).\n"
        'Ответь строго JSON: {"should_split": true/false}'
    ),
    "contrastive": (
        "Ты классифицируешь объявления об услугах ремонта.\n\n"
        "ПРИМЕР true: «Плиточник. Укладка плитки, керамогранита. Цена за м2 от 900 руб.»\n"
        "ПРИМЕР false: «Ремонт квартир под ключ. Бригада выполнит все виды работ, "
        "включая плитку, электрику и сантехнику. Работаем по договору.»\n\n"
        'Ответь строго JSON: {"should_split": true/false}'
    ),
    "verbalized": (
        "Определи роль автора объявления: СПЕЦИАЛИСТ (продаёт отдельные услуги) "
        "или ПОДРЯДЧИК (продаёт ремонт целиком).\n"
        "СПЕЦИАЛИСТ -> should_split=true, ПОДРЯДЧИК -> should_split=false.\n"
        'Ответь строго JSON: {"should_split": true/false}'
    ),
}


async def predict_one(
    client: LLMClient,
    model: str,
    system_prompt: str,
    ad: LabeledAd,
    detected: list[int],
    semaphore: asyncio.Semaphore,
) -> bool:
    async with semaphore:
        try:
            content = await client.complete(
                messages=[
                    {"role": "system", "content": system_prompt},
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

    print(f"Модель: {args.model}, выборка: {len(dataset)}\n")
    try:
        for name, prompt in PROMPTS.items():
            predictions = await asyncio.gather(
                *(
                    predict_one(client, args.model, prompt, ad, ids, semaphore)
                    for ad, ids in zip(dataset, detected, strict=True)
                )
            )
            score = binary_accuracy(truth, list(predictions))
            print(
                f"  {name:<14} acc={score.accuracy:.3f}  "
                f"tp={score.tp} tn={score.tn} fp={score.fp} fn={score.fn}"
            )
    finally:
        await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--keyphrases", type=Path, default=DEFAULT_KEY_PHRASES_PATH)
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    parser.add_argument("--model", default="qwen/qwen3-235b-a22b-2507")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
