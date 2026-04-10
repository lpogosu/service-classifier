"""Что установил эксперимент: выбор модели для черновиков решается измерением.

Проверка простая и механическая: генерируем черновик, затем ищем в нём
конкретные факты, которых нет в оригинале, — виды работ, гарантии, стаж, цены.
На исходном наборе кейса Qwen3-235B добавлял выдумки в 36% черновиков,
Gemini 2.0 Flash — в 1%, поэтому стадия 3 работает на Gemini.

Проверка намеренно консервативна: она ловит только те выдумки, которые можно
опознать по словарю, и потому даёт нижнюю оценку.

Нужен OPENROUTER_API_KEY.

    python -m training.experiments.draft_hallucination_audit --sample 100
"""

from __future__ import annotations

import argparse
import asyncio
import random
from pathlib import Path

from service_classifier.config import DEFAULT_DATASET_PATH, DEFAULT_KEY_PHRASES_PATH, Settings
from service_classifier.datasets import load_dataset
from service_classifier.detection import KeywordDetector, normalize_text
from service_classifier.drafts import Draft, DraftGenerator
from service_classifier.llm import build_client

DEFAULT_MODELS = ("qwen/qwen3-235b-a22b-2507", "google/gemini-2.0-flash-001")
DEFAULT_SAMPLE = 100
RANDOM_SEED = 42

#: Факты, которые модель не имеет права придумывать: если слово есть в черновике,
#: но его нет в оригинале, это выдумка.
CLAIM_MARKERS: tuple[str, ...] = (
    "гарантия", "гарантию", "опыт", "стаж", "лет",
    "договор", "смета", "скидка", "бесплатн", "выезд",
    "многоуровнев", "теневой", "парящий", "подсветк",
    "керамогранит", "мозаика", "клинкер", "венециан",
    "лицензи", "сертификат", "рассрочк",
)


def invented_claims(original: str, draft: str) -> list[str]:
    source = normalize_text(original).lower()
    generated = normalize_text(draft).lower()
    return [
        marker
        for marker in CLAIM_MARKERS
        if marker in generated and marker not in source
    ]


async def run(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    client = build_client(settings)
    if client is None:
        raise SystemExit("Нужен OPENROUTER_API_KEY")

    rng = random.Random(RANDOM_SEED)
    dataset = load_dataset(args.dataset)
    detector = KeywordDetector(args.keyphrases)
    candidates = [
        (ad, detector.detect(ad.description, ad.source_mc_id)) for ad in dataset
    ]
    candidates = [pair for pair in candidates if pair[1]]
    sample = rng.sample(candidates, min(args.sample, len(candidates)))
    print(f"Выборка: {len(sample)} объявлений\n")

    categories = {
        mc_id: (category.mc_title, category.key_phrases)
        for mc_id, category in detector.categories.items()
    }

    try:
        for model in args.models:
            generator = DraftGenerator(client=client, model=model, categories=categories)
            drafts: list[tuple[str, Draft]] = []
            for ad, detected in sample:
                generated = await generator.generate_all(ad.description, detected[:1])
                drafts.extend((ad.description, draft) for draft in generated)

            flagged = [
                (draft, invented_claims(original, draft.text))
                for original, draft in drafts
                if not draft.is_fallback
            ]
            hallucinated = [pair for pair in flagged if pair[1]]
            rate = len(hallucinated) / len(flagged) if flagged else 0.0
            print(f"  {model:<40} черновиков={len(flagged)}  выдумок={rate:.0%}")
            for draft, claims in hallucinated[:3]:
                print(f"      [{draft.mc_id}] добавлено: {', '.join(claims)}")
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
