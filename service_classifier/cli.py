"""CLI: прогон пайплайна по датасету и расчёт метрик.

    python -m service_classifier --dataset data/synthetic_dataset.csv --evaluate
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from service_classifier.config import CATEGORY_TITLES, DEFAULT_DATASET_PATH, Settings
from service_classifier.datasets import LabeledAd, load_dataset
from service_classifier.evaluation import evaluate, format_report
from service_classifier.llm import build_client
from service_classifier.pipeline import Ad, PipelineResult, ServiceClassifier

logger = logging.getLogger("service_classifier")

PROGRESS_EVERY = 100


def _to_ad(labeled: LabeledAd) -> Ad:
    return Ad(
        item_id=labeled.item_id,
        mc_id=labeled.source_mc_id,
        mc_title=labeled.source_mc_title,
        description=labeled.description,
    )


def _serialize(result: PipelineResult) -> dict[str, object]:
    return {
        "itemId": result.item_id,
        "detectedMcIds": result.detected_mc_ids,
        "shouldSplit": result.should_split,
        "splitMcIds": result.split_mc_ids,
        "drafts": [
            {"mcId": d.mc_id, "mcTitle": d.mc_title, "text": d.text, "fallback": d.is_fallback}
            for d in result.drafts
        ],
    }


async def run(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    logging.basicConfig(
        level=settings.log_level, format="%(levelname)s %(name)s: %(message)s"
    )

    dataset = load_dataset(args.dataset)
    if args.max_items:
        dataset = dataset[: args.max_items]
    logger.info("Загружено объявлений: %d", len(dataset))

    client = build_client(settings)
    classifier = ServiceClassifier(settings, client)
    logger.info("Режим стадии 2: %s", classifier.split_mode.value)

    started = time.perf_counter()
    try:
        results = await classifier.process_many(
            [_to_ad(item) for item in dataset],
            concurrency=args.concurrency,
            generate_drafts=not args.no_drafts,
        )
    finally:
        if client is not None:
            await client.close()

    elapsed = time.perf_counter() - started
    print(
        f"Обработано {len(results)} объявлений за {elapsed:.1f}с "
        f"({len(results) / elapsed:.1f} items/sec)"
    )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps([_serialize(r) for r in results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Предсказания сохранены в {output}")

    if args.evaluate:
        report = evaluate(
            true_detected=[set(item.target_detected_mc_ids) for item in dataset],
            pred_detected=[set(r.detected_mc_ids) for r in results],
            true_should_split=[item.should_split for item in dataset],
            pred_should_split=[r.should_split for r in results],
            true_split=[set(item.target_split_mc_ids) for item in dataset],
            pred_split=[set(r.split_mc_ids) for r in results],
        )
        print(format_report(report, CATEGORY_TITLES))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m service_classifier",
        description="Прогон пайплайна по датасету объявлений",
    )
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET_PATH,
        help="CSV с объявлениями (по умолчанию синтетический набор)",
    )
    parser.add_argument("--output", type=Path, help="Куда сохранить предсказания (JSON)")
    parser.add_argument("--evaluate", action="store_true", help="Посчитать метрики")
    parser.add_argument("--no-drafts", action="store_true", help="Пропустить стадию 3")
    parser.add_argument("--max-items", type=int, help="Ограничить число объявлений")
    parser.add_argument("--concurrency", type=int, default=8, help="Параллельных объявлений")
    return parser


def main() -> None:
    asyncio.run(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
