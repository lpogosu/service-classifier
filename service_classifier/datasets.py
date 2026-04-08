"""Чтение датасета объявлений в формате кейса.

Разделитель — точка с запятой, списки категорий записаны как `[102, 105]`.
Формат сохранён как в исходном кейсе, чтобы обучающие скрипты и метрики
работали и с реальными данными, и с синтетическим набором из `data/synthetic.py`.
"""

from __future__ import annotations

import ast
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CSV_DELIMITER = ";"
CSV_FIELDNAMES: tuple[str, ...] = (
    "itemId",
    "sourceMcId",
    "sourceMcTitle",
    "description",
    "targetDetectedMcIds",
    "targetSplitMcIds",
    "shouldSplit",
    "caseType",
    "split",
)

_TRUE_VALUES = frozenset({"true", "1", "yes", "да"})


@dataclass
class LabeledAd:
    """Размеченное объявление."""

    item_id: int
    source_mc_id: int
    source_mc_title: str
    description: str
    target_detected_mc_ids: list[int]
    target_split_mc_ids: list[int]
    should_split: bool
    case_type: str
    split: str


def parse_mc_ids(value: Any) -> set[int]:
    """Разбирает список категорий из строки `[102, 105]`, списка или множества."""
    if isinstance(value, set):
        return {int(v) for v in value}
    if isinstance(value, (list, tuple)):
        return {int(v) for v in value}
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return set()
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return set()
        if isinstance(parsed, (list, tuple, set)):
            return {int(v) for v in parsed}
        if isinstance(parsed, int):
            return {parsed}
    return set()


def parse_bool(value: str) -> bool:
    return value.strip().lower() in _TRUE_VALUES


def load_dataset(path: str | Path) -> list[LabeledAd]:
    """Читает CSV-датасет целиком."""
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    ads: list[LabeledAd] = []
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter=CSV_DELIMITER):
            ads.append(
                LabeledAd(
                    item_id=int(row["itemId"]),
                    source_mc_id=int(row["sourceMcId"]),
                    source_mc_title=row["sourceMcTitle"],
                    description=row["description"],
                    target_detected_mc_ids=sorted(parse_mc_ids(row["targetDetectedMcIds"])),
                    target_split_mc_ids=sorted(parse_mc_ids(row["targetSplitMcIds"])),
                    should_split=parse_bool(row["shouldSplit"]),
                    case_type=row.get("caseType", ""),
                    split=row.get("split", "train"),
                )
            )
    return ads


def write_dataset(path: str | Path, ads: list[LabeledAd]) -> None:
    """Записывает датасет в том же формате."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(CSV_FIELDNAMES), delimiter=CSV_DELIMITER
        )
        writer.writeheader()
        for ad in ads:
            writer.writerow(
                {
                    "itemId": ad.item_id,
                    "sourceMcId": ad.source_mc_id,
                    "sourceMcTitle": ad.source_mc_title,
                    "description": ad.description,
                    "targetDetectedMcIds": str(ad.target_detected_mc_ids),
                    "targetSplitMcIds": str(ad.target_split_mc_ids),
                    "shouldSplit": str(ad.should_split),
                    "caseType": ad.case_type,
                    "split": ad.split,
                }
            )
