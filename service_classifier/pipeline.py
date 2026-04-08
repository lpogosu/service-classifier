"""Оркестрация четырёх стадий и пакетная обработка датасета."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from service_classifier.config import DEMOLITION_MC_ID, Settings
from service_classifier.detection import KeywordDetector, normalize_text
from service_classifier.drafts import Draft, DraftGenerator
from service_classifier.llm import LLMClient
from service_classifier.split import SplitDecider, SplitMode
from service_classifier.verification import (
    ContextVerifier,
    find_categories_needing_verification,
    should_verify_demolition,
)

logger = logging.getLogger(__name__)

#: Как стадия 1.5 изменила категорию — используется веб-интерфейсом.
VERDICT_ADDED = "added"
VERDICT_REMOVED = "removed"
VERDICT_CONFIRMED = "confirmed"


@dataclass
class Ad:
    """Входное объявление."""

    item_id: int | str
    mc_id: int
    description: str
    mc_title: str = ""


@dataclass
class PipelineResult:
    item_id: int | str
    detected_mc_ids: list[int] = field(default_factory=list)
    should_split: bool = False
    split_mc_ids: list[int] = field(default_factory=list)
    drafts: list[Draft] = field(default_factory=list)
    #: Сработавшие фразы по категориям — объяснение решения стадии 1.
    matched_phrases: dict[int, list[str]] = field(default_factory=dict)
    #: `{mc_id: added|removed|confirmed}` от стадии 1.5.
    verification: dict[int, str] = field(default_factory=dict)
    #: Категории, снятые верификацией, вместе с фразами, из-за которых сработали.
    rejected: dict[int, list[str]] = field(default_factory=dict)


class ServiceClassifier:
    """Полный пайплайн: детекция → верификация → решение о сплите → черновики."""

    def __init__(self, settings: Settings, client: LLMClient | None) -> None:
        self.settings = settings
        self.detector = KeywordDetector(settings.key_phrases_path)
        self.verifier = ContextVerifier(client, settings.verification_model)
        self.decider = SplitDecider(
            classifier_path=settings.split_model_path,
            client=client,
            model=settings.verification_model,
        )
        self.generator = DraftGenerator(
            client=client,
            model=settings.draft_model,
            categories={
                mc_id: (category.mc_title, category.key_phrases)
                for mc_id, category in self.detector.categories.items()
            },
            concurrency=settings.draft_concurrency,
        )

    @property
    def split_mode(self) -> SplitMode:
        return self.decider.mode

    async def process(self, ad: Ad, *, generate_drafts: bool = True) -> PipelineResult:
        result = PipelineResult(item_id=ad.item_id)

        # Стадия 1 — ключевые слова, без сети.
        details = self.detector.detect_with_details(ad.description, ad.mc_id)

        # Стадия 1.5 — LLM только для демонтажных текстов.
        await self._verify(ad.description, details, result)

        result.detected_mc_ids = sorted(details)
        result.matched_phrases = dict(details)
        if not result.detected_mc_ids:
            return result

        # Стадия 2 — бинарное решение о разделении.
        split = await self.decider.decide(ad.description, result.detected_mc_ids)
        result.should_split = split.should_split
        result.split_mc_ids = split.split_mc_ids

        # Стадия 3 — черновики только под подтверждённое разделение.
        if generate_drafts and split.split_mc_ids:
            result.drafts = await self.generator.generate_all(
                ad.description, split.split_mc_ids
            )
        return result

    async def _verify(
        self,
        description: str,
        details: dict[int, list[str]],
        result: PipelineResult,
    ) -> None:
        text_lower = normalize_text(description).lower()
        verify_demolition = should_verify_demolition(text_lower, sorted(details))
        suspicious = find_categories_needing_verification(text_lower, details)
        if not verify_demolition and not suspicious:
            return

        verdict = await self.verifier.verify(description, verify_demolition, suspicious)

        if verify_demolition and verdict.has_demolition_service:
            details[DEMOLITION_MC_ID] = ["демонтаж (подтверждено LLM)"]
            result.verification[DEMOLITION_MC_ID] = VERDICT_ADDED

        for mc_id, is_real_service in verdict.context.items():
            if is_real_service:
                result.verification[mc_id] = VERDICT_CONFIRMED
                continue
            result.rejected[mc_id] = details.pop(mc_id, [])
            result.verification[mc_id] = VERDICT_REMOVED
            logger.info("Stage 1.5: категория [%d] снята как демонтажный контекст", mc_id)

    async def process_many(
        self,
        ads: list[Ad],
        *,
        concurrency: int = 8,
        generate_drafts: bool = True,
    ) -> list[PipelineResult]:
        """Обрабатывает пакет объявлений, сохраняя исходный порядок."""
        semaphore = asyncio.Semaphore(concurrency)

        async def one(ad: Ad) -> PipelineResult:
            async with semaphore:
                return await self.process(ad, generate_drafts=generate_drafts)

        return list(await asyncio.gather(*(one(ad) for ad in ads)))
