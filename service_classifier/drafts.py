"""Стадия 3 — генерация черновиков объявлений под каждую выделенную услугу.

Единственная стадия, где LLM незаменима. Выбор модели решался измерением, а не
вкусом: на одной и той же выборке Qwen3-235B добавлял несуществующие факты в 36%
черновиков, Gemini 2.0 Flash — в 1%. Промпт содержит явные отрицательные примеры,
именно они и убирают выдумки.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

from service_classifier.llm import LLMClient, LLMError

logger = logging.getLogger(__name__)

#: Сколько характерных фраз категории подсказывать модели.
MAX_KEY_PHRASES_IN_PROMPT = 10
MAX_DRAFT_TOKENS = 1024


@dataclass
class Draft:
    mc_id: int
    mc_title: str
    text: str
    #: True, если текст собран по шаблону: LLM недоступна или не справилась.
    is_fallback: bool = False


DRAFT_SYSTEM_PROMPT = """Ты — копирайтер, пишешь тексты объявлений об услугах ремонта.

ПРАВИЛА:
1. Текст от лица исполнителя ("я/мы"), 3-5 предложений
2. Фокус ТОЛЬКО на указанной услуге
3. Сохрани стиль оригинала

ЗАПРЕТ НА ВЫДУМКИ — строго используй ТОЛЬКО факты из оригинала:
- НЕ добавляй виды работ, которых нет в тексте
- НЕ добавляй опыт, стаж, гарантии, цены, если их нет
- НЕ конкретизируй то, что в оригинале общее

ПРИМЕРЫ ОШИБОК (так делать НЕЛЬЗЯ):
- Оригинал: "натяжные потолки" → НЕЛЬЗЯ писать "многоуровневые, теневые, парящие, с подсветкой"
- Оригинал: без слова "гарантия" → НЕЛЬЗЯ писать "гарантия на работу"
- Оригинал: без опыта → НЕЛЬЗЯ писать "опыт более 10 лет"
- Оригинал: "плитка" → НЕЛЬЗЯ писать "мозаика, керамогранит, клинкер"

ПРАВИЛЬНО: если в оригинале мало деталей — черновик тоже короткий и без деталей.

JSON: {"text": "текст черновика"}"""


def build_draft_prompt(
    description: str,
    target_mc_id: int,
    target_mc_title: str,
    key_phrases: list[str],
) -> str:
    phrases = ", ".join(key_phrases[:MAX_KEY_PHRASES_IN_PROMPT])
    return f"""Напиши черновик объявления для микрокатегории [{target_mc_id}] "{target_mc_title}".

ОРИГИНАЛЬНОЕ ОБЪЯВЛЕНИЕ:
\"\"\"
{description}
\"\"\"

ЦЕЛЕВАЯ МИКРОКАТЕГОРИЯ: {target_mc_title}
ХАРАКТЕРНЫЕ ФРАЗЫ КАТЕГОРИИ: {phrases}

Извлеки из оригинала всю релевантную информацию об этой услуге и создай отдельное объявление.
Верни JSON: {{"text": "текст черновика"}}"""


def fallback_draft(mc_id: int, mc_title: str) -> Draft:
    """Заглушка на случай недоступной LLM.

    Намеренно бедная: лучше очевидно шаблонный текст, чем правдоподобный
    черновик с выдуманными фактами.
    """
    return Draft(
        mc_id=mc_id,
        mc_title=mc_title,
        text=f"Услуга: {mc_title}. Черновик не сгенерирован — LLM недоступна.",
        is_fallback=True,
    )


class DraftGenerator:
    """Генерирует черновики параллельно, по одному запросу на категорию."""

    def __init__(
        self,
        client: LLMClient | None,
        model: str,
        categories: dict[int, tuple[str, list[str]]],
        *,
        temperature: float = 0.3,
        concurrency: int = 10,
    ) -> None:
        self.client = client
        self.model = model
        self.categories = categories
        self.temperature = temperature
        self.concurrency = concurrency

    def _category(self, mc_id: int) -> tuple[str, list[str]]:
        return self.categories.get(mc_id, (f"Категория {mc_id}", []))

    async def generate(self, description: str, mc_id: int) -> Draft:
        mc_title, key_phrases = self._category(mc_id)
        if self.client is None:
            return fallback_draft(mc_id, mc_title)

        prompt = build_draft_prompt(description, mc_id, mc_title, key_phrases)
        try:
            content = await self.client.complete(
                messages=[
                    {"role": "system", "content": DRAFT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
                temperature=self.temperature,
                max_tokens=MAX_DRAFT_TOKENS,
            )
            text = str(json.loads(content).get("text", "")).strip()
            if not text:
                raise ValueError("модель вернула пустой черновик")
        except (LLMError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("Stage 3: черновик [%d] не сгенерирован (%s)", mc_id, exc)
            return fallback_draft(mc_id, mc_title)

        return Draft(mc_id=mc_id, mc_title=mc_title, text=text)

    async def generate_all(self, description: str, mc_ids: list[int]) -> list[Draft]:
        if not mc_ids:
            return []

        semaphore = asyncio.Semaphore(self.concurrency)

        async def one(mc_id: int) -> Draft:
            async with semaphore:
                return await self.generate(description, mc_id)

        return list(await asyncio.gather(*(one(mc_id) for mc_id in mc_ids)))
