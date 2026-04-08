"""Стадия 1.5 — LLM-верификация контекста для демонтажных объявлений.

Проблема, которую решает стадия: в объявлении «сносим стены, сбиваем плитку,
снимаем обои» ключевые слова «плитка» и «обои» есть, а услуг укладки и поклейки
нет. Стадия срабатывает редко (нужно не меньше DEMOLITION_VERIFICATION_MIN_MARKERS
маркеров сноса), поэтому почти не влияет на пропускную способность.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from service_classifier.config import CATEGORY_TITLES, DEMOLITION_MC_ID, SOURCE_MC_ID
from service_classifier.llm import LLMClient, LLMError

logger = logging.getLogger(__name__)

DEMOLITION_MARKERS: tuple[str, ...] = (
    "демонтаж", "снос", "снятие", "сбивка", "разборка",
    "удаление", "очистка", "разбор", "слом",
)

DEMOLITION_STANDALONE: tuple[str, ...] = (
    "демонтажные работы", "демонтаж любой сложности",
    "разборка", "снос перегородок", "демонтаж стен",
    "полный демонтаж", "частичный демонтаж", "демонтаж отделки",
    "демонтаж под ремонт", "подготовка квартиры к ремонту",
    "демонтаж перегородок", "алмазная резка", "вынос мусора",
    "демонтаж дверей", "демонтаж окон", "демонтаж кухни",
    "демонтаж полов", "демонтаж пола", "демонтаж покрытий",
)

#: Порог, ниже которого объявление считается обычным ремонтным: одно-два
#: упоминания демонтажа нормальны для отделочника и верификации не требуют.
DEMOLITION_VERIFICATION_MIN_MARKERS = 10
#: Окно слева от слова, в котором ищем маркер сноса.
CONTEXT_WINDOW_CHARS = 60
#: Отсечка текста в промпте — дальше идёт реклама, а не перечень услуг.
PROMPT_TEXT_LIMIT = 800
#: Сколько сработавших фраз показывать модели на категорию.
MAX_PHRASES_IN_PROMPT = 5
#: Хвост слова, отбрасываемый при поиске основы (окончания русских слов).
STEM_SUFFIX_LEN = 2
#: Слова короче этого обрезать нечем — ищем их целиком.
MIN_LEN_FOR_STEMMING = 5


@dataclass
class VerificationResult:
    """Что стадия 1.5 сообщила о категориях."""

    #: Является ли демонтаж самостоятельной услугой (проверяется, только если
    #: стадия 1 сняла категорию 111).
    has_demolition_service: bool = False
    #: `{mc_id: услуга реальна}` для проверенных категорий.
    context: dict[int, bool] = field(default_factory=dict)


VERIFY_CONTEXT_BLOCK = (
    "2. Для каждой указанной услуги определи — мастер ПРЕДЛАГАЕТ её клиентам, "
    "или она лишь упомянута как объект демонтажа/разборки/снятия?\n"
    "Правила: true = мастер выполняет эту работу (монтаж, установка, укладка, "
    "поклейка). false = услуга упомянута ТОЛЬКО как объект демонтажа. "
    "Если мастер И демонтирует, И устанавливает — true.\n"
    "\n"
    "ПРОВЕРИТЬ:\n"
    "{categories}"
)

VERIFY_DEMOLITION_BLOCK = (
    "1. Демонтаж — это УСЛУГА мастера в этом объявлении, "
    "или просто упоминание подготовительной работы?\n"
    '"да" — демонтаж перечислен как отдельная услуга мастера.\n'
    '"нет" — демонтаж упомянут как часть другой работы '
    "(например «демонтаж старой плитки» = часть укладки плитки)."
)

VERIFY_PROMPT = """Ты анализируешь объявление об услугах ремонта.

ТЕКСТ ОБЪЯВЛЕНИЯ:
\"\"\"{text}\"\"\"

{questions}

Ответь строго JSON, без пояснений:
{{{schema}}}"""


def _stem(word: str) -> str:
    """Основа слова для морфологически устойчивого поиска по подстроке."""
    if len(word) >= MIN_LEN_FOR_STEMMING:
        return word[: len(word) - STEM_SUFFIX_LEN]
    return word


def _is_demolition_word(word: str) -> bool:
    return any(
        word.startswith(marker) or marker.startswith(word)
        for marker in DEMOLITION_MARKERS
    )


def is_phrase_in_demolition_context(text_lower: str, phrase: str) -> bool:
    """Встречается ли фраза в тексте только рядом с маркерами сноса.

    Если фраза вообще не найдена буквально (сработало fuzzy-сравнение по
    лемматизированному тексту), считаем контекст демонтажным: нечёткое
    совпадение — не доказательство того, что услуга действительно предлагается.
    """
    phrase_lower = phrase.lower()
    if any(phrase_lower.startswith(marker) for marker in DEMOLITION_MARKERS):
        return True

    stem = _stem(phrase_lower.split()[0])
    position = 0
    while True:
        position = text_lower.find(stem, position)
        if position == -1:
            break

        word_start = position
        while word_start > 0 and text_lower[word_start - 1].isalpha():
            word_start -= 1
        word_end = position + len(stem)
        while word_end < len(text_lower) and text_lower[word_end].isalpha():
            word_end += 1

        # Основа могла попасть внутрь самого слова «демонтаж» — это не услуга.
        if _is_demolition_word(text_lower[word_start:word_end]):
            position = word_end
            continue

        prefix = text_lower[max(0, position - CONTEXT_WINDOW_CHARS) : position]
        if not any(marker in prefix for marker in DEMOLITION_MARKERS):
            return False

        position += len(stem)

    return True


def find_categories_needing_verification(
    text_lower: str,
    detection_details: dict[int, list[str]],
) -> dict[int, list[str]]:
    """Категории, все совпадения которых оказались в демонтажном контексте."""
    marker_count = sum(text_lower.count(marker) for marker in DEMOLITION_MARKERS)
    if marker_count < DEMOLITION_VERIFICATION_MIN_MARKERS:
        return {}

    suspicious: dict[int, list[str]] = {}
    for mc_id, phrases in detection_details.items():
        if mc_id in (SOURCE_MC_ID, DEMOLITION_MC_ID) or not phrases:
            continue
        if all(is_phrase_in_demolition_context(text_lower, p) for p in phrases):
            suspicious[mc_id] = phrases
    return suspicious


def should_verify_demolition(text_lower: str, detected: list[int]) -> bool:
    """Стадия 1 сняла демонтаж, но слово в тексте есть — стоит переспросить LLM."""
    if DEMOLITION_MC_ID in detected:
        return False
    if "демонтаж" not in text_lower:
        return False
    return not any(phrase in text_lower for phrase in DEMOLITION_STANDALONE)


def _format_category_line(mc_id: int, phrases: list[str]) -> str:
    quoted = ", ".join(f'"{p}"' for p in phrases[:MAX_PHRASES_IN_PROMPT])
    return f"- [{mc_id}] {CATEGORY_TITLES.get(mc_id, '?')} — найдено: {quoted}"


def build_prompt(
    text: str,
    verify_demolition: bool,
    suspicious: dict[int, list[str]],
) -> str:
    """Собирает один промпт под обе проверки — чтобы обойтись одним вызовом LLM."""
    questions: list[str] = []
    schema: list[str] = []
    if verify_demolition:
        questions.append(VERIFY_DEMOLITION_BLOCK)
        schema.append('"has_demolition_service": true/false')
    if suspicious:
        categories = "\n".join(
            _format_category_line(mc_id, phrases) for mc_id, phrases in suspicious.items()
        )
        questions.append(VERIFY_CONTEXT_BLOCK.format(categories=categories))
        schema.extend(f'"{mc_id}": true/false' for mc_id in suspicious)

    return VERIFY_PROMPT.format(
        text=text[:PROMPT_TEXT_LIMIT],
        questions="\n\n".join(questions),
        schema=", ".join(schema),
    )


class ContextVerifier:
    """Проверяет через LLM, реальны ли услуги, найденные в демонтажном тексте.

    Без клиента (нет API-ключа) работает в offline-режиме: ничего не отбрасывает
    и ничего не добавляет — то же поведение, что и при ошибке LLM.
    """

    def __init__(self, client: LLMClient | None, model: str) -> None:
        self.client = client
        self.model = model

    @staticmethod
    def _neutral(verify_demolition: bool, suspicious: dict[int, list[str]]) -> VerificationResult:
        # При отказе LLM сохраняем всё найденное: терять recall хуже, чем
        # оставить несколько ложных категорий, которые отфильтрует стадия 2.
        return VerificationResult(
            has_demolition_service=verify_demolition,
            context=dict.fromkeys(suspicious, True),
        )

    async def verify(
        self,
        text: str,
        verify_demolition: bool,
        suspicious: dict[int, list[str]],
    ) -> VerificationResult:
        if not verify_demolition and not suspicious:
            return VerificationResult()
        if self.client is None:
            return self._neutral(verify_demolition, suspicious)

        prompt = build_prompt(text, verify_demolition, suspicious)
        try:
            content = await self.client.complete(
                messages=[
                    {"role": "system", "content": "Отвечай строго JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
                temperature=0.6,
                max_tokens=192,
            )
            parsed = json.loads(content)
        except (LLMError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("Stage 1.5: верификация не удалась (%s), детекция сохранена", exc)
            return self._neutral(verify_demolition, suspicious)

        return VerificationResult(
            has_demolition_service=bool(parsed.get("has_demolition_service", False))
            if verify_demolition
            else False,
            context={
                mc_id: bool(parsed.get(str(mc_id), parsed.get(mc_id, True)))
                for mc_id in suspicious
            },
        )
