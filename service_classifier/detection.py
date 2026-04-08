"""Стадия 1 — детерминированная детекция микрокатегорий по тексту объявления.

Лемматизация (pymorphy3) приводит «электрику», «электрике», «электрики» к одной
форме, fuzzy-сравнение (rapidfuzz) закрывает опечатки. LLM здесь не участвует:
стадия должна быть воспроизводимой и дешёвой.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import ClassVar

import pymorphy3
from rapidfuzz import fuzz

from service_classifier.config import (
    DEMOLITION_MC_ID,
    SOURCE_MC_ID,
)

#: Подмена латиницы на визуально идентичную кириллицу встречалась в 16% текстов
#: исходного датасета — без нормализации такие объявления теряются целиком.
_HOMOGLYPH_MAP = str.maketrans(
    {
        "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К",
        "M": "М", "O": "О", "P": "Р", "T": "Т", "X": "Х",
        "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х",
        "y": "у", "k": "к",
    }
)

_WORD_RE = re.compile(r"[а-яa-z0-9]+")
_KEEP_CHARS_RE = re.compile(r"[^\w\s.,;:!?\-()«»\"'—–/\\]")

#: Однословные фразы дают больше всего ложных срабатываний при fuzzy-сравнении,
#: поэтому у них отдельный, более высокий порог.
SINGLE_WORD_FUZZY_FLOOR = 85
#: Порог 86 (вместо 80) убрал 476 ложных срабатываний: у многословных фраз
#: совпадал общий префикс («демонтаж старой ...»), а не сама услуга.
MULTI_WORD_FUZZY_FLOOR = 86

#: Если категорий много и нет явной «отдельной» демонтажной формулировки,
#: демонтаж почти всегда часть комплексного ремонта, а не самостоятельная услуга.
MAX_CATEGORIES_KEEPING_DEMOLITION = 3


@dataclass
class MicroCategory:
    """Микрокатегория и её словарь ключевых фраз."""

    mc_id: int
    mc_title: str
    key_phrases: list[str]
    key_phrases_lemmatized: list[str] = field(default_factory=list)
    description: str = ""


def load_categories(path: str | Path) -> dict[int, MicroCategory]:
    """Читает CSV-словарь ключевых фраз (`mcId;mcTitle;keyPhrases;description`)."""
    categories: dict[int, MicroCategory] = {}
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            mc_id = int(row["mcId"])
            phrases = [p.strip().lower() for p in row["keyPhrases"].split(";") if p.strip()]
            categories[mc_id] = MicroCategory(
                mc_id=mc_id,
                mc_title=row["mcTitle"],
                key_phrases=phrases,
                description=row.get("description", ""),
            )
    return categories


def normalize_text(text: str) -> str:
    """Гомоглифы → кириллица, ё → е, эмодзи и прочий юникод → пробел."""
    text = text.translate(_HOMOGLYPH_MAP)
    text = text.replace("ё", "е").replace("Ё", "Е")
    return _KEEP_CHARS_RE.sub(" ", text)


class KeywordDetector:
    """Детекция микрокатегорий через лемматизацию ключевых фраз."""

    #: Шумные категории требуют двух совпадений: одиночное слово вроде «розетка»
    #: встречается в любом описании ремонта и само по себе ничего не значит.
    PER_CATEGORY_MIN_MATCHES: ClassVar[dict[int, int]] = {
        102: 2,  # Сантехника — 152 FP при mm=1, якоря компенсируют recall
        103: 2,  # Электрика — 242 FP при mm=1
        104: 1,  # Натяжные потолки — mm=2 обрушивал recall 0.833 → 0.613
        105: 1,  # Укладка плитки
        106: 2,  # Поклейка обоев
        107: 2,  # Малярные работы
        108: 2,  # Штукатурные работы
        109: 2,  # Напольные покрытия
        110: 2,  # Гипсокартон — 136 FP при mm=1
        111: 2,  # Демонтажные работы — mm=1 ухудшал F1 до 0.543
    }

    #: Якоря — однозначные слова, которые сами по себе доказывают категорию
    #: и потому обходят требование PER_CATEGORY_MIN_MATCHES.
    ANCHOR_WORDS: ClassVar[dict[int, list[str]]] = {
        102: [
            "сантехник", "сантехника", "сантехнические работы",
            "водоснабжение", "канализация", "водоразводка",
            "полипропилен", "замена труб", "разводка труб",
            "установка сантехники", "ремонт сантехники",
            "замена стояка", "замена смесителя",
            "монтаж труб", "монтаж водоснабжения",
            "водопровод", "унитаз", "смеситель", "раковин",
            "ванной комнат", "ванных комнат", "санузл",
            "ремонт ванн", "отделка ванн",
        ],
        103: [
            "электрик", "электрика", "электромонтаж",
            "электромонтажные работы", "замена проводки",
            "монтаж электрики", "электропроводка",
            "электрические работы", "электроснабжение",
        ],
        104: [
            "натяжные потолки", "натяжной потолок",
            "установка натяжных потолков", "монтаж натяжных потолков",
            "натяжные", "потолки пвх",
            "потолки", "потолков", "потолка",
        ],
        105: [
            "кафель", "кафеля", "кафелем", "керамогранит", "мозаика",
            "плиточник", "плиточные работы", "укладка плитки",
            "облицовка плиткой", "кафельная плитка",
            "плитка", "плиткой", "плитку", "плиточных",
        ],
        106: ["обои", "обоев", "обоями", "поклейка обоев", "оклейка обоев"],
        107: [
            "малярка", "малярные", "покраска стен", "покраска потолка",
            "малярные работы", "малярных работ",
            "покраска", "покрасить", "окраска",
        ],
        108: [
            "штукатурка", "штукатурные работы", "штукатурка стен",
            "оштукатуривание", "штукатурных работ",
            "шпаклевка", "шпаклёвка", "шпатлевка", "шпатлёвка",
        ],
        109: [
            "ламинат", "линолеум", "кварцвинил", "паркет", "паркетный",
            "стяжка пола", "укладка пола", "укладка ламината",
        ],
        110: [
            "гкл", "гвл", "гипсокартон", "гипсокартоном", "гипсокартона",
            "монтаж гипсокартона", "работы с гипсокартоном",
            "перегородки из гипсокартона",
            "гипрок", "гипроком", "гипрока",
            "перегородки", "перегородок",
        ],
        111: [
            "демонтажные работы", "демонтаж стен", "снос перегородок",
            "демонтаж любой сложности", "демонтаж отделки",
            "демонтаж перегородок", "алмазная резка",
            "демонтаж полов", "демонтаж пола",
            "демонтаж", "демонтажных", "демонтажные",
        ],
    }

    #: Формулировки, по которым видно, что демонтаж продаётся отдельно.
    DEMOLITION_STANDALONE: ClassVar[list[str]] = [
        "демонтажные работы", "демонтаж любой сложности",
        "разборка", "снос перегородок", "демонтаж стен",
        "полный демонтаж", "частичный демонтаж", "демонтаж отделки",
        "демонтаж под ремонт", "подготовка квартиры к ремонту",
        "демонтаж перегородок", "алмазная резка", "вынос мусора",
        "демонтаж дверей", "демонтаж окон", "демонтаж кухни",
        "демонтаж полов", "демонтаж пола", "демонтаж покрытий",
    ]

    def __init__(
        self,
        keyphrases_path: str | Path,
        fuzzy_threshold: int = 80,
        min_phrase_len: int = 4,
        min_matches: dict[int, int] | None = None,
    ) -> None:
        self.morph = pymorphy3.MorphAnalyzer()
        self.fuzzy_threshold = fuzzy_threshold
        self.min_phrase_len = min_phrase_len
        #: Копия покатегорийных порогов: скрипт подбора меняет её на экземпляре,
        #: не трогая настройку по умолчанию.
        self.min_matches = dict(min_matches or self.PER_CATEGORY_MIN_MATCHES)
        self.categories: dict[int, MicroCategory] = load_categories(keyphrases_path)

        self._lemmatize_word = lru_cache(maxsize=100_000)(self._lemmatize_word_uncached)
        for category in self.categories.values():
            category.key_phrases_lemmatized = [
                self.lemmatize(p) for p in category.key_phrases
            ]
        self._anchors_lemmatized: dict[int, list[str]] = {
            mc_id: [self.lemmatize(anchor) for anchor in anchors]
            for mc_id, anchors in self.ANCHOR_WORDS.items()
        }

    def _lemmatize_word_uncached(self, word: str) -> str:
        parsed = self.morph.parse(word)
        return parsed[0].normal_form if parsed else word

    def lemmatize(self, text: str) -> str:
        """Нормализует текст и приводит каждое слово к начальной форме."""
        words = _WORD_RE.findall(normalize_text(text).lower())
        return " ".join(self._lemmatize_word(word) for word in words)

    def _text_contains_phrase(
        self,
        text_lemmatized: str,
        phrase_lemmatized: str,
        phrase_original: str,
    ) -> bool:
        """Есть ли фраза в тексте — точно или с поправкой на опечатки."""
        if len(phrase_original) < self.min_phrase_len:
            return False

        if phrase_lemmatized in text_lemmatized:
            return True

        phrase_words = phrase_lemmatized.split()
        if len(phrase_words) <= 1:
            threshold = max(self.fuzzy_threshold, SINGLE_WORD_FUZZY_FLOOR)
            return any(
                fuzz.ratio(word, phrase_lemmatized) >= threshold
                for word in text_lemmatized.split()
            )

        threshold = max(self.fuzzy_threshold, MULTI_WORD_FUZZY_FLOOR)
        text_words = text_lemmatized.split()
        window_size = len(phrase_words)
        for start in range(len(text_words) - window_size + 1):
            window = " ".join(text_words[start : start + window_size])
            if fuzz.token_sort_ratio(window, phrase_lemmatized) >= threshold:
                return True
        return False

    def _find_anchor(
        self,
        mc_id: int,
        text_normalized: str,
        text_lemmatized: str,
    ) -> str | None:
        """Первый сработавший якорь категории — по сырому и по лемматизированному тексту."""
        anchors = self.ANCHOR_WORDS.get(mc_id)
        if not anchors:
            return None
        for anchor in anchors:
            if anchor in text_normalized:
                return anchor
        for anchor, anchor_lemmatized in zip(anchors, self._anchors_lemmatized[mc_id], strict=True):
            if anchor_lemmatized in text_lemmatized:
                return anchor
        return None

    def detect_with_details(
        self,
        text: str,
        source_mc_id: int | None = None,
    ) -> dict[int, list[str]]:
        """Детекция с расшифровкой: какие именно фразы сработали.

        Returns:
            `{mc_id: [сработавшие фразы]}` без исходной категории и без 101.
        """
        text_lemmatized = self.lemmatize(text)
        text_normalized = normalize_text(text).lower()
        results: dict[int, list[str]] = {}

        for mc_id, category in self.categories.items():
            if mc_id in (source_mc_id, SOURCE_MC_ID):
                continue

            matched = [
                phrase
                for phrase, phrase_lemmatized in zip(
                    category.key_phrases, category.key_phrases_lemmatized
                , strict=True)
                if self._text_contains_phrase(text_lemmatized, phrase_lemmatized, phrase)
            ]

            anchor = self._find_anchor(mc_id, text_normalized, text_lemmatized)
            if anchor is not None and anchor not in matched:
                matched.append(anchor)

            min_matches = self.min_matches.get(mc_id, 1)
            if anchor is not None or len(matched) >= min_matches:
                results[mc_id] = matched

        self._apply_demolition_context_rule(results, text_normalized)
        return results

    def detect(self, text: str, source_mc_id: int | None = None) -> list[int]:
        """Список mc_id обнаруженных микрокатегорий."""
        return sorted(self.detect_with_details(text, source_mc_id))

    def _apply_demolition_context_rule(
        self,
        results: dict[int, list[str]],
        text_normalized: str,
    ) -> None:
        """Убирает демонтаж, если он выглядит подготовкой к другим работам.

        Снятые здесь категории отправляются на LLM-верификацию в стадии 1.5,
        так что правило смещает баланс в сторону precision без потери recall.
        """
        if DEMOLITION_MC_ID not in results:
            return
        if len(results) <= MAX_CATEGORIES_KEEPING_DEMOLITION:
            return
        if any(phrase in text_normalized for phrase in self.DEMOLITION_STANDALONE):
            return
        del results[DEMOLITION_MC_ID]
