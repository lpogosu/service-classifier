"""Стадия 2 — решение о том, стоит ли разбивать объявление на отдельные.

Здесь LLM проиграла: семь моделей и десять промптов дали максимум accuracy 0.747,
а GBM на 44 ручных признаках плюс TF-IDF — 0.848 на кросс-валидации. Граница
«специалист против комплексного ремонта» задаётся распределением разметки, а не
логикой, которую модель может вывести из инструкции.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from service_classifier.llm import LLMClient, LLMError

logger = logging.getLogger(__name__)

#: Порог по умолчанию, если он не сохранён вместе с моделью.
DEFAULT_SPLIT_THRESHOLD = 0.40
#: Обрезка текста в промпте LLM-фолбэка.
LLM_PROMPT_TEXT_LIMIT = 500
#: Короткое объявление с небольшим числом услуг — типичный признак специалиста.
SHORT_TEXT_CHARS = 300
SHORT_TEXT_MAX_CATEGORIES = 4
#: Длинное объявление с «под ключ» и множеством услуг — типичный комплексный ремонт.
LONG_TEXT_CHARS = 800
LONG_TEXT_MIN_CATEGORIES = 5
#: До трёх услуг мастер обычно перечисляет как свои специализации.
SPECIALIST_MAX_CATEGORIES = 3

_SENTENCE_SPLIT_RE = re.compile(r"[.!?\n]")
_PRICE_RE = re.compile(r"\d+\s*(руб|₽|р\.)")
_DASH_LIST_RE = re.compile(r"\n\s*[-–—]\s*\w")
_NUMBERED_LIST_RE = re.compile(r"\n\s*\d+[.)]\s*\w")
_EMOJI_RE = re.compile(r"[\U0001f300-\U0001f9ff]")

_LIST_MARKERS = ("✔", "•", "✅", "⚙", "🔹", "▪", "►")
_PUNCTUATION = ".,!?;:-"


class SplitMode(StrEnum):
    """Чем принято решение — видно в /health и в логах."""

    CLASSIFIER = "classifier"
    LLM = "llm"
    HEURISTIC = "heuristic"


@dataclass
class SplitResult:
    should_split: bool
    split_mc_ids: list[int]
    mode: SplitMode


def extract_features(text: str, detected_mc_ids: list[int]) -> dict[str, float]:
    """44 ручных признака: длина и структура текста, маркеры «комплексности».

    Признаки поделены на три группы: объём текста, сигналы комплексного ремонта
    («под ключ», «все виды», «бригада») и сигналы специалиста («отдельно», цены,
    короткий текст). Их разность оказалась самым весомым признаком модели.
    """
    text_lower = text.lower()
    text_len = len(text)
    n_words = len(text.split())
    n_sentences = max(1, len(_SENTENCE_SPLIT_RE.split(text)))
    n_detected = len(detected_mc_ids)
    words = text_lower.split()

    features: dict[str, float] = {
        "text_len": text_len,
        "n_words": n_words,
        "n_sentences": n_sentences,
        "words_per_sentence": n_words / n_sentences,
        "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "n_newlines": text.count("\n"),
        "n_exclamation": text.count("!"),
        "n_question": text.count("?"),
        "n_commas": text.count(","),
        "n_dots": text.count("."),
        "n_detected": n_detected,
        "det_per_100chars": n_detected / max(1, text_len) * 100,
        "det_per_word": n_detected / max(1, n_words),
        "n_detected_sq": n_detected**2,
        "has_pod_klyuch": int("под ключ" in text_lower),
        "has_pod_klyuch_strong": int(
            any(w in text_lower for w in ["ремонт под ключ", "отделка под ключ"])
        ),
        "has_kompleks": int(any(w in text_lower for w in ["комплексн", "капитальн"])),
        "has_vse_vidy": int(
            any(
                w in text_lower
                for w in ["все виды", "любые виды", "полный спектр", "весь спектр"]
            )
        ),
        "has_vsyo": int(
            any(
                w in text_lower
                for w in [
                    "делаю всё", "делаю все", "абсолютно все",
                    "полный цикл", "любой сложности",
                ]
            )
        ),
        "has_brigada": int(
            any(w in text_lower for w in ["бригада", "команда мастер", "наша команда"])
        ),
        "has_remont": int("ремонт" in text_lower),
        "has_kosmeticheskiy": int("косметическ" in text_lower),
        "has_ot_do": int(any(w in text_lower for w in ["от и до", "от демонтажа до"])),
        "has_vklyuchaya": int(
            any(w in text_lower for w in ["включая", "в том числе", "в составе", "входит"])
        ),
        "has_garant": int(any(w in text_lower for w in ["гарантия", "гарантию"])),
        "has_dogovor": int("договор" in text_lower),
        "has_dizain": int("дизайн" in text_lower),
        "has_sdaem": int(any(w in text_lower for w in ["сдаем объект", "сдаём объект"])),
        "has_otdelno": int(any(w in text_lower for w in ["отдельно", "как отдельн"])),
        "has_price": int(bool(_PRICE_RE.search(text_lower))),
        "has_experience": int(any(w in text_lower for w in ["опыт", "стаж"])),
        "has_list_markers": int(any(c in text for c in _LIST_MARKERS)),
        "has_dash_list": int(bool(_DASH_LIST_RE.search(text))),
        "has_emoji": int(bool(_EMOJI_RE.search(text))),
        "has_numbered_list": int(bool(_NUMBERED_LIST_RE.search(text))),
        "uppercase_ratio": sum(1 for c in text if c.isupper()) / max(1, text_len),
        "digit_ratio": sum(1 for c in text if c.isdigit()) / max(1, text_len),
        "newline_ratio": text.count("\n") / max(1, text_len),
        "punct_ratio": sum(1 for c in text if c in _PUNCTUATION) / max(1, text_len),
    }

    comprehensive = (
        features["has_pod_klyuch"]
        + features["has_kompleks"]
        + features["has_vse_vidy"]
        + features["has_vsyo"]
        + features["has_brigada"]
        + features["has_vklyuchaya"]
    )
    specialist = (
        features["has_otdelno"]
        + features["has_price"]
        + (1 if n_detected <= SPECIALIST_MAX_CATEGORIES else 0)
    )
    features["comprehensive_score"] = comprehensive
    features["specialist_score"] = specialist
    features["comp_minus_spec"] = comprehensive - specialist
    features["short_specialist"] = int(
        text_len < SHORT_TEXT_CHARS and n_detected <= SHORT_TEXT_MAX_CATEGORIES
    )
    features["long_comprehensive"] = int(
        text_len > LONG_TEXT_CHARS
        and n_detected >= LONG_TEXT_MIN_CATEGORIES
        and bool(features["has_pod_klyuch"])
    )
    return features


SPLIT_SYSTEM_PROMPT = (
    "Ты эксперт по объявлениям об услугах ремонта. Определи: нужно ли создавать "
    "ОТДЕЛЬНЫЕ объявления для обнаруженных услуг? should_split=true ТОЛЬКО если "
    'исполнитель предлагает услуги как САМОСТОЯТЕЛЬНЫЕ. JSON: {"should_split": true/false}'
)


class SplitDecider:
    """Решает, разбивать ли объявление.

    Порядок выбора режима: обученный GBM → LLM (если есть ключ) → эвристика.
    Метрики README получены на первом режиме; два остальных нужны, чтобы
    пайплайн оставался работоспособным без артефактов и без сети.
    """

    def __init__(
        self,
        classifier_path: Path | None = None,
        client: LLMClient | None = None,
        model: str = "",
        temperature: float = 0.5,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.classifier: Any = None
        self.tfidf: Any = None
        self.feature_names: list[str] = []
        self.threshold = DEFAULT_SPLIT_THRESHOLD

        has_classifier = classifier_path is not None and classifier_path.exists()
        if has_classifier and classifier_path is not None:
            self._load_classifier(classifier_path)

        self.mode: SplitMode
        if has_classifier:
            self.mode = SplitMode.CLASSIFIER
        elif client is not None:
            self.mode = SplitMode.LLM
            logger.warning(
                "Классификатор %s не найден — стадия 2 работает через LLM "
                "(accuracy 0.747 против 0.848 у GBM)",
                classifier_path,
            )
        else:
            self.mode = SplitMode.HEURISTIC
            logger.warning(
                "Ни классификатора, ни LLM — стадия 2 работает на эвристике "
                "comprehensive_score/specialist_score, качество не измерялось. "
                "Обучите модель: make train"
            )

    def _load_classifier(self, path: Path) -> None:
        # Артефакт создаётся своим же `make train` и лежит рядом с кодом,
        # чужие pickle сюда не попадают.
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        self.classifier = data["classifier"]
        self.tfidf = data.get("tfidf")
        self.feature_names = list(data["feature_names"])
        self.threshold = float(data.get("threshold", DEFAULT_SPLIT_THRESHOLD))
        logger.info(
            "Split-классификатор загружен из %s (CV accuracy=%s, threshold=%.2f)",
            path, data.get("cv_accuracy", "n/a"), self.threshold,
        )

    async def decide(self, description: str, detected_mc_ids: list[int]) -> SplitResult:
        if not detected_mc_ids:
            return SplitResult(should_split=False, split_mc_ids=[], mode=self.mode)

        detected = sorted(detected_mc_ids)
        if self.mode is SplitMode.CLASSIFIER:
            should_split = self._predict_classifier(description, detected)
        elif self.mode is SplitMode.LLM:
            should_split = await self._predict_llm(description, detected)
        else:
            should_split = self._predict_heuristic(description, detected)

        # Ключевое наблюдение из разметки: targetSplitMcIds всегда совпадал с
        # targetDetectedMcIds, поэтому подзадача сводится к бинарному решению.
        return SplitResult(
            should_split=should_split,
            split_mc_ids=detected if should_split else [],
            mode=self.mode,
        )

    def _predict_classifier(self, description: str, detected: list[int]) -> bool:
        features = extract_features(description, detected)
        hand_crafted = np.array([[features[name] for name in self.feature_names]])

        if self.tfidf is not None:
            from scipy.sparse import csr_matrix, hstack

            tfidf_features = self.tfidf.transform([description.lower()])
            matrix = hstack([csr_matrix(hand_crafted), tfidf_features]).toarray()
        else:
            matrix = hand_crafted

        probability = float(self.classifier.predict_proba(matrix)[0][1])
        return probability >= self.threshold

    async def _predict_llm(self, description: str, detected: list[int]) -> bool:
        assert self.client is not None
        categories = ", ".join(str(mc_id) for mc_id in detected)
        try:
            content = await self.client.complete(
                messages=[
                    {"role": "system", "content": SPLIT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Обнаружены микрокатегории: {categories}\n"
                            f'Текст: "{description[:LLM_PROMPT_TEXT_LIMIT]}"'
                        ),
                    },
                ],
                model=self.model,
                temperature=self.temperature,
                max_tokens=64,
            )
            return bool(json.loads(content).get("should_split", False))
        except (LLMError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("Stage 2: LLM не ответила (%s), разделение не предлагается", exc)
            return False

    @staticmethod
    def _predict_heuristic(description: str, detected: list[int]) -> bool:
        """Знак comp_minus_spec — самый весомый признак обученной модели."""
        features = extract_features(description, detected)
        return features["comp_minus_spec"] <= 0
