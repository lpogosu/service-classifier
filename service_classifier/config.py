"""Конфигурация пайплайна: микрокатегории, пути к артефактам, настройки LLM."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

#: Исходная микрокатегория объявления. Из результатов детекции исключается всегда:
#: «ремонт под ключ» — это контейнер, а не отдельная услуга.
SOURCE_MC_ID: Final[int] = 101

#: Демонтаж обрабатывается особо: слова вроде «плитка» в демонтажном объявлении
#: означают объект сноса, а не предлагаемую услугу.
DEMOLITION_MC_ID: Final[int] = 111

CATEGORY_TITLES: Final[dict[int, str]] = {
    101: "Ремонт квартир и домов под ключ",
    102: "Сантехника",
    103: "Электрика",
    104: "Натяжные потолки",
    105: "Укладка плитки",
    106: "Поклейка обоев",
    107: "Малярные работы",
    108: "Штукатурные работы",
    109: "Напольные покрытия",
    110: "Гипсокартон",
    111: "Демонтажные работы",
}

#: Категории, которые может вернуть детектор (без контейнерной 101).
TARGET_MC_IDS: Final[tuple[int, ...]] = tuple(
    mc_id for mc_id in CATEGORY_TITLES if mc_id != SOURCE_MC_ID
)

DEFAULT_KEY_PHRASES_PATH: Final[Path] = PROJECT_ROOT / "data" / "key_phrases.csv"
DEFAULT_SPLIT_MODEL_PATH: Final[Path] = PROJECT_ROOT / "models" / "split_classifier.pkl"
DEFAULT_DATASET_PATH: Final[Path] = PROJECT_ROOT / "data" / "synthetic_dataset.csv"

OPENROUTER_URL: Final[str] = "https://openrouter.ai/api/v1/chat/completions"

#: Верификация контекста — задача на рассуждение, нужна сильная модель.
DEFAULT_VERIFICATION_MODEL: Final[str] = "qwen/qwen3-235b-a22b-2507"
#: Генерация черновиков — измеренный hallucination rate 1% против 36% у Qwen3.
DEFAULT_DRAFT_MODEL: Final[str] = "google/gemini-2.0-flash-001"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} должен быть целым числом, получено {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Настройки процесса, собранные из окружения."""

    key_phrases_path: Path = DEFAULT_KEY_PHRASES_PATH
    split_model_path: Path = DEFAULT_SPLIT_MODEL_PATH
    openrouter_api_key: str = ""
    verification_model: str = DEFAULT_VERIFICATION_MODEL
    draft_model: str = DEFAULT_DRAFT_MODEL
    request_timeout_sec: int = 60
    max_retries: int = 2
    draft_concurrency: int = 10
    log_level: str = "INFO"

    @property
    def llm_enabled(self) -> bool:
        """Есть ли ключ для обращений к LLM.

        Без ключа стадии 1.5 и 3 работают в offline-режиме: верификация ничего
        не отбрасывает, черновики собираются по шаблону. Детекция и решение
        о разделении от LLM не зависят и работают всегда.
        """
        return bool(self.openrouter_api_key)

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            key_phrases_path=Path(
                os.getenv("KEYPHRASES_PATH", str(DEFAULT_KEY_PHRASES_PATH))
            ),
            split_model_path=Path(
                os.getenv("SPLIT_MODEL_PATH", str(DEFAULT_SPLIT_MODEL_PATH))
            ),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
            verification_model=os.getenv(
                "VERIFICATION_MODEL", DEFAULT_VERIFICATION_MODEL
            ),
            draft_model=os.getenv("DRAFT_MODEL", DEFAULT_DRAFT_MODEL),
            request_timeout_sec=_env_int("LLM_TIMEOUT_SEC", 60),
            max_retries=_env_int("LLM_MAX_RETRIES", 2),
            draft_concurrency=_env_int("DRAFT_CONCURRENCY", 10),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
