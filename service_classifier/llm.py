"""Клиент LLM-провайдера (OpenRouter) с повторами на временных сбоях.

Транспорт вынесен в отдельный метод `_post`, чтобы тесты подменяли его фейком
и проверяли логику повторов без сети.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol

import aiohttp

from service_classifier.config import OPENROUTER_URL, Settings

logger = logging.getLogger(__name__)

Message = dict[str, str]

#: Коды, при которых повтор осмыслен: троттлинг и сбои на стороне провайдера.
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
#: База экспоненциальной задержки: 0.5с, 1с, 2с ...
RETRY_BACKOFF_BASE_SEC = 0.5


class LLMError(RuntimeError):
    """Обращение к LLM не удалось после всех повторов."""


class LLMClient(Protocol):
    """Минимальный контракт, которого достаточно всем стадиям пайплайна."""

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Возвращает текст ответа модели (ожидается JSON-объект)."""
        ...


def strip_code_fence(content: str) -> str:
    """Убирает обрамление ```json ... ```, которое модели добавляют вопреки промпту."""
    content = content.strip()
    if not content.startswith("```"):
        return content
    without_opening = content.split("\n", 1)[-1]
    return without_opening.rsplit("```", 1)[0].strip()


class OpenRouterClient:
    """Асинхронный клиент OpenRouter с повторами и общей HTTP-сессией."""

    def __init__(
        self,
        api_key: str,
        *,
        timeout_sec: int = 60,
        max_retries: int = 2,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OpenRouterClient требует непустой API-ключ")
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self._session = session
        self._owns_session = session is None

    async def __aenter__(self) -> OpenRouterClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_session and self._session is not None and not self._session.closed:
            await self._session.close()
        if self._owns_session:
            self._session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self._session

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Один HTTP-запрос. Подменяется в тестах."""
        session = await self._get_session()
        async with session.post(
            OPENROUTER_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=aiohttp.ClientTimeout(total=self.timeout_sec),
        ) as response:
            if response.status in RETRYABLE_STATUS_CODES:
                raise aiohttp.ClientResponseError(
                    response.request_info,
                    response.history,
                    status=response.status,
                    message=await response.text(),
                )
            response.raise_for_status()
            data: dict[str, Any] = await response.json()
            return data

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        if isinstance(exc, aiohttp.ClientResponseError):
            return exc.status in RETRYABLE_STATUS_CODES
        return isinstance(exc, (asyncio.TimeoutError, aiohttp.ClientConnectionError))

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }

        last_error: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                data = await self._post(payload)
                return strip_code_fence(data["choices"][0]["message"]["content"])
            except (KeyError, IndexError, TypeError) as exc:
                raise LLMError(f"Неожиданная структура ответа {model}: {exc}") from exc
            except Exception as exc:
                if not self._is_retryable(exc):
                    raise LLMError(f"Запрос к {model} не удался: {exc}") from exc
                last_error = exc
                if attempt < self.max_retries:
                    delay = RETRY_BACKOFF_BASE_SEC * (2**attempt)
                    logger.warning(
                        "LLM %s: попытка %d/%d не удалась (%s), повтор через %.1fс",
                        model, attempt + 1, self.max_retries + 1, exc, delay,
                    )
                    await asyncio.sleep(delay)

        raise LLMError(
            f"Запрос к {model} не удался после {self.max_retries + 1} попыток: {last_error}"
        )


def build_client(settings: Settings) -> OpenRouterClient | None:
    """Клиент, если ключ задан; иначе None — пайплайн уйдёт в offline-режим."""
    if not settings.llm_enabled:
        logger.warning(
            "OPENROUTER_API_KEY не задан — стадии 1.5 и 3 работают offline: "
            "верификация ничего не отбрасывает, черновики собираются по шаблону"
        )
        return None
    return OpenRouterClient(
        settings.openrouter_api_key,
        timeout_sec=settings.request_timeout_sec,
        max_retries=settings.max_retries,
    )
