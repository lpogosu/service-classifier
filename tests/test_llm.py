"""Клиент LLM: повторы, классификация ошибок, разбор ответа. Сеть не используется."""

from __future__ import annotations

from typing import Any

import aiohttp
import pytest
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

from service_classifier.config import Settings
from service_classifier.llm import (
    LLMError,
    OpenRouterClient,
    build_client,
    strip_code_fence,
)

MESSAGES = [{"role": "user", "content": "привет"}]


def response_payload(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}}]}


def http_error(status: int) -> aiohttp.ClientResponseError:
    url = URL("https://openrouter.example/api")
    return aiohttp.ClientResponseError(
        request_info=aiohttp.RequestInfo(url, "POST", CIMultiDictProxy(CIMultiDict()), url),
        history=(),
        status=status,
        message=f"HTTP {status}",
    )


class RecordingClient(OpenRouterClient):
    """OpenRouterClient с подменённым транспортом: считает попытки."""

    def __init__(self, outcomes: list[Any], **kwargs: Any) -> None:
        super().__init__("test-key", **kwargs)
        self.outcomes = list(outcomes)
        self.attempts = 0

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.attempts += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        assert isinstance(outcome, dict)
        return outcome


async def complete(client: OpenRouterClient) -> str:
    return await client.complete(
        MESSAGES, model="test-model", temperature=0.0, max_tokens=32
    )


def test_api_key_is_required() -> None:
    with pytest.raises(ValueError, match="API-ключ"):
        OpenRouterClient("")


def test_strip_code_fence_removes_json_wrapper() -> None:
    assert strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_code_fence('  {"a": 1}  ') == '{"a": 1}'


async def test_successful_call_makes_one_attempt() -> None:
    client = RecordingClient([response_payload('{"ok": true}')])
    assert await complete(client) == '{"ok": true}'
    assert client.attempts == 1


async def test_rate_limit_is_retried_until_success() -> None:
    client = RecordingClient(
        [http_error(429), http_error(503), response_payload("готово")],
        max_retries=2,
    )
    assert await complete(client) == "готово"
    assert client.attempts == 3


async def test_timeout_is_retried() -> None:
    client = RecordingClient([TimeoutError(), response_payload("готово")], max_retries=1)
    assert await complete(client) == "готово"
    assert client.attempts == 2


async def test_client_error_is_not_retried() -> None:
    """401 не пройдёт со второй попытки — повторять бессмысленно."""
    client = RecordingClient([http_error(401)], max_retries=3)
    with pytest.raises(LLMError):
        await complete(client)
    assert client.attempts == 1


async def test_retries_are_exhausted() -> None:
    client = RecordingClient([http_error(429)] * 3, max_retries=2)
    with pytest.raises(LLMError, match="3 попыток"):
        await complete(client)
    assert client.attempts == 3


async def test_malformed_response_structure_raises_immediately() -> None:
    client = RecordingClient([{"choices": []}], max_retries=3)
    with pytest.raises(LLMError, match="структура ответа"):
        await complete(client)
    assert client.attempts == 1


async def test_response_fence_is_stripped() -> None:
    client = RecordingClient([response_payload('```json\n{"should_split": true}\n```')])
    assert await complete(client) == '{"should_split": true}'


def test_build_client_returns_none_without_key() -> None:
    assert build_client(Settings(openrouter_api_key="")) is None


def test_build_client_returns_client_with_key() -> None:
    client = build_client(Settings(openrouter_api_key="test-key", max_retries=5))
    assert isinstance(client, OpenRouterClient)
    assert client.max_retries == 5
