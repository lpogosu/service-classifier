from __future__ import annotations

import pytest

from service_classifier.config import DEFAULT_KEY_PHRASES_PATH
from service_classifier.detection import KeywordDetector
from service_classifier.llm import LLMError, Message


#: pymorphy3 инициализируется около секунды — один детектор на всю сессию.
@pytest.fixture(scope="session")
def detector() -> KeywordDetector:
    return KeywordDetector(DEFAULT_KEY_PHRASES_PATH)


class FakeLLMClient:
    """Провайдер без сети: отдаёт заранее заданные ответы по очереди."""

    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        self.calls.append(messages)
        if not self.responses:
            raise LLMError("У фейкового клиента закончились ответы")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response
