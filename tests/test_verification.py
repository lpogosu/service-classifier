"""Стадия 1.5: распознавание демонтажного контекста и поведение при отказе LLM."""

from __future__ import annotations

from service_classifier.llm import LLMError
from service_classifier.verification import (
    DEMOLITION_MARKERS,
    DEMOLITION_VERIFICATION_MIN_MARKERS,
    ContextVerifier,
    build_prompt,
    find_categories_needing_verification,
    is_phrase_in_demolition_context,
    should_verify_demolition,
)
from tests.conftest import FakeLLMClient

TILING = 105
WALLPAPER = 106
DEMOLITION = 111

PURE_DEMOLITION_TEXT = (
    "демонтаж квартир и домов. демонтаж стен, демонтаж перегородок, "
    "демонтаж плитки, демонтаж обоев, демонтаж штукатурки, демонтаж полов, "
    "демонтаж потолков, демонтаж дверей, снос конструкций, слом стен, "
    "разборка коробов, снятие покрытий, удаление старой отделки, "
    "очистка помещения, разбор мебели, вынос мусора."
)


def test_phrase_only_next_to_demolition_markers() -> None:
    assert is_phrase_in_demolition_context("демонтаж плитки со стен", "плитка") is True


def test_phrase_used_as_a_real_service() -> None:
    text = "укладка плитки под ключ, демонтаж стен по желанию"
    assert is_phrase_in_demolition_context(text, "плитка") is False


def test_phrase_absent_from_text_counts_as_demolition_context() -> None:
    """Фраза сработала по fuzzy и буквально в тексте не встречается."""
    assert is_phrase_in_demolition_context("сносим перегородки", "керамогранит") is True


def test_phrase_starting_with_marker_is_demolition() -> None:
    assert is_phrase_in_demolition_context("что угодно", "демонтаж плитки") is True


def test_verification_skipped_for_ordinary_renovation_ads() -> None:
    text = "укладка плитки, демонтаж старой плитки, штукатурка стен"
    assert find_categories_needing_verification(text, {TILING: ["плитка"]}) == {}


def test_verification_triggered_for_demolition_heavy_ads() -> None:
    marker_count = sum(
        PURE_DEMOLITION_TEXT.count(marker) for marker in DEMOLITION_MARKERS
    )
    assert marker_count >= DEMOLITION_VERIFICATION_MIN_MARKERS

    suspicious = find_categories_needing_verification(
        PURE_DEMOLITION_TEXT, {TILING: ["плитка"], WALLPAPER: ["обои"]}
    )
    assert set(suspicious) == {TILING, WALLPAPER}


def test_demolition_category_itself_is_never_suspicious() -> None:
    suspicious = find_categories_needing_verification(
        PURE_DEMOLITION_TEXT, {DEMOLITION: ["демонтажные работы"]}
    )
    assert suspicious == {}


def test_should_verify_demolition_only_when_stage_one_suppressed_it() -> None:
    assert should_verify_demolition("демонтаж старой плитки", detected=[TILING]) is True
    assert should_verify_demolition("демонтаж старой плитки", detected=[DEMOLITION]) is False
    assert should_verify_demolition("укладка плитки", detected=[TILING]) is False
    # Явно самостоятельная формулировка — переспрашивать LLM незачем.
    assert should_verify_demolition("демонтажные работы, демонтаж", detected=[]) is False


def test_prompt_contains_both_questions_and_schema() -> None:
    prompt = build_prompt("текст", verify_demolition=True, suspicious={TILING: ["плитка"]})
    assert "has_demolition_service" in prompt
    assert f'"{TILING}": true/false' in prompt
    assert "плитка" in prompt


async def test_verifier_removes_category_rejected_by_llm() -> None:
    client = FakeLLMClient(['{"105": false, "106": true}'])
    verifier = ContextVerifier(client, model="test-model")

    result = await verifier.verify(
        PURE_DEMOLITION_TEXT,
        verify_demolition=False,
        suspicious={TILING: ["плитка"], WALLPAPER: ["обои"]},
    )
    assert result.context == {TILING: False, WALLPAPER: True}
    assert len(client.calls) == 1


async def test_verifier_confirms_demolition_service() -> None:
    client = FakeLLMClient(['{"has_demolition_service": true}'])
    verifier = ContextVerifier(client, model="test-model")

    result = await verifier.verify("демонтаж", verify_demolition=True, suspicious={})
    assert result.has_demolition_service is True


async def test_verifier_keeps_everything_when_llm_fails() -> None:
    """Отказ LLM не должен стоить recall — детекция сохраняется целиком."""
    client = FakeLLMClient([LLMError("provider down")])
    verifier = ContextVerifier(client, model="test-model")

    result = await verifier.verify(
        PURE_DEMOLITION_TEXT, verify_demolition=True, suspicious={TILING: ["плитка"]}
    )
    assert result.context == {TILING: True}
    assert result.has_demolition_service is True


async def test_verifier_survives_malformed_json() -> None:
    client = FakeLLMClient(["это не json"])
    verifier = ContextVerifier(client, model="test-model")

    result = await verifier.verify(
        "текст", verify_demolition=False, suspicious={TILING: ["плитка"]}
    )
    assert result.context == {TILING: True}


async def test_offline_verifier_makes_no_calls() -> None:
    verifier = ContextVerifier(None, model="test-model")
    result = await verifier.verify(
        PURE_DEMOLITION_TEXT, verify_demolition=True, suspicious={TILING: ["плитка"]}
    )
    assert result.context == {TILING: True}
    assert result.has_demolition_service is True


async def test_nothing_to_verify_returns_empty_result() -> None:
    client = FakeLLMClient(['{"unused": true}'])
    verifier = ContextVerifier(client, model="test-model")

    result = await verifier.verify("текст", verify_demolition=False, suspicious={})
    assert result.context == {}
    assert result.has_demolition_service is False
    assert client.calls == []
