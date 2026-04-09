"""Стадия 2: извлечение признаков и выбор режима принятия решения."""

from __future__ import annotations

from service_classifier.llm import LLMError
from service_classifier.split import SplitDecider, SplitMode, extract_features
from tests.conftest import FakeLLMClient

ELECTRICS = 103
TILING = 105
PLASTER = 108

#: Тот же материал, что и в test_detection.py: стадия 1 находит электрику
#: в обеих формулировках, а различать их — работа признаков стадии 2.
PACKAGE_TEXT = (
    "Ремонт квартир под ключ. Наша бригада выполнит все виды работ, "
    "включая электрику, плитку и штукатурку. Работаем по договору."
)
SPECIALIST_TEXT = (
    "Электрику делаем отдельно, плитку отдельно, штукатурку отдельно.\n"
    "Цена за м2 от 900 руб."
)

FEATURE_COUNT = 44


def test_feature_set_is_stable() -> None:
    features = extract_features(SPECIALIST_TEXT, [ELECTRICS, TILING, PLASTER])
    assert len(features) == FEATURE_COUNT
    assert set(features) == set(extract_features("", []))


def test_packaging_language_separates_the_two_texts() -> None:
    """«включая» против «отдельно» — то, ради чего стадия 2 вообще существует."""
    package = extract_features(PACKAGE_TEXT, [ELECTRICS, TILING, PLASTER])
    specialist = extract_features(SPECIALIST_TEXT, [ELECTRICS, TILING, PLASTER])

    assert package["has_vklyuchaya"] == 1
    assert package["has_otdelno"] == 0
    assert specialist["has_otdelno"] == 1
    assert specialist["has_vklyuchaya"] == 0
    assert package["comp_minus_spec"] > specialist["comp_minus_spec"]


def test_comprehensive_and_specialist_scores_are_derived_consistently() -> None:
    features = extract_features(PACKAGE_TEXT, [ELECTRICS, TILING, PLASTER])
    assert (
        features["comp_minus_spec"]
        == features["comprehensive_score"] - features["specialist_score"]
    )


def test_structural_features_read_the_layout() -> None:
    bulleted = "Мастер.\n- укладка плитки\n- штукатурка стен\n1) электрика"
    features = extract_features(bulleted, [ELECTRICS, TILING, PLASTER])
    assert features["has_dash_list"] == 1
    assert features["has_numbered_list"] == 1
    assert features["n_newlines"] == 3


def test_ratio_features_are_safe_on_empty_text() -> None:
    features = extract_features("", [])
    assert features["text_len"] == 0
    assert features["avg_word_len"] == 0
    assert features["uppercase_ratio"] == 0
    assert features["det_per_word"] == 0


def test_price_and_experience_markers() -> None:
    features = extract_features("Опыт 10 лет. Цена 1500 руб. за м2.", [TILING])
    assert features["has_price"] == 1
    assert features["has_experience"] == 1


async def test_heuristic_mode_without_model_and_without_llm() -> None:
    decider = SplitDecider(classifier_path=None, client=None)
    assert decider.mode is SplitMode.HEURISTIC

    specialist = await decider.decide(SPECIALIST_TEXT, [ELECTRICS, TILING, PLASTER])
    package = await decider.decide(PACKAGE_TEXT, [ELECTRICS, TILING, PLASTER])
    assert specialist.should_split is True
    assert package.should_split is False


async def test_split_ids_repeat_detected_ids() -> None:
    """Разметка кейса: targetSplitMcIds всегда совпадал с targetDetectedMcIds."""
    decider = SplitDecider(classifier_path=None, client=None)
    result = await decider.decide(SPECIALIST_TEXT, [TILING, ELECTRICS])
    assert result.split_mc_ids == [ELECTRICS, TILING]


async def test_no_categories_means_no_split() -> None:
    decider = SplitDecider(classifier_path=None, client=None)
    result = await decider.decide(SPECIALIST_TEXT, [])
    assert result.should_split is False
    assert result.split_mc_ids == []


async def test_llm_mode_used_when_model_file_is_absent() -> None:
    client = FakeLLMClient(['{"should_split": true}'])
    decider = SplitDecider(classifier_path=None, client=client, model="test-model")
    assert decider.mode is SplitMode.LLM

    result = await decider.decide(PACKAGE_TEXT, [ELECTRICS, TILING])
    assert result.should_split is True
    assert len(client.calls) == 1


async def test_llm_failure_defaults_to_no_split() -> None:
    """Лишний черновик хуже отсутствующего: при отказе не предлагаем разделение."""
    client = FakeLLMClient([LLMError("provider down")])
    decider = SplitDecider(classifier_path=None, client=client, model="test-model")

    result = await decider.decide(SPECIALIST_TEXT, [ELECTRICS])
    assert result.should_split is False
