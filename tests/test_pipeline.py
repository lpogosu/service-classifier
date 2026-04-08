"""Сквозной проход по стадиям — офлайн, без сети."""

from __future__ import annotations

from pathlib import Path

from service_classifier.config import Settings
from service_classifier.detection import KeywordDetector
from service_classifier.evaluation import binary_accuracy, micro_f1, per_category_f1
from service_classifier.pipeline import Ad, ServiceClassifier
from service_classifier.split import SplitMode
from tests.conftest import FakeLLMClient

TILING = 105
PLASTER = 108

SPECIALIST_AD = Ad(
    item_id=1,
    mc_id=101,
    mc_title="Ремонт квартир и домов под ключ",
    description=(
        "Мастер с собственным инструментом.\n"
        "- укладка плитки и керамогранита\n"
        "- штукатурка стен по маякам\n"
        "Каждый вид работ считаю отдельно, цена от 900 руб. за м2."
    ),
)

PURE_DEMOLITION_AD = Ad(
    item_id=2,
    mc_id=101,
    mc_title="Ремонт квартир и домов под ключ",
    description=(
        "Демонтаж квартир и домов. ДЕМОНТАЖ, СЛОМ, СНОС: демонтаж стен, "
        "демонтаж перегородок, демонтаж плитки и кафеля, демонтаж обоев, "
        "демонтаж штукатурки, демонтаж полов, снос конструкций, разборка "
        "коробов ГКЛ, снятие покрытий, удаление старой отделки. Вынос мусора."
    ),
)


#: Тесты не должны зависеть от того, лежит ли рядом артефакт `make train`:
#: путь заведомо несуществующий, поэтому режим стадии 2 предсказуем.
NO_CLASSIFIER = Settings(split_model_path=Path("models") / "not-trained.pkl")


def offline_classifier() -> ServiceClassifier:
    return ServiceClassifier(NO_CLASSIFIER, client=None)


async def test_offline_pipeline_runs_end_to_end() -> None:
    classifier = offline_classifier()
    assert classifier.split_mode is SplitMode.HEURISTIC

    result = await classifier.process(SPECIALIST_AD)
    assert TILING in result.detected_mc_ids
    assert PLASTER in result.detected_mc_ids
    assert result.should_split is True
    assert {draft.mc_id for draft in result.drafts} == set(result.split_mc_ids)
    assert all(draft.is_fallback for draft in result.drafts)


async def test_offline_drafts_are_marked_as_templates() -> None:
    result = await offline_classifier().process(SPECIALIST_AD)
    assert result.drafts
    for draft in result.drafts:
        assert draft.is_fallback is True
        assert draft.mc_title.lower() in draft.text.lower()


async def test_drafts_can_be_skipped() -> None:
    result = await offline_classifier().process(SPECIALIST_AD, generate_drafts=False)
    assert result.should_split is True
    assert result.drafts == []


async def test_verification_removes_categories_and_records_the_reason() -> None:
    """Демонтажное объявление: LLM отменяет ложные категории стадии 1."""
    detector = KeywordDetector(NO_CLASSIFIER.key_phrases_path)
    before = detector.detect(PURE_DEMOLITION_AD.description, PURE_DEMOLITION_AD.mc_id)
    assert TILING in before

    client = FakeLLMClient(['{"105": false, "106": false, "108": false, "110": false}'])
    classifier = ServiceClassifier(NO_CLASSIFIER, client=client)
    result = await classifier.process(PURE_DEMOLITION_AD, generate_drafts=False)

    assert TILING not in result.detected_mc_ids
    assert result.verification[TILING] == "removed"
    assert TILING in result.rejected


async def test_batch_preserves_order() -> None:
    classifier = offline_classifier()
    ads = [
        Ad(item_id=index, mc_id=101, description=SPECIALIST_AD.description)
        for index in range(5)
    ]
    results = await classifier.process_many(ads, generate_drafts=False)
    assert [result.item_id for result in results] == list(range(5))


async def test_empty_detection_short_circuits() -> None:
    ad = Ad(item_id=9, mc_id=101, description="Продаю велосипед, торг уместен.")
    result = await offline_classifier().process(ad)
    assert result.detected_mc_ids == []
    assert result.should_split is False
    assert result.drafts == []


def test_micro_f1_counts_pairs_not_items() -> None:
    truth = [{102, 103}, {105}]
    predicted = [{102}, {105, 108}]
    score = micro_f1(truth, predicted)
    assert (score.tp, score.fp, score.fn) == (2, 1, 1)
    assert score.precision == round(2 / 3, 4)


def test_binary_accuracy_breakdown() -> None:
    score = binary_accuracy([True, True, False, False], [True, False, False, True])
    assert (score.tp, score.tn, score.fp, score.fn) == (1, 1, 1, 1)
    assert score.accuracy == 0.5


def test_per_category_scores_report_support() -> None:
    scores = per_category_f1([{102}, {102, 103}], [{102}, {102}], [102, 103])
    assert scores[102].f1 == 1.0
    assert scores[103].recall == 0.0
    assert scores[103].support == 1
