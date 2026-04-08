"""Стадия 1: лемматизация, опечатки, гомоглифы и пороги совпадений."""

from __future__ import annotations

import pytest

from service_classifier.config import DEFAULT_KEY_PHRASES_PATH, SOURCE_MC_ID
from service_classifier.detection import KeywordDetector, normalize_text

ELECTRICS = 103
TILING = 105
WALLPAPER = 106
PLASTER = 108
DEMOLITION = 111

#: Обе формулировки называют одну и ту же услугу — стадия 1 обязана найти её
#: в обеих. Различает их стадия 2, а не детекция (см. test_split.py).
INCLUDED_IN_PACKAGE = "Делаем ремонт квартир под ключ, включая электрику и сантехнику."
SOLD_SEPARATELY = "Электрику делаем отдельно, сантехнику отдельно. Цена от 1200 руб."


@pytest.mark.parametrize("text", [INCLUDED_IN_PACKAGE, SOLD_SEPARATELY])
def test_electrics_detected_regardless_of_packaging(
    detector: KeywordDetector, text: str
) -> None:
    assert ELECTRICS in detector.detect(text, SOURCE_MC_ID)


def test_lemmatisation_collapses_oblique_cases(detector: KeywordDetector) -> None:
    """«плитку», «плитки», «плиткой» — одна лемма, одно совпадение по словарю."""
    forms = ["плитку", "плитки", "плиткой", "плитке"]
    assert {detector.lemmatize(form) for form in forms} == {"плитка"}


def test_homonymous_forms_split_into_two_lemmas(detector: KeywordDetector) -> None:
    """«электрику» — это и услуга, и профессия; pymorphy3 выбирает профессию.

    Слово «электрика» в разных падежах распадается на две леммы, «электрика» и
    «электрик». Ровно поэтому в списке якорей есть обе формы: полагаться на одну
    лемму для этой категории нельзя.
    """
    assert detector.lemmatize("электрикой") == "электрика"
    assert detector.lemmatize("электрику") == "электрик"
    assert {"электрик", "электрика"} <= set(KeywordDetector.ANCHOR_WORDS[ELECTRICS])


def test_oblique_forms_reach_the_category(detector: KeywordDetector) -> None:
    for form in ["электрику", "электрике", "электрикой", "электрики"]:
        text = f"Монтаж проводки, занимаюсь {form} в квартирах."
        assert ELECTRICS in detector.detect(text, SOURCE_MC_ID), form


def test_typo_is_absorbed_by_fuzzy_matching(detector: KeywordDetector) -> None:
    correct = "Штукатурка стен по маякам, выравнивание стен под покраску."
    typo = "Штукатрука стен по маякам, выравнивание стен под покраску."
    assert PLASTER in detector.detect(correct, SOURCE_MC_ID)
    assert PLASTER in detector.detect(typo, SOURCE_MC_ID)


def test_normalize_text_maps_latin_homoglyphs_to_cyrillic() -> None:
    assert normalize_text("oбoи") == "обои"
    assert normalize_text("плёнka") == "пленка"
    assert normalize_text("плитка 🔨 ремонт") == "плитка   ремонт"


def test_homoglyph_substitution_still_detected(detector: KeywordDetector) -> None:
    clean = "Поклейка обоев, оклейка обоев с подготовкой стен."
    disguised = clean.replace("о", "o").replace("е", "e")
    assert clean != disguised
    assert WALLPAPER in detector.detect(disguised, SOURCE_MC_ID)


def test_emoji_do_not_break_matching(detector: KeywordDetector) -> None:
    text = "🔨 Укладка плитки и керамогранита ✔ облицовка плиткой с затиркой швов"
    assert TILING in detector.detect(text, SOURCE_MC_ID)


def test_source_category_is_excluded(detector: KeywordDetector) -> None:
    text = "Электромонтаж: щиток, розетки, освещение. Замена электропроводки."
    assert ELECTRICS not in detector.detect(text, source_mc_id=ELECTRICS)
    assert ELECTRICS in detector.detect(text, source_mc_id=SOURCE_MC_ID)


def test_container_category_never_returned(detector: KeywordDetector) -> None:
    text = "Комплексный ремонт квартир под ключ, черновая и чистовая отделка."
    assert SOURCE_MC_ID not in detector.detect(text, SOURCE_MC_ID)


def test_unrelated_text_detects_nothing(detector: KeywordDetector) -> None:
    assert detector.detect("Продаю велосипед, торг уместен.", SOURCE_MC_ID) == []


def test_demolition_suppressed_inside_full_renovation(detector: KeywordDetector) -> None:
    """Демонтаж как подготовка к другим работам — не самостоятельная услуга."""
    text = (
        "Ремонт под ключ. Демонтаж старой плитки, штукатурка стен по маякам, "
        "укладка плитки, поклейка обоев, электромонтаж, монтаж гипсокартона, "
        "укладка ламината."
    )
    detected = detector.detect(text, SOURCE_MC_ID)
    assert len(detected) > 3
    assert DEMOLITION not in detected


def test_standalone_demolition_survives_the_context_rule(
    detector: KeywordDetector,
) -> None:
    text = (
        "Демонтажные работы любой сложности: снос перегородок, демонтаж стен. "
        "Также штукатурка, укладка плитки, поклейка обоев, электромонтаж, "
        "монтаж гипсокартона."
    )
    detected = detector.detect(text, SOURCE_MC_ID)
    assert len(detected) > 3
    assert DEMOLITION in detected


def test_details_and_ids_agree(detector: KeywordDetector) -> None:
    text = "Штукатурка стен по маякам и укладка плитки в ванной."
    details = detector.detect_with_details(text, SOURCE_MC_ID)
    assert sorted(details) == detector.detect(text, SOURCE_MC_ID)
    assert all(phrases for phrases in details.values())


def test_min_matches_blocks_lone_ambiguous_hit(detector: KeywordDetector) -> None:
    """«Выравниваю стены» — общая фраза; порог в два совпадения её отсекает."""
    assert KeywordDetector.PER_CATEGORY_MIN_MATCHES[WALLPAPER] == 2
    text = "Ремонтирую квартиры. Стены выравниваю аккуратно."
    assert WALLPAPER not in detector.detect(text, SOURCE_MC_ID)


def test_anchor_word_bypasses_min_matches(detector: KeywordDetector) -> None:
    """Одно якорное слово доказывает категорию даже при пороге в два совпадения."""
    assert KeywordDetector.PER_CATEGORY_MIN_MATCHES[ELECTRICS] == 2
    assert ELECTRICS in detector.detect("Электрика. Больше ничем не занимаюсь.", SOURCE_MC_ID)


def test_phrases_shorter_than_min_length_are_skipped() -> None:
    strict = KeywordDetector(DEFAULT_KEY_PHRASES_PATH, min_phrase_len=8)
    lemmatized = strict.lemmatize("Укладка плитки и керамогранита.")
    assert strict._text_contains_phrase(lemmatized, "плитка", "плитка") is False
    assert strict._text_contains_phrase(lemmatized, "керамогранит", "керамогранит") is True


def test_single_word_fuzzy_threshold_is_stricter_than_multi_word(
    detector: KeywordDetector,
) -> None:
    """Однословные фразы дают больше всего FP, поэтому у них порог выше."""
    lemmatized = detector.lemmatize("Работаю с ламинатом и линолеумом.")
    assert detector._text_contains_phrase(lemmatized, "ламинат", "ламинат") is True
    # «лаборант» отличается от «ламинат» сильнее, чем допускает порог 85.
    assert detector._text_contains_phrase(lemmatized, "лаборант", "лаборант") is False
