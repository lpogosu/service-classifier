"""Генератор датасета: воспроизводимость, схема и согласованность разметки."""

from __future__ import annotations

from pathlib import Path

from data.synthetic import DEFAULT_SEED, generate
from service_classifier.config import SOURCE_MC_ID, TARGET_MC_IDS
from service_classifier.datasets import CSV_FIELDNAMES, load_dataset, write_dataset


def test_generation_is_reproducible() -> None:
    first = generate(rows=60, seed=DEFAULT_SEED)
    second = generate(rows=60, seed=DEFAULT_SEED)
    assert [ad.description for ad in first] == [ad.description for ad in second]


def test_different_seeds_give_different_data() -> None:
    first = generate(rows=60, seed=1)
    second = generate(rows=60, seed=2)
    assert [ad.description for ad in first] != [ad.description for ad in second]


def test_labels_are_internally_consistent() -> None:
    for ad in generate(rows=200, seed=DEFAULT_SEED):
        assert ad.source_mc_id == SOURCE_MC_ID
        assert ad.description.strip()
        assert set(ad.target_detected_mc_ids) <= set(TARGET_MC_IDS)
        assert ad.split in {"train", "test"}
        if ad.should_split:
            assert ad.target_split_mc_ids == ad.target_detected_mc_ids
        else:
            assert ad.target_split_mc_ids == []


def test_both_classes_are_present() -> None:
    ads = generate(rows=200, seed=DEFAULT_SEED)
    positives = sum(1 for ad in ads if ad.should_split)
    assert 0 < positives < len(ads)


def test_demolition_ads_mention_other_categories_only_as_objects() -> None:
    """Ради этих строк и существует стадия 1.5."""
    ads = generate(rows=300, seed=DEFAULT_SEED)
    demolition = [ad for ad in ads if ad.case_type == "demolition"]
    assert demolition
    for ad in demolition:
        assert ad.target_detected_mc_ids == [111]
        assert "демонтаж" in ad.description.lower()


def test_csv_roundtrip_preserves_records(tmp_path: Path) -> None:
    ads = generate(rows=40, seed=DEFAULT_SEED)
    path = tmp_path / "dataset.csv"
    write_dataset(path, ads)

    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header == ";".join(CSV_FIELDNAMES)
    assert load_dataset(path) == ads
