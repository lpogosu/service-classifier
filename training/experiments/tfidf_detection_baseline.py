"""Что установил эксперимент: обучаемая детекция слабее словарной.

TF-IDF + OneVsRest логистическая регрессия сравнивается со стадией 1 на одних и
тех же данных, плюс три способа их объединить. На исходном наборе кейса
TF-IDF дал F1 0.800 против 0.929 у ключевых слов, объединения тоже не выиграли —
поэтому стадия 1 осталась детерминированной.

    python -m training.experiments.tfidf_detection_baseline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from service_classifier.config import (
    CATEGORY_TITLES,
    DEFAULT_DATASET_PATH,
    DEFAULT_KEY_PHRASES_PATH,
    TARGET_MC_IDS,
)
from service_classifier.datasets import load_dataset
from service_classifier.detection import KeywordDetector
from service_classifier.evaluation import MultiLabelScore, micro_f1, per_category_f1

CV_FOLDS = 5
TFIDF_MAX_FEATURES = 2000
TFIDF_MIN_DF = 3
TFIDF_MAX_DF = 0.9


def report(name: str, score: MultiLabelScore) -> None:
    print(
        f"  {name:<28} f1={score.f1:.4f}  p={score.precision:.4f}  "
        f"r={score.recall:.4f}  fp={score.fp}  fn={score.fn}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--keyphrases", type=Path, default=DEFAULT_KEY_PHRASES_PATH)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    detector = KeywordDetector(args.keyphrases)

    texts = [ad.description.lower() for ad in dataset]
    truth = [set(ad.target_detected_mc_ids) for ad in dataset]
    keyword_pred = [set(detector.detect(ad.description, ad.source_mc_id)) for ad in dataset]

    binarizer = MultiLabelBinarizer(classes=list(TARGET_MC_IDS))
    y = binarizer.fit_transform([sorted(labels) for labels in truth])

    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(1, 2),
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
    )
    matrix = tfidf.fit_transform(texts)
    print(f"Объявлений: {len(dataset)}, TF-IDF признаков: {matrix.shape[1]}")

    classifier = OneVsRestClassifier(
        LogisticRegression(max_iter=2000, solver="liblinear")
    )
    predicted = cross_val_predict(classifier, matrix, y, cv=CV_FOLDS)
    tfidf_pred = [
        set(np.array(binarizer.classes_)[np.where(row)[0]].tolist()) for row in predicted
    ]

    print("\nСтратегии:")
    report("keyword (стадия 1)", micro_f1(truth, keyword_pred))
    report("tfidf", micro_f1(truth, tfidf_pred))
    report(
        "union (keyword | tfidf)",
        micro_f1(truth, [k | t for k, t in zip(keyword_pred, tfidf_pred, strict=True)]),
    )
    report(
        "intersect (keyword & tfidf)",
        micro_f1(truth, [k & t for k, t in zip(keyword_pred, tfidf_pred, strict=True)]),
    )
    report(
        "fallback (tfidf при пустом keyword)",
        micro_f1(truth, [k or t for k, t in zip(keyword_pred, tfidf_pred, strict=True)]),
    )

    print("\nПокатегорийно, keyword против union:")
    keyword_scores = per_category_f1(truth, keyword_pred, list(TARGET_MC_IDS))
    union_scores = per_category_f1(
        truth, [k | t for k, t in zip(keyword_pred, tfidf_pred, strict=True)], list(TARGET_MC_IDS)
    )
    for mc_id in TARGET_MC_IDS:
        delta = union_scores[mc_id].f1 - keyword_scores[mc_id].f1
        print(
            f"  [{mc_id}] {CATEGORY_TITLES[mc_id]:<28} "
            f"keyword={keyword_scores[mc_id].f1:.3f}  union={union_scores[mc_id].f1:.3f}  "
            f"delta={delta:+.3f}"
        )


if __name__ == "__main__":
    main()
