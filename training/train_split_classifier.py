"""Обучение классификатора стадии 2 (решение о разделении).

Схема, которой получены цифры README: 44 ручных признака + TF-IDF по символьным
и словесным n-граммам, сравнение четырёх семейств моделей по 5-fold CV, затем
подбор порога вероятности по CV-предсказаниям. Порог сохраняется вместе с моделью,
иначе он подбирался бы на тех же данных, на которых модель обучалась.

    python -m training.train_split_classifier --dataset data/synthetic_dataset.csv
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from service_classifier.config import DEFAULT_DATASET_PATH, DEFAULT_SPLIT_MODEL_PATH
from service_classifier.datasets import load_dataset
from service_classifier.detection import KeywordDetector
from service_classifier.split import extract_features

RANDOM_STATE = 42
CV_FOLDS = 5
TFIDF_MAX_FEATURES = 3000
TFIDF_MIN_DF = 3
THRESHOLD_GRID = tuple(round(0.05 * step, 2) for step in range(2, 17))
TOP_FEATURES_SHOWN = 12


def build_candidates() -> dict[str, Any]:
    """Модели-кандидаты. Линейные — в пайплайне со скейлером, деревья — без."""
    return {
        "logreg_balanced": Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "logreg": Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        ),
        "gbm": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, random_state=RANDOM_STATE
        ),
        "gbm_regularized": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=20, random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def build_matrix(
    descriptions: list[str],
    detected: list[list[int]],
    tfidf: TfidfVectorizer,
    *,
    fit: bool,
) -> tuple[NDArray[np.float64], list[str]]:
    """Ручные признаки, склеенные с TF-IDF-представлением текста."""
    feature_dicts = [
        extract_features(text, ids) for text, ids in zip(descriptions, detected, strict=True)
    ]
    feature_names = sorted(feature_dicts[0])
    hand = np.array([[row[name] for name in feature_names] for row in feature_dicts])

    lowered = [text.lower() for text in descriptions]
    text_matrix = tfidf.fit_transform(lowered) if fit else tfidf.transform(lowered)
    combined: NDArray[np.float64] = hstack([csr_matrix(hand), text_matrix]).toarray()
    return combined, feature_names


def pick_threshold(y_true: NDArray[np.int_], proba: NDArray[np.float64]) -> tuple[float, float]:
    """Порог, максимизирующий accuracy на CV-предсказаниях."""
    best_threshold, best_accuracy = 0.5, 0.0
    for threshold in THRESHOLD_GRID:
        accuracy = float(np.mean((proba >= threshold).astype(int) == y_true))
        if accuracy > best_accuracy:
            best_threshold, best_accuracy = threshold, accuracy
    return best_threshold, best_accuracy


def f1(y_true: NDArray[np.int_], y_pred: NDArray[np.int_]) -> float:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    if not tp:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_SPLIT_MODEL_PATH)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    print(f"Объявлений: {len(dataset)}")

    detector = KeywordDetector(
        Path(__file__).resolve().parent.parent / "data" / "key_phrases.csv"
    )
    descriptions = [ad.description for ad in dataset]
    detected = [detector.detect(ad.description, ad.source_mc_id) for ad in dataset]
    y = np.array([int(ad.should_split) for ad in dataset])
    print(f"Баланс классов: shouldSplit=true у {y.sum()} из {len(y)}")

    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES, ngram_range=(1, 2), min_df=TFIDF_MIN_DF
    )
    matrix, feature_names = build_matrix(descriptions, detected, tfidf, fit=True)
    print(f"Признаков: {len(feature_names)} ручных + {matrix.shape[1] - len(feature_names)} TF-IDF")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    best_name, best_model, best_accuracy, best_proba = "", None, 0.0, None

    print(f"\n{CV_FOLDS}-fold кросс-валидация:")
    for name, model in build_candidates().items():
        proba = cross_val_predict(model, matrix, y, cv=cv, method="predict_proba")[:, 1]
        threshold, accuracy = pick_threshold(y, proba)
        score = f1(y, (proba >= threshold).astype(int))
        print(f"  {name:<18} acc={accuracy:.3f}  f1={score:.3f}  threshold={threshold:.2f}")
        if accuracy > best_accuracy:
            best_name, best_model, best_accuracy, best_proba = name, model, accuracy, proba

    if best_model is None or best_proba is None:
        raise SystemExit("Не удалось обучить ни одной модели")

    threshold, _ = pick_threshold(y, best_proba)
    print(
        f"\nЛучшая модель: {best_name} "
        f"(CV accuracy={best_accuracy:.3f}, threshold={threshold:.2f})"
    )

    best_model.fit(matrix, y)
    importances = getattr(best_model, "feature_importances_", None)
    if importances is not None:
        print("\nВклад ручных признаков:")
        hand_importances = importances[: len(feature_names)]
        for index in np.argsort(hand_importances)[::-1][:TOP_FEATURES_SHOWN]:
            print(f"  {feature_names[index]:<24} {hand_importances[index]:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as handle:
        pickle.dump(
            {
                "classifier": best_model,
                "tfidf": tfidf,
                "feature_names": feature_names,
                "threshold": threshold,
                "cv_accuracy": round(best_accuracy, 4),
                "model_name": best_name,
            },
            handle,
        )
    print(f"\nМодель сохранена: {args.output}")
    print(
        "Обратите внимание: на синтетическом наборе архетипы объявлений "
        "разделимы по построению, поэтому accuracy там близка к единице и "
        "ничего не говорит о качестве. Цифры README получены на данных кейса."
    )


if __name__ == "__main__":
    main()
