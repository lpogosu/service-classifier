"""Метрики качества: multi-label F1 детекции и accuracy решения о разделении."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MultiLabelScore:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    support: int = 0


@dataclass
class BinaryScore:
    accuracy: float
    total: int
    tp: int
    tn: int
    fp: int
    fn: int


@dataclass
class EvaluationReport:
    detection: MultiLabelScore
    should_split: BinaryScore
    split: MultiLabelScore
    per_category_detection: dict[int, MultiLabelScore] = field(default_factory=dict)


def _score(tp: int, fp: int, fn: int, support: int = 0) -> MultiLabelScore:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return MultiLabelScore(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        tp=tp,
        fp=fp,
        fn=fn,
        support=support,
    )


def micro_f1(y_true: list[set[int]], y_pred: list[set[int]]) -> MultiLabelScore:
    """Micro-F1: каждая пара (объявление, категория) считается независимо."""
    tp = sum(len(t & p) for t, p in zip(y_true, y_pred, strict=True))
    fp = sum(len(p - t) for t, p in zip(y_true, y_pred, strict=True))
    fn = sum(len(t - p) for t, p in zip(y_true, y_pred, strict=True))
    return _score(tp, fp, fn, support=sum(len(t) for t in y_true))


def binary_accuracy(y_true: list[bool], y_pred: list[bool]) -> BinaryScore:
    pairs = list(zip(y_true, y_pred, strict=True))
    tp = sum(1 for t, p in pairs if t and p)
    tn = sum(1 for t, p in pairs if not t and not p)
    fp = sum(1 for t, p in pairs if not t and p)
    fn = sum(1 for t, p in pairs if t and not p)
    total = len(pairs)
    return BinaryScore(
        accuracy=round((tp + tn) / total, 4) if total else 0.0,
        total=total,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
    )


def per_category_f1(
    y_true: list[set[int]],
    y_pred: list[set[int]],
    mc_ids: list[int] | None = None,
) -> dict[int, MultiLabelScore]:
    if mc_ids is None:
        mc_ids = sorted(set().union(*y_true, *y_pred)) if y_true or y_pred else []

    scores: dict[int, MultiLabelScore] = {}
    for mc_id in mc_ids:
        tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if mc_id in t and mc_id in p)
        fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if mc_id not in t and mc_id in p)
        fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if mc_id in t and mc_id not in p)
        support = sum(1 for t in y_true if mc_id in t)
        scores[mc_id] = _score(tp, fp, fn, support)
    return scores


def evaluate(
    true_detected: list[set[int]],
    pred_detected: list[set[int]],
    true_should_split: list[bool],
    pred_should_split: list[bool],
    true_split: list[set[int]],
    pred_split: list[set[int]],
) -> EvaluationReport:
    return EvaluationReport(
        detection=micro_f1(true_detected, pred_detected),
        should_split=binary_accuracy(true_should_split, pred_should_split),
        split=micro_f1(true_split, pred_split),
        per_category_detection=per_category_f1(true_detected, pred_detected),
    )


def format_report(report: EvaluationReport, category_titles: dict[int, str]) -> str:
    """Отчёт для терминала."""
    lines = ["", "=" * 62, "  EVALUATION REPORT", "=" * 62]

    d = report.detection
    lines += [
        "",
        "Detection (targetDetectedMcIds)",
        f"  precision={d.precision:.4f}  recall={d.recall:.4f}  f1={d.f1:.4f}",
        f"  tp={d.tp}  fp={d.fp}  fn={d.fn}",
    ]

    s = report.should_split
    lines += [
        "",
        "shouldSplit",
        f"  accuracy={s.accuracy:.4f} ({s.tp + s.tn}/{s.total})",
        f"  tp={s.tp}  tn={s.tn}  fp={s.fp}  fn={s.fn}",
    ]

    sp = report.split
    lines += [
        "",
        "Split (targetSplitMcIds)",
        f"  precision={sp.precision:.4f}  recall={sp.recall:.4f}  f1={sp.f1:.4f}",
        f"  tp={sp.tp}  fp={sp.fp}  fn={sp.fn}",
    ]

    lines += ["", "Detection F1 по категориям"]
    for mc_id, score in sorted(report.per_category_detection.items()):
        title = category_titles.get(mc_id, str(mc_id))
        lines.append(
            f"  [{mc_id}] {title:<28} f1={score.f1:.3f}  "
            f"p={score.precision:.3f}  r={score.recall:.3f}  n={score.support}"
        )

    lines += ["=" * 62, ""]
    return "\n".join(lines)
