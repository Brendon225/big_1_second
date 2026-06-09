import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


OUTPUT_RE = re.compile(r"^\s*relation:\s*([^\s]+)\s*$", re.IGNORECASE)


@dataclass
class EvaluationResult:
    metrics: Dict[str, float]
    per_class: Dict[str, Dict[str, float]]
    confusion: Dict[str, Dict[str, int]]


def parse_relation_output(raw_output: str, relation_labels: Iterable[str]) -> Tuple[Optional[str], bool, bool]:
    match = OUTPUT_RE.match(raw_output or "")
    if not match:
        return None, False, False
    label = match.group(1)
    return label, True, label in set(relation_labels)


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_predictions(
    records: List[Dict[str, Any]],
    relation_labels: List[str],
    rare_relations: Optional[List[str]] = None,
    no_relation_labels: Optional[List[str]] = None,
) -> EvaluationResult:
    rare_relations = rare_relations or []
    no_relation_labels = no_relation_labels or []
    positive_labels = [label for label in relation_labels if label not in set(no_relation_labels)]

    per_class: Dict[str, Dict[str, float]] = {}
    micro_tp = micro_fp = micro_fn = 0
    for label in positive_labels:
        tp = sum(1 for item in records if item.get("gold_label") == label and item.get("pred_label") == label)
        fp = sum(1 for item in records if item.get("gold_label") != label and item.get("pred_label") == label)
        fn = sum(1 for item in records if item.get("gold_label") == label and item.get("pred_label") != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1_score(precision, recall),
            "support": float(tp + fn),
        }
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

    precision = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
    recall = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
    macro_f1 = sum(item["f1"] for item in per_class.values()) / max(len(per_class), 1)
    rare_values = [per_class[label]["f1"] for label in rare_relations if label in per_class]
    rare_macro_f1 = sum(rare_values) / max(len(rare_values), 1)
    valid_count = sum(1 for item in records if item.get("valid_output"))
    relation_valid_count = sum(1 for item in records if item.get("relation_valid"))
    overall_correct = sum(1 for item in records if item.get("gold_label") == item.get("pred_label"))
    no_relation_stats = aggregate_label_group(records, no_relation_labels)

    prototype_available = [item for item in records if item.get("prototype_topk")]
    prototype_top1 = sum(1 for item in prototype_available if item.get("prototype_top1") == item.get("gold_label"))
    prototype_top3 = sum(
        1 for item in prototype_available if item.get("gold_label") in item.get("prototype_topk", [])[:3]
    )
    agreement_items = [item for item in prototype_available if item.get("pred_label") is not None]
    agreement_count = sum(1 for item in agreement_items if item.get("pred_label") == item.get("prototype_top1"))

    metrics = {
        "micro_f1": f1_score(precision, recall),
        "macro_f1": macro_f1,
        "precision": precision,
        "recall": recall,
        "rare_relation_macro_f1": rare_macro_f1,
        "valid_output_rate": valid_count / max(len(records), 1),
        "relation_validity_rate": relation_valid_count / max(len(records), 1),
        "overall_accuracy": overall_correct / max(len(records), 1),
        "positive_support": float(sum(1 for item in records if item.get("gold_label") in positive_labels)),
        "no_relation_support": float(no_relation_stats["support"]),
        "no_relation_precision": no_relation_stats["precision"],
        "no_relation_recall": no_relation_stats["recall"],
        "no_relation_f1": no_relation_stats["f1"],
        "prototype_top1_accuracy": prototype_top1 / max(len(prototype_available), 1),
        "prototype_top3_accuracy": prototype_top3 / max(len(prototype_available), 1),
        "generation_vs_prototype_agreement": agreement_count / max(len(agreement_items), 1),
    }
    return EvaluationResult(metrics=metrics, per_class=per_class, confusion=build_confusion(records, relation_labels))


def aggregate_label_group(records: List[Dict[str, Any]], labels: List[str]) -> Dict[str, float]:
    label_set = set(labels)
    if not label_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0.0}
    tp = sum(1 for item in records if item.get("gold_label") in label_set and item.get("pred_label") in label_set)
    fp = sum(1 for item in records if item.get("gold_label") not in label_set and item.get("pred_label") in label_set)
    fn = sum(1 for item in records if item.get("gold_label") in label_set and item.get("pred_label") not in label_set)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
        "support": float(tp + fn),
    }


def build_confusion(records: List[Dict[str, Any]], relation_labels: List[str]) -> Dict[str, Dict[str, int]]:
    labels = list(relation_labels) + ["INVALID"]
    matrix = {gold: {pred: 0 for pred in labels} for gold in labels}
    for item in records:
        gold = item.get("gold_label") if item.get("gold_label") in relation_labels else "INVALID"
        pred = item.get("pred_label") if item.get("pred_label") in relation_labels else "INVALID"
        matrix[gold][pred] += 1
    return matrix


def write_per_class_csv(path: str, per_class: Dict[str, Dict[str, float]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "precision", "recall", "f1", "support"])
        writer.writeheader()
        for label, row in per_class.items():
            writer.writerow({"label": label, **row})


def write_confusion_csv(path: str, confusion: Dict[str, Dict[str, int]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    labels = list(confusion.keys())
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gold\\pred", *labels])
        for gold in labels:
            writer.writerow([gold, *[confusion[gold].get(pred, 0) for pred in labels]])
