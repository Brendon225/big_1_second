import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from src.stage1.data_io import read_json_config, read_jsonl
from src.stage1.metrics import evaluate_predictions
from src.stage1.schema import load_relation_schema


SUMMARY_FIELDS = [
    "experiment_id",
    "dataset",
    "model",
    "method",
    "backend",
    "train_samples",
    "dev_samples",
    "test_samples",
    "micro_f1",
    "macro_f1",
    "precision",
    "recall",
    "rare_relation_macro_f1",
    "overall_accuracy",
    "no_relation_f1",
    "valid_output_rate",
    "relation_validity_rate",
    "prototype_top1_accuracy",
    "prototype_top3_accuracy",
    "generation_vs_prototype_agreement",
    "loss",
    "dev_loss",
    "metrics_path",
]


def summarize_metrics(root_dir: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    root = Path(root_dir)
    for metrics_path in sorted(root.rglob("metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        backfill_missing_diagnostics(payload, metrics_path.parent)
        row = {field: payload.get(field, "") for field in SUMMARY_FIELDS}
        row["metrics_path"] = str(metrics_path)
        rows.append(row)
    return sorted(rows, key=lambda item: str(item.get("experiment_id", "")))


def backfill_missing_diagnostics(payload: Dict[str, Any], run_dir: Path) -> None:
    missing_fields = [
        field
        for field in ["overall_accuracy", "no_relation_precision", "no_relation_recall", "no_relation_f1"]
        if field not in payload
    ]
    if not missing_fields:
        return
    predictions_path = run_dir / "predictions.jsonl"
    config_path = run_dir / "run_config.yaml"
    if not predictions_path.exists() or not config_path.exists():
        return
    try:
        config = read_json_config(str(config_path))
        schema = load_relation_schema(config["schema_file"])
        predictions = read_jsonl(str(predictions_path))
        evaluation = evaluate_predictions(
            predictions,
            relation_labels=schema.labels,
            rare_relations=schema.rare_relations,
            no_relation_labels=schema.no_relation_labels,
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return
    for field in missing_fields:
        payload[field] = evaluation.metrics.get(field, "")
    for field in ["positive_support", "no_relation_support"]:
        payload.setdefault(field, evaluation.metrics.get(field, ""))


def write_summary_csv(rows: List[Dict[str, Any]], output_path: str) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})
