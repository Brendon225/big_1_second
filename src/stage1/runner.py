import csv
import shutil
from pathlib import Path
from typing import Any, Dict, List

from src.stage1.data_io import read_json_config, read_jsonl, write_json, write_jsonl
from src.stage1.metrics import evaluate_predictions, write_confusion_csv, write_per_class_csv
from src.stage1.models.rsg_biore import RsgBioREModel
from src.stage1.models.text2text_baseline import Text2TextBaseline
from src.stage1.schema import load_relation_schema


def run_experiment(config_path: str) -> Dict[str, Any]:
    config = read_json_config(config_path)
    schema = load_relation_schema(config["schema_file"])
    batch = read_jsonl(config["test_file"])

    method = config["method"]
    if method.startswith("R"):
        model = RsgBioREModel(
            schema=schema,
            semantic_field=config["semantic_field"],
            alignment_lambda=float(config.get("alignment_lambda", 0.1)),
            temperature_tau=float(config.get("temperature_tau", 0.1)),
            prototype_type=config.get("prototype_type", "learnable"),
            backend=config.get("backend", "mock"),
        )
    else:
        model = Text2TextBaseline(
            schema=schema,
            semantic_field=config["semantic_field"],
            backend=config.get("backend", "mock"),
        )

    model_output = model.forward(batch)
    evaluation = evaluate_predictions(
        model_output.predictions,
        relation_labels=schema.labels,
        rare_relations=schema.rare_relations,
        no_relation_labels=schema.no_relation_labels,
    )

    metrics = {
        "experiment_id": config["experiment_id"],
        "dataset": config["dataset"],
        "model": config["model"],
        "method": config["method"],
        "loss": model_output.loss,
        "generation_loss": model_output.generation_loss,
        "alignment_loss": model_output.alignment_loss,
        **evaluation.metrics,
    }

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, output_dir / "run_config.yaml")
    write_json(str(output_dir / "metrics.json"), metrics)
    write_jsonl(str(output_dir / "predictions.jsonl"), model_output.predictions)
    write_per_class_csv(str(output_dir / "per_class_metrics.csv"), evaluation.per_class)
    write_confusion_csv(str(output_dir / "confusion_matrix.csv"), evaluation.confusion)
    write_error_cases(str(output_dir / "error_cases.md"), config, model_output.predictions)

    if model_output.prototype_scores:
        write_jsonl(str(output_dir / "prototype_scores.jsonl"), model_output.prototype_scores)
        write_prototype_analysis(str(output_dir / "prototype_analysis.csv"), model_output.prototype_scores)

    return metrics


def write_error_cases(path: str, config: Dict[str, Any], predictions: List[Dict[str, Any]]) -> None:
    errors = [item for item in predictions if item.get("gold_label") != item.get("pred_label")]
    invalid_generation = sum(1 for item in predictions if not item.get("valid_output"))
    has_raw_generation = any("generation_valid_output" in item for item in predictions)
    invalid_raw_generation = sum(1 for item in predictions if item.get("generation_valid_output") is False)
    prototype_fusion_fallback = sum(
        1
        for item in predictions
        if item.get("prototype_fusion_applied") and item.get("generation_relation_valid") is False
    )
    lines = [
        "# Error Cases",
        "",
        "## Summary",
        "",
        f"- Dataset: {config['dataset']}",
        f"- Model: {config['model']}",
        f"- Method: {config['method']}",
        f"- Seed: {config.get('seed', '')}",
        "",
        "## Error Type Counts",
        "",
        "| Error Type | Count | Description |",
        "|---|---:|---|",
        f"| label_confusion | {len(errors)} | Predicted label differs from gold label |",
        f"| invalid_generation | {invalid_generation} | Final output could not be parsed or mapped to schema |",
    ]
    if has_raw_generation:
        lines.extend(
            [
                f"| invalid_raw_generation | {invalid_raw_generation} | Raw decoder output before prototype fusion was invalid |",
                f"| prototype_fusion_fallback | {prototype_fusion_fallback} | Prototype fusion replaced an invalid raw decoder label |",
            ]
        )
    lines.extend(["", "## Representative Cases", ""])
    for index, item in enumerate(errors[:5], start=1):
        lines.extend(
            [
                f"### Case {index}",
                "",
                f"- id: {item.get('id')}",
                f"- gold: {item.get('gold_label')}",
                f"- prediction: {item.get('pred_label')}",
                f"- raw output: {item.get('raw_output')}",
                "- analysis: inspect relation semantic text, instance wording, and prototype scores.",
                "",
            ]
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")


def write_prototype_analysis(path: str, prototype_scores: List[Dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "gold_label", "prototype_top1", "prototype_top3", "gold_score", "top1_score"],
        )
        writer.writeheader()
        for row in prototype_scores:
            scores = row["scores"]
            top1 = row["prototype_top1"]
            gold = row["gold_label"]
            writer.writerow(
                {
                    "id": row["id"],
                    "gold_label": gold,
                    "prototype_top1": top1,
                    "prototype_top3": " ".join(row["prototype_top3"]),
                    "gold_score": scores.get(gold, 0.0),
                    "top1_score": scores.get(top1, 0.0),
                }
            )
