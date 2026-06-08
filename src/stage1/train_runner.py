from pathlib import Path
from typing import Any, Dict

from src.stage1.backend_factory import build_stage1_model
from src.stage1.data_io import read_json_config, read_jsonl, write_json, write_jsonl
from src.stage1.metrics import evaluate_predictions, write_confusion_csv, write_per_class_csv
from src.stage1.model_outputs import ModelOutput
from src.stage1.runner import write_error_cases, write_prototype_analysis
from src.stage1.schema import load_relation_schema


def run_training(config_path: str) -> Dict[str, Any]:
    config = read_json_config(config_path)
    schema = load_relation_schema(config["schema_file"])
    train_samples = limit_samples(read_jsonl(config["train_file"]), config.get("max_train_samples"))
    dev_samples = limit_samples(read_jsonl(config["dev_file"]), config.get("max_dev_samples"))
    test_samples = limit_samples(read_jsonl(config["test_file"]), config.get("max_test_samples"))
    validate_run_size(config_path, config, train_samples, dev_samples, test_samples)

    model = build_stage1_model(
        method=config["method"],
        schema=schema,
        semantic_field=config["semantic_field"],
        backend=config.get("backend", "mock"),
        alignment_lambda=config.get("alignment_lambda", 0.1),
        temperature_tau=config.get("temperature_tau", 0.1),
        prototype_type=config.get("prototype_type", "learnable"),
        model_name_or_path=config.get("model_name_or_path", config.get("model", "t5-small")),
        max_input_length=config.get("max_input_length", 512),
        max_output_length=config.get("max_output_length", 32),
        device=config.get("device"),
    )

    train_log = [
        f"experiment_id={config['experiment_id']}",
        f"backend={config.get('backend', 'mock')}",
        f"train_samples={len(train_samples)}",
        f"dev_samples={len(dev_samples)}",
        f"test_samples={len(test_samples)}",
        f"epochs={config.get('epochs', 1)}",
    ]

    if hasattr(model, "train_model"):
        train_log.extend(
            model.train_model(
                train_samples=train_samples,
                dev_samples=dev_samples,
                epochs=int(config.get("epochs", 1)),
                batch_size=int(config.get("batch_size", 2)),
                eval_batch_size=int(config.get("eval_batch_size", config.get("batch_size", 2))),
                learning_rate=float(config.get("learning_rate", 1e-4)),
                max_train_steps=config.get("max_train_steps"),
                gradient_clip_norm=config.get("gradient_clip_norm", 1.0),
                output_dir=config.get("output_dir"),
            )
        )
    else:
        train_log.append("note=backend has no train_model method; executed deterministic smoke/evaluation pass.")

    eval_batch_size = int(config.get("eval_batch_size", config.get("batch_size", 2)))
    dev_output = forward_in_batches(model, dev_samples, eval_batch_size)
    test_output = forward_in_batches(model, test_samples, eval_batch_size)
    evaluation = evaluate_predictions(
        test_output.predictions,
        relation_labels=schema.labels,
        rare_relations=schema.rare_relations,
        no_relation_labels=schema.no_relation_labels,
    )
    metrics = {
        "experiment_id": config["experiment_id"],
        "dataset": config["dataset"],
        "model": config["model"],
        "method": config["method"],
        "backend": config.get("backend", "mock"),
        "train_samples": len(train_samples),
        "dev_samples": len(dev_samples),
        "test_samples": len(test_samples),
        "dev_loss": dev_output.loss,
        "loss": test_output.loss,
        "generation_loss": test_output.generation_loss,
        "alignment_loss": test_output.alignment_loss,
        **evaluation.metrics,
    }

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(output_dir / "run_config.yaml").write_text(
        Path(config_path).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    Path(output_dir / "train_log.txt").write_text("\n".join(train_log) + "\n", encoding="utf-8")
    write_json(str(output_dir / "metrics.json"), metrics)
    write_jsonl(str(output_dir / "predictions.jsonl"), test_output.predictions)
    write_per_class_csv(str(output_dir / "per_class_metrics.csv"), evaluation.per_class)
    write_confusion_csv(str(output_dir / "confusion_matrix.csv"), evaluation.confusion)
    write_error_cases(str(output_dir / "error_cases.md"), config, test_output.predictions)
    if test_output.prototype_scores:
        write_jsonl(str(output_dir / "prototype_scores.jsonl"), test_output.prototype_scores)
        write_prototype_analysis(str(output_dir / "prototype_analysis.csv"), test_output.prototype_scores)
    return metrics


def limit_samples(samples: list[dict[str, str]], max_samples: Any) -> list[dict[str, str]]:
    if max_samples is None:
        return samples
    limit = int(max_samples)
    if limit < 0:
        raise ValueError("max sample limits must be non-negative")
    return samples[:limit]


def forward_in_batches(model: Any, samples: list[dict[str, str]], batch_size: int) -> ModelOutput:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not samples:
        return ModelOutput(loss=0.0, generation_loss=0.0, alignment_loss=0.0, predictions=[])
    if hasattr(model, "forward_in_batches"):
        return model.forward_in_batches(samples, batch_size)

    total_loss = 0.0
    total_generation_loss = 0.0
    total_alignment_loss = 0.0
    total_items = 0
    predictions = []
    prototype_scores = []

    for batch in iter_batches(samples, batch_size):
        output = model.forward(batch)
        weight = len(batch)
        total_items += weight
        total_loss += output.loss * weight
        total_generation_loss += output.generation_loss * weight
        total_alignment_loss += output.alignment_loss * weight
        predictions.extend(output.predictions)
        prototype_scores.extend(output.prototype_scores)

    return ModelOutput(
        loss=total_loss / total_items,
        generation_loss=total_generation_loss / total_items,
        alignment_loss=total_alignment_loss / total_items,
        predictions=predictions,
        prototype_scores=prototype_scores,
    )


def iter_batches(samples: list[dict[str, str]], batch_size: int):
    for start in range(0, len(samples), batch_size):
        yield samples[start : start + batch_size]


def validate_run_size(
    config_path: str,
    config: Dict[str, Any],
    train_samples: list[dict[str, str]],
    dev_samples: list[dict[str, str]],
    test_samples: list[dict[str, str]],
) -> None:
    if config.get("allow_small_full_run"):
        return
    marker = f"{config_path} {config.get('experiment_id', '')}".lower()
    if "full" not in marker:
        return
    limited_keys = [
        key
        for key in ["max_train_steps", "max_train_samples", "max_dev_samples", "max_test_samples"]
        if key in config
    ]
    split_sizes = {
        "train": len(train_samples),
        "dev": len(dev_samples),
        "test": len(test_samples),
    }
    too_small = {split: size for split, size in split_sizes.items() if size < 100}
    if limited_keys or too_small:
        raise ValueError(
            "This looks like a full run, but it is still using smoke limits or smoke-sized data. "
            f"limited_keys={limited_keys}, split_sizes={split_sizes}. "
            "For a real ChemProt full run, reconvert data without --max-samples-per-split and remove "
            "max_train_steps/max_*_samples from the config. "
            "Use allow_small_full_run=true only for intentional debugging."
        )
