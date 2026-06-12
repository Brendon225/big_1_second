from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from src.stage1.metrics import parse_relation_output
from src.stage1.model_outputs import ModelOutput
from src.stage1.numeric_utils import finite_or_default
from src.stage1.prompting import build_relation_prompt, build_target_text
from src.stage1.schema import RelationSchema


class HfText2TextModel:
    """Hugging Face seq2seq adapter for real T5/BioBART experiments.

    This class intentionally keeps all torch/transformers imports inside init so
    stdlib smoke tests can run on machines without the training stack installed.
    """

    def __init__(
        self,
        schema: RelationSchema,
        semantic_field: str,
        model_name_or_path: str,
        max_input_length: int = 512,
        max_output_length: int = 32,
        device: Optional[str] = None,
        model_dtype: str = "float32",
        decoding_strategy: str = "generate",
    ) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.schema = schema
        self.semantic_field = semantic_field
        self.model_name_or_path = model_name_or_path
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dtype = model_dtype
        self.decoding_strategy = self._normalize_decoding_strategy(decoding_strategy)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self._apply_model_dtype(torch, model_dtype)
        self.model.to(self.device)

    def _apply_model_dtype(self, torch_module: Any, model_dtype: str) -> None:
        normalized = str(model_dtype).lower()
        if normalized in {"auto", "as_loaded", "none"}:
            return
        if normalized in {"float32", "fp32"}:
            self.model.float()
            return
        if normalized in {"float16", "fp16"}:
            self.model.half()
            return
        if normalized in {"bfloat16", "bf16"}:
            self.model.to(dtype=torch_module.bfloat16)
            return
        raise ValueError(f"Unknown model_dtype: {model_dtype}")

    def build_examples(self, samples: List[Dict[str, str]]) -> List[Dict[str, str]]:
        return [
            {
                "id": sample["id"],
                "input_text": build_relation_prompt(sample, self.schema, self.semantic_field),
                "target_text": build_target_text(sample),
                "gold_label": sample["gold_relation"],
            }
            for sample in samples
        ]

    def forward(self, batch: List[Dict[str, str]]) -> ModelOutput:
        import torch

        self.model.eval()
        examples = self.build_examples(batch)
        inputs, labels = self._tokenize_batch(examples)
        labels[labels == self.tokenizer.pad_token_id] = -100

        with torch.no_grad():
            loss_output = self.model(**inputs, labels=labels)
            raw_outputs = self.decode_raw_outputs(inputs)

        predictions = []
        mismatches = 0
        for example, raw_output in zip(examples, raw_outputs):
            pred_label, valid_output, relation_valid = parse_relation_output(raw_output, self.schema.labels)
            if pred_label != example["gold_label"]:
                mismatches += 1
            predictions.append(
                {
                    "id": example["id"],
                    "gold_label": example["gold_label"],
                    "pred_label": pred_label,
                    "raw_output": raw_output,
                    "valid_output": valid_output,
                    "relation_valid": relation_valid,
                }
            )
        fallback_loss = mismatches / max(len(batch), 1)
        raw_generation_loss = float(loss_output.loss.detach().cpu().item()) if loss_output.loss is not None else fallback_loss
        generation_loss = finite_or_default(raw_generation_loss, fallback_loss)
        return ModelOutput(
            loss=generation_loss,
            generation_loss=generation_loss,
            alignment_loss=0.0,
            predictions=predictions,
        )

    def forward_in_batches(self, samples: List[Dict[str, str]], batch_size: int) -> ModelOutput:
        import torch

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not samples:
            return ModelOutput(loss=0.0, generation_loss=0.0, alignment_loss=0.0, predictions=[])

        total_loss = 0.0
        total_generation_loss = 0.0
        total_alignment_loss = 0.0
        total_items = 0
        predictions = []
        prototype_scores = []

        for batch in iter_batches(samples, batch_size):
            output = self.forward(batch)
            weight = len(batch)
            total_items += weight
            total_loss += output.loss * weight
            total_generation_loss += output.generation_loss * weight
            total_alignment_loss += output.alignment_loss * weight
            predictions.extend(output.predictions)
            prototype_scores.extend(output.prototype_scores)
            if str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()

        return ModelOutput(
            loss=total_loss / total_items,
            generation_loss=total_generation_loss / total_items,
            alignment_loss=total_alignment_loss / total_items,
            predictions=predictions,
            prototype_scores=prototype_scores,
        )

    def decode_raw_outputs(self, inputs: Dict[str, Any]) -> List[str]:
        if self.decoding_strategy == "generate":
            return self.generate_raw_outputs(inputs)
        if self.decoding_strategy == "label_scoring":
            return self.format_label_scoring_outputs(self.score_label_candidates(inputs))
        raise ValueError(f"Unknown decoding_strategy: {self.decoding_strategy}")

    def generate_raw_outputs(self, inputs: Dict[str, Any]) -> List[str]:
        generated = self.model.generate(
            **inputs,
            max_length=self.max_output_length,
            num_beams=1,
        )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

    def score_label_candidates(self, inputs: Dict[str, Any]) -> List[Dict[str, float]]:
        import torch

        labels = list(self.schema.labels)
        candidate_texts = [f"relation: {label}" for label in labels]
        candidate_label_ids = self.tokenizer(
            candidate_texts,
            padding=True,
            truncation=True,
            max_length=self.max_output_length,
            return_tensors="pt",
        )["input_ids"].to(self.device)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        masked_labels = candidate_label_ids.clone()
        if pad_token_id is not None:
            masked_labels[masked_labels == pad_token_id] = -100

        first_tensor = next(iter(inputs.values()))
        batch_size = int(first_tensor.shape[0])
        score_rows: List[Dict[str, float]] = []
        for row_index in range(batch_size):
            repeated_inputs = {}
            for key, value in inputs.items():
                repeat_shape = (len(labels),) + tuple(1 for _ in range(value.dim() - 1))
                repeated_inputs[key] = value[row_index : row_index + 1].repeat(*repeat_shape)
            output = self.model(**repeated_inputs, labels=masked_labels)
            logits = output.logits.float()
            token_losses = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                masked_labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).view(len(labels), -1)
            token_mask = masked_labels.ne(-100)
            sequence_losses = token_losses.sum(dim=1) / token_mask.sum(dim=1).clamp(min=1)
            score_rows.append(
                {label: float(sequence_losses[index].detach().cpu().item()) for index, label in enumerate(labels)}
            )
        return score_rows

    def format_label_scoring_outputs(self, score_rows: List[Dict[str, float]]) -> List[str]:
        raw_outputs = []
        for score_map in score_rows:
            best_label = min(score_map.items(), key=lambda item: item[1])[0]
            raw_outputs.append(f"relation: {best_label}")
        return raw_outputs

    def _normalize_decoding_strategy(self, decoding_strategy: str) -> str:
        normalized = str(decoding_strategy or "generate").lower()
        aliases = {
            "free": "generate",
            "free_generate": "generate",
            "generation": "generate",
            "constrained": "label_scoring",
            "constrained_label_scoring": "label_scoring",
            "label_score": "label_scoring",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"generate", "label_scoring"}:
            raise ValueError(f"Unknown decoding_strategy: {decoding_strategy}")
        return normalized

    def train_model(
        self,
        train_samples: List[Dict[str, str]],
        dev_samples: List[Dict[str, str]],
        epochs: int = 1,
        batch_size: int = 2,
        eval_batch_size: Optional[int] = None,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 1e-4,
        max_train_steps: Optional[int] = None,
        max_non_finite_batches: int = 10,
        gradient_clip_norm: Optional[float] = 1.0,
        output_dir: Optional[str] = None,
    ) -> List[str]:
        import torch

        logs: List[str] = []
        eval_batch_size = int(eval_batch_size or batch_size)
        gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
        max_non_finite_batches = max(1, int(max_non_finite_batches))
        logs.append(f"model_dtype={self.model_dtype}")
        logs.append(f"parameter_dtype={next(self.model.parameters()).dtype}")
        optimizer = torch.optim.AdamW(self.trainable_parameters(), lr=learning_rate)
        global_step = 0
        non_finite_batches = 0
        for epoch in range(1, epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            accumulated_batches = 0
            accumulated_loss = 0.0
            for batch in iter_batches(train_samples, batch_size):
                loss_items = self.compute_training_loss(batch)
                loss = loss_items["loss"]
                if not torch.isfinite(loss):
                    non_finite_batches += 1
                    logs.append(f"epoch={epoch} step={global_step + 1} train_loss=non_finite skipped=true")
                    optimizer.zero_grad()
                    accumulated_batches = 0
                    accumulated_loss = 0.0
                    if non_finite_batches >= max_non_finite_batches:
                        raise RuntimeError(
                            "Training stopped because non-finite loss repeated "
                            f"{non_finite_batches} times. Check model_dtype, learning_rate, and input labels."
                        )
                    continue
                non_finite_batches = 0
                loss_value = float(loss.detach().cpu().item())
                (loss / gradient_accumulation_steps).backward()
                accumulated_batches += 1
                accumulated_loss += loss_value
                if accumulated_batches >= gradient_accumulation_steps:
                    global_step = self._optimizer_step(
                        optimizer=optimizer,
                        torch_module=torch,
                        gradient_clip_norm=gradient_clip_norm,
                        global_step=global_step,
                    )
                    logs.append(
                        "epoch="
                        f"{epoch} step={global_step} "
                        f"train_loss={accumulated_loss / accumulated_batches:.6f} "
                        f"micro_batches={accumulated_batches}"
                        f"{self.format_training_loss_items(loss_items)}"
                    )
                    accumulated_batches = 0
                    accumulated_loss = 0.0
                    if max_train_steps is not None and global_step >= max_train_steps:
                        break
            if accumulated_batches and (max_train_steps is None or global_step < max_train_steps):
                global_step = self._optimizer_step(
                    optimizer=optimizer,
                    torch_module=torch,
                    gradient_clip_norm=gradient_clip_norm,
                    global_step=global_step,
                )
                logs.append(
                    "epoch="
                    f"{epoch} step={global_step} "
                    f"train_loss={accumulated_loss / accumulated_batches:.6f} "
                    f"micro_batches={accumulated_batches}"
                    f"{self.format_training_loss_items(loss_items)}"
                )
            dev_output = self.forward_in_batches(dev_samples, eval_batch_size)
            logs.append(f"epoch={epoch} dev_loss={dev_output.loss:.6f}")
            if max_train_steps is not None and global_step >= max_train_steps:
                break
        if output_dir:
            target = Path(output_dir) / "model"
            target.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(target)
            self.tokenizer.save_pretrained(target)
            self.save_extra_state(target)
            logs.append(f"saved_model={target}")
        return logs

    def trainable_parameters(self) -> Any:
        return self.model.parameters()

    def compute_training_loss(self, batch: List[Dict[str, str]]) -> Dict[str, Any]:
        examples = self.build_examples(batch)
        inputs, labels = self._tokenize_batch(examples)
        labels[labels == self.tokenizer.pad_token_id] = -100
        output = self.model(**inputs, labels=labels)
        return {"loss": output.loss, "generation_loss": output.loss, "alignment_loss": None}

    def format_training_loss_items(self, loss_items: Dict[str, Any]) -> str:
        alignment_loss = loss_items.get("alignment_loss")
        if alignment_loss is None:
            return ""
        generation_loss = loss_items["generation_loss"]
        return (
            f" generation_loss={float(generation_loss.detach().cpu().item()):.6f}"
            f" alignment_loss={float(alignment_loss.detach().cpu().item()):.6f}"
        )

    def save_extra_state(self, target: Path) -> None:
        return None

    def _optimizer_step(
        self,
        optimizer: Any,
        torch_module: Any,
        gradient_clip_norm: Optional[float],
        global_step: int,
    ) -> int:
        if gradient_clip_norm is not None:
            torch_module.nn.utils.clip_grad_norm_(self.model.parameters(), float(gradient_clip_norm))
        optimizer.step()
        optimizer.zero_grad()
        return global_step + 1

    def _tokenize_batch(self, examples: List[Dict[str, str]]) -> tuple[Dict[str, Any], Any]:
        inputs = self.tokenizer(
            [item["input_text"] for item in examples],
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
            return_tensors="pt",
        )
        labels = self.tokenizer(
            [item["target_text"] for item in examples],
            padding=True,
            truncation=True,
            max_length=self.max_output_length,
            return_tensors="pt",
        )["input_ids"]
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        labels = labels.to(self.device)
        return inputs, labels


def iter_batches(samples: List[Dict[str, str]], batch_size: int) -> Iterator[List[Dict[str, str]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(samples), batch_size):
        yield samples[start : start + batch_size]
