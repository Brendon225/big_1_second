from pathlib import Path
from typing import Any, Dict, List, Optional

from src.stage1.hf_text2text_backend import HfText2TextModel
from src.stage1.metrics import parse_relation_output
from src.stage1.model_outputs import ModelOutput
from src.stage1.numeric_utils import finite_or_default
from src.stage1.prompting import build_target_text
from src.stage1.schema import RelationSchema


class HfRsgBioREModel(HfText2TextModel):
    """HF encoder-decoder RSG-BioRE backend with prototype alignment loss."""

    def __init__(
        self,
        schema: RelationSchema,
        semantic_field: str,
        model_name_or_path: str,
        max_input_length: int = 512,
        max_output_length: int = 32,
        device: Optional[str] = None,
        model_dtype: str = "float32",
        alignment_lambda: float = 0.1,
        temperature_tau: float = 0.1,
        prototype_type: str = "learnable",
        prototype_semantic_field: Optional[str] = None,
    ) -> None:
        import torch

        super().__init__(
            schema=schema,
            semantic_field=semantic_field,
            model_name_or_path=model_name_or_path,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            device=device,
            model_dtype=model_dtype,
        )
        if temperature_tau <= 0:
            raise ValueError("temperature_tau must be positive")
        self.alignment_lambda = float(alignment_lambda)
        self.temperature_tau = float(temperature_tau)
        self.prototype_type = prototype_type
        self.prototype_semantic_field = prototype_semantic_field or semantic_field
        self.labels = list(schema.labels)
        self.label_to_id = {label: index for index, label in enumerate(self.labels)}
        hidden_size = int(getattr(self.model.config, "d_model", getattr(self.model.config, "hidden_size", 768)))
        self.relation_projection = torch.nn.Linear(hidden_size, hidden_size)
        self.instance_projection = torch.nn.Linear(hidden_size, hidden_size)
        self.prototype_offsets = torch.nn.Embedding(len(self.labels), hidden_size)
        self.prototype_norm = torch.nn.LayerNorm(hidden_size)
        if prototype_type == "static":
            self.prototype_offsets.weight.requires_grad_(False)
            torch.nn.init.zeros_(self.prototype_offsets.weight)
        elif prototype_type != "learnable":
            raise ValueError(f"Unknown prototype_type: {prototype_type}")
        self.relation_projection.to(self.device)
        self.instance_projection.to(self.device)
        self.prototype_offsets.to(self.device)
        self.prototype_norm.to(self.device)

    def trainable_parameters(self) -> Any:
        for parameter in self.model.parameters():
            yield parameter
        for module in [self.relation_projection, self.instance_projection, self.prototype_offsets, self.prototype_norm]:
            for parameter in module.parameters():
                if parameter.requires_grad:
                    yield parameter

    def compute_training_loss(self, batch: List[Dict[str, str]]) -> Dict[str, Any]:
        import torch

        examples = self.build_examples(batch)
        inputs, labels = self._tokenize_batch(examples)
        labels[labels == self.tokenizer.pad_token_id] = -100
        generation_output = self.model(**inputs, labels=labels)
        alignment_loss, _logits, _scores = self.compute_alignment_loss(examples, inputs)
        total_loss = generation_output.loss + self.alignment_lambda * alignment_loss
        return {
            "loss": total_loss,
            "generation_loss": generation_output.loss,
            "alignment_loss": alignment_loss,
        }

    def forward(self, batch: List[Dict[str, str]]) -> ModelOutput:
        import torch

        self.model.eval()
        examples = self.build_examples(batch)
        inputs, labels = self._tokenize_batch(examples)
        labels[labels == self.tokenizer.pad_token_id] = -100

        with torch.no_grad():
            generation_output = self.model(**inputs, labels=labels)
            alignment_loss, _logits, scores = self.compute_alignment_loss(examples, inputs)
            generated = self.model.generate(
                **inputs,
                max_length=self.max_output_length,
                num_beams=1,
            )
        raw_outputs = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        predictions = []
        prototype_rows = []
        mismatches = 0
        for row_index, (example, raw_output) in enumerate(zip(examples, raw_outputs)):
            pred_label, valid_output, relation_valid = parse_relation_output(raw_output, self.schema.labels)
            if pred_label != example["gold_label"]:
                mismatches += 1
            score_values = scores[row_index].detach().cpu().tolist()
            score_map = {label: float(score_values[index]) for index, label in enumerate(self.labels)}
            sorted_scores = sorted(score_map.items(), key=lambda item: item[1], reverse=True)
            top_labels = [label for label, _score in sorted_scores[:3]]
            prototype_top1 = top_labels[0]
            predictions.append(
                {
                    "id": example["id"],
                    "gold_label": example["gold_label"],
                    "pred_label": pred_label,
                    "raw_output": raw_output,
                    "valid_output": valid_output,
                    "relation_valid": relation_valid,
                    "prototype_top1": prototype_top1,
                    "prototype_topk": top_labels,
                }
            )
            prototype_rows.append(
                {
                    "id": example["id"],
                    "gold_label": example["gold_label"],
                    "prototype_top1": prototype_top1,
                    "prototype_top3": top_labels,
                    "scores": score_map,
                }
            )
        fallback_loss = mismatches / max(len(batch), 1)
        raw_generation_loss = (
            float(generation_output.loss.detach().cpu().item()) if generation_output.loss is not None else fallback_loss
        )
        generation_loss = finite_or_default(raw_generation_loss, fallback_loss)
        raw_alignment_loss = float(alignment_loss.detach().cpu().item())
        alignment_loss_value = finite_or_default(raw_alignment_loss, fallback_loss)
        total_loss = generation_loss + self.alignment_lambda * alignment_loss_value
        return ModelOutput(
            loss=total_loss,
            generation_loss=generation_loss,
            alignment_loss=alignment_loss_value,
            predictions=predictions,
            prototype_scores=prototype_rows,
        )

    def build_examples(self, samples: List[Dict[str, str]]) -> List[Dict[str, str]]:
        examples = super().build_examples(samples)
        for example, sample in zip(examples, samples):
            example["target_text"] = build_target_text(sample)
        return examples

    def compute_alignment_loss(self, examples: List[Dict[str, str]], inputs: Dict[str, Any]) -> tuple[Any, Any, Any]:
        import torch

        instance_vectors = self.encode_inputs(inputs)
        instance_vectors = torch.nn.functional.normalize(self.instance_projection(instance_vectors), dim=-1)
        prototypes = torch.nn.functional.normalize(self.build_relation_prototypes(), dim=-1)
        scores = instance_vectors @ prototypes.transpose(0, 1)
        logits = scores / self.temperature_tau
        target = torch.tensor([self.label_to_id[item["gold_label"]] for item in examples], device=self.device)
        alignment_loss = torch.nn.functional.cross_entropy(logits, target)
        return alignment_loss, logits, scores

    def build_relation_prototypes(self) -> Any:
        import torch

        semantic_texts = [text for _label, text in self.schema.semantic_items(self.prototype_semantic_field)]
        encoded = self.encode_texts(semantic_texts, self.max_input_length)
        projected = self.relation_projection(encoded)
        offsets = self.prototype_offsets(torch.arange(len(self.labels), device=self.device))
        return self.prototype_norm(projected + offsets)

    def encode_inputs(self, inputs: Dict[str, Any]) -> Any:
        encoder = self.model.get_encoder()
        outputs = encoder(**inputs)
        return masked_mean_pool(outputs.last_hidden_state, inputs["attention_mask"])

    def encode_texts(self, texts: List[str], max_length: int) -> Any:
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokenized = {key: value.to(self.device) for key, value in tokenized.items()}
        return self.encode_inputs(tokenized)

    def save_extra_state(self, target: Path) -> None:
        import torch

        torch.save(
            {
                "relation_projection": self.relation_projection.state_dict(),
                "instance_projection": self.instance_projection.state_dict(),
                "prototype_offsets": self.prototype_offsets.state_dict(),
                "prototype_norm": self.prototype_norm.state_dict(),
                "alignment_lambda": self.alignment_lambda,
                "temperature_tau": self.temperature_tau,
                "prototype_type": self.prototype_type,
                "prototype_semantic_field": self.prototype_semantic_field,
                "labels": self.labels,
            },
            target / "rsg_head.pt",
        )


def masked_mean_pool(hidden_states: Any, attention_mask: Any) -> Any:
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    denominator = mask.sum(dim=1).clamp(min=1.0)
    return summed / denominator
