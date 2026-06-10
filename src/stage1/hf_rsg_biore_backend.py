from pathlib import Path
from typing import Any, Dict, List, Optional

from src.stage1.hf_text2text_backend import HfText2TextModel
from src.stage1.metrics import parse_relation_output
from src.stage1.model_outputs import ModelOutput
from src.stage1.numeric_utils import finite_or_default
from src.stage1.prompting import build_marked_relation_prompt, build_target_text
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
        instance_pooling: str = "mean",
        use_entity_markers: bool = False,
        use_prototype_fusion: bool = False,
        prototype_fusion_alpha: float = 0.0,
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
        self.instance_pooling = instance_pooling
        self.use_entity_markers = bool(use_entity_markers)
        self.use_prototype_fusion = bool(use_prototype_fusion)
        self.prototype_fusion_alpha = float(prototype_fusion_alpha)
        self.marker_token_ids: Dict[str, int] = {}
        if self.use_entity_markers:
            added = self.tokenizer.add_special_tokens(
                {"additional_special_tokens": ["<H>", "</H>", "<T>", "</T>"]}
            )
            if added:
                self.model.resize_token_embeddings(len(self.tokenizer))
                self._apply_model_dtype(torch, model_dtype)
                self.model.to(self.device)
            self.marker_token_ids = {
                token: int(self.tokenizer.convert_tokens_to_ids(token))
                for token in ["<H>", "</H>", "<T>", "</T>"]
            }
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
            generated_label, generated_valid, generated_relation_valid = parse_relation_output(
                raw_output, self.schema.labels
            )
            score_values = scores[row_index].detach().cpu().tolist()
            score_map = {label: float(score_values[index]) for index, label in enumerate(self.labels)}
            sorted_scores = sorted(score_map.items(), key=lambda item: item[1], reverse=True)
            top_labels = [label for label, _score in sorted_scores[:3]]
            prototype_top1 = top_labels[0]
            pred_label = self.fuse_prediction(generated_label, score_map)
            raw_prediction_output = f"relation: {pred_label}" if pred_label is not None else raw_output
            pred_label, valid_output, relation_valid = parse_relation_output(raw_prediction_output, self.schema.labels)
            if pred_label != example["gold_label"]:
                mismatches += 1
            predictions.append(
                {
                    "id": example["id"],
                    "gold_label": example["gold_label"],
                    "pred_label": pred_label,
                    "raw_output": raw_prediction_output,
                    "valid_output": valid_output,
                    "relation_valid": relation_valid,
                    "generated_label": generated_label,
                    "raw_generation_output": raw_output,
                    "generation_valid_output": generated_valid,
                    "generation_relation_valid": generated_relation_valid,
                    "prototype_top1": prototype_top1,
                    "prototype_topk": top_labels,
                    "prototype_fusion_applied": self.use_prototype_fusion,
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
        if not self.use_entity_markers:
            return super().build_examples(samples)
        return [
            {
                "id": sample["id"],
                "input_text": build_marked_relation_prompt(sample, self.schema, self.semantic_field),
                "target_text": build_target_text(sample),
                "gold_label": sample["gold_relation"],
            }
            for sample in samples
        ]

    def compute_alignment_loss(self, examples: List[Dict[str, str]], inputs: Dict[str, Any]) -> tuple[Any, Any, Any]:
        import torch

        instance_vectors = self.encode_inputs(inputs, self.instance_pooling)
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

    def encode_inputs(self, inputs: Dict[str, Any], pooling: str = "mean") -> Any:
        encoder = self.model.get_encoder()
        outputs = encoder(**inputs)
        if pooling == "entity_pair" and self.marker_token_ids:
            return marker_pair_pool(
                outputs.last_hidden_state,
                inputs["input_ids"],
                inputs["attention_mask"],
                self.marker_token_ids["<H>"],
                self.marker_token_ids["<T>"],
            )
        if pooling != "mean":
            raise ValueError(f"Unknown instance_pooling: {pooling}")
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
        return self.encode_inputs(tokenized, "mean")

    def fuse_prediction(self, generated_label: Optional[str], score_map: Dict[str, float]) -> Optional[str]:
        if not self.use_prototype_fusion:
            return generated_label
        if generated_label not in self.label_to_id:
            return max(score_map.items(), key=lambda item: item[1])[0]
        fused_scores = {
            label: (1.0 if label == generated_label else 0.0) + self.prototype_fusion_alpha * score
            for label, score in score_map.items()
        }
        return max(fused_scores.items(), key=lambda item: item[1])[0]

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
                "instance_pooling": self.instance_pooling,
                "use_entity_markers": self.use_entity_markers,
                "use_prototype_fusion": self.use_prototype_fusion,
                "prototype_fusion_alpha": self.prototype_fusion_alpha,
                "labels": self.labels,
            },
            target / "rsg_head.pt",
        )


def masked_mean_pool(hidden_states: Any, attention_mask: Any) -> Any:
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    denominator = mask.sum(dim=1).clamp(min=1.0)
    return summed / denominator


def marker_pair_pool(hidden_states: Any, input_ids: Any, attention_mask: Any, head_marker_id: int, tail_marker_id: int) -> Any:
    import torch

    rows = []
    fallback = masked_mean_pool(hidden_states, attention_mask)
    for row_index in range(input_ids.shape[0]):
        row_ids = input_ids[row_index]
        head_positions = (row_ids == head_marker_id).nonzero(as_tuple=False)
        tail_positions = (row_ids == tail_marker_id).nonzero(as_tuple=False)
        if len(head_positions) and len(tail_positions):
            head_vector = hidden_states[row_index, int(head_positions[0].item())]
            tail_vector = hidden_states[row_index, int(tail_positions[0].item())]
            rows.append((head_vector + tail_vector) / 2.0)
        else:
            rows.append(fallback[row_index])
    return torch.stack(rows, dim=0)
