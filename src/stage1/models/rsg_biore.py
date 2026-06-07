from typing import Dict, List

from src.stage1.metrics import parse_relation_output
from src.stage1.model_outputs import ModelOutput
from src.stage1.prompting import build_relation_prompt
from src.stage1.schema import RelationSchema
from src.stage1.vectorizer import (
    add_vectors,
    cosine,
    l2_normalize,
    softmax_cross_entropy,
    stable_label_vector,
    vectorize_text,
)


class RsgBioREModel:
    """Dependency-free RSG-BioRE framework for smoke tests and interface checks."""

    def __init__(
        self,
        schema: RelationSchema,
        semantic_field: str,
        alignment_lambda: float = 0.1,
        temperature_tau: float = 0.1,
        prototype_type: str = "learnable",
        backend: str = "mock",
    ) -> None:
        if backend != "mock":
            raise RuntimeError(
                "The smoke implementation only enables backend='mock'. "
                "Install torch and transformers before enabling trainable T5/BioBART."
            )
        if temperature_tau <= 0:
            raise ValueError("temperature_tau must be positive")
        self.schema = schema
        self.semantic_field = semantic_field
        self.alignment_lambda = alignment_lambda
        self.temperature_tau = temperature_tau
        self.prototype_type = prototype_type
        self.prototypes = self._build_prototypes()

    def forward(self, batch: List[Dict[str, str]]) -> ModelOutput:
        predictions = []
        prototype_rows = []
        alignment_losses = []
        generation_mismatches = 0

        for sample in batch:
            instance_vector = self._encode_instance(sample)
            raw_scores = {
                label: cosine(instance_vector, prototype)
                for label, prototype in self.prototypes.items()
            }
            scaled_scores = {
                label: score / self.temperature_tau for label, score in raw_scores.items()
            }
            sorted_scores = sorted(raw_scores.items(), key=lambda item: item[1], reverse=True)
            top_labels = [label for label, _score in sorted_scores[:3]]
            pred_label = top_labels[0]
            raw_output = f"relation: {pred_label}"
            parsed_label, valid_output, relation_valid = parse_relation_output(raw_output, self.schema.labels)
            gold_label = sample["gold_relation"]
            if parsed_label != gold_label:
                generation_mismatches += 1
            alignment_losses.append(softmax_cross_entropy(scaled_scores, gold_label))

            prompt = build_relation_prompt(sample, self.schema, self.semantic_field)
            predictions.append(
                {
                    "id": sample["id"],
                    "gold_label": gold_label,
                    "pred_label": parsed_label,
                    "raw_output": raw_output,
                    "valid_output": valid_output,
                    "relation_valid": relation_valid,
                    "prototype_top1": pred_label,
                    "prototype_topk": top_labels,
                    "prompt": prompt,
                }
            )
            prototype_rows.append(
                {
                    "id": sample["id"],
                    "gold_label": gold_label,
                    "prototype_top1": pred_label,
                    "prototype_top3": top_labels,
                    "scores": raw_scores,
                }
            )

        generation_loss = generation_mismatches / max(len(batch), 1)
        alignment_loss = sum(alignment_losses) / max(len(alignment_losses), 1)
        total_loss = generation_loss + self.alignment_lambda * alignment_loss
        return ModelOutput(
            loss=total_loss,
            generation_loss=generation_loss,
            alignment_loss=alignment_loss,
            predictions=predictions,
            prototype_scores=prototype_rows,
        )

    def _build_prototypes(self) -> Dict[str, List[float]]:
        prototypes: Dict[str, List[float]] = {}
        for label, semantic_text in self.schema.semantic_items(self.semantic_field):
            semantic_vector = vectorize_text(semantic_text)
            if self.prototype_type == "static":
                prototypes[label] = l2_normalize(semantic_vector)
            elif self.prototype_type == "learnable":
                prototypes[label] = l2_normalize(add_vectors(semantic_vector, stable_label_vector(label)))
            else:
                raise ValueError(f"Unknown prototype_type: {self.prototype_type}")
        return prototypes

    def _encode_instance(self, sample: Dict[str, str]) -> List[float]:
        text = (
            f"{sample['text']} head {sample['head_entity']} {sample['head_type']} "
            f"tail {sample['tail_entity']} {sample['tail_type']}"
        )
        return vectorize_text(text)
