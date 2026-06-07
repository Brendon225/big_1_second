from typing import Dict, List

from src.stage1.metrics import parse_relation_output
from src.stage1.model_outputs import ModelOutput
from src.stage1.prompting import build_relation_prompt
from src.stage1.schema import RelationSchema
from src.stage1.vectorizer import cosine, vectorize_text


class Text2TextBaseline:
    """Text-to-text baseline interface with a dependency-free mock backend."""

    def __init__(self, schema: RelationSchema, semantic_field: str, backend: str = "mock") -> None:
        if backend != "mock":
            raise RuntimeError(
                "The smoke implementation only enables backend='mock'. "
                "Install torch and transformers before enabling a Hugging Face backend."
            )
        self.schema = schema
        self.semantic_field = semantic_field
        self.relation_vectors = {
            label: vectorize_text(text)
            for label, text in self.schema.semantic_items(self.semantic_field)
        }

    def forward(self, batch: List[Dict[str, str]]) -> ModelOutput:
        predictions = []
        mismatches = 0
        for sample in batch:
            prompt = build_relation_prompt(sample, self.schema, self.semantic_field)
            pred_label = self._predict_label(sample)
            raw_output = f"relation: {pred_label}"
            parsed_label, valid_output, relation_valid = parse_relation_output(raw_output, self.schema.labels)
            if parsed_label != sample["gold_relation"]:
                mismatches += 1
            predictions.append(
                {
                    "id": sample["id"],
                    "gold_label": sample["gold_relation"],
                    "pred_label": parsed_label,
                    "raw_output": raw_output,
                    "valid_output": valid_output,
                    "relation_valid": relation_valid,
                    "prompt": prompt,
                }
            )
        loss = mismatches / max(len(batch), 1)
        return ModelOutput(loss=loss, generation_loss=loss, alignment_loss=0.0, predictions=predictions)

    def _predict_label(self, sample: Dict[str, str]) -> str:
        instance_text = (
            f"{sample['text']} {sample['head_entity']} {sample['head_type']} "
            f"{sample['tail_entity']} {sample['tail_type']}"
        )
        instance_vector = vectorize_text(instance_text)
        scores = {
            label: cosine(instance_vector, relation_vector)
            for label, relation_vector in self.relation_vectors.items()
        }
        return max(scores.items(), key=lambda item: item[1])[0]
