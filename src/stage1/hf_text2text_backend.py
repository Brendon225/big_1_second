from typing import Any, Dict, List

from src.stage1.metrics import parse_relation_output
from src.stage1.model_outputs import ModelOutput
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
    ) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.schema = schema
        self.semantic_field = semantic_field
        self.model_name_or_path = model_name_or_path
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

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

        examples = self.build_examples(batch)
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
        labels[labels == self.tokenizer.pad_token_id] = -100

        with torch.no_grad():
            loss_output = self.model(**inputs, labels=labels)
            generated = self.model.generate(
                **inputs,
                max_length=self.max_output_length,
                num_beams=1,
            )
        raw_outputs = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

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
        generation_loss = float(loss_output.loss.detach().cpu().item()) if loss_output.loss is not None else fallback_loss
        return ModelOutput(
            loss=generation_loss,
            generation_loss=generation_loss,
            alignment_loss=0.0,
            predictions=predictions,
        )
