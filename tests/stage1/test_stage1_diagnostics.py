import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


class Stage1DiagnosticsTest(unittest.TestCase):
    def test_error_cases_counts_invalid_final_outputs(self):
        from src.stage1.runner import write_error_cases

        predictions = [
            {
                "id": "x1",
                "gold_label": "CPR:4",
                "pred_label": None,
                "raw_output": "CPR:4 CPR:4",
                "valid_output": False,
                "relation_valid": False,
            },
            {
                "id": "x2",
                "gold_label": "CPR:4",
                "pred_label": "CPR:4",
                "raw_output": "relation: CPR:4",
                "valid_output": True,
                "relation_valid": True,
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "error_cases.md"
            write_error_cases(str(target), minimal_config(), predictions)
            text = target.read_text(encoding="utf-8")

        self.assertIn("| invalid_generation | 1 |", text)

    def test_error_cases_counts_invalid_raw_generation_before_fusion(self):
        from src.stage1.runner import write_error_cases

        predictions = [
            {
                "id": "x1",
                "gold_label": "CPR:4",
                "pred_label": "CPR:4",
                "raw_output": "relation: CPR:4",
                "valid_output": True,
                "relation_valid": True,
                "generated_label": None,
                "raw_generation_output": "CPR:4 CPR:4",
                "generation_valid_output": False,
                "generation_relation_valid": False,
                "prototype_fusion_applied": True,
            },
            {
                "id": "x2",
                "gold_label": "CPR:4",
                "pred_label": "CPR:4",
                "raw_output": "relation: CPR:4",
                "valid_output": True,
                "relation_valid": True,
                "generated_label": "CPR:4",
                "raw_generation_output": "relation: CPR:4",
                "generation_valid_output": True,
                "generation_relation_valid": True,
                "prototype_fusion_applied": True,
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "error_cases.md"
            write_error_cases(str(target), minimal_config(), predictions)
            text = target.read_text(encoding="utf-8")

        self.assertIn("| invalid_raw_generation | 1 |", text)
        self.assertIn("| prototype_fusion_fallback | 1 |", text)

    def test_metrics_include_generation_validity_and_fusion_fallback_rates(self):
        from src.stage1.metrics import evaluate_predictions

        records = [
            {
                "gold_label": "CPR:4",
                "pred_label": "CPR:4",
                "valid_output": True,
                "relation_valid": True,
                "generated_label": None,
                "generation_valid_output": False,
                "generation_relation_valid": False,
                "prototype_fusion_applied": True,
                "prototype_top1": "CPR:4",
                "prototype_topk": ["CPR:4", "NO_RELATION"],
            },
            {
                "gold_label": "NO_RELATION",
                "pred_label": "NO_RELATION",
                "valid_output": True,
                "relation_valid": True,
                "generated_label": "NO_RELATION",
                "generation_valid_output": True,
                "generation_relation_valid": True,
                "prototype_fusion_applied": True,
                "prototype_top1": "NO_RELATION",
                "prototype_topk": ["NO_RELATION", "CPR:4"],
            },
        ]

        result = evaluate_predictions(
            records,
            relation_labels=["CPR:4", "NO_RELATION"],
            no_relation_labels=["NO_RELATION"],
        )

        self.assertEqual(result.metrics["generation_valid_output_rate"], 0.5)
        self.assertEqual(result.metrics["generation_relation_validity_rate"], 0.5)
        self.assertEqual(result.metrics["prototype_fusion_fallback_rate"], 0.5)

    def test_metrics_treat_final_outputs_as_generation_outputs_when_no_fusion_fields_exist(self):
        from src.stage1.metrics import evaluate_predictions

        records = [
            {"gold_label": "CPR:4", "pred_label": "CPR:4", "valid_output": True, "relation_valid": True},
            {"gold_label": "CPR:4", "pred_label": None, "valid_output": False, "relation_valid": False},
        ]

        result = evaluate_predictions(records, relation_labels=["CPR:4", "NO_RELATION"])

        self.assertEqual(result.metrics["generation_valid_output_rate"], 0.5)
        self.assertEqual(result.metrics["generation_relation_validity_rate"], 0.5)

    def test_label_scoring_decoding_formats_lowest_loss_candidate(self):
        from src.stage1.hf_text2text_backend import HfText2TextModel

        model = HfText2TextModel.__new__(HfText2TextModel)
        model.schema = SimpleNamespace(labels=["CPR:4", "NO_RELATION"])
        model.decoding_strategy = "label_scoring"
        model.score_label_candidates = lambda _inputs: [{"CPR:4": 2.0, "NO_RELATION": 0.5}]

        self.assertEqual(model.decode_raw_outputs({}), ["relation: NO_RELATION"])


def minimal_config():
    return {
        "dataset": "ChemProt",
        "model": "test-model",
        "method": "test-method",
        "seed": 42,
    }


if __name__ == "__main__":
    unittest.main()
