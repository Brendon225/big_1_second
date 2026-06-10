import unittest


class Stage1SmokeTest(unittest.TestCase):
    def test_schema_prompt_and_parser(self):
        from src.stage1.prompting import build_relation_prompt
        from src.stage1.schema import load_relation_schema
        from src.stage1.metrics import parse_relation_output

        schema = load_relation_schema("data/stage1/tiny/relation_schema.yaml")
        sample = {
            "id": "tiny_001",
            "text": "Aspirin inhibits COX1 activity.",
            "head_entity": "Aspirin",
            "head_type": "chemical",
            "tail_entity": "COX1",
            "tail_type": "protein",
            "gold_relation": "CPR:4",
            "split": "test",
        }

        prompt = build_relation_prompt(sample, schema, semantic_field="entity_type_aware_description")

        self.assertIn("Aspirin", prompt)
        self.assertIn("COX1", prompt)
        self.assertIn("relation:", prompt)
        self.assertEqual(parse_relation_output("relation: CPR:4", schema.labels), ("CPR:4", True, True))
        self.assertEqual(parse_relation_output("CPR:4", schema.labels), (None, False, False))

    def test_marked_relation_prompt_adds_entity_markers(self):
        from src.stage1.prompting import build_marked_relation_prompt
        from src.stage1.schema import load_relation_schema

        schema = load_relation_schema("data/stage1/tiny/relation_schema.yaml")
        sample = {
            "id": "tiny_001",
            "text": "Aspirin inhibits COX1 activity.",
            "head_entity": "Aspirin",
            "head_type": "chemical",
            "tail_entity": "COX1",
            "tail_type": "protein",
            "gold_relation": "CPR:4",
            "split": "test",
        }

        prompt = build_marked_relation_prompt(sample, schema, semantic_field="relation_description")

        self.assertIn("<H> Aspirin </H>", prompt)
        self.assertIn("<T> COX1 </T>", prompt)
        self.assertIn("Relation schema:", prompt)

    def test_metrics_include_generation_and_prototype_fields(self):
        from src.stage1.metrics import evaluate_predictions

        records = [
            {
                "gold_label": "CPR:4",
                "pred_label": "CPR:4",
                "raw_output": "relation: CPR:4",
                "valid_output": True,
                "relation_valid": True,
                "prototype_topk": ["CPR:4", "CPR:3"],
                "prototype_top1": "CPR:4",
            },
            {
                "gold_label": "CPR:5",
                "pred_label": "CPR:4",
                "raw_output": "relation: CPR:4",
                "valid_output": True,
                "relation_valid": True,
                "prototype_topk": ["CPR:4", "CPR:5"],
                "prototype_top1": "CPR:4",
            },
        ]

        result = evaluate_predictions(
            records,
            relation_labels=["CPR:3", "CPR:4", "CPR:5"],
            rare_relations=["CPR:5"],
        )

        self.assertAlmostEqual(result.metrics["valid_output_rate"], 1.0)
        self.assertAlmostEqual(result.metrics["relation_validity_rate"], 1.0)
        self.assertAlmostEqual(result.metrics["prototype_top1_accuracy"], 0.5)
        self.assertAlmostEqual(result.metrics["prototype_top3_accuracy"], 1.0)
        self.assertIn("CPR:5", result.per_class)

    def test_metrics_include_no_relation_diagnostics(self):
        from src.stage1.metrics import evaluate_predictions

        records = [
            {"gold_label": "CPR:4", "pred_label": "CPR:4", "valid_output": True, "relation_valid": True},
            {"gold_label": "NO_RELATION", "pred_label": "CPR:4", "valid_output": True, "relation_valid": True},
            {"gold_label": "NO_RELATION", "pred_label": "NO_RELATION", "valid_output": True, "relation_valid": True},
            {"gold_label": "CPR:5", "pred_label": "NO_RELATION", "valid_output": True, "relation_valid": True},
        ]

        result = evaluate_predictions(
            records,
            relation_labels=["CPR:4", "CPR:5", "NO_RELATION"],
            no_relation_labels=["NO_RELATION"],
        )

        self.assertAlmostEqual(result.metrics["overall_accuracy"], 0.5)
        self.assertAlmostEqual(result.metrics["no_relation_precision"], 0.5)
        self.assertAlmostEqual(result.metrics["no_relation_recall"], 0.5)
        self.assertAlmostEqual(result.metrics["no_relation_f1"], 0.5)
        self.assertAlmostEqual(result.metrics["positive_support"], 2.0)
        self.assertAlmostEqual(result.metrics["no_relation_support"], 2.0)

    def test_text2text_baseline_mock_forward(self):
        from src.stage1.data_io import read_jsonl
        from src.stage1.models.text2text_baseline import Text2TextBaseline
        from src.stage1.schema import load_relation_schema

        schema = load_relation_schema("data/stage1/tiny/relation_schema.yaml")
        batch = read_jsonl("data/stage1/tiny/test.jsonl")
        model = Text2TextBaseline(schema=schema, semantic_field="entity_type_aware_description")

        output = model.forward(batch)

        self.assertEqual(len(output.predictions), len(batch))
        self.assertGreaterEqual(output.loss, 0.0)
        self.assertTrue(all(item["raw_output"].startswith("relation: ") for item in output.predictions))

    def test_rsg_biore_mock_forward_outputs_alignment_artifacts(self):
        from src.stage1.data_io import read_jsonl
        from src.stage1.models.rsg_biore import RsgBioREModel
        from src.stage1.schema import load_relation_schema

        schema = load_relation_schema("data/stage1/tiny/relation_schema.yaml")
        batch = read_jsonl("data/stage1/tiny/test.jsonl")
        model = RsgBioREModel(
            schema=schema,
            semantic_field="entity_type_aware_description",
            alignment_lambda=0.1,
            temperature_tau=0.1,
        )

        output = model.forward(batch)

        self.assertEqual(len(output.predictions), len(batch))
        self.assertEqual(len(output.prototype_scores), len(batch))
        self.assertGreaterEqual(output.generation_loss, 0.0)
        self.assertGreaterEqual(output.alignment_loss, 0.0)
        self.assertGreaterEqual(output.loss, output.generation_loss)
        self.assertIn("prototype_top1", output.predictions[0])


if __name__ == "__main__":
    unittest.main()
