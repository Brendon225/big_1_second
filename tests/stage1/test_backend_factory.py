import unittest


class BackendFactoryTest(unittest.TestCase):
    def test_mock_backend_factory_builds_text2text_model(self):
        from src.stage1.backend_factory import build_stage1_model
        from src.stage1.schema import load_relation_schema

        schema = load_relation_schema("data/stage1/tiny/relation_schema.yaml")
        model = build_stage1_model(
            method="P2_entity_type_description",
            schema=schema,
            semantic_field="entity_type_aware_description",
            backend="mock",
        )

        output = model.forward(
            [
                {
                    "id": "x1",
                    "text": "Aspirin inhibits COX1.",
                    "head_entity": "Aspirin",
                    "head_type": "chemical",
                    "tail_entity": "COX1",
                    "tail_type": "protein",
                    "gold_relation": "CPR:4",
                    "split": "test",
                }
            ]
        )

        self.assertEqual(len(output.predictions), 1)

    def test_optional_dependency_error_names_install_file(self):
        from src.stage1.optional_deps import OptionalDependencyError, require_modules

        with self.assertRaises(OptionalDependencyError) as ctx:
            require_modules(["stage1_definitely_missing_dependency"], "test backend")

        message = str(ctx.exception)
        self.assertIn("stage1_definitely_missing_dependency", message)
        self.assertIn("requirements-stage1.txt", message)


if __name__ == "__main__":
    unittest.main()
