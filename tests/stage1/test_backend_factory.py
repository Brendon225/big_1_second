import unittest
import importlib.util


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

    def test_backend_factory_recognizes_hf_rsg_backend(self):
        import src.stage1.backend_factory as backend_factory

        original_require = backend_factory.require_modules
        backend_factory.require_modules = lambda *_args, **_kwargs: None
        try:
            backend_class = backend_factory.resolve_hf_backend_class("hf_rsg")
        finally:
            backend_factory.require_modules = original_require

        self.assertEqual(backend_class.__name__, "HfRsgBioREModel")

    def test_backend_factory_passes_decoding_strategy_to_hf_backend(self):
        import src.stage1.backend_factory as backend_factory
        from src.stage1.schema import load_relation_schema

        captured_kwargs = {}

        class FakeHfBackend:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

        original_require = backend_factory.require_modules
        original_resolve = backend_factory.resolve_hf_backend_class
        backend_factory.require_modules = lambda *_args, **_kwargs: None
        backend_factory.resolve_hf_backend_class = lambda _backend: FakeHfBackend
        try:
            backend_factory.build_stage1_model(
                method="P3_knowledge_enhanced_description",
                schema=load_relation_schema("data/stage1/tiny/relation_schema.yaml"),
                semantic_field="knowledge_enhanced_description",
                backend="hf",
                decoding_strategy="label_scoring",
            )
        finally:
            backend_factory.require_modules = original_require
            backend_factory.resolve_hf_backend_class = original_resolve

        self.assertEqual(captured_kwargs["decoding_strategy"], "label_scoring")

    def test_rsg_fusion_falls_back_to_prototype_for_invalid_generation(self):
        from src.stage1.hf_rsg_biore_backend import HfRsgBioREModel

        model = HfRsgBioREModel.__new__(HfRsgBioREModel)
        model.use_prototype_fusion = True
        model.prototype_fusion_alpha = 2.0
        model.label_to_id = {"CPR:4": 0, "NO_RELATION": 1}

        label = model.fuse_prediction(None, {"CPR:4": 0.1, "NO_RELATION": 0.3})

        self.assertEqual(label, "NO_RELATION")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
    def test_rsg_head_modules_follow_bfloat16_model_dtype(self):
        import torch

        from src.stage1.hf_rsg_biore_backend import HfRsgBioREModel

        rsg = HfRsgBioREModel.__new__(HfRsgBioREModel)
        rsg.device = "cpu"
        rsg.model = torch.nn.Linear(2, 2).to(dtype=torch.bfloat16)
        rsg.relation_projection = torch.nn.Linear(2, 2)
        rsg.instance_projection = torch.nn.Linear(2, 2)
        rsg.prototype_offsets = torch.nn.Embedding(2, 2)
        rsg.prototype_norm = torch.nn.LayerNorm(2)

        rsg.sync_rsg_head_modules_to_model_dtype(torch)

        self.assertEqual(rsg.relation_projection.weight.dtype, torch.bfloat16)
        self.assertEqual(rsg.instance_projection.weight.dtype, torch.bfloat16)
        self.assertEqual(rsg.prototype_offsets.weight.dtype, torch.bfloat16)
        self.assertEqual(rsg.prototype_norm.weight.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
