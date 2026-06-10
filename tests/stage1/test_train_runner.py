import json
import tempfile
import unittest
from pathlib import Path

from src.stage1.model_outputs import ModelOutput


class RecordingForwardModel:
    def __init__(self):
        self.batch_sizes = []

    def forward(self, batch):
        self.batch_sizes.append(len(batch))
        return ModelOutput(
            loss=float(len(batch)),
            generation_loss=float(len(batch)) + 0.5,
            alignment_loss=0.0,
            predictions=[{"id": sample["id"], "gold_label": "CPR:4", "pred_label": "CPR:4"} for sample in batch],
        )


class RecordingTrainModel(RecordingForwardModel):
    def __init__(self):
        super().__init__()
        self.train_kwargs = None

    def train_model(self, **kwargs):
        self.train_kwargs = kwargs
        return ["trained=true"]


class TrainRunnerTest(unittest.TestCase):
    def test_forward_in_batches_splits_evaluation_batches_and_aggregates_loss(self):
        from src.stage1.train_runner import forward_in_batches

        model = RecordingForwardModel()
        samples = [{"id": f"s{i}"} for i in range(5)]

        output = forward_in_batches(model, samples, batch_size=2)

        self.assertEqual(model.batch_sizes, [2, 2, 1])
        self.assertEqual(len(output.predictions), 5)
        self.assertAlmostEqual(output.loss, 1.8)
        self.assertAlmostEqual(output.generation_loss, 2.3)

    def test_train_runner_passes_low_memory_training_controls(self):
        import src.stage1.train_runner as runner_module

        model = RecordingTrainModel()
        build_kwargs = {}
        original_builder = runner_module.build_stage1_model

        def record_builder(**kwargs):
            build_kwargs.update(kwargs)
            return model

        runner_module.build_stage1_model = record_builder
        try:
            with tempfile.TemporaryDirectory() as tmp:
                output_dir = Path(tmp) / "run"
                config_path = Path(tmp) / "config.json"
                config_path.write_text(
                    json.dumps(
                        {
                            "experiment_id": "tiny_fake_train",
                            "dataset": "TinyChemProtSmoke",
                            "model": "fake-seq2seq",
                            "method": "P2_entity_type_description",
                            "backend": "fake_train",
                            "semantic_field": "entity_type_aware_description",
                            "schema_file": "data/stage1/tiny/relation_schema.yaml",
                            "train_file": "data/stage1/tiny/train.jsonl",
                            "dev_file": "data/stage1/tiny/dev.jsonl",
                            "test_file": "data/stage1/tiny/test.jsonl",
                            "output_dir": str(output_dir),
                            "epochs": 1,
                            "batch_size": 2,
                            "eval_batch_size": 1,
                            "gradient_accumulation_steps": 4,
                            "model_dtype": "float32",
                            "max_non_finite_batches": 3,
                            "alignment_lambda": 0.2,
                            "temperature_tau": 0.07,
                            "prototype_type": "learnable",
                            "prototype_semantic_field": "knowledge_enhanced_description",
                            "instance_pooling": "entity_pair",
                            "use_entity_markers": True,
                            "use_prototype_fusion": True,
                            "prototype_fusion_alpha": 0.2,
                            "max_train_samples": 2,
                            "max_dev_samples": 1,
                            "max_test_samples": 1,
                        }
                    ),
                    encoding="utf-8",
                )

                runner_module.run_training(str(config_path))
        finally:
            runner_module.build_stage1_model = original_builder

        self.assertEqual(model.train_kwargs["batch_size"], 2)
        self.assertEqual(model.train_kwargs["eval_batch_size"], 1)
        self.assertEqual(model.train_kwargs["gradient_accumulation_steps"], 4)
        self.assertEqual(model.train_kwargs["max_non_finite_batches"], 3)
        self.assertEqual(build_kwargs["model_dtype"], "float32")
        self.assertEqual(build_kwargs["alignment_lambda"], 0.2)
        self.assertEqual(build_kwargs["temperature_tau"], 0.07)
        self.assertEqual(build_kwargs["prototype_type"], "learnable")
        self.assertEqual(build_kwargs["prototype_semantic_field"], "knowledge_enhanced_description")
        self.assertEqual(build_kwargs["instance_pooling"], "entity_pair")
        self.assertEqual(build_kwargs["use_entity_markers"], True)
        self.assertEqual(build_kwargs["use_prototype_fusion"], True)
        self.assertEqual(build_kwargs["prototype_fusion_alpha"], 0.2)

    def test_train_runner_uses_fake_backend_and_writes_outputs(self):
        from src.stage1.train_runner import run_training

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            config_path = Path(tmp) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "experiment_id": "tiny_fake_train",
                        "dataset": "TinyChemProtSmoke",
                        "model": "fake-seq2seq",
                        "method": "P2_entity_type_description",
                        "backend": "fake_train",
                        "semantic_field": "entity_type_aware_description",
                        "schema_file": "data/stage1/tiny/relation_schema.yaml",
                        "train_file": "data/stage1/tiny/train.jsonl",
                        "dev_file": "data/stage1/tiny/dev.jsonl",
                        "test_file": "data/stage1/tiny/test.jsonl",
                        "output_dir": str(output_dir),
                        "epochs": 1,
                        "seed": 42,
                        "max_train_samples": 2,
                        "max_dev_samples": 1,
                        "max_test_samples": 1,
                    }
                ),
                encoding="utf-8",
            )

            metrics = run_training(str(config_path))

            self.assertEqual(metrics["experiment_id"], "tiny_fake_train")
            self.assertIn("micro_f1", metrics)
            self.assertEqual(metrics["train_samples"], 2)
            self.assertEqual(metrics["dev_samples"], 1)
            self.assertEqual(metrics["test_samples"], 1)
            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "predictions.jsonl").exists())
            self.assertTrue((output_dir / "train_log.txt").exists())
            self.assertTrue((output_dir / "run_config.yaml").exists())

    def test_full_run_rejects_smoke_sized_data(self):
        from src.stage1.train_runner import run_training

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "run"
            config_path = Path(tmp) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "experiment_id": "tiny_full_train",
                        "dataset": "TinyChemProtSmoke",
                        "model": "fake-seq2seq",
                        "method": "P2_entity_type_description",
                        "backend": "fake_train",
                        "semantic_field": "entity_type_aware_description",
                        "schema_file": "data/stage1/tiny/relation_schema.yaml",
                        "train_file": "data/stage1/tiny/train.jsonl",
                        "dev_file": "data/stage1/tiny/dev.jsonl",
                        "test_file": "data/stage1/tiny/test.jsonl",
                        "output_dir": str(output_dir),
                        "epochs": 1,
                        "seed": 42,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "looks like a full run"):
                run_training(str(config_path))


if __name__ == "__main__":
    unittest.main()
