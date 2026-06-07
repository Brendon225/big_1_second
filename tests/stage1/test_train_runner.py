import json
import tempfile
import unittest
from pathlib import Path


class TrainRunnerTest(unittest.TestCase):
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
                    }
                ),
                encoding="utf-8",
            )

            metrics = run_training(str(config_path))

            self.assertEqual(metrics["experiment_id"], "tiny_fake_train")
            self.assertIn("micro_f1", metrics)
            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "predictions.jsonl").exists())
            self.assertTrue((output_dir / "train_log.txt").exists())
            self.assertTrue((output_dir / "run_config.yaml").exists())


if __name__ == "__main__":
    unittest.main()
