import json
import tempfile
import unittest
from pathlib import Path


class ResultSummaryTest(unittest.TestCase):
    def test_summarize_results_writes_sorted_csv(self):
        from src.stage1.result_summary import summarize_metrics, write_summary_csv

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_b = root / "outputs" / "run_b"
            run_a = root / "outputs" / "run_a"
            run_b.mkdir(parents=True)
            run_a.mkdir(parents=True)
            (run_b / "metrics.json").write_text(
                json.dumps({"experiment_id": "b", "micro_f1": 0.2, "macro_f1": 0.1}),
                encoding="utf-8",
            )
            (run_a / "metrics.json").write_text(
                json.dumps({"experiment_id": "a", "micro_f1": 0.4, "macro_f1": 0.3}),
                encoding="utf-8",
            )

            rows = summarize_metrics(str(root / "outputs"))
            output_csv = root / "summary.csv"
            write_summary_csv(rows, str(output_csv))

            text = output_csv.read_text(encoding="utf-8")
            self.assertEqual([row["experiment_id"] for row in rows], ["a", "b"])
            self.assertIn("experiment_id", text.splitlines()[0])
            self.assertIn("micro_f1", text.splitlines()[0])
            self.assertIn("a", text)
            self.assertIn("b", text)

    def test_summarize_results_backfills_diagnostics_from_predictions(self):
        from src.stage1.data_io import write_jsonl
        from src.stage1.result_summary import summarize_metrics

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "outputs" / "run"
            run_dir.mkdir(parents=True)
            (run_dir / "metrics.json").write_text(
                json.dumps({"experiment_id": "run", "micro_f1": 0.4}),
                encoding="utf-8",
            )
            (run_dir / "run_config.yaml").write_text(
                json.dumps({"schema_file": "data/stage1/tiny/relation_schema.yaml"}),
                encoding="utf-8",
            )
            write_jsonl(
                str(run_dir / "predictions.jsonl"),
                [
                    {"gold_label": "CPR:4", "pred_label": "CPR:4", "valid_output": True, "relation_valid": True},
                    {
                        "gold_label": "NO_RELATION",
                        "pred_label": "NO_RELATION",
                        "valid_output": True,
                        "relation_valid": True,
                    },
                ],
            )

            rows = summarize_metrics(str(Path(tmp) / "outputs"))

        self.assertEqual(rows[0]["overall_accuracy"], 1.0)
        self.assertEqual(rows[0]["no_relation_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
