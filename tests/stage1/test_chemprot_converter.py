import tempfile
import unittest
from pathlib import Path


class ChemProtConverterTest(unittest.TestCase):
    def test_convert_split_normalizes_chemical_protein_pairs(self):
        from src.stage1.chemprot_converter import convert_split

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_dir = root / "chemprot_training"
            split_dir.mkdir()
            (split_dir / "chemprot_training_abstracts.tsv").write_text(
                "1001\tAspirin inhibits COX1\tAspirin inhibits COX1 activity in cells.\n",
                encoding="utf-8",
            )
            (split_dir / "chemprot_training_entities.tsv").write_text(
                "1001\tT1\tCHEMICAL\t0\t7\tAspirin\n"
                "1001\tT2\tGENE-Y\t17\t21\tCOX1\n"
                "1001\tT3\tCHEMICAL\t30\t38\tCaffeine\n",
                encoding="utf-8",
            )
            (split_dir / "chemprot_training_gold_standard.tsv").write_text(
                "1001\tCPR:4\tArg1:T1\tArg2:T2\n",
                encoding="utf-8",
            )

            samples = convert_split(
                split_dir=split_dir,
                split_name="train",
                include_negatives=True,
                max_negative_per_doc=1,
            )

        positive = [item for item in samples if item["gold_relation"] == "CPR:4"]
        negatives = [item for item in samples if item["gold_relation"] == "NO_RELATION"]

        self.assertEqual(len(positive), 1)
        self.assertEqual(positive[0]["head_entity"], "Aspirin")
        self.assertEqual(positive[0]["head_type"], "chemical")
        self.assertEqual(positive[0]["tail_entity"], "COX1")
        self.assertEqual(positive[0]["tail_type"], "protein")
        self.assertEqual(positive[0]["split"], "train")
        self.assertEqual(len(negatives), 1)

    def test_convert_split_accepts_test_gs_filenames(self):
        from src.stage1.chemprot_converter import convert_split

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_dir = root / "chemprot_test_gs"
            split_dir.mkdir()
            (split_dir / "chemprot_test_abstracts_gs.tsv").write_text(
                "2001\tDrug inhibits target\tDrug inhibits target activity.\n",
                encoding="utf-8",
            )
            (split_dir / "chemprot_test_entities_gs.tsv").write_text(
                "2001\tT1\tCHEMICAL\t0\t4\tDrug\n"
                "2001\tT2\tGENE-Y\t14\t20\ttarget\n",
                encoding="utf-8",
            )
            (split_dir / "chemprot_test_gold_standard.tsv").write_text(
                "2001\tCPR:4\tArg1:T1\tArg2:T2\n",
                encoding="utf-8",
            )

            samples = convert_split(split_dir=split_dir, split_name="test", include_negatives=False)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["gold_relation"], "CPR:4")
        self.assertEqual(samples[0]["split"], "test")


if __name__ == "__main__":
    unittest.main()
