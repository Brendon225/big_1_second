import unittest


class ConfigBuilderTest(unittest.TestCase):
    def test_scifive_derivation_updates_backbone_identity_and_keeps_training_inputs(self):
        from src.stage1.config_builder import SCIFIVE_BASE_PUBMED_PMC, derive_backbone_config

        base_config = {
            "experiment_id": "chemprot_hf_biobart_P1_full_seed42",
            "dataset": "ChemProt",
            "model": "GanjinZero/biobart-base",
            "model_name_or_path": "models/stage1/biobart-base",
            "method": "P1_relation_description",
            "backend": "hf",
            "semantic_field": "relation_description",
            "train_file": "data/stage1/chemprot/train.jsonl",
            "dev_file": "data/stage1/chemprot/dev.jsonl",
            "test_file": "data/stage1/chemprot/test.jsonl",
            "schema_file": "data/stage1/chemprot/relation_schema.yaml",
            "output_dir": "outputs/stage1/chemprot/biobart_text2text/P1_relation_description_seed42_full",
        }

        derived = derive_backbone_config(base_config, SCIFIVE_BASE_PUBMED_PMC)

        self.assertEqual(derived["experiment_id"], "chemprot_hf_scifive_P1_full_seed42")
        self.assertEqual(derived["model"], "razent/SciFive-base-Pubmed_PMC")
        self.assertEqual(derived["model_name_or_path"], "models/stage1/scifive-base-pubmed-pmc")
        self.assertEqual(
            derived["output_dir"],
            "outputs/stage1/chemprot/scifive_text2text/P1_relation_description_seed42_full",
        )
        self.assertEqual(derived["method"], base_config["method"])
        self.assertEqual(derived["backend"], base_config["backend"])
        self.assertEqual(derived["train_file"], base_config["train_file"])
        self.assertEqual(derived["schema_file"], base_config["schema_file"])

    def test_scifive_derivation_names_rsg_config_without_losing_alignment_controls(self):
        from src.stage1.config_builder import SCIFIVE_BASE_PUBMED_PMC, derive_backbone_config

        base_config = {
            "experiment_id": "chemprot_hf_rsg_R2_1_full_seed42",
            "dataset": "ChemProt",
            "model": "GanjinZero/biobart-base",
            "model_name_or_path": "models/stage1/biobart-base",
            "method": "R2_1_marker_pooling_fusion",
            "backend": "hf_rsg",
            "semantic_field": "relation_description",
            "prototype_semantic_field": "knowledge_enhanced_description",
            "alignment_lambda": 0.05,
            "instance_pooling": "entity_pair",
            "use_entity_markers": True,
            "use_prototype_fusion": True,
            "output_dir": "outputs/stage1/chemprot/rsg_biore/R2_1_marker_pooling_fusion_seed42_full",
        }

        derived = derive_backbone_config(base_config, SCIFIVE_BASE_PUBMED_PMC)

        self.assertEqual(derived["experiment_id"], "chemprot_hf_rsg_scifive_R2_1_full_seed42")
        self.assertEqual(
            derived["output_dir"],
            "outputs/stage1/chemprot/rsg_biore_scifive/R2_1_marker_pooling_fusion_seed42_full",
        )
        self.assertEqual(derived["backend"], "hf_rsg")
        self.assertEqual(derived["alignment_lambda"], 0.05)
        self.assertEqual(derived["instance_pooling"], "entity_pair")
        self.assertEqual(derived["use_entity_markers"], True)
        self.assertEqual(derived["use_prototype_fusion"], True)

    def test_scifive_large_derivation_applies_low_memory_training_overrides(self):
        from src.stage1.config_builder import SCIFIVE_LARGE_PUBMED_PMC, derive_backbone_config

        base_config = {
            "experiment_id": "chemprot_hf_rsg_R2_1_full_seed42",
            "dataset": "ChemProt",
            "model": "GanjinZero/biobart-base",
            "model_name_or_path": "models/stage1/biobart-base",
            "method": "R2_1_marker_pooling_fusion",
            "backend": "hf_rsg",
            "batch_size": 4,
            "eval_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "learning_rate": 0.00003,
            "model_dtype": "float32",
            "output_dir": "outputs/stage1/chemprot/rsg_biore/R2_1_marker_pooling_fusion_seed42_full",
        }

        derived = derive_backbone_config(base_config, SCIFIVE_LARGE_PUBMED_PMC)

        self.assertEqual(derived["experiment_id"], "chemprot_hf_rsg_scifive_large_R2_1_full_seed42")
        self.assertEqual(derived["model"], "razent/SciFive-large-Pubmed_PMC")
        self.assertEqual(derived["model_name_or_path"], "models/stage1/scifive-large-pubmed-pmc")
        self.assertEqual(
            derived["output_dir"],
            "outputs/stage1/chemprot/rsg_biore_scifive_large/R2_1_marker_pooling_fusion_seed42_full",
        )
        self.assertEqual(derived["model_dtype"], "bfloat16")
        self.assertEqual(derived["batch_size"], 1)
        self.assertEqual(derived["eval_batch_size"], 1)
        self.assertEqual(derived["gradient_accumulation_steps"], 8)
        self.assertEqual(derived["learning_rate"], 0.00002)


if __name__ == "__main__":
    unittest.main()
