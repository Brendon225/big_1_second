# Stage 1 RSG-BioRE Experiment Handoff Summary

> Date: 2026-06-10
> Project: `D:\Desktop_D\postgraduate\研二\big\big_1_second`
> Thesis topic: 基于生成式模型的生物医学关系抽取研究
> Stage 1 method: RSG-BioRE, Relation Semantic Prototype Guided Generative Biomedical Relation Extraction

## 1. ISO Snapshot

### Insight

The current thesis line is generative biomedical relation extraction. The core problem is not dependency parsing or closed LLM prompting. The core contradiction is:

- Biomedical relation labels such as ChemProt CPR labels are semantically dense and hard for a generator to interpret from raw labels alone.
- Pure prompt/schema wording can improve a BioBART/T5 baseline, but it is easy to be criticized as prompt engineering.
- The method therefore needs a model-level mechanism: relation semantic prototypes and instance-prototype alignment.

### Strategy

Stage 1 has been implemented around this route:

1. Build relation semantic schema for ChemProt:
   - raw label
   - label words
   - relation description
   - entity-type-aware description
   - knowledge-enhanced description
2. Train BioBART text-to-text baselines.
3. Implement RSG-BioRE:
   - encode relation semantic text into prototypes
   - encode marked entity-pair instances
   - add instance-prototype alignment loss
   - keep decoder output as `relation: <label>`
4. Evaluate not only F1 but also:
   - valid output rate
   - relation validity rate
   - prototype top-1/top-3 accuracy
   - generation/prototype agreement

### Operation

The current best Stage 1 version is R2.1:

- entity markers: `<H> ... </H>` and `<T> ... </T>`
- entity-pair pooling over marker hidden states
- relation semantic prototype alignment
- optional prototype fusion path, although the current full run did not actually rewrite predictions through fusion

## 2. Important Files

### Core docs

- `thesis_research_opening_plan.md`
- `docs/stage1_docs/stage1_relation_label_semantic_modeling_experiment_guide.md`
- `docs/stage1_docs/stage1_rsg_biore_handoff_summary.md`

### Core code

- `src/stage1/hf_text2text_backend.py`
  - Hugging Face BioBART/T5 text-to-text baseline backend.
- `src/stage1/hf_rsg_biore_backend.py`
  - Main RSG-BioRE backend.
  - Implements relation prototypes, alignment loss, entity markers, entity-pair pooling, and prototype diagnostics.
- `src/stage1/prompting.py`
  - Builds relation prompts and marked relation prompts.
- `src/stage1/metrics.py`
  - Computes micro-F1, macro-F1, rare-F1, no-relation diagnostics, valid output rate, and prototype diagnostics.
- `src/stage1/train_runner.py`
  - Main training/evaluation runner.
- `scripts/stage1/train_stage1.py`
  - CLI entry for experiments.
- `scripts/stage1/summarize_results.py`
  - Summarizes result folders into `stage1_metrics_summary.csv`.

### Current full configs to keep

- `configs/stage1/chemprot_hf_biobart_P1_full_seed42.yaml`
- `configs/stage1/chemprot_hf_biobart_P2_full_seed42.yaml`
- `configs/stage1/chemprot_hf_biobart_P3_full_seed42.yaml`
- `configs/stage1/chemprot_hf_rsg_R2_full_seed42.yaml`
- `configs/stage1/chemprot_hf_rsg_R2_1_full_seed42.yaml`
- Low-memory variants of the above are also kept for 5090 runs.

## 3. Current Experiment Results

The main completed ChemProt runs are under:

- `outputs/stage1/chemprot/biobart_text2text/`
- `outputs/stage1/chemprot/rsg_biore/`
- summary table: `outputs/stage1/chemprot/stage1_metrics_summary.csv`

### Main ChemProt Results

| Method | Backend | micro-F1 | macro-F1 | precision | recall | rare-F1 | no-rel F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| P1 relation description | BioBART text-to-text | 0.7420 | 0.6997 | 0.7717 | 0.7146 | 0.6546 | 0.5110 |
| P2 entity-type description | BioBART text-to-text | 0.7081 | 0.6743 | 0.6993 | 0.7172 | 0.6309 | 0.5077 |
| P3 knowledge-enhanced description | BioBART text-to-text | 0.7392 | 0.6943 | 0.7526 | 0.7262 | 0.6779 | 0.5015 |
| R2 prototype alignment | BioBART + RSG | 0.6997 | 0.6432 | 0.6925 | 0.7070 | 0.5885 | 0.4729 |
| R2.1 marker pooling fusion | BioBART + RSG | 0.7570 | 0.7053 | 0.7421 | 0.7726 | 0.6317 | 0.5088 |

### Prototype Diagnostics

| Method | prototype top1 | prototype top3 | generation/prototype agreement |
|---|---:|---:|---:|
| R2 | 0.6516 | 0.9514 | 0.8926 |
| R2.1 | 0.7195 | 0.9562 | 0.9062 |

### Result Interpretation

R2.1 is the current best internal result:

- It improves over R2 by about +5.73 micro-F1 and +6.21 macro-F1 points.
- It improves over the best BioBART prompt baseline P1 by about +1.50 micro-F1 points.
- It improves prototype top-1 accuracy from 0.6516 to 0.7195.

Important diagnostic:

- In the R2.1 full run, final prediction equals generated label for all test cases.
- Prototype fusion was enabled but did not actually rewrite predictions.
- Therefore the gain should be attributed mainly to entity-aware representation and alignment, not to post-hoc label correction.

## 4. SOTA Boundary

Do not claim that the current method beats overall ChemProt SOTA.

The project is generative BioRE, so the main comparison should be against generative/text-to-text methods, not against all encoder-only discriminative models.

Recommended framing:

- PubMedBERT, BioBERT, BioLinkBERT, RE-SciBERT, and NBRNLI are discriminative or classification/matching style references.
- They are useful as diagnostic baselines or upper-bound references, but they are not the primary comparison group for the thesis title.
- T5, BioBART, SciFive, ClinicalT5, and BioGPT-style methods are more relevant generative/text-to-text comparisons.

Recent references to consider in the next exploration:

- BioBART: https://aclanthology.org/2022.bionlp-1.9/
- NBRNLI: https://aclanthology.org/2023.acl-long.138/
- LEAP: https://pmc.ncbi.nlm.nih.gov/articles/PMC11339510/
- Prompt Tuning in Biomedical Relation Extraction: https://pmc.ncbi.nlm.nih.gov/articles/PMC11052745/
- ClinicalT5 / transformer fine-tuning vs inference paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC12796933/

The next Codex session should first verify whether ClinicalT5 was evaluated as a true text-to-text generator or as a fine-tuned classifier-like transformer before treating it as a direct generative SOTA baseline.

## 5. Data Caveat

Current local `data/stage1/chemprot/train.jsonl`, `dev.jsonl`, and `test.jsonl` appear to contain 30 lines each. The completed full outputs report:

- train samples: 5114
- dev samples: 2991
- test samples: 4199

This means the local JSONL files may currently be smoke-sized while the full outputs came from the 5090 machine. Before rerunning full experiments locally or on 5090, verify that the ChemProt JSONL files are the intended full converted data.

## 6. Suggested Next Work

The next exploration should not continue blindly optimizing BioBART. A stronger and cleaner route is:

1. Identify true generative/text-to-text BioRE baselines for ChemProt.
2. Add one stronger generative backbone, preferably SciFive or ClinicalT5 if the model and tokenizer are available.
3. Run the same ladder:
   - P1 relation description
   - P2 entity-type-aware description
   - P3 knowledge-enhanced description
   - R2.1 marker pooling alignment
4. Check whether RSG-BioRE improves over the corresponding backbone baseline.
5. If yes, run multi-seed experiments and then move to DDI 2013.

Recommended thesis claim at the current stage:

> RSG-BioRE improves BioBART-based generative ChemProt relation extraction by modeling relation descriptions as learnable semantic prototypes and aligning marked entity-pair representations with the target relation prototype. The current result supports the effectiveness of relation semantic prototype guidance within a generative framework, but does not yet establish overall ChemProt SOTA.

## 7. Commands for New Session

Inspect current summary:

```bash
python scripts/stage1/summarize_results.py --root-dir outputs/stage1/chemprot --output outputs/stage1/chemprot/stage1_metrics_summary.csv
```

Run current best R2.1 full experiment on 5090:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/stage1/train_stage1.py --config configs/stage1/chemprot_hf_rsg_R2_1_full_seed42_lowmem.yaml
```

Run current best BioBART baseline:

```bash
python scripts/stage1/train_stage1.py --config configs/stage1/chemprot_hf_biobart_P1_full_seed42_lowmem.yaml
```

Before any full rerun, confirm full data size:

```bash
wc -l data/stage1/chemprot/train.jsonl data/stage1/chemprot/dev.jsonl data/stage1/chemprot/test.jsonl
```
