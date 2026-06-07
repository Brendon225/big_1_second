# Stage 1 Implementation Plan: RSG-BioRE

> 本文档是研究内容一的工程落地计划，不替代已有开题规划和实验指南。  
> 当前目标：先跑通可复现最小闭环，再接入真实 ChemProt、DDI 2013 与 Hugging Face 训练。

## I. Insight

当前核心矛盾不是“把 relation description 写进 prompt”，而是要证明关系语义可以从 prompt 文本升级为模型内部可学习、可对齐的 relation semantic prototypes。

内部矛盾：

1. BioRE relation label 抽象、缩写化，生成式模型直接输出 label 容易混淆。
2. prompt/schema comparison 只能作为 baseline 和消融，不能作为最终创新点。
3. RSG-BioRE 必须包含 prototype memory 与 instance-prototype alignment loss。

外部矛盾：

1. 不走 dependency parsing 作为核心路线。
2. 不依赖闭源重型 LLM 或完整 RAG。
3. PubMedBERT/BioBERT 只能作为 diagnostic baseline，不是生成式主线。
4. 实验同学需要按 smoke -> ChemProt -> DDI -> optional BioRED 的顺序复现。

## S. Strategy

当前 repo 原本只有文档和 PDF，没有现成 `data/`、`src/`、`configs/`、`scripts/`、`outputs/`、`checkpoints/`。因此 Stage 1 采用轻量新目录，不侵入 thesis 文档。

执行顺序：

1. 建立统一 JSONL 数据契约与 relation schema。
2. 先实现 T5/BioBART text-to-text baseline 接口，目前用 `backend=mock` 做无依赖 smoke。
3. 再实现 RSG-BioRE 框架：relation semantic encoder、instance encoder、prototype memory、alignment loss、generation loss、total loss。
4. 先用 tiny/mock 数据验证完整输出。
5. 后续安装最小依赖后，把 mock backend 替换为 Hugging Face T5/BioBART backend。

## O. Operation

### A. Data Layer

统一 JSONL 字段：

```json
{
  "id": "ChemProt_train_000001",
  "text": "sentence or passage text",
  "head_entity": "entity mention 1",
  "head_type": "chemical",
  "tail_entity": "entity mention 2",
  "tail_type": "protein",
  "gold_relation": "CPR:4",
  "split": "train"
}
```

ChemProt 转换策略：

1. 从 raw ChemProt 文件读取 sentence/abstract、chemical mention、protein mention、relation annotation。
2. 枚举候选 chemical-protein entity pair。
3. 有标注关系的 pair 写入 `gold_relation=CPR:x`。
4. 无目标关系但需要负例的 pair 写入 `gold_relation=NO_RELATION`。
5. 输出到 `data/stage1/chemprot/train.jsonl`、`dev.jsonl`、`test.jsonl`。

DDI 2013 转换策略：

1. 从 DDI XML 读取 sentence、drug entities、pair annotations。
2. 对每个 drug-drug pair 输出一个 JSONL 样本。
3. `ddi=true` 的 pair 映射到 `mechanism/effect/advice/int`。
4. `ddi=false` 的 pair 映射到 `false`。
5. 输出到 `data/stage1/ddi2013/train.jsonl`、`dev.jsonl`、`test.jsonl`。

`relation_schema.yaml` 当前采用 JSON-compatible YAML，便于无 PyYAML 环境下用 Python stdlib 读取。每个 relation 至少包含：

```json
{
  "raw_label": "CPR:4",
  "label_words": "downregulator inhibitor",
  "relation_description": "a chemical decreases the activity or expression of a protein",
  "entity_type_aware_description": "given a chemical and a protein, the chemical inhibits or decreases the protein",
  "knowledge_enhanced_description": "the chemical inhibits, suppresses, reduces, blocks, or downregulates the protein activity or expression"
}
```

### B. Baseline Layer

必须支持的 schema/prompt variants：

| ID | semantic_field | 用途 |
|---|---|---|
| B1 | `raw_label` | raw label prompt |
| B2 | `label_words` | manual label words |
| P1 | `relation_description` | relation description prompt |
| P2 | `entity_type_aware_description` | entity-type-aware prompt |
| P3 | `knowledge_enhanced_description` | knowledge-enhanced prompt |

### C. Generative Baseline

接口要求：

1. 主线模型为 T5/BioBART 或同类 encoder-decoder/text-to-text 模型。
2. 输入模板读取 `relation_schema.yaml`。
3. 输出固定为 `relation: <label>`。
4. 解析预测结果后映射回 schema label。
5. 当前 smoke 使用 `Text2TextBaseline(backend="mock")`，不声称它是真实 T5 训练。

### D. RSG-BioRE Main Method

必须支持：

| ID | 模块 | 说明 |
|---|---|---|
| R1 | prototype only | 使用 relation semantic prototype，不加 alignment |
| R2 | prototype + alignment | 主方法，加入 `L_align` |
| R3 | guided decoder | 可选增强，等 R2 稳定后再做 |

核心模块：

1. `relation semantic encoder`: 编码 relation semantic text 得到 `z_i`。
2. `instance encoder`: 编码文本实体对得到 `h_x`。
3. `prototype memory`: `p_i = LayerNorm(W z_i + q_i)`。
4. `alignment loss`: `CrossEntropy(sim(h_x, p_i) / tau, y)`。
5. `generation loss`: decoder 生成 `relation: <label>` 的交叉熵。
6. `total loss`: `L = L_gen + lambda * L_align`。

当前 smoke 版本用纯 Python 向量器模拟上述接口，后续接入 PyTorch 后保持外部输出契约不变。

### E. Evaluation Layer

必须输出：

1. `micro_f1`
2. `macro_f1`
3. `precision`
4. `recall`
5. `per-class F1`
6. `rare_relation_macro_f1`
7. `valid_output_rate`
8. `relation_validity_rate`
9. `prototype_top1_accuracy`
10. `prototype_top3_accuracy`
11. `generation_vs_prototype_agreement`

默认主表排除 no-relation label，避免负类掩盖正类和 rare relation。

### F. Output Layer

每次实验输出：

```text
run_config.yaml
metrics.json
per_class_metrics.csv
predictions.jsonl
confusion_matrix.csv
error_cases.md
```

RSG-BioRE 额外输出：

```text
prototype_scores.jsonl
prototype_analysis.csv
```

## Smoke Commands

```powershell
python -m unittest discover -s tests/stage1 -v
python scripts/stage1/run_smoke.py --config configs/stage1/smoke_t5_baseline.yaml
python scripts/stage1/run_smoke.py --config configs/stage1/smoke_rsg_biore.yaml
```

## Next Engineering Steps

1. 准备真实 ChemProt raw files，补全 `scripts/stage1/convert_chemprot.py`。
2. 准备真实 DDI 2013 XML，补全 `scripts/stage1/convert_ddi2013.py`。
3. 增加 Hugging Face backend：`torch + transformers + sentencepiece`。
4. 增加 training loop、dev selection、checkpoint saving。
5. 跑 ChemProt smoke：B1、P1、R2。
6. 跑 ChemProt main：B1、B2、P1、P2、P3、R1、R2。
7. 趋势成立后迁移到 DDI 2013。
