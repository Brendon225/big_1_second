# 研究内容一实验指导指南：关系语义原型引导的生成式生物医学关系抽取方法

> 文档版本：v1.1  
> 适用论文：《基于生成式模型的生物医学关系抽取研究》  
> 适用阶段：研究内容一 / Stage 1  
> 面向对象：实验同学、论文作者、后续复现实验人员  
> 核心原则：不走依存分析路线；不依赖重型闭源大模型系统；优先做可复现、可消融、可解释的实验。

---

## 0. 文档用途

本指南用于指导实验同学完成“研究内容一：关系语义原型引导的生成式生物医学关系抽取方法”的实验验证。

实验同学读完本文档后，应能够独立完成以下工作：

1. 理解本阶段任务到底要验证什么。
2. 准备 ChemProt、DDI 2013、可选 BioRED 数据集。
3. 构造不同层级的 relation label semantic schema，并将其作为 relation semantic prototypes 的输入材料。
4. 搭建并运行 baseline、strong prompt baseline 与 RSG-BioRE proposed methods。
5. 输出统一格式的指标、预测结果和错误分析文件。
6. 将结果整理成论文第一章可用的实验表格与结论。

本阶段不是最终完整系统，也不是为了追求复杂模型结构。它的核心目标是回答一个清晰问题：

> 能否将抽象 relation label 编码为可学习的关系语义原型，并通过实例-关系语义对齐机制引导生成式模型更稳定地生成正确关系标签？

---

## 1. ISO 总体分析

### 1.1 Insight：核心矛盾

生物医学关系抽取通常要求模型判断两个实体之间是否存在某种语义关系，例如 chemical-protein interaction、drug-drug interaction、gene-disease relation 等。

传统分类式方法通常直接预测数据集给出的 relation label。例如：

- ChemProt 中的 `CPR:3`、`CPR:4`、`CPR:5`、`CPR:6`、`CPR:9`
- DDI 2013 中的 `mechanism`、`effect`、`advice`、`int`
- BioRED 中的多类型 biomedical relation label

这些 label 对人类研究者有定义，但对模型而言语义并不充分。特别是 `CPR:4`、`int` 这类标签，如果不提供解释，模型很难仅凭标签名称理解其医学含义。

本阶段的内部矛盾是：

1. 生成式模型或 prompt-based 模型需要自然语言语义提示；
2. BioRE 数据集中的原始标签往往抽象、缩写化、领域化；
3. 关系标签语义表达不充分会导致类别混淆、少数类召回低、跨数据集泛化弱。

外部约束是：

1. 本研究不再走 dependency parsing 路线；
2. 研究生论文需要方法可落地、可复现、可解释；
3. 不能把主实验建立在高成本闭源 LLM 或复杂 RAG 系统上；
4. 论文题目保持为《基于生成式模型的生物医学关系抽取研究》，因此最终必须有 T5/BioBART 等 text-to-text 生成式验证。

### 1.2 Strategy：研究策略

本阶段不直接从复杂生成系统开始，而是先把“关系标签如何表达”作为独立变量拆出来，再进一步把 relation description 从普通 prompt 文本升级为模型内部可学习、可对齐的 relation semantic prototypes。

我们设计一组从弱语义到强语义的 relation label 表达方式：

1. 原始标签：`CPR:4`
2. 短 label words：`downregulator`
3. 关系描述：`a chemical decreases the expression or activity of a protein`
4. 实体类型感知关系描述：`given a chemical and a protein, the chemical downregulates the protein`
5. 轻量知识增强描述：加入同义词、触发词、医学定义或知识库解释

然后在相同数据集、相同模型、相同训练设置下比较这些表达方式对 BioRE 的影响。需要注意：上述 schema/prompt 对比是必要基线，不是最终创新本身；本阶段主方法是 RSG-BioRE，即 Relation Semantic Prototype Guided Generative Biomedical Relation Extraction。

### 1.3 Operation：实验拆解

本阶段实验拆为四层：

1. 诊断层：PubMedBERT / BioBERT prompt classification  
   作用是控制变量，快速验证 relation semantics 是否有效。注意：这一层是判别式/encoder-only 基线，不是论文题目的最终生成式主方法。

2. 生成式主验证层：T5 / BioBART text-to-text  
   作用是支撑论文题目中的“生成式模型”，验证同一套 relation semantic schema 是否能提升生成式 BioRE。

3. 主方法层：RSG-BioRE  
   作用是把 relation description/entity type/knowledge 从输入 prompt 升级为 relation semantic prototypes，并使用 instance-prototype alignment loss 显式约束实例语义与关系语义的匹配。

4. 补充分析层：可选 LLM zero-shot/few-shot  
   作用是与近年 LLM prompting 文献对齐，但不作为主实验，因为成本与可复现性较弱。

---

## 2. 任务定义

### 2.1 标准 BioRE 任务

给定一段生物医学文本和一对候选实体：

```text
Sentence: In activated microglia, 15d-PGJ2 suppressed iNOS promoter activity, mRNA, and protein levels.
Entity 1: 15d-PGJ2
Entity 1 type: chemical
Entity 2: iNOS
Entity 2 type: protein
```

模型需要判断实体 1 和实体 2 之间的关系：

```text
Relation: CPR:4 / downregulator
```

### 2.2 本阶段研究任务

本阶段不是单纯研究“哪个模型更强”，而是研究：

> 如何构造更适合生成式模型理解的 BioRE relation semantics，并将其进一步建模为可学习的 relation semantic prototypes？

这里的 relation label semantic schema 指：

```json
{
  "raw_label": "CPR:4",
  "label_words": "downregulator",
  "relation_description": "a chemical decreases the activity or expression of a protein",
  "entity_type_aware_description": "given a chemical and a protein, the chemical downregulates the protein",
  "knowledge_enhanced_description": "the chemical inhibits, suppresses, reduces, or downregulates the protein's activity, expression, or production"
}
```

实验要验证：

1. 原始 label 是否不如自然语言 label words？
2. relation description 是否优于短 label words？
3. entity type 是否能减少标签歧义？
4. 轻量医学知识是否能提升少数类关系识别？
5. 这些语义增强是否对生成式 text-to-text BioRE 同样有效？
6. 将 relation semantics 编码为 prototype，并加入实例-关系语义对齐损失，是否比单纯把 description 拼接到 prompt 更稳定？

---

## 3. 相关工作主线

本阶段相关工作不需要继续展开 dependency parsing。实验同学只需理解以下几条主线。

### 3.1 Prompt tuning for BioRE

代表论文：

- He et al., 2024, *Prompt Tuning in Biomedical Relation Extraction*  
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11052745/

该论文将 BioRE 改写为 masked language modeling 任务，在 ChemProt 和 DDI 上比较 prompt tuning 与普通 fine-tuned PLMs。它证明 prompt template 和 label words 对 BioRE 有效。

对本研究的启发：

- 可作为 prompt classification baseline。
- 说明 label words 有用。
- 但它主要依赖人工短语，没有系统比较 relation description、entity type、knowledge-enhanced description。

### 3.2 LLM instruction/example prompting for BioRE

代表论文：

- Zhou et al., 2024, *LEAP: LLM instruction-example adaptive prompting framework for biomedical relation extraction*  
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11339510/

该论文在 DDI、ChemProt、BioRED 上研究 instruction、options、option descriptions、examples 对 LLM BioRE 的影响。

对本研究的启发：

- relation options 和 option descriptions 很重要。
- examples 对 LLM BioRE 帮助明显。
- 我们第一阶段可借鉴 option description 的思想，但不把复杂 LLM tuning 作为主实验。

### 3.3 Soft prompt / model tuning vs prompt tuning

代表论文：

- *Model Tuning or Prompt Tuning? A Study of LLMs for Clinical Concept and Relation Extraction*, JBI 2024  
  https://www.sciencedirect.com/science/article/pii/S1532046424000480

对本研究的启发：

- soft prompt / learnable prompt 可作为可选 method。
- 但第一阶段主创新点应放在可解释的 relation semantic prototype 与实例-关系语义对齐机制，而不是只比较不可解释的软向量或手工 prompt。

### 3.4 Directionality and entity-role-aware BioRE

代表论文：

- *Enhancing biomedical relation extraction with directionality*, Bioinformatics 2025  
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12261447/

对本研究的启发：

- BioRE 不只是关系类别，还包含 subject/object role、方向性和实体类型。
- 我们的 entity-type-aware relation description 可以看作对关系角色语义的轻量建模。

### 3.5 Knowledge-enhanced BioRE

代表论文：

- *Knowledge-augmented pre-trained language models for biomedical relation extraction*, 2025  
  https://link.springer.com/10.1186/s12859-025-06262-6

对本研究的启发：

- 知识增强可能有用，但不是越复杂越好。
- 本阶段只做轻量知识增强，不做完整知识图谱推理。

### 3.6 Ontology/schema grounding for LLM BioRE

代表论文：

- *RELATE: Relation Extraction in Biomedical Abstracts with LLMs and Ontology Constraints*, 2026  
  https://proceedings.mlr.press/v297/olasunkanmi26a.html

对本研究的启发：

- 近年趋势是让 LLM 输出关系后映射到标准 ontology/schema。
- 我们第一阶段做的是更基础、更可复现的 relation label semantic schema，为后续结构化生成和 schema-constrained decoding 铺垫。

### 3.7 Zero-shot BioRE benchmark

代表论文：

- *A benchmark for end-to-end zero-shot biomedical relation extraction with LLMs*, WASP 2025  
  https://aclanthology.org/2025.wasp-main.6/

对本研究的启发：

- LLM zero-shot 尚未稳定解决 BioRE。
- 因此主实验不应只依赖 LLM，而应使用可复现的开源模型。

---

## 4. 现有方法不足

本阶段要解决的不是“没有人做 prompt”，而是现有 prompt / LLM BioRE 仍存在以下不足。

### 4.1 label words 依赖人工经验

很多方法把关系类别映射为短 label words，例如：

```text
CPR:4 -> downregulator
DDI-mechanism -> mechanism
```

这种做法简单，但存在问题：

1. label words 怎么写缺少统一原则；
2. 同一关系可以有多个表达，性能可能波动；
3. 换一个数据集就需要重新设计；
4. 短 label words 对复杂生物医学关系解释不足。

### 4.2 relation description 研究不足

近年 LLM prompting 研究说明 option descriptions 有帮助，但多数 BioRE 工作没有系统比较：

1. 短 label words；
2. 完整 relation description；
3. 加入实体类型的 relation description；
4. 加入知识解释的 relation description。

这正是本阶段实验要填补的空缺。

### 4.3 实体类型和方向信息常被忽略

例如 `downregulator` 如果脱离实体类型，会比较模糊；但写成：

```text
given a chemical and a protein, the chemical decreases the activity or expression of the protein
```

语义就更明确。

DDI 中的 `mechanism`、`effect`、`advice` 也类似。如果不说明两个实体都是 drug，模型可能无法稳定理解 relation boundary。

### 4.4 外部知识增强容易过重

BioKnowPrompt、KG-enhanced、RAG-enhanced 方法可能引入知识库、检索、图谱、复杂推理。对研究生论文来说，工程成本和复现成本较高。

本阶段只做轻量知识增强：

1. 关系定义；
2. 同义词；
3. 常见触发词；
4. 典型表达；
5. 必要时加入 UMLS/MeSH/DrugBank/CTD 的简短概念说明。

### 4.5 生成式模型输出不稳定

生成式 BioRE 容易出现：

1. 生成非法关系标签；
2. 生成不在输入中的实体；
3. 输出格式不可解析；
4. 多关系漏抽；
5. no relation 与正类混淆。

第一阶段虽然不完全解决结构化输出问题，但 relation semantic schema 是后续 schema-constrained generation 的基础。

---

## 5. 我们的核心创新点

本阶段创新点要谨慎表述，避免夸大。

### 5.1 创新点一：关系语义原型引导的生成式 BioRE

本文不把 raw label、label words、relation description 的简单替换作为最终创新点，而是将它们作为关系语义输入材料，进一步编码为 relation semantic prototypes。

核心思想是：

1. 每个关系类别都有一个语义描述文本 `s_i`；
2. 使用 relation semantic encoder 将 `s_i` 编码为向量 `z_i`；
3. 将 `z_i` 与可学习关系原型向量 `q_i` 融合，得到关系语义原型 `p_i`；
4. 用 `p_i` 引导生成式模型判断当前实体对更接近哪个关系类别；
5. 最终仍由 T5/BioBART decoder 生成标准关系标签。

推荐方法命名：

```text
RSG-BioRE: Relation Semantic Prototype Guided Generative Biomedical Relation Extraction
中文：关系语义原型引导的生成式生物医学关系抽取方法
```

### 5.2 创新点二：实例-关系语义对齐损失

传统 prompt 方法通常只是把 description 拼接到输入文本中，模型是否真的学习了“实例语义”和“关系语义”的匹配并不明确。

本方法设计 instance-prototype alignment loss，使当前文本实体对表示 `h_x` 与真实关系原型 `p_y` 更接近，同时远离其他关系原型 `p_i`：

```text
L_align = - log exp(sim(h_x, p_y) / tau) / sum_i exp(sim(h_x, p_i) / tau)
```

最终训练目标：

```text
L = L_gen + lambda * L_align
```

其中：

1. `L_gen` 是生成式模型输出 `relation: <label>` 的交叉熵损失；
2. `L_align` 是实例-关系语义对齐损失；
3. `lambda` 是权重超参数，建议从 `[0.05, 0.1, 0.2, 0.5]` 中选择。

### 5.3 创新点三：可消融的关系语义构成机制

多层 relation semantic schema 仍然保留，但定位为“原型输入构成”和“消融变量”，而不是单独作为算法创新。

关系语义输入可以逐步扩展为：

1. raw label；
2. manual label words；
3. relation description；
4. entity-type-aware description；
5. knowledge-enhanced description。

这样可以回答导师最关心的问题：性能提升到底来自“换了一句 prompt”，还是来自“关系语义原型 + 对齐机制”这一模型层设计。

### 5.4 与论文题目的关系

本论文题目是《基于生成式模型的生物医学关系抽取研究》。

因此本阶段必须明确：

- PubMedBERT / BioBERT prompt classification 是诊断性基线，用于隔离 relation semantics 的影响；
- T5 / BioBART text-to-text 是生成式 baseline；
- RSG-BioRE 是研究内容一的生成式主方法；
- 最终论文表述中，不能把 PubMedBERT 说成生成式模型；
- 可以说 relation semantic schema 同时服务于 prompt-based classifier 和 generative text-to-text BioRE，但主贡献落在生成式 BioRE 的关系语义原型表示与实例-关系对齐机制上。

推荐论文表述：

> 本研究首先构建关系标签语义化表示，并通过判别式 prompt baseline 进行诊断性验证；随后在 T5/BioBART 等 text-to-text 生成式框架中，将关系语义描述编码为可学习的 relation semantic prototypes，并通过实例-关系语义对齐损失引导模型生成正确关系标签。

---

## 5A. RSG-BioRE 主方法设计

本节是实验同学实现主方法时必须理解的核心内容。

### 5A.1 输入

每条样本包含：

```json
{
  "text": "...",
  "head_entity": "15d-PGJ2",
  "head_type": "chemical",
  "tail_entity": "iNOS",
  "tail_type": "protein",
  "gold_relation": "CPR:4"
}
```

每个关系类别包含一段语义文本：

```json
{
  "label": "CPR:4",
  "semantic_text": "Given a chemical and a protein, the chemical decreases or inhibits the expression, amount, or activity of the protein."
}
```

### 5A.2 模块一：文本实体对编码器

使用 T5/BioBART encoder 对输入文本、实体标记和实体类型进行编码，得到样本实例表示：

```text
h_x = Pooling(Encoder(input_text))
```

实现建议：

1. 先使用 encoder 最后一层 hidden states 的 mean pooling；
2. 或者使用特殊标记 `<H>`、`<T>` 对应 hidden states 的拼接后投影；
3. 第一版优先 mean pooling，降低实现复杂度。

### 5A.3 模块二：关系语义编码器

使用同一个 T5/BioBART encoder 或一个共享参数 encoder 编码每个关系描述：

```text
z_i = Pooling(Encoder(semantic_text_i))
```

第一版建议共享 encoder，避免引入额外大模型。

### 5A.4 模块三：关系语义原型记忆

将关系描述向量 `z_i` 转换为关系原型 `p_i`：

```text
p_i = LayerNorm(W z_i + q_i)
```

其中：

1. `W` 是线性投影层；
2. `q_i` 是每个关系类别对应的可学习向量；
3. `p_i` 是最终用于匹配和引导生成的 relation semantic prototype。

第一版可以实现两个版本：

1. `prototype_static`: `p_i = LayerNorm(W z_i)`；
2. `prototype_learnable`: `p_i = LayerNorm(W z_i + q_i)`。

### 5A.5 模块四：实例-关系语义对齐

计算实例表示 `h_x` 与所有关系原型 `p_i` 的相似度：

```text
score_i = cosine(h_x, p_i) / tau
```

对 gold relation `y` 使用交叉熵：

```text
L_align = CrossEntropy(score, y)
```

这等价于让样本向量靠近正确关系原型、远离错误关系原型。

### 5A.6 模块五：生成式关系输出

decoder 输出固定格式：

```text
relation: CPR:4
```

生成损失：

```text
L_gen = CrossEntropy(decoder_output, target_text)
```

总损失：

```text
L = L_gen + lambda * L_align
```

### 5A.7 可选增强：prototype-guided decoder

如果 V2 已经跑通，可以尝试 V3：

```text
g = softmax(score)
r = sum_i g_i * p_i
decoder_input = decoder_input + Project(r)
```

或者把 top-1 prototype 对应的 relation semantic text 加入 decoder prompt。

注意：V3 是增强实验，不是第一阶段必须完成项。最小可行主方法是 V2：prototype + alignment。

---

## 6. 总体实验设计

### 6.1 实验总目标

验证 relation semantic prototypes 与 instance-prototype alignment 是否能在生成式 BioRE 中带来稳定收益，并分析收益是否来自关系语义表示、实体类型/方向信息、轻量知识信息和对齐损失。

### 6.2 核心假设

| 编号 | 假设 |
|---|---|
| H1 | 原始 relation label 语义不足，性能低于自然语言 label words。 |
| H2 | 完整 relation description 优于短 label words，尤其能提升 macro-F1。 |
| H3 | entity-type-aware description 能减少关系混淆，提高少数类 per-class F1。 |
| H4 | 轻量 knowledge-enhanced description 能提升 rare relation recall，但可能引入噪声。 |
| H5 | 同一套 relation semantic schema 在 T5/BioBART text-to-text 生成式模型中仍然有效。 |
| H6 | relation semantic prototype 优于单纯 description prompt，说明关系语义作为模型内部表示有价值。 |
| H7 | instance-prototype alignment loss 能提升 macro-F1、rare relation F1 或生成输出合法率。 |

### 6.3 实验层级

| 层级 | 模型 | 目的 |
|---|---|---|
| L0 | majority / simple classifier | 检查数据与评估脚本是否正常 |
| L1 | PubMedBERT / BioBERT classifier | 传统判别式 baseline |
| L2 | PubMedBERT / BioBERT prompt classification | 诊断 relation label semantics |
| L3 | T5 / BioBART text-to-text | 生成式强基线 |
| L4 | RSG-BioRE | 研究内容一主方法 |
| L5 可选 | LLM zero/few-shot | 近年文献对齐和案例分析 |

---

## 7. 数据集选择

### 7.1 主数据集一：ChemProt

推荐作为第一主数据集。

#### 任务特点

ChemProt 是 chemical-protein relation extraction 数据集，常用正类关系包括：

| Label | 关系含义 |
|---|---|
| CPR:3 | upregulator / activator / inducer |
| CPR:4 | downregulator / inhibitor |
| CPR:5 | agonist |
| CPR:6 | antagonist |
| CPR:9 | substrate / product |
| No relation | 无目标关系 |

#### 为什么适合本阶段

1. `CPR:x` 标签高度抽象，非常适合研究 verbalization。
2. chemical-protein 类型固定，适合 entity-type-aware description。
3. 类别不平衡明显，适合分析 rare relation。
4. 近年 Prompt Tuning、LEAP 等论文均使用 ChemProt，方便对齐。

### 7.2 主数据集二：DDI 2013

推荐作为第二主数据集。

#### 任务特点

DDI 2013 是 drug-drug interaction extraction 数据集，常用关系包括：

| Label | 关系含义 |
|---|---|
| mechanism | 一个药物影响另一个药物的药代/药效机制 |
| effect | 两个药物共同使用产生某种效果或不良反应 |
| advice | 文本中给出联合用药建议或警告 |
| int | 存在 interaction，但类型不够具体 |
| No relation | 无药物相互作用 |

#### 为什么适合本阶段

1. 两个实体都是 drug，便于构造 drug-drug description。
2. `int` 标签语义较抽象，适合测试 relation description。
3. 类别不平衡，适合做 rare relation analysis。
4. 与 ChemProt 互补：一个是 chemical-protein，一个是 drug-drug。

### 7.3 扩展数据集：BioRED

BioRED 建议作为扩展实验，不建议作为第一阶段唯一主数据集。

#### 任务特点

BioRED 是文档级、多实体类型、多关系类型数据集，包含 genes/proteins、diseases、chemicals 等实体类型及多种关系。

#### 使用建议

1. 第一轮可以只做 ChemProt + DDI。
2. 当主实验跑通后，再用 BioRED 做小规模扩展验证。
3. BioRED 的 label、novelty、directionality 设置必须以官方数据文件为准。
4. 如果实验资源有限，BioRED 可以只跑最关键的 3 组：raw label、relation description、entity-type-aware description。

---

## 8. 数据预处理要求

所有数据集预处理后必须转成统一 JSONL 格式。

### 8.1 实例格式

每一行是一个候选实体对实例：

```json
{
  "id": "ChemProt_train_000001",
  "dataset": "ChemProt",
  "split": "train",
  "text": "In activated microglia, 15d-PGJ2 suppressed iNOS promoter activity, mRNA, and protein levels.",
  "head_entity": "15d-PGJ2",
  "head_type": "chemical",
  "tail_entity": "iNOS",
  "tail_type": "protein",
  "gold_label": "CPR:4",
  "is_positive": true
}
```

DDI 示例：

```json
{
  "id": "DDI_train_000001",
  "dataset": "DDI2013",
  "split": "train",
  "text": "Concomitant use of DrugA and DrugB may increase the risk of bleeding.",
  "head_entity": "DrugA",
  "head_type": "drug",
  "tail_entity": "DrugB",
  "tail_type": "drug",
  "gold_label": "effect",
  "is_positive": true
}
```

### 8.2 预处理规则

1. 保留原始文本。
2. 为目标实体加入实体标记或占位符。
3. 每个候选实体对生成一条 instance。
4. 正类使用数据集 gold relation。
5. 负类统一标记为 `No relation`。
6. train/dev/test 划分必须使用官方划分，除非数据集没有官方划分。
7. 所有实验使用同一份预处理文件，避免不同方法之间数据不一致。

### 8.3 推荐目录结构

```text
data/
  stage1/
    chemprot/
      train.jsonl
      dev.jsonl
      test.jsonl
      relation_schema.yaml
    ddi2013/
      train.jsonl
      dev.jsonl
      test.jsonl
      relation_schema.yaml
    biored/
      train.jsonl
      dev.jsonl
      test.jsonl
      relation_schema.yaml
```

---

## 9. Relation Semantic Schema 构造

每个数据集都需要一个 `relation_schema.yaml` 文件。

### 9.1 schema 字段

```yaml
dataset: ChemProt
relations:
  - raw_label: CPR:4
    label_words: downregulator
    relation_description: "A chemical decreases the expression, activity, or amount of a protein."
    entity_type_aware_description: "Given a chemical and a protein, the chemical decreases the expression, activity, or amount of the protein."
    knowledge_enhanced_description: "The chemical inhibits, suppresses, reduces, blocks, or downregulates the protein's activity, expression, production, or function."
```

字段解释：

| 字段 | 含义 | 是否必须 |
|---|---|---|
| raw_label | 数据集原始标签 | 必须 |
| label_words | 短自然语言标签 | 必须 |
| relation_description | 完整关系定义 | 必须 |
| entity_type_aware_description | 带实体类型和方向的关系定义 | 必须 |
| knowledge_enhanced_description | 加入同义词/触发词/轻量知识的关系解释 | 必须 |

### 9.2 ChemProt 初始 schema 草案

实验同学可以先使用以下 schema，后续根据官方定义微调。

```yaml
dataset: ChemProt
relations:
  - raw_label: CPR:3
    label_words: upregulator
    relation_description: "A chemical increases the expression, activity, or amount of a protein."
    entity_type_aware_description: "Given a chemical and a protein, the chemical increases or activates the protein."
    knowledge_enhanced_description: "The chemical activates, induces, stimulates, increases, or upregulates the protein's expression, production, activity, or function."

  - raw_label: CPR:4
    label_words: downregulator
    relation_description: "A chemical decreases the expression, activity, or amount of a protein."
    entity_type_aware_description: "Given a chemical and a protein, the chemical decreases or inhibits the protein."
    knowledge_enhanced_description: "The chemical inhibits, suppresses, reduces, blocks, or downregulates the protein's expression, production, activity, or function."

  - raw_label: CPR:5
    label_words: agonist
    relation_description: "A chemical acts as an agonist of a protein or receptor."
    entity_type_aware_description: "Given a chemical and a protein, the chemical binds to or activates the protein as an agonist."
    knowledge_enhanced_description: "The chemical functions as an agonist, activating or enhancing the biological activity of the protein or receptor."

  - raw_label: CPR:6
    label_words: antagonist
    relation_description: "A chemical acts as an antagonist of a protein or receptor."
    entity_type_aware_description: "Given a chemical and a protein, the chemical blocks or counteracts the protein as an antagonist."
    knowledge_enhanced_description: "The chemical functions as an antagonist, blocking, opposing, or reducing the biological activity of the protein or receptor."

  - raw_label: CPR:9
    label_words: substrate_or_product
    relation_description: "A chemical is a substrate or product in a reaction involving a protein."
    entity_type_aware_description: "Given a chemical and a protein, the chemical is a substrate or product of a reaction catalyzed or mediated by the protein."
    knowledge_enhanced_description: "The chemical participates as a substrate, product, or reaction participant in a biochemical process involving the protein or enzyme."

  - raw_label: No relation
    label_words: no_relation
    relation_description: "The two entities do not express any target relation in the given text."
    entity_type_aware_description: "Given the entity pair, the text does not state any target relation between them."
    knowledge_enhanced_description: "The text does not provide evidence that the entities have one of the predefined biomedical relations."
```

### 9.3 DDI 2013 初始 schema 草案

```yaml
dataset: DDI2013
relations:
  - raw_label: mechanism
    label_words: mechanism
    relation_description: "One drug changes the pharmacological mechanism or metabolism of another drug."
    entity_type_aware_description: "Given two drugs, one drug affects the mechanism, metabolism, absorption, distribution, or clearance of the other drug."
    knowledge_enhanced_description: "The interaction describes a pharmacokinetic or pharmacodynamic mechanism, such as altered metabolism, enzyme inhibition, absorption, clearance, or concentration."

  - raw_label: effect
    label_words: effect
    relation_description: "The combined use of two drugs causes a specific effect or adverse outcome."
    entity_type_aware_description: "Given two drugs, their co-administration leads to a clinical effect, therapeutic effect, or adverse reaction."
    knowledge_enhanced_description: "The interaction describes increased toxicity, reduced efficacy, adverse effects, side effects, or other observable clinical outcomes caused by using the drugs together."

  - raw_label: advice
    label_words: advice
    relation_description: "The text provides advice, warning, or recommendation about using two drugs together."
    entity_type_aware_description: "Given two drugs, the text recommends, warns against, avoids, monitors, or adjusts their combined use."
    knowledge_enhanced_description: "The interaction is expressed as clinical guidance, such as avoid combination, use caution, monitor patients, adjust dosage, or contraindicated use."

  - raw_label: int
    label_words: interaction
    relation_description: "The text states that two drugs interact, but the specific mechanism, effect, or advice is not clearly specified."
    entity_type_aware_description: "Given two drugs, the text indicates an interaction between them without specifying whether it is mechanism, effect, or advice."
    knowledge_enhanced_description: "The text reports a drug-drug interaction in a general way, using vague evidence such as interacts with or interaction occurs, without enough detail for a more specific type."

  - raw_label: No relation
    label_words: no_relation
    relation_description: "The two drugs do not express a target drug-drug interaction in the given text."
    entity_type_aware_description: "Given the two drugs, the text does not state any target interaction between them."
    knowledge_enhanced_description: "The text does not provide evidence for mechanism, effect, advice, or general interaction between the two drugs."
```

---

## 10. 实验组设计

### 10.1 总览

所有实验组尽量保持数据划分、生成基座、训练轮数、随机种子一致。前半部分只改变 relation semantic schema，用于建立强基线；后半部分加入 relation semantic prototype 和 alignment loss，用于验证模型层创新。

| 实验编号 | 名称 | 模型/机制 | 使用的 relation 表达 | 目的 |
|---|---|---|---|---|
| B0 | classifier baseline | PubMedBERT classifier | raw label | 传统判别式 baseline |
| B1 | raw label prompt | PubMedBERT/T5/BioBART prompt | raw label | 验证直接使用原始标签的效果 |
| B2 | manual label words | PubMedBERT/T5/BioBART prompt | label_words | 复现/对齐已有 prompt tuning |
| P1 | relation description prompt | T5/BioBART prompt | relation_description | 强 prompt baseline |
| P2 | entity-type-aware prompt | T5/BioBART prompt | entity_type_aware_description | 验证实体类型和方向信息 |
| P3 | knowledge-enhanced prompt | T5/BioBART prompt | knowledge_enhanced_description | 验证轻量知识增强 |
| R1 | RSG prototype only | T5/BioBART + prototype | entity_type_aware 或 knowledge_enhanced | 验证原型表示是否优于拼接 prompt |
| R2 | RSG prototype + alignment | T5/BioBART + prototype + `L_align` | entity_type_aware 或 knowledge_enhanced | 第一阶段主方法 |
| R3 可选 | RSG guided decoder | T5/BioBART + prototype + `L_align` + guided decoder | entity_type_aware 或 knowledge_enhanced | 可选增强，验证原型引导生成 |
| M4 可选 | soft verbalizer / soft prompt | learnable prompt | learnable prompt | 验证可学习 prompt 上界 |

### 10.2 诊断层实验：PubMedBERT prompt classification

注意：这是判别式/encoder-only 诊断基线，不是最终生成式主模型。

#### 输入模板

```text
Sentence: {marked_text}
Head entity: {head_entity}
Head type: {head_type}
Tail entity: {tail_entity}
Tail type: {tail_type}
Relation candidates: {relation_semantic_text}
Question: What is the relation between the head entity and the tail entity?
Answer: [MASK]
```

不同实验组中，`relation_semantic_text` 分别使用 raw label、label words、relation description 等。

#### 输出

模型输出某个 label 或 label word，然后映射回标准 relation label。

### 10.3 生成式基线实验：T5 / BioBART text-to-text

这是本阶段支撑论文题目的生成式强基线，也是 RSG-BioRE 必须超过或至少显著改善部分指标的参照。

#### 输入模板

```text
Task: Extract the biomedical relation between the given head entity and tail entity.

Text: {marked_text}
Head entity: {head_entity}
Head type: {head_type}
Tail entity: {tail_entity}
Tail type: {tail_type}

Relation schema:
{relation_semantic_schema_text}

Output one relation label from the schema.
```

#### 输出格式

推荐使用固定格式：

```text
relation: CPR:4
```

或者 JSON 格式：

```json
{"relation": "CPR:4"}
```

第一阶段建议先使用简单文本格式 `relation: <label>`，降低解析难度。

### 10.4 RSG-BioRE 主实验

#### R1：prototype only

目标：验证 relation semantic prototype 本身是否比单纯 prompt 拼接更有效。

训练目标：

```text
L = L_gen
```

区别：模型内部仍构造 `p_i`，但不使用 `L_align`。可将 top-k prototype scores 作为辅助特征，或仅在 decoder prompt 中加入最相近 prototype 的关系描述。

#### R2：prototype + alignment，主方法

目标：验证实例-关系语义对齐损失是否有效。

训练目标：

```text
L = L_gen + lambda * L_align
```

推荐设置：

```yaml
lambda: [0.05, 0.1, 0.2]
tau: 0.07 或 0.1
prototype_type: learnable
semantic_field: entity_type_aware_description
```

优先跑：

1. ChemProt + BioBART/T5 + R2；
2. DDI + BioBART/T5 + R2；
3. 如果 R2 相比 P2/P3 有提升，再补多种子和 BioRED。

#### R3：prototype + alignment + guided decoder，可选

目标：验证关系原型不仅能做对齐，还能直接引导生成。

可选实现之一：

```text
prototype_context = weighted_sum(score_i, p_i)
decoder_hidden = decoder_hidden + Project(prototype_context)
```

如果实现复杂，可以只做简化版：

```text
把 top-1/top-3 prototype 对应的 relation descriptions 加入输入 prompt
```

R3 不作为最低成功标准。

### 10.5 LLM 补充实验，可选

如果需要与 LEAP、zero-shot benchmark 对齐，可以做少量 LLM 测试。

推荐只做：

1. zero-shot raw label；
2. zero-shot relation description；
3. few-shot relation description + 1 example per class。

LLM 实验只做案例与补充，不进入主结果表的核心对比。

---

## 11. 训练与运行设置

### 11.1 随机种子

每个主要实验至少跑 3 个随机种子：

```text
seed = 42, 43, 44
```

如果算力有限，先跑 seed 42，确认趋势后再补多种子。

### 11.2 模型建议

| 层级 | 推荐模型 |
|---|---|
| 传统分类 baseline | PubMedBERT、BioBERT |
| prompt classification | PubMedBERT、BioBERT |
| text-to-text generation | T5-base、BioBART |
| RSG-BioRE | T5-base、BioBART + prototype/alignment modules |
| 可选 LLM | 开源 Llama/MedLLaMA 或 API 模型，仅做补充 |

### 11.3 超参数建议

#### PubMedBERT / BioBERT

```yaml
max_seq_length: 256
batch_size: 16 或 32
learning_rate: [2e-5, 3e-5, 5e-5]
epochs: 5 到 15
early_stopping_patience: 3
seed: [42, 43, 44]
```

#### T5 / BioBART

```yaml
max_input_length: 512
max_output_length: 32
batch_size: 8 或 16
learning_rate: [1e-4, 5e-5, 3e-5]
epochs: 5 到 10
beam_size: 1 或 4
seed: [42, 43, 44]
```

#### RSG-BioRE

```yaml
base_model: T5-base 或 BioBART
prototype_dim: 与 encoder hidden size 一致
prototype_type: static 或 learnable
semantic_field: entity_type_aware_description 或 knowledge_enhanced_description
alignment_lambda: [0.05, 0.1, 0.2]
temperature_tau: [0.07, 0.1]
pooling: mean_pooling
max_input_length: 512
max_relation_text_length: 128
batch_size: 8 或 16
learning_rate: [5e-5, 3e-5, 1e-4]
epochs: 5 到 10
seed: [42, 43, 44]
```

第一轮建议只固定：

```yaml
prototype_type: learnable
semantic_field: entity_type_aware_description
alignment_lambda: 0.1
temperature_tau: 0.1
```

确认主趋势后再做 `lambda` 和 `semantic_field` 消融。

### 11.4 训练选择标准

1. 使用 dev set 选择最佳 checkpoint。
2. 主指标优先看 dev macro-F1。
3. 若 macro-F1 接近，则看 rare relation F1。
4. 最终只在 test set 上评估一次，避免反复调 test。

---

## 12. 评价指标

### 12.1 主指标

必须报告：

1. micro-F1；
2. macro-F1；
3. precision；
4. recall；
5. per-class F1。

### 12.2 positive-class 评价

主表建议排除 `No relation`，只统计正类关系。

原因：

1. `No relation` 数量通常很大；
2. 包含 `No relation` 会掩盖少数正类表现；
3. BioRE 论文通常更关心目标关系识别能力。

但附录可以报告包含 `No relation` 的完整指标。

### 12.3 rare relation performance

ChemProt 中重点关注：

1. `CPR:5`
2. `CPR:6`

DDI 中重点关注：

1. `int`
2. `advice`

需要单独输出：

```text
rare_relation_macro_f1
rare_relation_recall
rare_relation_precision
```

### 12.4 生成式附加指标

T5 / BioBART 必须额外报告：

| 指标 | 含义 |
|---|---|
| valid_output_rate | 输出能否被解析 |
| relation_validity_rate | 输出关系是否属于 schema |
| exact_label_match | 生成 label 是否与 gold label 一致 |
| invalid_label_count | 生成了多少非法 label |
| empty_output_count | 空输出数量 |

### 12.5 RSG-BioRE 附加分析指标

RSG-BioRE 需要额外输出：

| 指标 | 含义 |
|---|---|
| prototype_top1_accuracy | `argmax_i sim(h_x, p_i)` 是否等于 gold relation |
| prototype_top3_accuracy | gold relation 是否在 top-3 prototype 中 |
| alignment_loss_dev | dev set 上的 `L_align` |
| generation_vs_prototype_agreement | decoder 生成标签与 prototype top-1 是否一致 |
| confusion_by_prototype | 按 prototype 相似度统计的类别混淆 |

这些指标不一定写入主结果表，但必须用于判断 RSG-BioRE 是否真的学到了关系语义对齐。

---

## 13. 消融实验

### 13.1 relation semantics 消融

第一组消融用于验证 relation semantic text 的构成。

| 组别 | raw label | label words | description | entity type | knowledge |
|---|---|---|---|---|---|
| B1 | yes | no | no | no | no |
| B2 | no | yes | no | no | no |
| P1 | no | no | yes | no | no |
| P2 | no | no | yes | yes | no |
| P3 | no | no | yes | yes | yes |

### 13.2 RSG-BioRE 模型模块消融

第二组消融用于验证模型层创新。

| 组别 | relation prototype | learnable `q_i` | `L_align` | guided decoder | 目的 |
|---|---|---|---|---|---|
| P2 | no | no | no | no | 强 prompt baseline |
| R1-static | yes | no | no | no | 验证静态 description prototype |
| R1-learnable | yes | yes | no | no | 验证可学习 prototype |
| R2-static-align | yes | no | yes | no | 验证 alignment 是否有效 |
| R2-learnable-align | yes | yes | yes | no | 主方法 |
| R3-guided | yes | yes | yes | yes | 可选增强 |

最低必须完成：

1. P2；
2. R1-learnable；
3. R2-learnable-align。

如果 R2 明显优于 P2 和 R1，说明创新点可以成立。

### 13.3 description 长度敏感性

对 P1/P2/P3 或 R2 的 semantic text 额外测试三种长度：

1. short：一句话，10-20 words；
2. medium：一句完整定义，20-40 words；
3. long：定义 + 同义词 + 触发词，40-80 words。

目的：

验证描述越长是否越好。预期不一定越长越好，过长可能引入噪声。

### 13.4 prompt robustness

至少测试两种 prompt 顺序：

版本 A：

```text
Text -> Entities -> Relation schema -> Question
```

版本 B：

```text
Relation schema -> Text -> Entities -> Question
```

如果性能差异很大，说明方法对 prompt 顺序敏感，需要在论文中说明。

### 13.5 prototype sensitivity

至少测试以下设置：

1. `semantic_field = relation_description`
2. `semantic_field = entity_type_aware_description`
3. `semantic_field = knowledge_enhanced_description`
4. `prototype_type = static`
5. `prototype_type = learnable`

目的：

验证 RSG-BioRE 是否依赖某一种手工描述。如果模型只在某一句 prompt 上有效，创新说服力会下降；如果在 entity-type-aware 和 knowledge-enhanced 两种语义输入下均稳定，则说明原型与对齐机制更可靠。

### 13.6 few-shot 实验

建议使用：

```text
K = 1, 8, 16
```

每个 relation class 抽取 K 个训练样本，dev set 同样每类 K 个，test set 使用完整测试集。

目的：

验证 relation semantic prototype 与 alignment loss 是否在低资源场景更有价值。

---

## 14. 输出文件要求

每次实验必须保存独立目录。

### 14.1 目录命名

```text
outputs/stage1/
  chemprot/
    pubmedbert_prompt/
      B1_raw_label_seed42/
      B2_label_words_seed42/
      P1_relation_description_seed42/
      P2_entity_type_description_seed42/
      P3_knowledge_description_seed42/
    t5_text2text/
      B1_raw_label_seed42/
      P1_relation_description_seed42/
      P2_entity_type_description_seed42/
      P3_knowledge_description_seed42/
    rsg_biore/
      R1_prototype_only_seed42/
      R2_prototype_alignment_seed42/
  ddi2013/
    ...
```

### 14.2 每个实验目录必须包含

```text
run_config.yaml
metrics.json
per_class_metrics.csv
predictions.jsonl
confusion_matrix.csv
error_cases.md
train_log.txt
```

### 14.3 run_config.yaml 示例

```yaml
experiment_id: chemprot_rsg_R2_seed42
dataset: ChemProt
model: t5-base
method: R2_prototype_alignment
relation_schema_field: entity_type_aware_description
prototype_type: learnable
alignment_lambda: 0.1
temperature_tau: 0.1
seed: 42
train_file: data/stage1/chemprot/train.jsonl
dev_file: data/stage1/chemprot/dev.jsonl
test_file: data/stage1/chemprot/test.jsonl
schema_file: data/stage1/chemprot/relation_schema.yaml
max_input_length: 512
max_output_length: 32
batch_size: 8
learning_rate: 5e-5
epochs: 10
evaluation_positive_only: true
```

### 14.4 metrics.json 示例

```json
{
  "experiment_id": "chemprot_rsg_R2_seed42",
  "dataset": "ChemProt",
  "model": "t5-base",
  "method": "R2_prototype_alignment",
  "micro_f1": 0.0,
  "macro_f1": 0.0,
  "precision": 0.0,
  "recall": 0.0,
  "rare_relation_macro_f1": 0.0,
  "valid_output_rate": 0.0,
  "relation_validity_rate": 0.0,
  "prototype_top1_accuracy": 0.0,
  "generation_vs_prototype_agreement": 0.0
}
```

### 14.5 predictions.jsonl 示例

```json
{"id":"ChemProt_test_000001","gold_label":"CPR:4","pred_label":"CPR:4","raw_output":"relation: CPR:4","valid_output":true}
{"id":"ChemProt_test_000002","gold_label":"CPR:5","pred_label":"CPR:4","raw_output":"relation: CPR:4","valid_output":true}
```

### 14.6 error_cases.md 模板

```markdown
# Error Cases

## Summary

- Dataset:
- Model:
- Method:
- Seed:

## Error Type Counts

| Error Type | Count | Description |
|---|---:|---|
| label_confusion | 0 | 正类之间混淆 |
| false_positive | 0 | No relation 误判为正类 |
| false_negative | 0 | 正类误判为 No relation |
| invalid_generation | 0 | 生成非法 label |
| rare_relation_miss | 0 | 少数类漏召回 |

## Representative Cases

### Case 1

- id:
- text:
- head entity:
- tail entity:
- gold:
- prediction:
- relation schema used:
- analysis:
```

---

## 15. 实验执行顺序

实验同学按以下顺序执行，不要一开始就跑所有模型。

### Step 1：数据预处理检查

1. 准备 ChemProt 数据。
2. 准备 DDI 2013 数据。
3. 转成统一 JSONL。
4. 检查每个 split 的样本数、正负类比例、每类数量。
5. 输出 `dataset_statistics.md`。

### Step 2：relation schema 构造

1. 为 ChemProt 写 `relation_schema.yaml`。
2. 为 DDI 2013 写 `relation_schema.yaml`。
3. 检查每个 relation 是否有五个字段。
4. 确保所有实验使用同一份 schema。

### Step 3：跑最小 smoke test

只跑：

```text
ChemProt + T5 + B1_raw_label + seed42
ChemProt + T5 + P1_relation_description + seed42
ChemProt + RSG-BioRE + R2_prototype_alignment + seed42
```

目的：

1. 确认训练代码能跑通；
2. 确认输出能解析；
3. 确认指标脚本正常。

### Step 4：跑 ChemProt 主实验

先跑 T5/BioBART 生成式主验证：

```text
B1 raw label
B2 label words
P1 relation description
P2 entity-type-aware description
P3 knowledge-enhanced description
R1 prototype only
R2 prototype + alignment
```

每组先 seed42；趋势成立后补 seed43、seed44。

### Step 5：跑 DDI 主实验

同 ChemProt。

### Step 6：跑 PubMedBERT 诊断实验

目的不是支撑生成式标题，而是验证 relation semantics 是否在 encoder-only prompt classification 下也有效。

### Step 7：few-shot 实验

优先跑：

```text
ChemProt + T5 + B2/P1/P2/P3 + K=1/8/16
ChemProt + RSG-BioRE + R2 + K=1/8/16
DDI + T5 + B2/P1/P2/P3 + K=1/8/16
DDI + RSG-BioRE + R2 + K=1/8/16
```

### Step 8：BioRED 扩展实验，可选

只跑最小三组：

```text
B1 raw label
P2 entity-type-aware description
R2 prototype + alignment
```

### Step 9：汇总结果

生成：

```text
checkpoints/stage1_summary/
  main_results.csv
  main_results.md
  per_class_summary.csv
  rare_relation_summary.csv
  prompt_sensitivity_summary.md
  error_analysis_summary.md
```

---

## 16. 结果表格模板

### 16.1 主结果表

| Dataset | Model | Method | Micro-F1 | Macro-F1 | Precision | Recall | Rare-F1 | Valid Output |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ChemProt | T5 | B1 raw label |  |  |  |  |  |  |
| ChemProt | T5 | B2 label words |  |  |  |  |  |  |
| ChemProt | T5 | P1 description |  |  |  |  |  |  |
| ChemProt | T5 | P2 entity type |  |  |  |  |  |  |
| ChemProt | T5 | P3 knowledge |  |  |  |  |  |  |
| ChemProt | RSG-BioRE | R1 prototype only |  |  |  |  |  |  |
| ChemProt | RSG-BioRE | R2 prototype + alignment |  |  |  |  |  |  |

### 16.2 per-class F1 表

| Dataset | Model | Method | CPR:3 | CPR:4 | CPR:5 | CPR:6 | CPR:9 |
|---|---|---|---:|---:|---:|---:|---:|
| ChemProt | T5 | B1 |  |  |  |  |  |
| ChemProt | T5 | P2 |  |  |  |  |  |
| ChemProt | RSG-BioRE | R2 |  |  |  |  |  |

### 16.3 消融表

| Dataset | Model | Method | Description | Entity Type | Knowledge | Prototype | Align | Macro-F1 | Rare-F1 |
|---|---|---|---|---|---|---|---|---:|---:|
| ChemProt | T5 | B2 | no | no | no | no | no |  |  |
| ChemProt | T5 | P1 | yes | no | no | no | no |  |  |
| ChemProt | T5 | P2 | yes | yes | no | no | no |  |  |
| ChemProt | T5 | P3 | yes | yes | yes | no | no |  |  |
| ChemProt | RSG-BioRE | R1 | yes | yes | no | yes | no |  |  |
| ChemProt | RSG-BioRE | R2 | yes | yes | no | yes | yes |  |  |

---

## 17. 成功标准

本阶段不要求每个数据集都显著超过所有 SOTA。成功标准应设为可验证、可写论文。

### 17.1 最低成功标准

满足以下条件即可支撑论文第一章：

1. P1/P2/P3 中至少一种方法在 ChemProt 或 DDI 上优于 B1/B2，说明 relation semantic text 有价值。
2. 至少一个数据集上 macro-F1 或 rare relation F1 有明显提升。
3. 能通过 per-class 分析说明哪些 relation 受益。
4. R2 在至少一个主数据集上优于 P2 或 P3，说明 relation semantic prototype + alignment 不只是 prompt wording。
5. 能通过 prototype_top1_accuracy、generation_vs_prototype_agreement 或消融实验说明模型确实学习了实例-关系语义对齐。
6. T5/BioBART 生成式实验能复现同方向趋势。

### 17.2 理想成功标准

1. P2 在 ChemProt 和 DDI 上均优于 B2。
2. R2 在 ChemProt 和 DDI 上均优于 P2/P3。
3. R2 对 rare relation recall 或 macro-F1 有提升。
4. R2 的 valid output rate 高于 raw label 设置。
5. BioRED 扩展实验中 R2 优于 description prompt baseline。

### 17.3 如果结果不理想

如果 P3 不如 P2，不代表失败。可以解释为：

> 轻量知识增强可能引入冗余信息或噪声；相比之下，entity-type-aware relation description 在可解释性和稳定性之间取得更好平衡。

如果 R2 只提升 macro-F1 不提升 micro-F1，也可以接受，因为说明该方法更有利于少数类关系。

如果 R2 不如 P2/P3，需要检查：

1. `lambda` 是否过大导致生成目标被干扰；
2. prototype pooling 是否过粗；
3. relation semantic text 是否过长或噪声过多；
4. negative relation / no relation 是否主导了对齐损失。

---

## 18. 风险与规避

### 18.1 被认为只是 prompt engineering

规避方式：

1. 固定模型和训练设置；
2. 把 prompt/schema 对比作为 baseline，不作为最终创新；
3. 主方法必须包含 relation semantic prototype、learnable `q_i` 和 `L_align`；
4. 做 P2 vs R1 vs R2 的模块消融；
5. 报告 prototype_top1_accuracy、generation_vs_prototype_agreement 等原型分析指标；
6. 把方法表述为 relation semantic prototype guided generation，而不是简单 prompt 调参。

### 18.2 与已有 Prompt Tuning / LEAP 重复

规避方式：

1. Prompt Tuning 主要证明 prompt tuning 有效，我们研究 label semantics 的层级化设计；
2. LEAP 主要研究 LLM instruction/example adaptive prompting，我们主实验用可复现开源模型；
3. 我们不止比较 raw label -> description -> entity-type-aware -> knowledge-enhanced，而是进一步提出 prototype memory 和 instance-prototype alignment；
4. 论文中明确将 BioKnowPrompt、LEAP、RELATE 等作为相关工作和强基线思想来源，不宣称这些组件本身是全新概念。

### 18.3 PubMedBERT 不是生成式模型

规避方式：

1. 文档中明确 PubMedBERT 是 diagnostic baseline；
2. 主实验必须包含 T5/BioBART；
3. 论文中不要把 PubMedBERT 说成生成式方法。

### 18.4 外部知识引入过重

规避方式：

1. 不做完整 KG reasoning；
2. 不做重型 RAG；
3. 只使用简短定义、同义词、触发词；
4. 所有知识内容写入 schema 文件，保证可复现。

### 18.5 数据类别不平衡

规避方式：

1. 报告 macro-F1；
2. 报告 per-class F1；
3. 单独报告 rare relation performance；
4. 不只看 micro-F1。

---

## 19. 给实验同学的最终执行清单

### 19.1 数据准备

- [ ] 下载并准备 ChemProt。
- [ ] 下载并准备 DDI 2013。
- [ ] 可选准备 BioRED。
- [ ] 转换成统一 JSONL。
- [ ] 输出 dataset statistics。

### 19.2 Schema 准备

- [ ] 编写 ChemProt `relation_schema.yaml`。
- [ ] 编写 DDI `relation_schema.yaml`。
- [ ] 每个 relation 包含五种表达。
- [ ] 人工检查 schema 是否有明显错误。

### 19.3 模型实验

- [ ] 跑 T5/BioBART smoke test。
- [ ] 跑 ChemProt T5/BioBART B1-B2-P1-P2-P3。
- [ ] 跑 DDI T5/BioBART B1-B2-P1-P2-P3。
- [ ] 跑 ChemProt RSG-BioRE R1/R2。
- [ ] 跑 DDI RSG-BioRE R1/R2。
- [ ] 跑 PubMedBERT prompt classification 诊断实验。
- [ ] 补充 3 个随机种子。
- [ ] 如果 R2 有提升，补 R2 的 `lambda` 和 prototype sensitivity 消融。
- [ ] 跑 few-shot K=1/8/16。
- [ ] 可选跑 BioRED 扩展实验。

### 19.4 结果输出

- [ ] 每个实验输出 `run_config.yaml`。
- [ ] 每个实验输出 `metrics.json`。
- [ ] 每个实验输出 `per_class_metrics.csv`。
- [ ] 每个实验输出 `predictions.jsonl`。
- [ ] 每个实验输出 `error_cases.md`。
- [ ] RSG-BioRE 实验额外输出 `prototype_scores.jsonl`。
- [ ] RSG-BioRE 实验额外输出 `prototype_analysis.csv`。
- [ ] 汇总主结果表。
- [ ] 汇总 per-class 表。
- [ ] 汇总 rare relation 表。
- [ ] 汇总 P2/R1/R2 模块消融表。
- [ ] 写 `error_analysis_summary.md`。

---

## 20. 论文第一章可用表述

### 20.1 研究问题表述

> 生物医学关系抽取数据集中的 relation label 通常具有较强领域语义和缩写化特点，直接使用原始标签难以充分表达关系含义。为提升生成式模型对关系类别的理解能力，本文研究关系语义原型引导的生成式 BioRE 方法，将关系标签描述、实体类型信息和轻量领域知识编码为可学习的 relation semantic prototypes，并通过实例-关系语义对齐机制引导模型生成正确关系标签。

### 20.2 创新点表述

> 本文提出一种关系语义原型引导的生成式生物医学关系抽取方法。该方法首先构建多层关系语义描述，再将其编码为 relation semantic prototypes，并通过实例-关系语义对齐损失约束文本实体对表示与目标关系原型之间的匹配关系。该方法在不依赖依存分析和复杂知识图谱推理的前提下，将关系语义从 prompt 文本提升为可学习、可消融的模型表示。

### 20.3 实验设计表述

> 实验在 ChemProt 和 DDI 2013 数据集上进行，并以 BioRED 作为扩展验证。本文首先使用 PubMedBERT/BioBERT prompt classification 作为诊断性基线，随后构建 T5/BioBART text-to-text 生成式强基线，并进一步比较 RSG-BioRE 的 prototype-only、prototype+alignment 和可选 guided decoder 版本。评价指标包括 micro-F1、macro-F1、precision、recall、per-class F1、rare relation performance、生成式输出合法率以及 prototype alignment 分析指标。

---

## 21. 参考文献链接

1. He et al. Prompt Tuning in Biomedical Relation Extraction. Journal of Healthcare Informatics Research, 2024.  
   https://pmc.ncbi.nlm.nih.gov/articles/PMC11052745/

2. Zhou et al. LEAP: LLM instruction-example adaptive prompting framework for biomedical relation extraction. JAMIA, 2024.  
   https://pmc.ncbi.nlm.nih.gov/articles/PMC11339510/

3. Model Tuning or Prompt Tuning? A Study of LLMs for Clinical Concept and Relation Extraction. Journal of Biomedical Informatics, 2024.  
   https://www.sciencedirect.com/science/article/pii/S1532046424000480

4. Enhancing biomedical relation extraction with directionality. Bioinformatics, 2025.  
   https://pmc.ncbi.nlm.nih.gov/articles/PMC12261447/

5. Knowledge-augmented pre-trained language models for biomedical relation extraction. BMC Bioinformatics, 2025.  
   https://link.springer.com/10.1186/s12859-025-06262-6

6. A benchmark for end-to-end zero-shot biomedical relation extraction with LLMs. WASP, 2025.  
   https://aclanthology.org/2025.wasp-main.6/

7. RELATE: Relation Extraction in Biomedical Abstracts with LLMs and Ontology Constraints. ML4H / PMLR, 2026.  
   https://proceedings.mlr.press/v297/olasunkanmi26a.html

---

## 22. 当前版本边界

本指南是 v1.1 实验指导书，当前边界如下：

1. 不包含具体训练代码。
2. 不规定唯一模型实现框架，实验同学可基于 Hugging Face Transformers 实现。
3. 不做 dependency parsing。
4. 不做完整 RAG。
5. 不把 PubMedBERT 作为生成式主模型。
6. 不追求第一阶段直接达到 SOTA，重点是验证 relation semantic prototype 与 alignment loss 的有效性和可解释性。

后续如果实验同学开始实现代码，可基于本文档再生成：

```text
docs/stage1_docs/stage1_implementation_plan.md
docs/stage1_docs/stage1_relation_schema_templates.md
docs/stage1_docs/stage1_result_reporting_template.md
```
