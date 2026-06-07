# 《基于生成式模型的生物医学关系抽取研究》研究路线规划文档

> 当前版本：v0.2  
> 生成日期：2026-06-07  
> 用途：用于研究生大论文选题收束、开题前讨论、后续文献矩阵与实验计划扩展。  
> 说明：论文题目保持为《基于生成式模型的生物医学关系抽取研究》，本文档只收束研究内容与技术路线。v0.2 根据 2024-2026 年 BioRE / LLM prompting / schema grounding 相关文献，对研究内容一进行收束更新；当前进一步将研究内容一从“关系描述/prompt 对比”升级为“关系语义原型引导的生成式 BioRE 方法”。

## 1. 研究定位

生物医学关系抽取旨在从生物医学文本中识别实体之间的语义关系，例如药物-药物相互作用、化学物质-疾病关系、基因-疾病关系、药物-不良反应关系等。传统方法通常将该任务建模为分类问题，即给定实体对和上下文，预测预定义关系类别。

近年来，生成式模型在信息抽取任务中受到关注。与分类式方法相比，生成式模型可以将关系抽取转化为文本生成、问答、三元组生成或结构化输出任务，从而在低资源、跨数据集、文档级抽取等场景中展现出新的潜力。然而，现有生成式生物医学关系抽取方法仍存在标签语义表达不稳定、生成结果不可控、复杂文档证据利用不足、跨数据集泛化能力有限等问题。

因此，本研究不再沿用“依存分析 + 生成式模型”的旧路线，而是聚焦生成式关系抽取本身的核心问题：

> 如何使生成式模型更准确地理解生物医学关系语义，并稳定、可控地生成关系抽取结果？

## 2. ISO 分析

### 2.1 Insight：核心矛盾

本课题的核心矛盾可以分为内部矛盾和外部矛盾。

内部矛盾：

1. 生成式模型能够自然地表达关系三元组，但输出容易出现格式错误、实体边界错误、非法关系标签和漏抽取。
2. 生物医学关系标签通常具有强领域语义，例如 ChemProt、DDI、BioRED 等数据集中的关系类别并非普通自然语言标签，模型难以直接理解。
3. 文档级关系常依赖跨句证据，单纯输入文本或简单 prompt 难以稳定捕捉证据链。
4. LLM zero-shot 方法虽然降低了标注成本，但在复杂多关系、多谓词、长文本输入中仍然不稳定。

外部矛盾：

1. 导师不希望继续走依存分析路线，因此本研究必须放下 dependency parsing 作为核心技术点。
2. 研究生大论文需要形成 2-3 个环环相扣的研究点，而不是单点 prompt 调参或单个模型改造。
3. 实验路线需要可复现、可落地，不能依赖成本过高的大规模闭源 LLM 调用或复杂工程系统。

### 2.2 Strategy：研究策略

本研究采用由浅入深的三阶段策略：

1. 首先研究关系标签语义表达，使生成式模型更好理解关系类别。
2. 然后研究结构化关系生成，使模型输出稳定、合法、可解析。
3. 最后研究低资源和文档级场景下的示例/知识增强，提高泛化能力与证据利用能力。

### 2.3 Operation：可执行拆解

后续研究将围绕以下三个研究点展开：

1. 面向生成式 BioRE 的关系语义原型引导生成方法。
2. 面向生物医学关系抽取的结构化生成方法。
3. 面向低资源与文档级 BioRE 的示例或知识增强生成方法。

这三个研究点共同服务于固定论文题目《基于生成式模型的生物医学关系抽取研究》。

## 3. 代表性文献矩阵

| 类别 | 代表论文 | 方法特点 | 对本研究的启发 |
|---|---|---|---|
| 生物医学生成式基座模型 | BioGPT, 2022 | 使用生成式预训练模型处理 BC5CDR、KD-DTI、DDI 等任务，将关系抽取转化为端到端文本生成 | 说明 BioRE 可以被建模为生成任务，但输出解析和结构控制仍是问题 |
| 生物医学 encoder-decoder 模型 | BioBART, 2022 | 面向生物医学文本生成任务预训练 BART 类模型 | 为使用 BioBART/T5/BART 类模型进行生成式 BioRE 提供基座参考 |
| Prompt tuning BioRE | Prompt Tuning in Biomedical Relation Extraction, 2024 | 在 ChemProt 和 DDI 上将 BioRE 转化为 mask prediction，比较 prompt tuning 与普通 fine-tuned PLMs | 说明 label words 和 prompt template 对 BioRE 有效，可作为研究内容一的诊断性基线 |
| LLM instruction/example prompting | LEAP, 2024 | 在 DDI、ChemProt、BioRED 上系统比较 instruction、options、option descriptions、examples 及 adaptive prompting | 说明 relation options、option descriptions 和 examples 对 LLM BioRE 影响显著，支撑 relation description 和 example-enhanced prompt |
| Prompt/model tuning 对比 | Model Tuning or Prompt Tuning?, 2024 | 比较 clinical concept/relation extraction 中 model tuning、hard prompt、soft prompt 等策略 | 支撑将 soft verbalizer / learnable prompt 作为可选增强，而非第一阶段主创新 |
| 方向性与实体角色 | Enhancing biomedical relation extraction with directionality, 2025 | 在 BioRED 上强化关系方向性、novelty 和 soft-prompt 多任务建模 | 支撑 entity-type-aware / role-aware relation description |
| 知识增强 PLM | Knowledge-augmented PLMs for Biomedical Relation Extraction, 2025 | 比较实体描述、KG relation、分子特征等知识增强对多个 BioRE 数据集的影响 | 说明知识增强应轻量、可控，并需要消融验证是否真正有效 |
| LLM zero-shot benchmark | WASP 2025 zero-shot BioRE benchmark | 在 7 个 BioRE 数据集上评估 GPT-4、o1、GPT-OSS-120B | 证明 LLM zero-shot 尚未完全解决复杂 BioRE，尤其是多关系、多谓词输入 |
| 零样本三元组抽取 | Benchmarking zero-shot biomedical relation triplet extraction, 2025 | 跨大量生物医学语料评估不同语言模型的零样本三元组抽取能力 | 说明大模型并不能自动解决 relation schema 理解问题，仍需关系语义约束 |
| Ontology/schema grounding | RELATE, 2026 | 将 LLM 抽取关系映射到 Biolink 等标准 ontology predicates，并在 ChemProt 等任务上验证 | 说明 relation schema/ontology 对 BioRE 很重要；本研究进一步将 schema 文本编码为可学习关系语义原型，并用于生成式模型对齐 |
| 文档级 prompt learning | Biomedical document RE with prompt learning and KNN, 2023 | 将 document-level RE 转化为 T5 prompt learning，并结合 KNN | 说明文档级 BioRE 可以用生成式/文本到文本框架重构 |
| 结构化三元组生成 | PLRTE, 2024 | 采用 progressive learning 处理文档级 biomedical relation triplet extraction | 支撑“结构化三元组生成”和“跨数据集泛化”研究点 |
| 知识库引导生成 | Knowledge Base-Guided Generation, 2024 | 使用 UMLS concept mapping 引导 GPT 做文档级临床实体和关系抽取 | 说明知识增强不一定要完整 RAG，也可以从医学概念映射入手 |
| RAG + CoT 文档级 BioRE | SyRACT, 2025 | 将 BioDocRE 重构为 QA，结合 PubMed RAG 和 CoT 降低幻觉并增强推理 | 支撑“文档级证据增强”研究点，但工程复杂度需控制 |

## 4. 现有方法的共同不足

### 4.1 关系标签语义表达不足

许多 BioRE 数据集的关系标签具有领域专业性，例如 ChemProt 中的关系类别、DDI 中的药物相互作用类型、BioRED 中的多类型生物医学关系。若直接使用原始标签或简单 label word，模型可能难以捕捉标签背后的医学语义。

近两三年的 prompt tuning、LLM prompting 和 ontology grounding 研究已经说明，BioRE 的性能不仅取决于模型大小，也取决于 relation options、option descriptions、entity role 和 schema grounding 的表达方式。现有方法已经证明 verbalizer、instruction、examples 和 ontology constraints 有价值，但多数工作仍停留在“把标签写成更好的文本提示”这一层，缺少将关系语义显式建模为可学习表示、并与样本实例语义进行对齐的模型机制。因此仍存在以下不足：

1. label words 多依赖人工设计，迁移到新数据集时需要重新调整。
2. 不同 relation label 的语义粒度不统一，例如 `CPR:4`、`mechanism`、`int` 等标签本身并不能充分说明关系边界。
3. 现有研究通常证明 prompt 或 option description 有效，但较少系统比较 raw label、label words、relation description、entity-type-aware description 和 knowledge-enhanced description 的差异。
4. 关系标签常与实体类型、实体角色和方向性脱节，导致模型难以区分相近关系。
5. 知识增强方法容易走向复杂 KG/RAG 系统，复现成本高，不适合作为本研究第一阶段的主路线。
6. relation description 往往只作为 prompt 文本拼接到输入中，模型是否真正学到了“样本语义应靠近正确关系语义、远离错误关系语义”缺少显式约束。

### 4.2 生成结果缺乏结构约束

生成式 BioRE 方法常将关系抽取结果表示为自然语言句子或线性化三元组，但模型可能生成非法格式。例如：

1. 关系标签不在预定义集合中。
2. 生成的实体与输入实体不一致。
3. 多关系场景中漏掉部分三元组。
4. 生成文本需要复杂后处理才能转回标准评价格式。

因此，生成式 BioRE 不能只关注 F1 分数，还需要关注输出合法性、可解析性和结构一致性。

### 4.3 文档级证据利用不足

文档级 BioRE 关系常跨句出现，模型需要识别实体对之间的关键证据。现有 LLM 和 prompt 方法常面临：

1. 长文本输入导致关键信息被稀释。
2. 模型生成关系时缺乏显式证据依据。
3. RAG 方法可能引入无关知识或冗余上下文。
4. CoT 虽提升解释性，但也可能生成看似合理但不可靠的推理链。

因此，更适合本研究的方向是轻量级证据增强，而不是一开始构建完整复杂 RAG 系统。

### 4.4 低资源与跨数据集泛化能力有限

BioRE 数据集之间关系 schema 差异较大。一个模型在 ChemProt 上有效，不一定能迁移到 DDI、BioRED、CDR、GDA 等数据集。LLM zero-shot 方法在简单场景上表现接近监督模型，但在复杂多关系输入中仍然不稳定。

这说明本研究需要关注：

1. relation label 的跨数据集语义对齐。
2. 示例选择对 few-shot / low-resource BioRE 的影响。
3. 结构化输出对跨数据集评价的一致性帮助。

## 5. 拟定研究内容

### 5.1 研究内容一：面向生成式 BioRE 的关系语义原型引导生成方法

研究问题：

> 如何将 BioRE 数据集中的抽象 relation label 建模为可学习的 relation semantic prototypes，并通过实例-关系语义对齐机制引导 T5/BioBART 等生成式模型稳定生成正确关系标签？

基本思路：

1. 首先构造 relation semantic schema，包括 raw label、manual label words、relation description、entity-type-aware description 和 knowledge-enhanced description；该 schema 不作为最终创新本身，而作为关系语义原型的输入材料。
2. 使用 relation semantic encoder 将每个关系类别的语义描述编码为 relation semantic prototype，例如 `p_i = LayerNorm(W z_i + q_i)`，其中 `z_i` 来自关系描述编码，`q_i` 是可学习关系原型向量。
3. 使用 T5/BioBART encoder 对文本、实体对和实体类型进行编码，得到 instance representation `h_x`。
4. 设计 instance-prototype semantic alignment loss，使正样本实例靠近真实关系原型，远离其他关系原型：`L_align = - log exp(sim(h_x, p_y)/τ) / sum_i exp(sim(h_x, p_i)/τ)`。
5. 使用生成式 decoder 输出固定格式关系标签，例如 `relation: CPR:4`，训练目标为 `L = L_gen + λ L_align`；可选加入原型引导 decoder 或辅助分类损失作为增强实验。
6. 重点分析 macro-F1、per-class F1、rare relation performance、valid output rate、label sensitivity 和 prompt robustness，而不仅是总体 micro-F1。

可能方法：

1. Baseline：raw label / manual label words / relation description prompt。
2. Strong baseline：T5/BioBART + entity-type-aware 或 knowledge-enhanced relation description。
3. RSG-BioRE-V1：Relation Semantic Prototype Only，只引入关系语义原型表示。
4. RSG-BioRE-V2：Prototype + Alignment，引入实例-关系语义对齐损失，作为第一阶段主方法。
5. RSG-BioRE-V3：Prototype + Alignment + Guided Decoder，将原型匹配分数作为生成 decoder 的辅助引导，作为可选增强。
6. 可选 soft verbalizer / soft prompt：作为性能上界或补充实验，不作为第一阶段主创新点。

可用数据集：

1. ChemProt。
2. DDI 2013。
3. BioRED 作为扩展验证。

预期贡献：

1. 提出一种关系语义原型引导的生成式 BioRE 方法，将 relation label semantics 从文本 prompt 提升为可学习、可对齐的模型表示。
2. 设计实例-关系语义对齐机制，使生成式模型在生成标签前显式学习“文本实体对语义”与“关系类别语义”的匹配关系。
3. 通过 raw label、description、entity-type-aware、knowledge-enhanced、prototype-only、prototype+alignment 等消融，证明性能提升来自语义原型与对齐机制，而不是单纯 prompt wording。
4. 为后续结构化生成中的 relation schema constraint、ontology/schema grounding 和低资源迁移提供可复用的关系语义表示。

### 5.2 研究内容二：面向 BioRE 的结构化关系三元组生成

研究问题：

> 如何使生成式模型稳定输出合法、可解析的生物医学关系三元组？

基本思路：

1. 将 BioRE 输出统一为结构化三元组格式。
2. 约束生成关系标签必须来自预定义 relation schema。
3. 约束生成实体必须与输入实体或候选实体集合一致。
4. 设计生成后校验机制，修正非法输出或过滤不一致结果。

可能方法：

1. text-to-text triplet generation baseline。
2. schema-constrained decoding。
3. relation-set constrained generation。
4. entity-consistency validation。
5. JSON-style 或 bracket-style structured output。

可用数据集：

1. BC5CDR。
2. DDI 2013。
3. ChemProt。

预期贡献：

1. 将生成式 BioRE 从自由文本生成推进到可控结构化生成。
2. 降低非法输出和后处理错误。
3. 提升多关系场景下的稳定性和可复现性。

### 5.3 研究内容三：低资源与文档级场景下的示例/知识增强生成

研究问题：

> 在标注数据有限或关系证据分散于文档级上下文时，如何增强生成式 BioRE 的泛化能力？

基本思路：

1. 使用 relation description 和 entity type 构造任务 instruction。
2. 通过示例检索选择与当前样本相似的 few-shot examples。
3. 在必要时引入轻量医学知识，例如 UMLS concept mapping 或关系定义。
4. 避免构建过重的完整 RAG 系统，优先做可控、可复现的轻量增强。

可能方法：

1. random examples baseline。
2. similarity-based example retrieval。
3. relation-aware example selection。
4. UMLS concept-enhanced prompt。
5. evidence sentence / evidence snippet 提示。

可用数据集：

1. CDR。
2. GDA。
3. ADE。
4. BioRED。

预期贡献：

1. 提出适合低资源 BioRE 的示例增强生成方法。
2. 分析示例质量、关系类型和文档证据对生成性能的影响。
3. 连接 LLM prompting 与可复现的 BioRE 实验设置。

## 6. 实验设计初稿

### 6.1 数据集选择

优先级较高的数据集：

| 数据集 | 任务类型 | 适用研究内容 |
|---|---|---|
| ChemProt | chemical-protein relation classification | 研究内容一、二 |
| DDI 2013 | drug-drug interaction extraction | 研究内容一、二 |
| BC5CDR / CDR | chemical-disease relation extraction | 研究内容二、三 |
| BioRED | 多类型 biomedical relation extraction | 研究内容一、三 |
| GDA | gene-disease association | 研究内容三 |
| ADE | drug-adverse event relation extraction | 研究内容三 |

### 6.2 基线方法

分类式基线：

1. BioBERT / PubMedBERT + classifier。
2. SciBERT / BioLinkBERT + classifier。

诊断性 prompt 基线：

1. PubMedBERT / BioBERT prompt classification + raw label。
2. PubMedBERT / BioBERT prompt classification + manual label words。
3. PubMedBERT / BioBERT prompt classification + relation description。

说明：PubMedBERT / BioBERT 属于判别式或 encoder-only 模型，只作为诊断性基线，用于隔离 relation semantic schema 的影响，不作为论文题目中“生成式模型”的主方法。

生成式基线：

1. T5 / BART text-to-text baseline。
2. BioBART 或 BioGPT baseline。
3. T5 / BioBART + raw label schema。
4. T5 / BioBART + relation description schema。
5. T5 / BioBART + entity-type-aware / knowledge-enhanced schema。

研究内容一主方法：

1. RSG-BioRE-V1：T5/BioBART + relation semantic prototypes。
2. RSG-BioRE-V2：T5/BioBART + relation semantic prototypes + instance-prototype alignment loss。
3. RSG-BioRE-V3 可选：T5/BioBART + relation semantic prototypes + alignment loss + prototype-guided decoding。

LLM / prompting 基线：

1. zero-shot instruction。
2. few-shot random example。
3. few-shot retrieved example。
4. 仅作为补充分析，不作为主实验路线。

### 6.3 评价指标

主指标：

1. micro-F1。
2. macro-F1。
3. precision。
4. recall。

生成式附加指标：

1. valid output rate：生成结果格式合法率。
2. entity consistency rate：生成实体与输入实体一致率。
3. relation validity rate：生成关系是否在 schema 内。
4. triplet exact match：三元组精确匹配。
5. error type breakdown：实体错误、关系错误、漏抽取、非法格式、幻觉输出。

### 6.4 消融实验

研究内容一：

1. 原始 label。
2. 手工 label words。
3. relation description。
4. entity-type-aware description。
5. knowledge-enhanced description。
6. 去除 relation semantic prototype，仅保留 description prompt。
7. 去除 instance-prototype alignment loss，仅使用生成损失。
8. 去除可学习关系原型向量 `q_i`，只使用 description encoder 输出。
9. 去除 entity type / direction 信息。
10. 去除 knowledge-enhanced 信息。
11. 可选 soft verbalizer / soft prompt。
12. description 长度敏感性分析。
13. prompt 顺序鲁棒性分析。

研究内容二：

1. 无约束生成。
2. relation schema constraint。
3. entity constraint。
4. schema + entity 双重约束。
5. 加入生成后校验。

研究内容三：

1. zero-shot。
2. random few-shot。
3. similarity-based examples。
4. relation-aware examples。
5. relation-aware examples + light knowledge。

## 7. 预期创新点

创新点一：

> 提出一种关系语义原型引导的生成式 BioRE 方法，将 raw label、relation description、entity-type-aware description 和 knowledge-enhanced description 编码为 relation semantic prototypes，并通过实例-关系语义对齐损失引导生成式模型学习文本实体对与目标关系类别之间的匹配关系，从而缓解原始关系标签抽象、prompt wording 敏感和少数类关系混淆问题。

创新点二：

> 提出面向生成式 BioRE 的结构化关系三元组生成与校验机制，提高生成结果的合法性、可解析性和实体一致性。

创新点三：

> 提出面向低资源和文档级 BioRE 的示例/知识增强生成方法，在不依赖复杂依存分析和重型 RAG 系统的前提下，提高模型泛化能力与证据利用能力。

## 8. 研究路线图

```text
阶段一：文献综述与问题确认
  -> 生成式 BioRE 文献矩阵
  -> 数据集与评价指标整理
  -> 确定可复现实验环境

阶段二：关系语义原型引导生成
  -> relation semantic schema 设计
  -> relation semantic prototype 构造
  -> instance-prototype alignment loss
  -> raw label / description prompt / prototype-only / prototype+alignment 对比
  -> ChemProt、DDI 主实验，BioRED 扩展验证
  -> 分析 macro-F1、per-class F1、rare relation、valid output rate、prompt robustness

阶段三：结构化关系三元组生成
  -> text-to-text triplet generation baseline
  -> schema/entity constrained generation
  -> 输出合法性与错误类型分析

阶段四：低资源与文档级增强
  -> few-shot example selection
  -> light knowledge / concept-enhanced prompt
  -> CDR、GDA、ADE、BioRED 实验

阶段五：论文整合
  -> 方法统一表述
  -> 实验结果汇总
  -> 消融实验与案例分析
  -> 完成大论文写作
```

## 9. 风险与控制

| 风险 | 表现 | 控制方式 |
|---|---|---|
| 方向过散 | prompt、LLM、RAG、生成都想做 | 统一到“关系语义建模与结构化生成”主线 |
| 工程量过大 | 完整 RAG 或 LLM fine-tuning 成本高 | 优先做轻量知识增强和示例检索 |
| 创新点不清 | 容易被看成 prompt 调参 | 将 prompt/schema 作为基线和输入材料，主创新落在 relation semantic prototype、alignment loss 和 prototype-guided generation |
| 结果不可复现 | 闭源 API 版本变化 | 主实验优先使用开源模型，闭源 LLM 只做补充分析 |
| 与旧依存路线混淆 | 文献中可能出现 dependency parser | 不将依存分析作为本研究核心方法，只作为被排除路线说明 |

## 10. 当前推荐的正式研究表述

论文题目保持：

> 《基于生成式模型的生物医学关系抽取研究》

研究主线表述：

> 本研究围绕生成式生物医学关系抽取中的关系语义理解、结构化输出控制和低资源泛化问题展开，依次研究关系语义原型引导生成、结构化关系三元组生成以及示例/知识增强生成方法，以提升生成式模型在生物医学关系抽取任务中的准确性、可控性和泛化能力。

三项研究内容表述：

1. 面向生成式 BioRE 的关系语义原型引导生成方法。
2. 面向 BioRE 的结构化关系三元组生成与输出约束方法。
3. 面向低资源与文档级 BioRE 的示例/知识增强生成方法。

## 11. 下一步工作

1. 以 `docs/stage1_docs/stage1_relation_label_semantic_modeling_experiment_guide.md` 作为研究内容一的实验执行指南。
2. 为 ChemProt 和 DDI 2013 构造 relation semantic schema，并将 BioRED 作为扩展验证；该 schema 用作 prompt baseline 和 relation prototype 输入。
3. 搭建第一阶段最小可行实验：PubMedBERT/BioBERT 诊断性 prompt baseline + T5/BioBART text-to-text 生成式 baseline + RSG-BioRE 主方法。
4. 完成 raw label、label words、relation description、entity-type-aware description、knowledge-enhanced description、prototype-only、prototype+alignment 的消融实验。
5. 根据第一阶段实验结果决定第二章采用 BioBART/T5/BioGPT 中的哪类生成基座，并复用第一阶段得到的 relation semantic prototypes。

## 12. 参考文献与来源

1. Luo et al. BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining. Briefings in Bioinformatics, 2022. https://academic.oup.com/bib/article/23/6/bbac409/6713511
2. Yuan et al. BioBART: Pretraining and Evaluation of A Biomedical Generative Language Model. BioNLP, 2022. https://aclanthology.org/2022.bionlp-1.9/
3. He et al. Prompt Tuning in Biomedical Relation Extraction. Journal of Healthcare Informatics Research, 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC11052745/
4. Zhou et al. LEAP: LLM instruction-example adaptive prompting framework for biomedical relation extraction. JAMIA, 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC11339510/
5. Model Tuning or Prompt Tuning? A Study of LLMs for Clinical Concept and Relation Extraction. Journal of Biomedical Informatics, 2024. https://www.sciencedirect.com/science/article/pii/S1532046424000480
6. Bhattarai et al. Document-level Clinical Entity and Relation extraction via Knowledge Base-Guided Generation. BioNLP, 2024. https://aclanthology.org/2024.bionlp-1.24/
7. Progressive learning method for biomedical relation triplet extraction. Journal of Biomedical Informatics, 2024. https://www.sciencedirect.com/science/article/pii/S1532046424001564
8. Brokman et al. A benchmark for end-to-end zero-shot biomedical relation extraction with LLMs: experiments with OpenAI models. WASP, 2025. https://aclanthology.org/2025.wasp-main.6/
9. Benchmarking zero-shot biomedical relation triplet extraction across language model architectures. BioNLP, 2025. https://aclanthology.org/2025.bionlp-1.9/
10. Enhancing biomedical relation extraction with directionality. Bioinformatics, 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12261447/
11. Knowledge-augmented pre-trained language models for biomedical relation extraction. BMC Bioinformatics, 2025. https://link.springer.com/10.1186/s12859-025-06262-6
12. Dong et al. SyRACT: zero-shot biomedical document-level relation extraction with synergistic RAG and CoT. Bioinformatics, 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12237500/
13. RELATE: Relation Extraction in Biomedical Abstracts with LLMs and Ontology Constraints. ML4H / PMLR, 2026. https://proceedings.mlr.press/v297/olasunkanmi26a.html
