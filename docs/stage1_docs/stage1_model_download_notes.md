# Stage 1 Model Download Notes

## I. Insight

BioBART 不是随代码自动存在的本地文件。Hugging Face 模型有两种使用方式：

1. 直接在配置中写 `GanjinZero/biobart-base`，首次运行时自动下载到 Hugging Face cache。
2. 先显式下载并保存到项目目录，再在配置中写本地路径。

本项目采用第二种方式，便于迁移到 5090 机器时确认模型已经存在。

## S. Strategy

当前本地 BioBART 路径：

```text
models/stage1/biobart-base
```

该目录被 `.gitignore` 忽略，不提交到 git。5090 机器可以重新执行下载命令，或复制该目录。

官方/权威来源：

```text
https://huggingface.co/GanjinZero/biobart-base
https://github.com/GanjinZero/BioBART
```

其中 Hugging Face 模型页标注 `text2text-generation`，并给出 `AutoTokenizer` 和 `AutoModelForSeq2SeqLM` 加载方式；BioBART GitHub README 列出 `GanjinZero/biobart-base`、`GanjinZero/biobart-large`、`GanjinZero/biobart-v2-base` 等 checkpoint。

## O. Operation

下载模型：

```powershell
.\.venv\Scripts\python.exe scripts/stage1/download_model.py --model-name GanjinZero/biobart-base --output-dir models/stage1/biobart-base
```

本地 BioBART smoke：

```powershell
.\.venv\Scripts\python.exe scripts/stage1/train_stage1.py --config configs/stage1/chemprot_hf_biobart_smoke.yaml
```

当前 smoke 设置：

```text
max_train_steps = 1
max_train_samples = 2
max_dev_samples = 2
max_test_samples = 2
batch_size = 1
learning_rate = 1e-6
```

说明：

1. 本地 smoke 只验证真实 BioBART 加载、训练一步、生成、解析、评价和输出文件写入。
2. `valid_output_rate=0` 在 1-step CPU smoke 中可以接受，因为模型还没有学会固定格式输出。
3. 正式实验应在 5090 上取消 `max_*_samples` 和 `max_train_steps` 限制，并使用 dev macro-F1 选择 checkpoint。
