# ChemProt Raw Download Notes

## ISO

### I. Insight

本轮只下载 ChemProt raw files，并用本机虚拟环境做小范围 smoke。当前不做正式训练，不安装 PyTorch/Transformers，不把 mock 指标解释为实验结果。

核心矛盾：

1. 必须保留 raw corpus，避免把处理后的 parquet 当作原始数据。
2. 必须验证 raw -> unified JSONL -> Stage 1 runner 的最小闭环。
3. 本机只承担流程验证，正式大规模训练迁移到实验室 5090 机器。

### S. Strategy

1. raw zip 保留在 `data/stage1/raw/chemprot/ChemProt_Corpus.zip`。
2. 解压文件保留在 `data/stage1/raw/chemprot/extracted/`。
3. 转换后的小范围 smoke JSONL 写入 `data/stage1/chemprot/`。
4. 使用 `.venv` 运行 stdlib-only smoke。

### O. Operation

下载命令：

```powershell
python scripts/stage1/download_chemprot_raw.py
```

转换小样本命令：

```powershell
.\.venv\Scripts\python.exe scripts/stage1/convert_chemprot.py --raw-dir data/stage1/raw/chemprot/extracted --output-dir data/stage1/chemprot --max-samples-per-split 30 --max-negative-per-doc 1
```

smoke 命令：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests/stage1 -v
.\.venv\Scripts\python.exe scripts/stage1/run_smoke.py --config configs/stage1/chemprot_smoke_t5_baseline.yaml
.\.venv\Scripts\python.exe scripts/stage1/run_smoke.py --config configs/stage1/chemprot_smoke_rsg_biore.yaml
```

Raw manifest:

```text
data/stage1/raw/chemprot/manifest.json
```

Downloaded raw zip:

```text
data/stage1/raw/chemprot/ChemProt_Corpus.zip
```

SHA256:

```text
492e3d607f38e2727b799e9d60263b776ebd2a5e61cf0fb59bea2b3eb68e1c28
```

Source:

```text
https://huggingface.co/datasets/bigbio/chemprot/resolve/main/ChemProt_Corpus.zip
```
