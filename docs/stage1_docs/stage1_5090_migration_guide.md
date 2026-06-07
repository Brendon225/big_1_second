# Stage 1 5090 Migration Guide

## ISO

### I. Insight

本机目标是验证 ChemProt raw files、统一 JSONL、schema、prompt、metric、runner 和输出目录是否连通。正式训练需要 GPU、PyTorch、Transformers 和模型缓存，因此应迁移到实验室 5090 机器执行。

### S. Strategy

1. 本机保留 `backend=mock` 或 `backend=fake_train` 做流程 smoke。
2. 5090 机器安装 `requirements-stage1.txt`。
3. 先用 `t5-small` 跑极小 batch smoke。
4. 再切换 BioBART/T5-base，跑 ChemProt 主实验。
5. DDI 2013 和 BioRED 不提前扩张，等 ChemProt P2/R2 流程稳定后再迁移。

### O. Operation

本机 smoke：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests/stage1 -v
.\.venv\Scripts\python.exe scripts/stage1/train_stage1.py --config configs/stage1/chemprot_train_fake_smoke.yaml
```

5090 Linux/Conda 示例：

```bash
conda create -n rsg-biore python=3.10 -y
conda activate rsg-biore
pip install -r requirements-stage1.txt
python -m unittest discover -s tests/stage1 -v
python scripts/stage1/download_model.py --model-name GanjinZero/biobart-base --output-dir models/stage1/biobart-base
python scripts/stage1/train_stage1.py --config configs/stage1/chemprot_hf_t5_smoke_template.yaml
python scripts/stage1/train_stage1.py --config configs/stage1/chemprot_hf_biobart_smoke.yaml
```

注意：

1. `chemprot_hf_t5_smoke_template.yaml` 是真实 Hugging Face T5 backend 模板。
2. `chemprot_hf_biobart_smoke.yaml` 使用本地 `models/stage1/biobart-base`。
3. 当前 `HfText2TextModel` 已支持真实 seq2seq fine-tuning smoke；RSG-BioRE 的真实 PyTorch alignment training 下一步再补。
4. 不要把 raw zip、`.venv`、models、outputs、checkpoints 提交到 git。
