# Stage 1 5090 GitHub SSH Migration Runbook

> 适用项目：RSG-BioRE / Stage 1  
> 目标：把当前本地项目通过 WinSCP 整包迁移到实验室 5090 机器，并建立 GitHub SSH 远程仓库同步机制。  
> 原则：先让 5090 机器复现实验 smoke，再进入正式实验；先建立 GitHub 远程同步，再做大规模训练改动。

---

## 0. ISO 总览

### I. Insight：核心矛盾

当前真正要解决的不是“5090 上重新下载数据和模型”，因为你计划通过 WinSCP 直接传整个项目文件夹。

当前核心矛盾是：

1. 项目迁移后，5090 必须能识别同一套代码、数据、模型和配置。
2. GitHub 远程仓库必须用 SSH 连接，后续本机和 5090 可以通过 `git push / git pull` 同步。
3. smoke 配置和正式实验配置必须分开，避免误把 2 条样本、1 step 的 smoke 当正式实验。
4. 实验结果必须按 run 目录传回，不能只传 `metrics.json`，否则后续无法做错误分析、per-class 分析和 prototype 分析。

### S. Strategy：迁移策略

本次推荐流程：

1. 本机确认项目状态干净。
2. 用 WinSCP 把整个项目文件夹传到 5090。
3. 5090 上创建 conda 环境并安装依赖。
4. 5090 上跑测试与 BioBART smoke，确认迁移成功。
5. 在 GitHub 网页创建空远程仓库。
6. 本机和 5090 分别配置 GitHub SSH key。
7. 本机仓库绑定 GitHub remote 并 push。
8. 5090 仓库绑定同一 GitHub remote，后续通过 pull/push 同步代码。
9. 5090 上创建 full experiment config，开始正式实验。
10. 实验结束后打包 `outputs/stage1/...` 回传分析。

### O. Operation：执行顺序

后文每一节都包含：

```text
命令
目的
成功标志
失败时检查
```

---

## 1. 本机迁移前检查

在本机项目根目录执行：

```powershell
cd D:\Desktop_D\postgraduate\研二\big\big_1_second
git status --short
git log --oneline -3
```

目的：

确认本机代码已经提交，迁移时不会漏掉未保存改动。

成功标志：

```text
git status --short
```

没有输出，或只有你明确知道不用迁移的本地生成文件。

```text
git log --oneline -3
```

能看到最近提交，例如：

```text
2a631c6 feat: add biobart hf training smoke
7ed6183 chore: initialize stage1 rsg biore scaffold
```

失败时检查：

如果 `git status --short` 有 `M` 或 `??`，先确认是不是需要提交的代码。需要提交就执行：

```powershell
git add .
git commit -m "chore: save local migration state"
```

---

## 2. WinSCP 整包迁移

在 WinSCP 中：

```text
本机目录：
D:\Desktop_D\postgraduate\研二\big\big_1_second

5090 建议目录：
/home/<your_user>/projects/big_1_second
```

建议传输内容：

```text
整个 big_1_second 文件夹
```

包括：

```text
.git/
configs/
data/
docs/
models/
scripts/
src/
tests/
requirements-stage1.txt
thesis_research_opening_plan.md
```

目的：

直接把本机已经下载好的 ChemProt raw、转换后的 JSONL、BioBART 模型、本地 git 历史一起迁移到 5090。

成功标志：

在 5090 上执行：

```bash
cd /home/<your_user>/projects/big_1_second
ls
ls models/stage1/biobart-base
ls data/stage1/chemprot
git log --oneline -2
```

应能看到：

```text
models/stage1/biobart-base/model.safetensors
data/stage1/chemprot/train.jsonl
data/stage1/chemprot/dev.jsonl
data/stage1/chemprot/test.jsonl
2a631c6 feat: add biobart hf training smoke
```

失败时检查：

1. 如果没有 `.git/`，说明 WinSCP 隐藏文件没传。开启显示隐藏文件后重新传。
2. 如果没有 `models/stage1/biobart-base/model.safetensors`，说明模型目录没传。
3. 如果没有 `data/stage1/chemprot/*.jsonl`，说明转换后数据没传。

---

## 3. 5090 上创建实验环境

在 5090 shell 中执行：

```bash
cd /home/<your_user>/projects/big_1_second
conda create -n rsg-biore python=3.10 -y
conda activate rsg-biore
pip install -r requirements-stage1.txt
```

目的：

创建独立环境，安装 Stage 1 所需依赖。

成功标志：

```bash
python -c "import torch, transformers; print(torch.__version__); print(transformers.__version__); print(torch.cuda.is_available())"
```

应看到：

```text
torch version
transformers version
True
```

其中 `torch.cuda.is_available()` 在 5090 上应为 `True`。

失败时检查：

1. 如果 `conda: command not found`，说明服务器没有初始化 conda，需要联系管理员或使用已有环境。
2. 如果 `torch.cuda.is_available()` 是 `False`，先检查：

```bash
nvidia-smi
python -c "import torch; print(torch.version.cuda)"
```

3. 如果 pip 安装太慢，可以换实验室常用镜像源，但不要改代码。

---

## 4. 5090 上做迁移 smoke

### 4.1 跑单元测试

```bash
python -m unittest discover -s tests/stage1 -v
```

目的：

确认代码、路径、schema、converter、metrics、runner 都能正常工作。

成功标志：

```text
Ran 10 tests ... OK
```

失败时检查：

1. 是否在项目根目录执行。
2. 是否激活 `rsg-biore` 环境。
3. 是否缺少 `data/stage1/tiny` 或 `data/stage1/chemprot`。

### 4.2 跑 BioBART smoke

```bash
python scripts/stage1/train_stage1.py --config configs/stage1/chemprot_hf_biobart_smoke.yaml
```

目的：

确认 5090 能加载本地 BioBART、执行 1 step 训练、生成预测、解析输出、写 metrics。

成功标志：

命令结束后有 JSON 输出，并生成：

```text
outputs/stage1/chemprot/biobart_text2text/P2_entity_type_description_seed42_hf_smoke/
  run_config.yaml
  metrics.json
  predictions.jsonl
  per_class_metrics.csv
  confusion_matrix.csv
  error_cases.md
  train_log.txt
  model/
```

注意：

当前 smoke config 只有：

```text
max_train_steps = 1
max_train_samples = 2
max_dev_samples = 2
max_test_samples = 2
```

所以结果不是正式实验结果，只是流程验证。

失败时检查：

1. 如果报找不到模型：

```bash
ls models/stage1/biobart-base
```

2. 如果报 CUDA out of memory，先把 smoke config 的 `batch_size` 保持为 1。
3. 如果生成结果格式非法，不代表流程失败；1 step smoke 本来不会学会稳定输出。

---

## 5. GitHub 远程仓库与 SSH 配置

GitHub 官方 SSH 文档建议流程是：生成 SSH key、把 public key 添加到 GitHub、测试 SSH 连接，然后使用 SSH remote 做 Git 操作。

官方参考：

```text
https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent
https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account
https://docs.github.com/en/authentication/connecting-to-github-with-ssh/testing-your-ssh-connection
```

### 5.1 在 GitHub 网页创建空仓库

在 GitHub 创建一个新仓库，例如：

```text
rsg-biore-stage1
```

建议：

```text
不要勾选 README
不要勾选 .gitignore
不要勾选 License
```

目的：

因为本地已经是一个完整 git repo，GitHub 远程仓库应为空，避免首次 push 冲突。

成功标志：

创建后 GitHub 显示类似 SSH 地址：

```text
git@github.com:<your_github_name>/rsg-biore-stage1.git
```

---

## 6. 本机配置 GitHub SSH

在本机 PowerShell 执行：

```powershell
ssh-keygen -t ed25519 -C "your_email@example.com"
```

一路回车即可，或给 key 设置 passphrase。

查看公钥：

```powershell
type $env:USERPROFILE\.ssh\id_ed25519.pub
```

复制输出内容，添加到 GitHub：

```text
GitHub -> Settings -> SSH and GPG keys -> New SSH key
```

测试连接：

```powershell
ssh -T git@github.com
```

成功标志：

会出现类似：

```text
Hi <username>! You've successfully authenticated, but GitHub does not provide shell access.
```

失败时检查：

1. 公钥是不是 `.pub` 文件内容，不要复制私钥。
2. GitHub 上是否添加到正确账号。
3. 如果是第一次连接，看到 fingerprint 提示，输入 `yes`。

---

## 7. 本机绑定远程仓库并 push

在本机项目根目录执行：

```powershell
cd D:\Desktop_D\postgraduate\研二\big\big_1_second
git remote add origin git@github.com:<your_github_name>/rsg-biore-stage1.git
git branch -M main
git push -u origin main
```

目的：

把当前本机 git 历史推到 GitHub。

成功标志：

GitHub 仓库页面出现项目文件，并且本机执行：

```powershell
git remote -v
git status --short
```

能看到：

```text
origin git@github.com:<your_github_name>/rsg-biore-stage1.git (fetch)
origin git@github.com:<your_github_name>/rsg-biore-stage1.git (push)
```

且 `git status --short` 没有输出。

失败时检查：

1. 如果 remote 已存在：

```powershell
git remote set-url origin git@github.com:<your_github_name>/rsg-biore-stage1.git
```

2. 如果 push 被拒绝，确认 GitHub 仓库是否为空。
3. 如果 SSH 失败，回到第 6 节重新测试 `ssh -T git@github.com`。

---

## 8. 5090 配置 GitHub SSH

在 5090 上执行：

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub
```

把 5090 的 public key 也添加到 GitHub 的 SSH keys。

测试连接：

```bash
ssh -T git@github.com
```

成功标志：

```text
Hi <username>! You've successfully authenticated, but GitHub does not provide shell access.
```

重要说明：

本机和 5090 最好各自生成自己的 SSH key，不要把本机私钥复制到服务器。

---

## 9. 5090 绑定同一个 GitHub remote

因为你是通过 WinSCP 整包迁移，5090 上已经有 `.git/`。在 5090 项目根目录执行：

```bash
cd /home/<your_user>/projects/big_1_second
git remote -v
```

如果没有 remote：

```bash
git remote add origin git@github.com:<your_github_name>/rsg-biore-stage1.git
git branch -M main
```

如果 remote 地址不对：

```bash
git remote set-url origin git@github.com:<your_github_name>/rsg-biore-stage1.git
```

然后执行：

```bash
git pull --ff-only origin main
```

目的：

让 5090 上的项目和 GitHub 主分支对齐。

成功标志：

```bash
git status --short
git log --oneline -2
```

`git status --short` 没有输出；`git log` 能看到和本机相同的提交。

失败时检查：

1. 如果提示 divergent branches，不要强行 merge，先把错误信息发回来。
2. 如果有本地未提交修改，先执行：

```bash
git status --short
```

确认是否是实验输出文件。如果是 outputs，通常应被 `.gitignore` 忽略。

---

## 10. 以后本机和 5090 如何同步

### 10.1 本机修改代码后同步到 GitHub

本机：

```powershell
git status --short
git add .
git commit -m "feat: describe your change"
git push
```

成功标志：

GitHub 页面能看到新 commit。

### 10.2 5090 拉取最新代码

5090：

```bash
git pull --ff-only
```

成功标志：

```bash
git log --oneline -2
```

能看到本机刚 push 的 commit。

### 10.3 5090 如果只跑实验，不改代码

不要提交 outputs、models、raw data。它们应被 `.gitignore` 忽略。

检查：

```bash
git status --short
```

理想状态是没有输出。

---

## 11. 正式实验前必须改配置

当前 `chemprot_hf_biobart_smoke.yaml` 是 smoke，不是正式实验。

正式实验必须新建 full config，例如：

```text
configs/stage1/chemprot_hf_biobart_P2_full_seed42.yaml
```

必须删除或不设置：

```text
max_train_steps
max_train_samples
max_dev_samples
max_test_samples
```

建议正式参数起点：

```json
{
  "experiment_id": "chemprot_hf_biobart_P2_full_seed42",
  "dataset": "ChemProt",
  "model": "GanjinZero/biobart-base",
  "model_name_or_path": "models/stage1/biobart-base",
  "method": "P2_entity_type_description",
  "backend": "hf",
  "semantic_field": "entity_type_aware_description",
  "seed": 42,
  "epochs": 5,
  "max_input_length": 512,
  "max_output_length": 32,
  "batch_size": 8,
  "learning_rate": 0.00003,
  "gradient_clip_norm": 1.0,
  "train_file": "data/stage1/chemprot/train.jsonl",
  "dev_file": "data/stage1/chemprot/dev.jsonl",
  "test_file": "data/stage1/chemprot/test.jsonl",
  "schema_file": "data/stage1/chemprot/relation_schema.yaml",
  "output_dir": "outputs/stage1/chemprot/biobart_text2text/P2_entity_type_description_seed42_full"
}
```

运行：

```bash
python scripts/stage1/train_stage1.py --config configs/stage1/chemprot_hf_biobart_P2_full_seed42.yaml
```

成功标志：

输出目录中至少有：

```text
metrics.json
predictions.jsonl
per_class_metrics.csv
confusion_matrix.csv
error_cases.md
train_log.txt
model/
```

重要：

当前 `HfText2TextModel` 已支持 text-to-text baseline fine-tuning smoke。RSG-BioRE 的真实 PyTorch prototype alignment 训练还需要继续实现。因此正式大实验应先跑 P2/P3 text-to-text baseline，等 RSG 真实训练代码补齐后再跑 R1/R2。

---

## 12. 实验结果如何传回本机分析

每次实验结果都在：

```text
outputs/stage1/<dataset>/<model_or_method>/<run_name>/
```

不要只传 `metrics.json`。至少传整个 run 目录。

5090 上打包 ChemProt 全部结果：

```bash
tar -czf stage1_chemprot_outputs.tar.gz outputs/stage1/chemprot
```

或者只打包某个 run：

```bash
tar -czf chemprot_biobart_P2_seed42_full.tar.gz outputs/stage1/chemprot/biobart_text2text/P2_entity_type_description_seed42_full
```

成功标志：

```bash
ls -lh *.tar.gz
```

能看到压缩包，并且大小不是 0。

用 WinSCP 把压缩包传回本机后，建议放到：

```text
D:\Desktop_D\postgraduate\研二\big\big_1_second\imported_results\
```

解压后告诉我目录，例如：

```text
D:\Desktop_D\postgraduate\研二\big\big_1_second\imported_results\outputs\stage1\chemprot
```

我后续可以做：

1. 主结果表。
2. per-class F1 表。
3. rare relation F1 分析。
4. valid output / relation validity 分析。
5. error_cases 汇总。
6. 论文可写的实验结论。

---

## 13. 最小执行清单

### 第一次迁移必须做

```bash
cd /home/<your_user>/projects/big_1_second
conda create -n rsg-biore python=3.10 -y
conda activate rsg-biore
pip install -r requirements-stage1.txt
python -m unittest discover -s tests/stage1 -v
python scripts/stage1/train_stage1.py --config configs/stage1/chemprot_hf_biobart_smoke.yaml
```

### GitHub SSH 必须做

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub
ssh -T git@github.com
git remote add origin git@github.com:<your_github_name>/rsg-biore-stage1.git
git branch -M main
git pull --ff-only origin main
```

### 正式实验前必须确认

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
git status --short
```

成功标志：

```text
GPU 可见
torch cuda True
git 工作区干净
smoke 已跑通
正式 config 不再包含 max_train_steps / max_*_samples
```
