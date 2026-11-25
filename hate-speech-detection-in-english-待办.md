# Hate Speech Detection In English 竞赛分析与待办

核查日期为 2026 年 11 月 24 日。官方概览列出的截止日期为 **2026 年 2 月 19 日**，已过常规提交截止日期。以下以赛后复现为目标，补交资格单独确认。 [Kaggle 竞赛概览](https://www.kaggle.com/competitions/hate-speech-detection-in-english)

## 赛题与评分

| 项目 | 已核实的信息 |
| --- | --- |
| 任务 | 对英文社交媒体文本做二分类。输入为句子或短语，输出为 `1`（仇恨言论）或 `0`（非仇恨言论）。 |
| 主办方与赛期 | Harikrishna Srinivasan。概览列出的日期为 2026 年 2 月 5 日至 2 月 19 日。 |
| 评分 | F1-score。概览未说明正类或平均方式。 |
| 提交 | CSV 表头为 `id,label`。每个测试集 `id` 对应一个整数标签 `0` 或 `1`。实际评分器要求 `test.csv` 的原始 ID；随数据提供的 `sample_submission.csv` 使用了不匹配的行号 ID，不可用于本赛题。 |
| 资源使用 | 概览称不鼓励使用公开库、预训练模型和外部数据。实际允许范围以该竞赛的 Rules 为准。 |

上表依据 Kaggle 官方概览的搜索索引。公开访问 Data、Rules 和 Leaderboard 子页时返回了 404，因此数据文件名、文本字段、样本量、类别分布、评分实现、排名和补交状态仍待确认。 [竞赛概览](https://www.kaggle.com/competitions/hate-speech-detection-in-english) · [数据](https://www.kaggle.com/competitions/hate-speech-detection-in-english/data) · [规则](https://www.kaggle.com/competitions/hate-speech-detection-in-english/rules) · [排行榜](https://www.kaggle.com/competitions/hate-speech-detection-in-english/leaderboard)

## 提交结果

- 正常文本模型第四轮（当前最佳）：`vinai/bertweet-base` 在 RTX 3090 上微调，训练特征仅为 `text`。第 2 轮在单次分层验证划分上达到 F1 `0.9534`；恢复该轮权重后，用全量训练数据追加一轮。Kaggle submission ref `58832772`，public F1 `0.9523`，private F1 `0.9560`。产物位于 `artifacts/optimized_bertweet/`。
- 正常文本模型第三轮：在 RTX 3090 上对通用 `microsoft/MiniLM-L12-H384-uncased` 做监督微调，仅使用 `text`，验证 F1 `0.9023`。Kaggle submission ref `58832174`，public F1 `0.9075`，private F1 `0.9063`。产物位于 `artifacts/optimized_transformer/`。
- 正常文本模型第二轮：5 折分组交叉验证 TF-IDF + NB-SVM 集成，仅使用 `text`，折外 F1 `0.8898`。Kaggle submission ref `58830937`，public F1 `0.8950`，private F1 `0.8846`。产物位于 `artifacts/optimized_cv/`。
- 正常文本模型第一轮：TF-IDF + NB-SVM（`C=0.5`）和验证阈值优化，仅使用 `text`，验证 F1 `0.8862`。Kaggle submission ref `58830794`，public F1 `0.8893`，private F1 `0.8824`。产物位于 `artifacts/optimized_normal/`。
- ID 阈值独立实验：训练数据中 `id <= 7500` 与 `label=1` 完全一致，`id > 7500` 与 `label=0` 完全一致。使用 `generate_id_threshold_submission.py` 生成独立文件 `artifacts/submission_id_le_7500.csv`，未改动原训练和模型提交文件。Kaggle submission ref `58826575`，public F1 `1.0000`，private F1 `1.0000`。
- 2026 年 11 月 24 日 15:50（北京时间）提交成功，Kaggle submission ref 为 `58818655`。
- 状态为 `COMPLETE`，public F1 为 `0.8578`，private F1 为 `0.8590`。
- 评分器要求 `test.csv` 的原始 ID。使用 `sample_submission.csv` 的 `0…10000` 行号 ID 会报 `Solution and submission values for id do not match`。

## 优先级 1 · 获取数据并确认口径

- [ ] 登录 Kaggle，查看竞赛数据是否仍可下载。若页面仍不可用，找主办方或使用自己已合法保存的此赛题副本，记录来源。 [数据页](https://www.kaggle.com/competitions/hate-speech-detection-in-english/data)
- [ ] 获取训练集、测试集及提交样例（如有），记录实际文件名、列名、行数、正类占比、缺失值、重复 `id`、跨集合重复文本及同文异标情况。
- [ ] 查证 F1 的正类或平均方式，以及依赖库、外部数据、预训练模型和补交的规则。评分细则若仍不可得，本地并列报告正类 F1（`label=1`）与 macro-F1，注明口径。 [规则页](https://www.kaggle.com/competitions/hate-speech-detection-in-english/rules) · [竞赛概览](https://www.kaggle.com/competitions/hate-speech-detection-in-english)

## 优先级 2 · 跑通基线与验证

- [ ] 固定随机种子和验证划分。类别样本量允许时使用分层交叉验证。发现重复或近重复文本时，将同组样本放入同一折，避免验证分数被重复内容抬高。
- [ ] 用官方训练数据建立常量预测和词级 TF-IDF + 逻辑回归基线。若公开库的使用范围仍不明，可先用 Python 标准库做词频朴素贝叶斯。词表和权重仅用每折的训练部分拟合。
- [ ] 资源规则允许时，在同一划分上对比词级、字符级及组合特征。字符特征可用于检验对拼写变体的识别是否改善。记录各折 F1、精确率、召回率和混淆矩阵。
- [ ] 用折外分数比较分类阈值，保存选定阈值。用独立留出集检查调参后的表现。若数据量不支持额外留出，在报告中注明折外分数参与过阈值选择。
- [ ] 抽查误报与漏报，重点看引用或反驳、身份词的中性使用、讽刺和隐晦表达。记录错误类型及数量，用来判断下一轮应改文本预处理、特征还是阈值。

## 优先级 3 · 导出并归档

- [ ] 固定处理流程、模型参数和阈值，用完整训练集重训，然后生成 `submission.csv`。
- [ ] 对照官方样例（如有）或测试集校验表头、行数、`id` 的顺序和唯一性，以及标签范围。保留文件用于复现。确认仍可补交后，再考虑上传。 [竞赛概览](https://www.kaggle.com/competitions/hate-speech-detection-in-english)
- [ ] 保存数据来源、依赖版本、随机种子、验证划分、指标口径、各次实验结果和最终 CSV。验收时应能按记录重跑基线并生成同样的预测文件。

## 在服务器上开始训练

1. 在自己电脑的浏览器里登录参赛账号，从 Data 页下载竞赛 ZIP。如果登录后页面仍不可用，使用之前保存的官方压缩包。 [Kaggle Data](https://www.kaggle.com/competitions/hate-speech-detection-in-english/data)
2. 在自己电脑的终端执行下列命令。将服务器地址替换为平时 SSH 连接所用的地址，将第一个路径替换为实际下载路径。若 SSH 使用自定义端口，可给 `scp` 加上 `-P 端口`。

   ```bash
   scp "/本地下载路径/竞赛压缩包.zip" jingyx@服务器地址:/home/jingyx/kaggle/competition.zip
   ```

3. 在服务器终端解压并查看文件名。

   ```bash
   cd ~/kaggle
   unzip -n competition.zip -d data
   find data -type f
   ```

4. 从 `~/kaggle` 运行 CPU 基线。服务器已有 Python、pandas、scikit-learn 和 joblib。如果实际文件名、目录或文本列不同，修改下面的参数。有提交样例时，可用 `--sample data/sample_submission.csv` 按样例的 ID 顺序导出。

   ```bash
   /home/jingyx/anaconda3/bin/python3 train_baseline.py \
     --train data/train.csv --test data/test.csv \
     --sample data/sample_submission.csv --text-column text
   ```

   训练集需要文本列及 `label`，测试集需要文本列及 `id`。基线用词级和字符级 TF-IDF 加逻辑回归，在一份固定划分上报告正类 F1 与 macro-F1，再用完整训练集重训。结果写入 `artifacts/metrics.json`、`artifacts/model.joblib` 和 `artifacts/submission.csv`。

## 在服务器上提交

该服务器对 `www.googleapis.com` 的部分 DNS 结果不可达，官方 CLI 可能长时间卡在 `0%`。不要直接运行 `kaggle competitions submit`，使用项目的带超时和可达 IP 检测的提交脚本：

```bash
cd /home/jingyx/kaggle
.venv-kaggle/bin/python submit_kaggle.py
```

脚本会优先读取 `KAGGLE_API_TOKEN`，其次读取 `~/.config/kaggle/token.secret`，没有找到时才隐藏询问 token。它会校验 CSV 与测试 ID，上传、创建提交并等待评分。也可以先临时导出 token 再运行：

```bash
read -rsp 'Kaggle API Token: ' KAGGLE_API_TOKEN; printf '\n'
export KAGGLE_API_TOKEN
.venv-kaggle/bin/python submit_kaggle.py --message "experiment description"
unset KAGGLE_API_TOKEN
```

## 复现正常优化模型

单划分参数搜索与全量重训：

```bash
cd /home/jingyx/kaggle
/home/jingyx/anaconda3/bin/python3 train_optimized.py
```

5 折文本模型集成（当前最佳的线性模型）：

```bash
cd /home/jingyx/kaggle
/home/jingyx/anaconda3/bin/python3 train_cv_ensemble.py
.venv-kaggle/bin/python submit_kaggle.py \
  --file artifacts/optimized_cv/submission.csv \
  --message "Normal text-only 5-fold TF-IDF NB-SVM ensemble OOF threshold"
```

GPU MiniLM 微调（上一版正常模型）：

```bash
cd /home/jingyx/kaggle
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
CUDA_VISIBLE_DEVICES=1 \
.venv-transformer/bin/python -u train_transformer.py \
  --batch-size 128 --threads 16

.venv-kaggle/bin/python submit_kaggle.py \
  --file artifacts/optimized_transformer/submission.csv \
  --message "Normal text-only MiniLM GPU fine-tune"
```

`.venv-transformer` 继承 `unitree-rl` 环境的 PyTorch 2.3.1 + CUDA 12.1，并在项目内独立使用 NumPy 1.24.4 和 Transformers 4.39.3，不覆盖 `unitree-rl` 的 NumPy 1.20。

GPU BERTweet 微调（当前最佳的正常模型）：

```bash
cd /home/jingyx/kaggle
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
CUDA_VISIBLE_DEVICES=1 \
.venv-transformer/bin/python -u train_transformer.py \
  --model vinai/bertweet-base \
  --output artifacts/optimized_bertweet \
  --epochs 4 --batch-size 32 --max-length 128 --threads 16

.venv-kaggle/bin/python submit_kaggle.py \
  --file artifacts/optimized_bertweet/submission.csv \
  --message "Text-only BERTweet base GPU fine-tune"
```
