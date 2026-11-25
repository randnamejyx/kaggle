下面是一套从训练到提交的完整终端指令。请先在 Kaggle 撤销之前暴露的 token，并创建新 token。

## 1. 进入项目

```bash
cd /home/jingyx/kaggle
```

查看数据：

```bash
ls -lh data
head data/train.csv
head data/test.csv
```

预期文件：

```text
data/train.csv
data/test.csv
data/sample_submission.csv
```

## 2. 训练并生成提交文件

推荐命令：

```bash
/home/jingyx/anaconda3/bin/python3 train_baseline.py \
  --train data/train.csv \
  --test data/test.csv \
  --text-column text \
  --output artifacts
```

这里不要强制使用错误的 `sample_submission.csv` ID；脚本会直接采用评分器要求的 `test.csv` 原始 ID。

输出文件：

```text
artifacts/model.joblib
artifacts/metrics.json
artifacts/submission.csv
```

查看验证指标：

```bash
cat artifacts/metrics.json
```

查看提交文件：

```bash
head artifacts/submission.csv
wc -l artifacts/submission.csv
```

应为：

```text
10002 artifacts/submission.csv
```

即一行表头加 10,001 行预测。

## 3. 提交前校验

```bash
/home/jingyx/anaconda3/bin/python3 - <<'PY'
import pandas as pd

test = pd.read_csv("data/test.csv", dtype={"id": str})
sub = pd.read_csv("artifacts/submission.csv", dtype={"id": str})

assert list(sub.columns) == ["id", "label"]
assert len(sub) == len(test) == 10001
assert sub["id"].equals(test["id"])
assert sub["id"].is_unique
assert not sub.isna().any().any()
assert set(sub["label"]).issubset({0, 1})

print("提交文件校验通过")
print(sub["label"].value_counts().sort_index())
PY
```

必须看到：

```text
提交文件校验通过
```

## 4. 临时配置 Kaggle token

在 Kaggle 账户设置里创建一个新的 API token。不要把 token 写进命令历史或聊天。

```bash
read -rsp 'Kaggle API Token: ' KAGGLE_API_TOKEN
printf '\n'

export KAGGLE_API_TOKEN='KGAT_e32dab5500749153b1a24eeef6d85a1d'
export KAGGLE_USERNAME='randname114514'
export KAGGLE_KEY="$KAGGLE_API_TOKEN"
```

这三个变量只在当前终端有效。检查变量是否存在，但不要打印内容：

```bash
printf 'token=%s username=%s key=%s\n' \
  "${KAGGLE_API_TOKEN:+configured}" \
  "${KAGGLE_USERNAME:+configured}" \
  "${KAGGLE_KEY:+configured}"
```

预期：

```text
token=configured username=configured key=configured
```

## 5. 检查历史提交

```bash
.venv-kaggle/bin/kaggle competitions submissions \
  -c hate-speech-detection-in-english
```

## 6. 正式提交

```bash
.venv-kaggle/bin/kaggle competitions submit \
  -c hate-speech-detection-in-english \
  -f artifacts/submission.csv \
  -m "word-char TF-IDF logistic regression"
```

提交后查询评分：

```bash
.venv-kaggle/bin/kaggle competitions submissions \
  -c hate-speech-detection-in-english
```

正常状态变化：

```text
PENDING
COMPLETE
```

当前已完成提交的最好成绩是：

```text
Submission ref: 58818655
Public F1:      0.8578
Private F1:     0.8590
```

## 7. 清除终端凭据

提交完成后执行：

```bash
unset KAGGLE_API_TOKEN KAGGLE_USERNAME KAGGLE_KEY
```

确认已清除：

```bash
printf 'token=%s key=%s\n' \
  "${KAGGLE_API_TOKEN:+still-present}" \
  "${KAGGLE_KEY:+still-present}"
```

应显示空值。

## 一键流程摘要

每次重新训练和提交时主要执行：

```bash
cd /home/jingyx/kaggle

/home/jingyx/anaconda3/bin/python3 train_baseline.py \
  --train data/train.csv \
  --test data/test.csv \
  --text-column text \
  --output artifacts

read -rsp 'Kaggle API Token: ' KAGGLE_API_TOKEN
printf '\n'
export KAGGLE_API_TOKEN
export KAGGLE_USERNAME='randname114514'
export KAGGLE_KEY="$KAGGLE_API_TOKEN"

.venv-kaggle/bin/kaggle competitions submit \
  -c hate-speech-detection-in-english \
  -f artifacts/submission.csv \
  -m "word-char TF-IDF logistic regression"

.venv-kaggle/bin/kaggle competitions submissions \
  -c hate-speech-detection-in-english

unset KAGGLE_API_TOKEN KAGGLE_USERNAME KAGGLE_KEY
```

注意：这个赛题附带的 `sample_submission.csv` 使用 `0…10000`，但评分器实际要求 `test.csv` 原始 ID。不要手动把提交 ID 改成样例中的行号。