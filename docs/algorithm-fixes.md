# 影响预测成绩的问题修复与重跑

本次沿用已有模型、训练参数和数据划分，修复三类问题：同分概率下的阈值选择、融合分数的零方差，以及跨实验复用预测缓存时的样本错位。凭据、提交工具和依赖环境不在本次修改范围。

## 实施与验证

- [x] 用实际可实现的阈值验证 Transformer 的 F1 搜索；保持 `score >= threshold` 的预测约定，同分样本必须一起处理。推理时以 float32 计算 softmax。
- [x] 用常量候选检验融合；零标准差采用尺度 1，保留正常候选原有的归一化方式。
- [x] 保存并核对预测缓存对应的有序训练样本、测试样本、文本列与折划分；缺少来源信息的旧缓存需要重新生成。
- [x] 运行回归测试和小型合成数据的完整训练、融合流程。
- [x] GPT-6.1 Sol / high 独立代码审阅，未发现新增阻塞问题。

缓存检查用于避免将分数配到错误的标签或测试 ID。它不会把 ID 加入模型特征。为保持行为明确，输入顺序改变时拒绝旧缓存，要求重新运行基础五折模型。

原分支开发环境运行 `python -m pytest -q`，16 项测试通过：同分阈值与可实现阈值穷举结果一致；常量融合候选不再产生 NaN；样本顺序、内容、标签、ID 或折划分变化时拒绝旧缓存；匹配缓存可完成两种融合并导出 CSV。测试中的五折训练和融合使用合成数据；Transformer 仅执行真实的 NumPy 阈值函数。原分支提交时尚未运行真实数据的微调、GPU 推理或 Kaggle 评分；下文记录在原 GPU 服务器上的完整重跑。

## 原数据重跑与 Kaggle 验证（2026 年 11 月 25 日）

- 在 `fix/algorithm-score-correctness` 分支上运行 `python -m pytest -q`，16 项测试通过。
- 按下文原参数在 RTX 3090 上重跑 BERTweet，输出至 `artifacts/bertweet_algorithm_fixed/`。最佳仍为第 2 轮，验证 F1 为 `0.9534492428491307`；修复后的阈值为 `0.4877507984638214`。新 `test_scores.npy` 使用 float32，旧文件使用 float16。
- 按依赖顺序重跑五折 NB-SVM、普通融合和 `char_wb` 融合，分别输出至 `artifacts/cv_algorithm_fixed/`、`artifacts/blend_algorithm_fixed/`、`artifacts/charwb_algorithm_fixed/`。三组 OOF F1 分别为 `0.8898388688`、`0.8921098518`、`0.8934506354`；缓存签名检查通过。
- 四份重跑后的 `submission.csv` 均与各自旧版逐字节相同，测试集预测标签变化数均为 `0`。本次修复提高了阈值计算精度和缓存复用校验，没有改变最终预测。
- 提交 `artifacts/bertweet_algorithm_fixed/submission.csv` 至 Kaggle：submission ref `58875972`，状态 `COMPLETE`，public F1 `0.9523`，private F1 `0.9560`。这与旧版 BERTweet 提交 `58832772` 的两项分数相同。

## 验证与最终 CSV 的关系

比赛最终上传 `submission.csv`，其中只有 `id,label` 两列。但生成文件前仍要用训练集中的带标签验证样本选择 epoch 和阈值。完整流程是：训练与验证 → 恢复选中的模型 → 按原流程全量追加训练 → 预测测试集 → 导出 CSV。Kaggle 用隐藏标签评估这个 CSV。

只将旧最终模型重新预测一次，不能检验这次阈值修复：旧脚本没有保存各轮验证概率或各轮检查点，最终保存权重也已经学习过原验证集。不能用这些最终权重在原验证集上重新调阈值并把分数当作独立验证结果。

### 重跑 BERTweet

在原 Linux GPU 服务器的项目目录运行。需要已经存在的 `data/train.csv`、`data/test.csv` 和原训练环境；GPU 编号按实际机器选择。以下固定种子和原报告参数，使用新目录保留旧结果。

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
CUDA_VISIBLE_DEVICES=1 \
.venv-transformer/bin/python -u train_transformer.py \
  --train data/train.csv --test data/test.csv \
  --model vinai/bertweet-base \
  --epochs 4 --batch-size 32 --max-length 128 --threads 16 \
  --learning-rate 2e-5 --positive-weight 1.5 --seed 42 \
  --output artifacts/bertweet_algorithm_fixed
```

新提交文件为 `artifacts/bertweet_algorithm_fixed/submission.csv`。同时保留该目录的 `metrics.json`、`test_scores.npy` 和模型，以便检查这次运行。`best_validation` 是全量追加训练前、参与模型选择的验证分数；它不等于最终权重的独立测试分数。

### 重跑传统模型融合

如果只准备提交 BERTweet，则不必重新训练这些传统模型。要比较融合结果时，先重跑五折模型生成带来源信息的新缓存，再运行融合：

```bash
python train_cv_ensemble.py --output artifacts/cv_algorithm_fixed
python train_blend_ensemble.py \
  --base artifacts/cv_algorithm_fixed --output artifacts/blend_algorithm_fixed
python train_charwb_ensemble.py \
  --previous artifacts/cv_algorithm_fixed --output artifacts/charwb_algorithm_fixed
```

三个模型各自生成一个 `submission.csv`；它们不是三个必须一起上传的文件。按预先确定的比较方式选定一个结果，再在 Kaggle 上传对应 CSV。算法回归测试通过只证明已知错误得到修复，不能保证隐藏集分数一定上升。
