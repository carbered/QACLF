# ABMIL/TransMIL 匹配基线实验说明

## 实验目的

该实验用于扩展 QACLF 论文的病理主干比较范围，回答 QACLF 的实验结论是否仅建立在 VLSA 一种病理模型之上。ABMIL 代表经典注意力多实例学习，TransMIL 代表 Transformer 多实例学习。

## 数据与协议

- 使用已经提取的 `N x 512` CONCH patch 特征，不读取原始 WSI。
- 使用与 matched VLSA 相同的五个 TCGA 队列和 outer 5-fold 患者划分。
- 固定训练随机种子为 42，禁用 early stopping 和学习率调度，评估原始最后一轮 checkpoint。
- ABMIL 和 TransMIL 使用 VLSA 项目自带的 SA 实现及其官方 SA 训练目标 `SurvIFMLE`。
- GBMLGG 使用 5 epoch、16 位患者梯度累积；其余队列使用 10 epoch、32 位患者梯度累积。这些设置继承自各队列的官方 `cfg_sa_base_conch.yaml`。
- 两种模型共享 CONCH 特征、患者划分、生存终点、训练种子与 SurvivalEVAL 评估代码；仅改变 WSI 聚合网络。

这种设置属于“输入、划分和评估协议匹配，模型训练方案遵循官方实现”，不应表述为所有模型具有完全相同的网络或损失函数。

## 云端运行

先执行单折烟雾测试：

```bash
cd /tmp/pycharm_project_456
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PY=/root/miniconda3/envs/vlsa_env/bin/python

$PY tools/run_matched_mil_backbones.py prepare --verify-tensors

WANDB_MODE=offline $PY tools/run_matched_mil_backbones.py run \
  --models ABMIL TransMIL \
  --datasets brca \
  --folds 0
```

烟雾测试成功后运行全部 50 个训练任务：

```bash
mkdir -p logs
LOG=logs/matched_mil_backbones_$(date +%Y%m%d_%H%M%S).log

WANDB_MODE=offline nohup $PY -u tools/run_matched_mil_backbones.py run \
  --models ABMIL TransMIL \
  --datasets blca brca gbmlgg luad ucec \
  --folds 0 1 2 3 4 \
  > "$LOG" 2>&1 &

PID=$!
echo "PID=$PID"
echo "LOG=$LOG"
```

运行器具有断点续跑能力。已有 `train_metrics-last.txt` 的任务会自动跳过。

## 查看进度

```bash
tail -f "$LOG"
```

```bash
find result/matched-mil-backbones/seed_42 \
  -name "train_metrics-last.txt" | wc -l
```

```bash
cat logs/matched_mil_backbones/run_status.csv
```

完整任务数为 50。若计数为 25，通常表示 ABMIL 已完成而 TransMIL 尚未完成。

## 汇总结果

```bash
$PY tools/run_matched_mil_backbones.py summarize
```

输出文件：

- `result/matched-mil-backbones/summary/fold_metrics.csv`
- `result/matched-mil-backbones/summary/dataset_summary.csv`
- `result/matched-mil-backbones/summary/model_summary.csv`
- `result/matched-mil-backbones/summary/summary_audit.json`

正式论文中应报告五折均值和标准差。Published VLSA 继续标记为原论文参考值，Matched VLSA、ABMIL-CONCH 与 TransMIL-CONCH 标记为统一输入和 outer-fold 协议下的本地复现结果。
