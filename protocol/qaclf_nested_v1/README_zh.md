# MCAT/CAT 规范重训 v1

日期：2026-10-26。复用原 AutoDL 实例、`vlsa_env`、CONCH 特征、组学表及 RN50 缓存。不安装环境，不下载模型，不修改旧代码和旧结果。

## 这次改变什么

- 保持原有 CONCH 适配架构、原生四段 sigmoid hazard、NLL(alpha=0)、Adam 学习率 1e-4、weight decay 1e-5、seed=1 和 16 患者梯度累积。
- 修正每轮开始清梯度、末尾不足 16 人的归一化与更新；每步检查 loss/梯度是否有限。
- 25 个外层划分不变。每个外层训练集再作固定 80/20 的事件状态分层划分，两个模型共用相同内部患者划分。
- 内部拟合患者独立拟合四段事件分位数边界和组学 StandardScaler。内部验证仅用于选择 1 至 20 轮中 NLL 最小的轮数，同值选更早一轮，不看外层测试成绩。
- 选定轮数后重新初始化，用所有外层训练患者重拟合分箱、缩放器并训练所选轮数，最后只评价一次外层测试。
- 改用普通随机打乱、每患者每轮一次，不再按事件/时间类别逆频率重采样。这是新协议中的明确改变，不冒充与旧训练完全相同。
- 保留旧模型实际使用的签名基因词表，仅对这些基因列做缩放，排除结局、来源划分标志和 CSV 索引列。旧接口按前 12 列判断元数据，在不同队列表上并不稳健；此次检查发现 LUAD 的 censorship 曾进入一般缩放表，但没有进入六类签名网络输入，不能据此宣称已经发现旧模型直接使用结局标签。

## 时间与评价协议

第一边界为 0，中间三个边界由当前拟合患者的事件时间分位数得到，末边界为当前拟合患者最大观测时间加 1e-6。该协议明确采用有限右端点，不再声称这些月份坐标由原始离散模型唯一决定。

内部验证或外层测试观测时间超过拟合上限时，仅在离散 NLL 目标中作行政删失。最终 C-index 与概率指标使用另行保留的原始连续随访时间和事件，不改动真实结局。外层 C-index 使用原生风险负生存概率和，不使用预测均值风险。

概率评价使用 S(0)=1 与四个有限区间右端点，按原生概率线性插值，不强制末端归零。统一报告 12/36/60 月 IPCW Brier 及 IBS[12,60]，删失分布仅由外层训练患者估计。任何折不能支持固定上限时明确报错，不静默跳折。

完整 Mean/Linear hinge MAE 仅保留为带额外尾部假设的诊断，不保证变小。若无法得到有限均值，输出非有限计数与空诊断值，不删除异常患者。不能用这些新范围指标直接替换旧表而不改方法、表头和全部八模型口径。

这是发现异常后的修订协议，不是预先注册或未被查看过测试集上的独立验证，也不是 VLSA 表 9 的精确复现。不能从历史和新版本挑选较好数值拼表。

## 上传

在本机 PowerShell 执行：

```powershell
scp -P 13489 "E:\VLSA-main\downloads\QACLF_MCAT_CAT_nested_v1.zip" root@connect.nmb2.seetacloud.com:/root/autodl-tmp/
```

## 第一步：检查与单患者 smoke

在云端执行，不需要重新安装依赖：

```bash
conda activate vlsa_env
python -m zipfile -e /root/autodl-tmp/QACLF_MCAT_CAT_nested_v1.zip /root/autodl-tmp/
cd /root/autodl-tmp/QACLF_MCAT_CAT_nested_v1
python -u run_cat_mcat_nested.py check
python -u run_cat_mcat_nested.py smoke --datasets blca --folds 0 --output-root /root/autodl-tmp/QACLF_nested_v1_smoke
```

`check` 核对源码、输入、缓存和全部 25 个固定内部划分；没有训练、旧权重读取或旧结果修改。`smoke` 每方法只用一位内部训练患者检查前后向，不能计入论文结果。smoke 与正式训练使用不同输出目录。

建议先把 smoke 输出发回核对，再启动正式训练。不要同时运行旧版训练或重复启动新进程。

## 第二步：小规模正式运行与时间估计

以下两项是新协议的 BLCA 第 0 外层折，主要用于检查流程和实际耗时，不据其成绩决定保留哪个模型或改协议：

```bash
mkdir -p logs
LOG="logs/nested_v1_pilot_$(date +%Y%m%d_%H%M%S).log"
nohup python -u run_cat_mcat_nested.py run --datasets blca --folds 0 > "$LOG" 2>&1 < /dev/null &
echo "训练 PID：$!"
echo "日志：$LOG"
tail -n 40 -f "$LOG"
```

内部阶段打印 `INNER ... NLL ...`，不打印外层测试表现。外层 refit 结束后才打印 `COMPLETE`。Ctrl+C 若仅作用于 tail，只停止看日志；不要在前台 worker 上按中断。

## 第三步：全部五队列五折

```bash
LOG="logs/nested_v1_all_$(date +%Y%m%d_%H%M%S).log"
nohup python -u run_cat_mcat_nested.py run > "$LOG" 2>&1 < /dev/null &
echo "训练 PID：$!"
echo "日志：$LOG"
tail -n 40 -f "$LOG"
```

同一正式输出根目录下，已经完成且权重、预测 SHA256、协议和划分均一致的折会自动跳过，不重复训练。根目录运行锁阻止并行重复启动。

如果进程在某折中间被终止，该折不会被冒充完成，也不会自动覆盖或精确续训。当前实现仅支持跳过已核验的完整折；不完整目录需要另行处理或换新输出根目录，不能直接删旧数据。

本次包含内部最多 20 轮和外层重新训练，不等同旧版只训练一次的预算。先以实际 COMPLETE 的 wall_seconds 估算，不预先承诺几小时。

## 汇总与导出

```bash
python -u run_cat_mcat_nested.py summarize
```

仅完整五折的队列产生队列均值，完整五队列才打印五队列宏平均，不将部分折均值冒充全部结果。

结果根目录为 `/root/autodl-tmp/QACLF_MCAT_CAT_nested_v1_results`。每个方法/队列/折保存选轮记录、训练日志、内部和外层缩放器、最终权重、外层预测与训练预测、概率和风险指标、SHA256；旧论文结果不自动替换。

完成后再打包下载预测/日志，先不搬运所有权重。只有完整核验后才能更新论文中的表、图和文字。
