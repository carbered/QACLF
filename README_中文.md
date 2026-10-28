# QACLF

QACLF（Quantile-Aligned Conservative Late Fusion，分位数对齐保守晚期融合）原名
FICPF。方法复用冻结的 VLSA 病理分支，在外层训练折内拟合 PCA-Cox 组学分支，
通过训练折 ECDF 对齐得分尺度，再按病理/组学权重 0.80/0.20 融合。组学缺失时
回退到病理分位数；生存曲线由训练折 Cox-Breslow 映射得到，尺度对齐不等于概率校准。

![QACLF 方法图](figures/qaclf_current/QACLF_method_overview.png)

仓库地址暂保留 `https://github.com/carbered/FICPF`，避免旧链接失效。历史文件名和
原始结果键保留 `ficpf/FICPF`，用于追溯，不表示两个不同模型。当前中文 Word 初稿不上传。

## 当前结果

五个 TCGA 队列共享固定患者级外层五折划分，八种模型使用 CONCH 病理输入。

| 方法 | 宏平均 C-index | 宏平均 IBS，12–60个月 |
|---|---:|---:|
| ABMIL | 0.6215 | 0.1670 |
| TransMIL | 0.6447 | 0.1839 |
| ILRA | 0.6364 | 0.2061 |
| DSMIL | 0.6591 | 0.1698 |
| VLSA | 0.6958 | 0.1797 |
| MCAT | 0.6793 | 0.1641 |
| CAT-with-TPL | 0.6697 | 0.1659 |
| QACLF | 0.7050 | 0.1545 |
| VLSA＋同曲线映射（对照） | 0.6959 | 0.1572 |

同曲线映射对照不是第九种模型架构。IBS 改善很大一部分来自曲线重建，不能全部归因于组学融合。
QACLF 的锁定主结果未改动，配对宏平均增益为 0.0092，95% bootstrap CI 为 0.0016–0.0169。

- [当前队列级比较](result/qaclf-nested-v1-unified-evaluation-20261027/updated_eight_method_dataset_summary.csv)
- [当前宏平均曲线指标](result/qaclf-nested-v1-unified-evaluation-20261027/macro_metrics.csv)
- [225条逐折评价](result/qaclf-nested-v1-unified-evaluation-20261027/all_fold_metrics.csv)
- [当前八方法图](figures/qaclf_current/QACLF_eight_method_comparison.pdf)
- [共同窗口概率评价图](figures/qaclf_current/QACLF_unified_probability.pdf)

## 本次更新及边界

- MCAT/CAT-with-TPL 改用已完成的50折 nested-v1 重跑结果：拟合患者内标准化与分箱，
  内层验证 NLL 选轮、全部外层训练患者重新拟合，外层测试只评价一次。
- 八种模型及同映射对照统一在12–60个月、97个时点评价概率误差，删失权重仅用外层训练患者估计。
- D-calibration 在60个月行政删失后计算，不拒绝折数仅作描述，不能解释为整体校准通过。
- 全时间均值 hinge MAE 仅保留作诊断，不声称已解决曲线尾部外推问题。
- 方案调整属于事后补充，复用了原有外层划分，不构成独立外部验证。
- 历史 MCAT/CAT 结果、旧窗口指标及 `docs/FICPF*` 草稿保留，但不是当前稿件，不能混用。
- 固定输入推理计时仍为 VLSA 25.61 ms、QACLF 62.17 ms，不声称 QACLF 推理更快。

## 安装与校验

分析包建议 Python 3.10+；云端训练包单独保留原 Python 3.8 协议。

```bash
python -m pip install -r requirements.txt
python tools/verify_qaclf_release.py
python tools/verify_qaclf_results.py --recompute-c-index --recompute-probability
```

第一条校验命令检查文件 SHA256、结果汇总、选轮记录和225份曲线输入。
后一条进一步复算当前概率指标、50折多模态原生 C-index 与100折历史 MIL C-index。
这些校验不训练模型，也不需要大型组学表、WSI、patch 特征或模型权重。

原始数据及完整重训仍需按 [DATA.md](DATA.md) 补齐上游输入。
当前包包含固定患者划分、生存标签、预测曲线、结果和制图代码，不包含权重、原始切片、
大型组学表、训练日志和当前中文 Word 稿。历史文件保持原始结果身份，不做全局替换改名。

详细边界见 [当前结果说明](docs/CURRENT_RESULTS.md)、[英文说明](README.md) 和 [NOTICE.md](NOTICE.md)。
