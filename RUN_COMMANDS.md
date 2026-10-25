# 手动运行顺序（仅代码，无报告）

以下命令由你手动执行。项目不会自动启动训练，也没有批量训练启动脚本。

**本工作目录已完成数据准备和验证**：官方 manifest、训练集 RGB 统计量、
DataLoader 检查均已完成。直接做一次不带 `--prepare` 的检查后，即可按下文手动开始
三组 LR 实验。不要重复创建 manifest 或统计文件；准备命令是供全新副本使用的。
代码 ZIP 不包含数据集和本机绝对路径的 manifest，若使用新解压副本，需要先放置数据。

## 进入项目并指定已验证的解释器

```powershell
Set-Location "C:\Users\ROG\Documents\Codex\2026-10-24\files-pasted-by-the-user-ai6103\outputs\AI6103_Assignment"
$py = 'D:\WORKSPACE\Anaconda\envs\ntuai\python.exe'
& $py scripts/check_environment.py
```

后续命令中的 `$py` 在同一个 PowerShell 窗口内有效。VS Code 也应选择这个解释器。

## 放置并检查数据（不训练）

当前数据已经放在：

`C:\Users\ROG\Documents\Codex\2026-10-24\files-pasted-by-the-user-ai6103\outputs\AI6103_Assignment\data`

配置已设为 `data_root: data`，对应 `train/images`、`val/images`、`test/images`
和三份 `*_labels.csv`。保留官方划分及目录结构。先运行：

```powershell
& $py scripts/inspect_dataset.py
```

已发现官方数据存在 15 组相同内容图片，其中 10 组跨集合、6 组具有冲突标签。配置明确设为
`content_duplicate_policy: preserve_official`，保持原官方数量与划分，输出完整审计并
在运行时提醒；不会静默删图或重分。各集合文件路径没有交叉，但图片内容有重复。

如果数据已有官方划分：

```powershell
& $py scripts/inspect_dataset.py --prepare
```

如果确认没有官方划分，才用下面的命令替代 `--prepare`：

```powershell
& $py scripts/inspect_dataset.py --create-splits
```

然后计算仅基于训练集的 RGB 统计量并检查 DataLoader：

```powershell
& $py scripts/compute_statistics.py
& $py scripts/inspect_dataset.py --verify-loaders --verify-hashes
& $py -m src.experiments --validate-configs
```

官方索引格式无法自动识别时，程序会报错并要求 `path,label,split` CSV 适配文件，
不会擅自重新划分。详见 `data/README.md`。

## 三组初始学习率实验

以下开始是真实训练命令。每条命令只运行一个实验，请逐个手动执行。

```powershell
& $py -m src.experiments --experiment lr_05
& $py -m src.experiments --experiment lr_005
& $py -m src.experiments --experiment lr_0001
& $py -m src.experiments --select-lr
```

`lr_005` 的实际学习率是 **0.05**。三组各 15 轮；依据**最后一轮训练交叉熵**选择 LR。
缺少任意完整实验时，选择会停止并列出需要执行的命令。

## 学习率调度与权重衰减

```powershell
& $py -m src.experiments --experiment constant
& $py -m src.experiments --experiment cosine
& $py -m src.experiments --experiment wd_5e4
& $py -m src.experiments --experiment wd_1e4
& $py -m src.experiments --select-baseline
```

每组 300 轮。会从这四组配置中按验证集表现选择最优的非 Mixup 配置。
顺序为最高验证准确率、最低验证 CE、最早轮次、实验 ID；使用所有已完成轮次。

## Mixup 与最终选择

```powershell
& $py -m src.experiments --experiment mixup
& $py -m src.experiments --select-final
```

Mixup 从随机初始化训练 300 轮，alpha=0.2，继承上一步选定的调度和权重衰减。
最终仍根据验证集比较 Mixup 与对应基线，不预设 Mixup 更好。

## 单独评估最终测试集

仅在完成最终选择后，由你手动运行：

```powershell
& $py -m src.evaluate --test
```

测试集不会用于调参，也不会被训练、绘图或汇总命令自动评估。

## 停止与恢复

Ctrl+C 停止。恢复示例：

```powershell
& $py -m src.experiments --experiment cosine --resume
```

从最后一个完整轮次的 `last.pt` 恢复；中断的那一轮会重新执行。
保持原配置、软件环境、源码和数据不变。若之前使用了额外运行参数，恢复时保持一致。
如果刚启动、第一轮尚未完成，则还没有可恢复的 checkpoint，需明确使用 `--overwrite`。
`--overwrite` 会将旧输出归档，不会静默覆盖。异常强制结束后如有锁文件，先确认旧进程
已经退出，再删除对应的 `results/.实验名.lock`。

## 绘图、汇总与数值导出（不训练）

```powershell
& $py scripts/summarize_results.py
& $py scripts/generate_figures.py
& $py scripts/generate_report_tables.py
```

最后一个命令沿用最初要求的名称，只导出数值表和预处理代码片段到 `results/tables/`，
不会生成报告正文。结果缺失会如实标记，不会填入虚构数值。

## 运行前的设置

默认使用 CUDA BF16、batch size=128、4 个 worker。若需要 FP32，在任何正式实验开始前
统一修改 `configs/base.yaml` 的 `precision: fp32`；DataLoader 出问题时同理统一改
`workers: 0`。不同精度、seed、运行配置的实验不能混在同一轮对比里。
不要在恢复时改变精度。显存不足时不会自动降低 batch size 或替换网络。

只检查配置、不训练：

```powershell
& $py -m src.experiments --experiment lr_005 --dry-run
```

完整实现说明、输出路径和故障排查见 `README.md`；文件清单见 `FILES.md`。
# 当前性能优化说明

2026-10-25：八组训练与唯一一次最终测试已完成。真实结果见 `results/final_summary.json`、
`results/final_summary.csv` 和 `results/final_test.json`；最终模型是 `checkpoints/mixup/best.pt`（第 273 轮）。
测试准确率 37.5489%，对应 1,534 张测试图像。`scripts/finalize_results.py` 仅核验、汇总已有结果，不执行训练或评估。

当前顺序续跑采用经过等价性验证的 `scripts/train_optimized.py`。已有结果保留，原始配方源码与配置不变。
实际用法、断点备份、计时口径和核验证据见 [PERFORMANCE.md](PERFORMANCE.md)。运行期间不要额外启动第二个训练进程。

