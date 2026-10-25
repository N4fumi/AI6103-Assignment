# 保持训练轨迹的运行优化

原始 `src/` 和 `configs/` 保持原样；已完成实验不重跑、不覆盖。
2026-10-24 从 cosine 第 141 轮的完整断点进行切换。原始断点、指标、源码、配置及锁文件备份位于
`results/performance_optimization/backup_20261024_212440/`。

## 优化内容

- 复用训练和验证 DataLoader 的 worker，减少 Windows 反复启动进程的等待。
- `CompatiblePersistentLoader` 补齐普通迭代器原有的随机数消耗，保持每轮抽样顺序。
- 梯度有限值检查仍逐元素执行，但在 GPU 汇总后只向 CPU 返回一次判断结果。
- `best.pt` 在最佳 epoch 改变时保存；`last.pt` 仍每轮完整保存。
- 模型、样本、增强、batch=128、BF16 训练、FP32 验证、SGD、学习率、调度器、seed、
  确定性算法、TF32 和内存布局均保持原有设置。

## 验证与计时

`scripts/verify_optimization.py` 从同一份断点分别运行原版和优化版，不写入正式实验指标，
不读取测试集。检查完整两轮的训练/验证指标，以及模型、优化器、调度器、CPU/CUDA/NumPy/Python
随机数状态和 DataLoader 随机数状态的精确一致性；另检查两轮 Mixup + weight decay 路径。
真实图像的数据加载检查覆盖训练和验证各连续三轮，逐 batch 比较增强后的张量和标签。

初始 GPU 对照中，worker 启动后的完整一轮（训练 + 验证）为原版 45.52 秒、优化版 10.31 秒。
正式恢复后的第 143–148 轮平均为 9.62 秒（9.44–9.75 秒），完整保存断点另计。
第 143 轮正式 checkpoint 的权重、优化器、调度器及 RNG 哈希也与独立对照完全一致。
首次启动 worker 的轮次较慢。最终证据见：

- `results/performance_optimization/loader_equivalence.json`
- `results/performance_optimization/equivalence.json`
- `results/performance_optimization/preservation.json`
- `results/<experiment>/performance.jsonl`

原 `source_signature` 对应未改动的实验配方源码；实际运行 adapter 的 SHA256、验证证据 SHA256、
生效起始轮次及实际 worker 策略另存于 checkpoint、metadata、summary 的 `runtime_optimization`。
旧轮次和新轮次的运行时间使用了不同执行方式，不能把它们作为同一性能配置比较。

## 手动入口

当前顺序训练由续跑器管理，运行期间不要再启动第二个训练进程。
下面只用于今后明确需要手动运行时，必须先确认没有活动训练；入口不会自动重跑或覆盖结果。

```powershell
& 'D:\WORKSPACE\Anaconda\envs\ntuai\python.exe' -u scripts/train_optimized.py --experiment cosine --resume
```

入口要求本地等价性验证通过，且 adapter 哈希与证据匹配。需要回退时，原始
`python -m src.experiments --experiment cosine --resume` 入口仍兼容新增的运行元数据。
失败不会自动改变精度、batch、seed 或重试，也不会自动执行最终 hold-out 测试。
