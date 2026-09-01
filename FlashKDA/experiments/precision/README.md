# Recurrent state precision experiment

本实验回答讨论点 5，即 FlashKDA 把片上 recurrent state 存成 bf16 后，误差会怎样随序列长度、输入分布和门控压力增长。

## 实验边界

`tests/torch_ref.py` 和当前 CUDA kernel 都会把 fp32 API state 转为 bf16，再进行 chunk 间递推。现有 `state_dtype=fp32` exact-match 测试只能证明入口和出口的转换正确，无法证明内部保留 fp32 state 没有收益。

`state_precision.py` 因而建立三条使用相同量化输入的路径。

1. `gold_fp64` 按 `state_precision.py` 第 239 至 250 行的逐 token 公式连续递推，全程 fp64。
2. `bf16_state` 用 fp32 做状态更新，每个 16-token chunk 结束后把 state 往返转换为 bf16，模拟当前 FlashKDA 的 chunk 间存储。
3. `fp32_state` 使用完全相同的 fp32 更新，但 chunk 边界不降精度，代表真正保留 fp32 state 的候选实现。

q、k、v、原始 gate 和 beta logits 都先量化为 bf16。三条路径共享归一化后的 q/k、激活后的 gate 和 beta，初始 state 也取 bf16 可表示值。因此模型对比主要测量 state 回写误差，不把近似 sigmoid、近似指数、矩阵求逆或输入量化误差混进结论。

## 覆盖范围

`smoke` 包含固定长度基线、弱遗忘加高 beta、变长交替强门控。`full` 另含 8192-token 长度增长、正值输入、强遗忘、低 beta、大幅值 v 和两组 varlen。固定 seed 默认是 `20260830`，真实 head dim 默认是 128。

结果文件如下。

- `output_window_metrics.csv` 记录每个序列逐窗口的 max absolute、mean absolute、RMSE 和 RMSE/reference-RMS。
- `state_checkpoint_metrics.csv` 在 16、64、256 等长度检查点和 final state 记录同一组指标。
- `flash_kernel_metrics.csv` 仅在加 `--flash-kda` 时生成，包含 kernel 对 fp64 gold 的误差，以及 bf16/fp32 API 两条路径的直接 parity。
- `summary.json` 保存 case、seed、设备、软件版本和聚合指标。

## 运行

所有 GPU 命令从仓库根目录通过 `srun.sh` 申请资源。

```bash
bash srun.sh 'source .venv-b300/bin/activate; cd FlashKDA; python experiments/precision/state_precision.py --profile smoke --flash-kda --output-dir experiments/precision/results/smoke-kernel'
```

完整实验使用下面的命令。

```bash
bash srun.sh 'source .venv-b300/bin/activate; cd FlashKDA; python experiments/precision/state_precision.py --profile full --flash-kda --output-dir experiments/precision/results/full-kernel'
```

只验证递推模型时可以省略 `--flash-kda`，也可以指定 `--device cpu`。不要把 `flash_bf16_api` 与 `flash_fp32_api` 的一致理解为 fp32 state 无价值。当前实现的两条 API 分支共享 bf16 片上 state，kernel parity 是对此的反向验证。

## 解读原则

先比较同一窗口和同一 state checkpoint 的 `bf16_state` 与 `fp32_state`。如果 bf16 的误差随长度持续增加，同时 fp32 保持接近 fp64，state 回写就是主要误差源。弱遗忘、高 beta 和大 value 用例提供压力样本，随机基线只能说明当前合成输入。当前实验只测 forward recurrence，不覆盖训练反向、真实 K3 权重与激活分布，也不替代端到端任务指标。
