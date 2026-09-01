# bf16 recurrent state 精度结果

实验在 NVIDIA B300 SXM6 AC 上完成，使用 CUDA 13.0、PyTorch 2.13.0 和固定 seed `20260830`。完整原始数据保存在 `results/full-kernel/`。结果可以复跑，CSV 里保留每个窗口、每个长度检查点和每条变长序列的数据。

## 先把 fp32 API 的含义说清楚

当前 kernel 的 fp32 initial state 会在入口转成 bf16，片上递推仍使用 bf16 state。final state 选择 fp32 时，kernel 只在出口把 bf16 state 转回 fp32。`tests/torch_ref.py` 也采用相同处理。

完整实验中的 8 个 case 同时跑了 bf16 和 fp32 API state。固定长度和变长输入合计得到 16 组 output 与 final-state parity，max absolute、mean absolute、RMSE 和 RMSE ratio 全部为零。因此，现有 fp32 API 不能当作真正 fp32 片上 state 的精度对照。

## state 回写的单因素结果

脚本用 fp64 逐 token recurrence 作为 gold。两条候选路径都在 fp32 中更新 state，`bf16_state` 每 16 token 回写一次 bf16，`fp32_state` 则保留 fp32。输入值、gate、beta 和初始 state 完全相同。

| 用例 | 长度 | bf16 final-state RMSE ratio | fp32 final-state RMSE ratio | bf16 max absolute |
|---|---:|---:|---:|---:|
| 随机基线 | 8192 | 1.688e-3 | 5.027e-8 | 9.380e-4 |
| 正值输入 | 2048 | 1.667e-3 | 4.927e-8 | 1.218e-4 |
| 弱遗忘加高 beta | 8192 | 4.092e-3 | 2.746e-6 | 1.997e-2 |
| 强遗忘加高 beta | 2048 | 1.660e-3 | 5.126e-8 | 9.703e-4 |
| 低 beta | 2048 | 1.673e-3 | 5.047e-8 | 6.642e-7 |
| 弱遗忘加高 beta，value 放大 8 倍 | 2048 | 4.074e-3 | 2.769e-6 | 1.581e-1 |

完整矩阵中，bf16 state 的最大 RMSE ratio 为 `4.125e-3`，fp32 state 的最大值为 `2.819e-6`。约 1463 倍来自两组各自的最大值，不是同一行的配对提升。逐行数据仍显示，保留 fp32 能显著降低 chunk 边界回写误差。这类误差差异不能直接换算成模型质量收益。

弱遗忘用例给出了最清楚的长度曲线。bf16 state 在 16、64、256、1024、4096 和 8192 token 的 ratio 依次为 `1.682e-3`、`2.597e-3`、`3.792e-3`、`4.122e-3`、`4.125e-3` 和 `4.092e-3`。这个 case 的每 token log-decay 约为 `-0.00168`，e-fold 记忆长度约 596 token，所以约 1024 token 后的 0.4% 平台符合该门控设置。它不能外推到 e-fold 为 8192 token 或更长的真实长记忆分布。随机 gate 和强遗忘会更快衰减历史状态，8192-token 随机基线约为 0.17%。

变长实验逐条重置 state。混合长度 17、65、257、1023 的 bf16 final-state ratio 落在 `1.63e-3` 到 `1.85e-3`。交替弱遗忘与强遗忘的压力组覆盖 31、128、511、2048，结果落在 `1.61e-3` 到 `1.68e-3`。强遗忘 token 会周期性清除历史误差，因此这组数据比持续弱遗忘温和。

## 真实 kernel 的整体误差

真实 FlashKDA 与独立 fp64 recurrence 的误差还包含 q/k 归一化、近似 sigmoid 与指数、16 x 16 求逆和 bf16 GEMM，数值应当视作完整实现误差。完整矩阵里，最坏 final-state RMSE ratio 为 `7.155e-3`，出现在 value 放大 8 倍的弱遗忘用例。最坏全序列 output RMSE ratio 为 `7.898e-3`，出现在 8192-token 弱遗忘用例。

弱遗忘加高 beta 的用例比随机输入更苛刻。大 value 用例同时改变了 seed、长度和 value scale，因此只能作为压力样本，不能单独归因于 value 幅值。当前数据支持在这些合成分布上采用 bf16 state，尚不能代表真实 K3 激活分布或端到端质量。

## 结论与边界

当前 bf16 state 在已测 forward recurrence 分布中保持稳定，最大相对 RMSE 约 0.41%。真正 fp32 state 的最大值低于 `3e-6`。若 v2 面向更长的门控记忆或高幅值激活，保留 fp32 state 值得作为可选路径评估。常规推理路径继续用 bf16 符合当前 kernel 对共享内存和吞吐的取舍，但生产结论仍需要真实 K3 分布验证。

尚未覆盖真实 K3 权重与激活分布、端到端生成质量、e-fold 达到 8192 token 以上的门控、训练反向、多个 seed 和多个 head 的联合统计，以及 fp32 state kernel 的真实性能代价。现有 fp32 API 不提供真正 fp32 递推，最后一项需要新增 kernel 实现才能完成。
