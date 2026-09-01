# FlashKDA 项目导读

第一次打开这个项目，很容易先被文件名和 GPU 术语挡住。`KDA`、`CHUNK`、`HMMA`、`TCGEN05`、`TMA`、`NCU` 连在一起，看起来像是同一件事。它们其实分属不同层次。KDA 是模型计算，FlashKDA 是 CUDA 实现，HMMA 和 TCGEN05 是矩阵指令，TMA 负责搬数据，NCU 用来观察 GPU 到底怎样执行这些代码。

我把项目源码、测试、B300 实验和最终报告重新顺了一遍。理解它有一条比较省力的路线。先看 KDA 怎样保存历史，再看 FlashKDA 为什么拆成 K1 和 K2，最后看项目怎样用实验判断 SM100 优化是否值得继续。

配套的[交互流程图](PROJECT_WORKFLOW.html)把这条路线放在一张图里。正文负责解释每一步为什么存在，图适合在读完一节后回去定位。

## 这个项目在解决什么问题

普通注意力会计算 token 之间的两两关系。序列长度从 T 增长时，注意力矩阵的规模接近 T 的平方。长上下文因此需要更多计算和显存。

KDA 属于线性注意力的一种。它把已经读过的历史压进一个固定大小的 recurrent state。这里的 recurrent 可以理解为循环使用。每来一批新 token，程序更新 state，再用 query 从 state 中读取结果。序列继续变长时，state 的形状保持不变。

可以先用下面这组概念式理解。它省略了 KDA 的归一化、delta update 和 chunk 内求逆，只保留数据关系。

```text
new_state = gate * old_state + beta * new_information
output = query @ new_state
```

项目当前只支持 head dimension 等于 128。每个 head 的 state 因而是一个 `128 x 128` 矩阵。Kimi K3 的生产配置有 96 个 KDA head。做 TP8 张量并行以后，每张 GPU 只负责 12 个 head。这个数字会在后面的性能分析里反复出现，因为 K2 的独立任务数直接受本卡 head 数影响。

## 输入里的 q、k、v、gate 和 beta 分别做什么

FlashKDA 的 Python 接口位于 `flash_kda/__init__.py`。主要输入可以按下面理解。

| 输入 | 作用 | 当前类型与形状 |
|---|---|---|
| `q` | query，用来读取 state | bf16，`[B, T, H, 128]` |
| `k` | key，决定新信息怎样写入 | bf16，`[B, T, H, 128]` |
| `v` | value，携带要写入和输出的内容 | bf16，`[B, T, H, 128]` |
| `g` | gate 激活前的输入，控制旧状态保留多少 | bf16，`[B, T, H, 128]` |
| `beta` | delta update 的写入强度 | bf16，`[B, T, H]` |
| `A_log` | 每个 head 的门控参数 | fp32，`[H]` |
| `dt_bias` | 每个通道的 gate 偏置 | fp32，`[H, 128]` |
| `initial_state` | 可选的起始历史状态 | bf16 或 fp32，`[N, H, 128, 128]` |
| `final_state` | 可选的最终历史状态输出 | 与 initial state 遵循相同规则 |
| `cu_seqlens` | 变长批处理的序列边界 | int64，长度为 `N + 1` |

固定长度输入通常使用 `[B, T, H, 128]`。变长模式把多条序列拼进一个长张量，再用 `cu_seqlens` 标出每条序列从哪里开始、在哪里结束。此时 `B` 必须等于 1，`N` 表示实际序列条数。

Python 层会先根据 token 数、head 数和序列数申请 workspace，随后调用编译出来的 `flash_kda_C` 扩展。`csrc/flash_kda.cpp` 负责检查设备、连续性、数据类型和形状。它还根据 fixed 或 varlen、有没有输入 state、要不要输出 state、API state 是 bf16 还是 fp32，选择对应的 CUDA 模板实例。

这里有一个容易误解的细节。fp32 state API 允许用户传入和取回 fp32 张量，当前 kernel 在片上递推时仍然使用 bf16 state。入口会把 fp32 state 转成 bf16，出口再转回 fp32。这个接口保证使用方便，没有提供真正的片上 fp32 递推。

## 为什么 forward 要拆成 K1 和 K2

FlashKDA 每 16 个 token 分成一个 chunk。源码在 `csrc/smxx/fwd_launch.cu` 里把 `CHUNK` 固定为 16。一次 forward 随后启动两个主要 kernel。

### K1 prepare 处理 chunk 内部

K1 的实现位于 `csrc/smxx/fwd_kernel1.cuh`。它为每个 chunk 和每个 head 建立一个 CTA。CTA 可以理解为一组会被安排到同一个 SM 上协作的 CUDA 线程。

K1 会加载 q、k、g、beta 和 `dt_bias`，完成 q 与 k 的归一化和 gate 计算。它随后构造 chunk 内部的三角矩阵，计算 `k_decayed`、`q_decayed`、`k_restored`、`L`、`INV` 和 `Mqk` 等中间量。`INV` 来自有限 Neumann 级数。这里利用了严格下三角矩阵的幂零性质，有限次矩阵乘就能完成所需求逆。

不同 chunk 之间暂时不需要前一块的 state，所以 K1 可以铺开很多 CTA。固定 `T=8192`、`H=96` 时有 512 个 chunk，K1 的 grid 是 `512 x 96`，一共 49152 个 CTA。B300 有 148 个 SM，这些 CTA 足以让 GPU 持续取得新任务。

K1 算出的中间量会写入 workspace。这个 workspace 不是模型长期保存的 state，它只服务当前这次 forward，让 K2 不必重新计算 chunk 内的 q、k 和 gate 关系。

### K2 recurrence 串起 chunk 之间的历史

K2 的实现位于 `csrc/smxx/fwd_kernel2.cuh`。一个 K2 CTA 负责一条 sequence 的一个 head。它先读 initial state，然后按照时间顺序读取各个 chunk 的 workspace，更新 `128 x 128` state，并写出 output。最后一个 chunk 结束后，K2 可以把 state 写进 `final_state`。

这条递推存在明确的先后依赖。第 2 个 chunk 要用第 1 个 chunk 更新后的 state，第 3 个 chunk继续使用第 2 个 chunk 的结果。一个 head 内的 chunk 不能直接拆成互不相关的 CTA。

K2 的 grid 因而只有 `N x H`。固定单序列 H96 只有 96 个 CTA，已经少于 B300 的 148 个 SM。TP8 后的 H12 更少，只剩 12 个 CTA。多数 SM 从 kernel 开始到结束都拿不到任务。这里形成了项目最重要的性能判断。K2 当前先受独立工作数量限制，换一条峰值更高的矩阵指令无法自动增加 CTA。

## TMA、HMMA 和 TCGEN05 各自处在哪一层

TMA 是 Tensor Memory Accelerator。它负责在 global memory 和 shared memory 之间异步搬运规则形状的数据。FlashKDA 已经在 K1、K2 和 workspace 读写中大量使用 TMA。最终 SASS 里有 1118 条 `UTMALDG` 和 458 条 `UTMASTG`，说明 B300 确实执行了这些搬运指令。

HMMA 是项目当前矩阵乘路径在 SASS 中显示的指令类别。源码使用 `m16n8k16` 的 SM80 MMA 原子。一个逻辑 `16 x 16 x 16` 矩阵乘可以用两条这样的指令完整覆盖，有效算术比例为 100%。最终扩展的 SASS 统计到 1544 条 HMMA，没有 HGMMA，也没有 TCGEN 指令。

TCGEN05 是 SM100 上的新一代 dense Tensor Core 指令接口。它的 BF16 dense SS 形状要求 M 至少为 64。FlashKDA 的核心小矩阵只有 16 行。若把一个 C16 工作直接塞进 M64，64 行里有 48 行只是 padding。GPU 仍然做了这些乘加，结果却不会被 KDA 使用。

项目为此写了真实 B300 microbenchmark，位置在 `experiments/tcgen05/`。相同逻辑输入、CTA 数和累计次数下，C16 HMMA 平均为 `4.371 us`，TCGEN05 padded 路径为 `12.319 us`。后者慢 2.82 倍。这个结果淘汰了逐个替换小矩阵原子的方案。它没有排除把四份独立 C16 工作打包进一个 M64 的新设计。

## CHUNK 为什么固定为 16

CHUNK 16 同时受到数值、算法工作量和矩阵形状影响。

gate 激活后的 log 值位于 `(-5, 0)`。一个长度为 C 的 chunk 在极端情况下会产生 `exp(-5C)` 的累计衰减和 `exp(5C)` 的恢复因子。C 等于 16 时，这两个数仍在 bf16 的有限正常范围内。C 等于 32 时，一边下溢，另一边上溢。没有额外 rescale 设计时，数值范围允许的整数上限约为 17。

chunk 内三角矩阵的 Neumann 求逆也会随 C 快速变贵。C 为 16 时估算为 49152 FLOP。C 增到 32 后变成 524288 FLOP，工作量达到 10.7 倍。C 为 64 时达到 5242880 FLOP。shared memory 下界也从 17920 bytes 增到 38912 和 90112 bytes，这里还没有算所有 padding、barrier 和编译器 spill。

最后再看指令形状。C16 正好匹配当前 `m16n8k16` 原子。C64 才能填满 TCGEN05 的最小 M。前两项约束已经让 C32 和 C64 需要新的数值与算法方案，单纯为了适配新指令放大 chunk 会同时引入更多问题。

## 仓库里的主要部分

| 路径 | 负责的事情 | 阅读建议 |
|---|---|---|
| `flash_kda/__init__.py` | 用户调用的 Python API，申请 workspace，调用 CUDA 扩展 | 第一个看 |
| `csrc/flash_kda.cpp` | PyTorch 绑定、输入检查、fixed 与 varlen dispatch | 理解接口边界时看 |
| `csrc/fwd.h` | CUDA launch 的声明 | 用来连接绑定层和 launch 层 |
| `csrc/smxx/fwd_launch.cu` | 建立 TMA descriptor，切 workspace，启动 K1 和 K2 | 理解完整 forward 时重点看 |
| `csrc/smxx/fwd_kernel1.cuh` | K1 prepare 与 chunk 内矩阵计算 | 研究 CHUNK、求逆和 workspace 时看 |
| `csrc/smxx/fwd_kernel2.cuh` | K2 state recurrence、pipeline 和 output | 研究并行度与 state 时重点看 |
| `csrc/smxx/utils.cuh` | MMA、pipeline 和布局辅助代码 | 遇到具体 CUDA 模板时查 |
| `cutlass/` | NVIDIA CUTLASS 依赖 | 不需要从头阅读，只按类型和指令定义查 |
| `tests/torch_ref.py` | 与 kernel 行为逐项匹配的 PyTorch 参考实现 | 理解算法细节很有用 |
| `tests/test_fwd.py` | 官方 fixed、varlen、FLA 对拍 | 看主正确性入口 |
| `tests/test_fwd_full.py` | 扩展的 580 case 回归 | 看覆盖范围和长序列 case |
| `benchmarks/` | CUDA event 性能基线和报告生成 | 区分 latency 与 profiler counter |
| `experiments/` | TCGEN05、NCU、roofline、precision 和 SASS 证据 | 按问题进入对应子目录 |
| `FINAL_REPORT.md` | 当前技术判断与全部关键数字 | 了解本导读后再读 |
| `PREPARATION.md` | 12 页 PRE 内容和答辩问题 | 准备演讲时使用 |

`TASK1_assignment_artifacts/` 保存早期复现材料。当前结论没有直接把它当成最终证据。项目在新环境里重新编译扩展，重跑测试、benchmark、NCU 和 SASS，并用扩展 sha256 把几类结果绑定在一起。

## 实验目录怎样回答不同问题

这个项目刻意保留了多种证据，因为它们回答的问题不同。

### baseline 看端到端时间

`experiments/baseline/` 保存当前 B300 benchmark、raw stdout、GPU 与 Python 环境信息，以及扩展哈希。正式 benchmark 显式设置 `FLA_FLASH_KDA=0`，避免 FLA 对照又分发回 FlashKDA。

H96 fixed 下 FlashKDA 为 `0.9990 ms`，FLA chunk KDA 为 `2.3490 ms`。mixed varlen 为 `0.8705 ms` 对 `2.3734 ms`。八条等长序列为 `0.7163 ms` 对 `2.3264 ms`。同一次 allocation 内，FlashKDA 的优势为 2.35 到 3.25 倍。

不同 allocation 的绝对时间会受时钟和节点状态影响。项目里同日较早的一次运行整体慢了 45% 到 73%，FlashKDA 和 FLA 对照一起变慢。正式结论只使用同一作业内的 A/B，避免把环境波动写成 kernel 改进。

### tests 看结果是否正确

官方 `tests/test_fwd.py` 检查 fixed 和 varlen。FlashKDA 与项目自己的 `torch_ref` 逐元素相等。20 组 FLA fp64 gold 的 output 检查通过。state 检查产生 8 条 warning，最大的 FlashKDA state ratio 为 fixed `0.009628` 和 varlen `0.008926`。测试源码对 state 设置了 `warning=True`，这些记录没有触发失败。

扩展回归覆盖 fixed、varlen、batch、bf16 和 fp32 API state、不同 state 输入输出组合，以及最长 1048576 token。JUnit 记录为 `580 passed in 701.59s`，没有失败、错误或跳过。

### NCU 看 GPU 在忙什么

`experiments/ncu/` 保存四份 `.ncu-rep` 和 CSV。NCU 记录 CTA 数、waves per SM、SM throughput、DRAM throughput、Tensor 指令、register、shared memory 和 achieved occupancy。

fixed H12 的 K2 只有 12 个 CTA，`0.04 waves/SM`，SM throughput 为 `2.51%`，DRAM throughput 为 `2.16%`。计算和显存都没有接近饱和。equal H96 提供 768 条独立链后，两项分别升到 `38.63%` 和 `36.19%`。这些数据支持增加独立工作的方向。

focused NCU 采集的是 benchmark 的 bf16-state 分支，正式 latency 表使用 fp32 API state。它们不能拼成同一 state 配置的联合测量。grid 形状和并行度判断不依赖这个 API dtype 差异。

### roofline 解释大 GEMM 与小递推的差别

`experiments/roofline/` 移植了 assignment 4.5 的 `in_proj_qkvgfab` 瘦 GEMM。TP8 本地形状为 `N=6288, K=7168`。M 很小时，cuBLAS latency 约停在 19 us，M 为 1、8、16 时只有 4.70、38.26、73.34 TFLOPS。M 达到 1024 后进入 compute roof，M 为 4096 到 65536 时形成约 1.15 到 1.29 PFLOPS 的平台。

这项实验说明 B300 能在合适的大矩阵上发挥很高的 Tensor Core 吞吐。FlashKDA K2 的工作形状和递推依赖完全不同，因此不能把大 GEMM 的 roofline 直接套到 K2。

### precision 分开 API 类型和真实递推精度

`experiments/precision/` 建立 fp64 gold、bf16 chunk-state 和真正 fp32 state 三条合成 recurrence。三条路径共享量化输入。bf16 candidate 每 16 token 把 state 往返转换为 bf16，fp32 candidate 始终保留 fp32。

完整矩阵中，bf16 candidate 最大 RMSE ratio 为 `4.125e-3`，fp32 candidate 最大值为 `2.819e-6`。真实 FlashKDA 相对独立 fp64 recurrence 的最大 final-state ratio 为 `7.155e-3`，最大 output ratio 为 `7.898e-3`。

这些 case 来自合成分布。弱遗忘 case 的 e-fold 只有约 596 token，结果不能外推到真实 K3 的长记忆、模型生成质量或训练反向。它支持把片上 fp32 state 留作后续候选，尚未证明生产路径必须采用它。

### SASS 确认最终二进制执行了哪些指令

源码里写了某种模板，并不等于最终机器码一定保留同样的指令。`experiments/sass/` 对最终 `.so` 导出 sm_103a SASS，再按指令类别计数。结果为 HMMA 1544、UTMALDG 1118、UTMASTG 458、HGMMA 0、TCGEN 0。

SASS、四份 NCU、baseline 和 580 case 使用的扩展 sha256 相同。这样可以避免测试一份二进制、分析另一份二进制。

## 整个项目是怎样推进到当前结论的

工作先在 B300 上重建 Python 和 CUDA 编译环境。GPU 作业都通过仓库根目录的 `srun.sh` 申请。扩展编译完成以后，项目先跑官方测试和端到端 baseline，确认实现能运行、结果可信、性能对照没有绕回自身。

随后进入六类专项问题。源码和纸面估算解释 CHUNK 16，TCGEN05 microbenchmark 检查新指令的形状代价。roofline 提供同卡的大 GEMM 参照，focused NCU 定位 K1 与 K2 的并行度。precision 实验拆开 state 回写误差，SASS 负责核对最终机器码。

每项实验都保留复现脚本、原始结果和分析文档。审计阶段又检查了测试数量、FLA warning、NCU 指标口径、TCGEN05 timed workload、哈希绑定和文档中的因果边界。最后一轮复核确认这些问题已经闭合。

这套顺序很重要。正确性先决定结果能不能讨论，benchmark 再回答端到端快不快。microbenchmark 负责淘汰局部方案，NCU 解释真实 kernel 为什么没有把 GPU 用满。SASS 提供最终二进制证据。任何一类结果都没有代替其他类别。

## 目前已经完成到哪里

当前生产 kernel 保持原样。它已经在 B300 上通过官方测试和 580 项扩展回归，端到端性能快于本次作业内的 FLA chunk KDA 对照。最终机器码、NCU 和测试二进制已经用同一个 sha256 绑定。

SM100 挑战完成了指令替换切面的可行性淘汰实验。项目写出了真实 TCGEN05 代码，验证了与计时相同的 workload，并在 B300 上与 HMMA 做局部 A/B。C16 直接替换慢 2.82 倍，因此没有把这条更慢的路径接进生产 kernel。

当前还没有一个端到端 sm100a FlashKDA candidate，也没有 candidate 对 FlashKDA 本体和 FLA 的直接端到端 A/B。若评审把这一项当成硬要求，SM100 challenge 仍然缺生产级候选。现有成果可以支撑一个范围明确的负结论，逐原子 TCGEN05 替换不值得集成。

下一条优先路线是把 K2 的 `128 x 128` state 沿 value 列拆成两个 `128 x 64` 半块，让两个 CTA 分担一条 head 的工作。H96 fixed 的 CTA 数可从 96 增到 192，能够超过 B300 的 148 个 SM。H12 只会从 12 增到 24，仍然是硬门槛。公共 q、k 和 gate 会重复读取，所以原型必须同时检查 DRAM bytes。

项目为下一版保留了三道检查。K2-only 原型先在 H96 fixed 获得至少 10% 收益，H12 不能退化。通过以后再解释额外 DRAM 流量，最后才接回完整 forward，对 FLA reference 对拍并重跑 580 case。

## 建议的阅读顺序

先打开[交互流程图](PROJECT_WORKFLOW.html)，沿主线看一次输入、K1、workspace、K2 和输出。随后阅读本文的 K1 与 K2 两节，再看 `tests/torch_ref.py`，这样能把公式、张量和 kernel 结构对应起来。

准备理解性能结论时，先看 `experiments/ncu/ANALYSIS.md` 和 `experiments/tcgen05/ANALYSIS.md`。前者解释真实 K2 为什么缺并行工作，后者解释新指令为什么被 padding 拖慢。`FINAL_REPORT.md` 适合在这些概念已经熟悉以后完整阅读。

准备 PRE 时可以直接使用 `PREPARATION.md`。它按 12 页、约 13 分钟组织了页面内容、讲述稿和 15 个答辩问题。遇到数字需要追溯时，再从 `experiments/README.md` 进入对应结果目录。

## 相关文件

- [完整技术报告](FINAL_REPORT.md)
- [PRE 准备稿](PREPARATION.md)
- [实验索引](../experiments/README.md)
- [交互流程图](PROJECT_WORKFLOW.html)
- [流程图源文件](PROJECT_WORKFLOW.json)
- [官方项目说明](../README.md)
