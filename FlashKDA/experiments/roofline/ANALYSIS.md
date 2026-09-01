# B300 roofline 与 FlashKDA 并行度分析

我先对照了 assignment02 的 4.5 题面和 `05_thin_gemm.cu`，随后读了 FlashKDA 的 launch、K1、K2 和 Task 1 留下的三份 NCU details。这里把输入投影 GEMM 与 KDA kernel 分开讨论。前者提供同一台 B300 上的形状参照，后者才是判断 FlashKDA 瓶颈的直接证据。

## 4.5 移植与实测

实验只取 KDA 对应的 `in_proj_qkvgfab`。TP8 本地权重形状为 `N=6288, K=7168`，数据类型为 BF16，cuBLAS 使用 FP32 累加与 BF16 输出。roof 口径沿用 assignment02 的 0.2，B300 dense BF16 峰值取 2250 TFLOPS，HBM 带宽取 8000 GB/s，机器平衡点为 281.25 FLOP/byte。

实测命令如下。

```bash
cd FlashKDA/experiments/roofline
bash run_b300.sh
```

Slurm job 11924 分配到 NVIDIA B300 SXM6 AC，compute capability 为 10.3。程序用 CUDA event 计时，每个 M 预热十次，自适应选择迭代数，做五轮后取中位数。完整 stdout 在 `results_b300.txt`，逐行数据在 `results_b300.csv`。

| M | 单次 us | AI | TFLOPS | 有效 GB/s | compute roof | memory roof | active roof |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 19.16 | 1.00 | 4.70 | 4705.33 | 0.21% | 58.82% | 58.82% |
| 8 | 18.85 | 7.98 | 38.26 | 4793.36 | 1.70% | 59.92% | 59.92% |
| 16 | 19.67 | 15.92 | 73.34 | 4605.34 | 3.26% | 57.57% | 57.57% |
| 64 | 18.61 | 62.80 | 310.01 | 4936.49 | 13.78% | 61.71% | 61.71% |
| 256 | 26.33 | 237.82 | 876.50 | 3685.48 | 38.96% | 46.07% | 46.07% |
| 1024 | 78.78 | 784.25 | 1171.73 | 1494.07 | 52.08% | 18.68% | 52.08% |
| 4096 | 320.99 | 1842.70 | 1150.31 | 624.25 | 51.12% | 7.80% | 51.12% |
| 16384 | 1160.34 | 2781.04 | 1272.85 | 457.69 | 56.57% | 5.72% | 56.57% |
| 65536 | 4565.98 | 3186.74 | 1293.86 | 406.01 | 57.50% | 5.08% | 57.50% |

M 从 1 增到 64，延迟一直贴着 19 us，增加的工作主要填满同一次 kernel 的 tile。M 小于等于 16 时只得到 4.70 到 73.34 TFLOPS，Tensor Core 的峰值在这个区间没有实际意义。vLLM 在这里换用 skinny CUDA Core kernel 有合理依据，它能绕开 cuBLAS 与 Tensor Core 路径的固定调度成本。

M 为 256 时 AI 仍低于机器平衡点，memory roof 是有效上限，实测达到 46.07%。M 到 1024 时已经越过平衡点，compute roof 开始生效。M 从 4096 增到 65536 后形成约 1.15 到 1.29 PFLOPS 的平台，compute roof 达成率为 51.12% 到 57.50%。平台开始的位置可以保守放在 M 等于 1024，4096 以后更稳定。

表里的有效带宽由题面流量模型推得。相同权重在重复迭代间可能命中缓存，所以它不能替代 NCU 的 DRAM bytes 和 throughput。这个限制不会改变 AI 的分界，也不会改变小 M 延迟基本恒定的观测。

## FlashKDA 的直接 NCU 证据

launch 代码把 K1 设为 `grid=(total_tiles,H)` 与 256 threads，把 K2 设为 `grid=(N,H)` 与 192 threads。固定长度 `T=8192` 时 total_tiles 为 512。varlen 的 K1 grid 使用上界，mixed 为 518，`1024 x 8` 为 520。K2 的每个 CTA 独占一条 sequence 和一个 head，chunk 间递推留在这个 CTA 内串行执行。

Task 1 没有随当前仓库保留原始 `.ncu-rep`。旧 details 只作为历史参考，当前结论不再引用其中的 counter 或 replay duration。

当前扩展已经在 `experiments/ncu/results/` 重采四组 focused `.ncu-rep`。新数据绑定扩展 sha256 `d2cd7a499664f918dd2ebac8533e62d8f65ad91905373a4f4324b01fbfaa3dc7`，包含 DRAM bytes、DRAM throughput、Tensor 指令、SM throughput、launch 与 occupancy。采集目标是 benchmark 的 bf16-state 分支，正式 CUDA event 延迟表使用 fp32 API state。完整表与计量限制见 `experiments/ncu/ANALYSIS.md`。

当前 focused NCU 把这个判断推进了一步。fixed H12 的 K2 只有 12 CTA 和 `0.04 waves/SM`，SM throughput 为 `2.51%`，DRAM throughput 为 `2.16%`。这条生产 TP8 形状首先缺少独立工作，两个 roof 都远未触顶。H96 equal-varlen 的 K2 增到 768 CTA 后，SM 与 DRAM throughput 分别升到 `38.63%` 和 `36.19%`。它仍没有单一 roof 饱和，增加独立递推链是当前最确定的收益来源。

后续 NCU 至少要保留 `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`、`dram__throughput.avg.pct_of_peak_sustained_elapsed`、`dram__bytes_read.sum` 与 `dram__bytes_write.sum`。SM Throughput、Achieved Occupancy、Waves Per SM、registers per thread 和 dynamic shared memory 用 full set 的 launch 与 occupancy section 一起取。前一组回答 Tensor 与 HBM 的使用程度，后一组负责识别 grid、寄存器和 shared memory 造成的空转。

K1 有四十多个 waves，静态并行度充足。K2 每 block 使用 98.43 KiB dynamic shared memory 和 73 registers per thread，每 SM 最多驻留两个 block，理论 occupancy 只有 18.75%。固定 H96 的 96 个 CTA 还少于 B300 的 148 个 SM，第一波最多覆盖 64.86% 的 SM。它的 `0.32 waves/SM` 直接说明 grid 不足。

varlen 把独立序列加入 grid。八条等长序列有 768 个 K2 CTA，当前 NCU 的 DRAM throughput 为 36.19%。mixed 只有六条序列，两组的总 token 数都是 8192。旧 CUDA event benchmark 中，mixed 与八条等长序列的完整 forward 分别为 0.8716 ms 和 0.7204 ms；当前保存 provenance 的复跑为 0.8705 ms 和 0.7163 ms。两次绝对值不能组成 A/B，但顺序一致。这组对照支持增加独立 sequence。序列数与长度分布同时变化，长度不均衡的独立影响还需要 N=8 mixed control。

## TP8 后的静态并行度

K3 有 96 个 head，TP8 后每卡只有 12 个。`grid_model.py` 用当前 launch 与 occupancy 数据复算了 CTA 与 waves，结果保存在 `results_grid.csv`。

| case | H | K1 CTA | K1 waves/SM | K2 CTA | K2 waves/SM | K2 第一波 SM 覆盖 |
| --- | --- | --- | --- | --- | --- | --- |
| Fixed 8192 | 96 | 49152 | 41.51 | 96 | 0.324 | 64.86% |
| Fixed 8192 | 12 | 6144 | 5.19 | 12 | 0.041 | 8.11% |
| Mixed varlen N6 | 96 | 49728 | 42.00 | 576 | 1.946 | 100% |
| Mixed varlen N6 | 12 | 6216 | 5.25 | 72 | 0.243 | 48.65% |
| Varlen N8 | 96 | 49920 | 42.16 | 768 | 2.595 | 100% |
| Varlen N8 | 12 | 6240 | 5.27 | 96 | 0.324 | 64.86% |

TP8 不会让 K1 缺工作。K2 的 fixed case 只剩 12 条独立递推链，理论上最多让 8.11% 的 SM 接到第一个 CTA。fixed H12 已有 focused NCU 实测，mixed H12 与 equal H12 仍然只是 launch 模型。

## 三类并行度候选

### 多 head 放进一个 CTA

这条路线只有在新的 MMA 指令需要更宽 tile 时才值得保留。可以把两个或四个 head 看成 block diagonal 矩阵，借更宽的 M 或 N 维度喂给 TCGEN05。代价是无效的跨 head 元素，或者额外的打包与拆包。

它不能解决 grid 不足。fixed TP8 原本有 12 个 CTA，两 head 合并后只剩 6 个。各 head 没有可复用的输入或状态，现有单 head CTA 已经使用 98.43 KiB shared memory，把两份完整状态同时放入 shared memory 还可能越过约 200.70 KiB 的配置上限。

可测判据应当同时看 K2 延迟和实际执行的 Tensor Core 指令数。只在 packed TCGEN05 比 HMMA 的 K2 中位延迟至少低 5%，无效 FLOP 比例可解释，shared memory 没有导致 cluster 或 block 驻留进一步下降时继续。H12 fixed 是必须通过的反例，H96 的好结果不能替代它。

### Persistent kernel

一个常驻 CTA 可以从工作队列领取不同 sequence 和 head。单次 forward 内，CUDA block scheduler 本来就会动态分派 576 个 mixed CTA，常驻队列无法拆开某个 head 内的递推链，也无法缩短最后那条 3063 token 序列。这个版本很可能只增加原子取任务的开销。

跨 decode step 的常驻版本有另一种收益。它可以把 state 留在片上，并连续接收新 token，省掉多次 kernel launch 与 state 搬运。这个方向会改变调用协议，还要解决 CTA 生命周期、请求加入退出和抢占，已经超出当前一次 forward 的局部重构。

fixed TP8 只有 12 条独立链。常驻 148 个 CTA 也只有 12 个能拿到工作，chunk 依赖没有消失。八条等长 varlen 已经有规整工作量，它是另一个反例，persistent 调度不应让这组退化。

单次 forward 的实验应构造三组相同总 token 的输入，分别用一条长序列、八条等长序列和一长多短序列。记录 K2 event latency、每 SM active cycles 分布、原子操作量和尾部时间。只有 mixed 至少快 5%，equal case 退化不超过 2%，fixed TP8 没有扩大启动成本时才值得合入。跨 step 版本要单独比较每 token 延迟和 state 的 HBM bytes，不能用 8192 token 的一次性 prefill 数字替它验收。

### 两 CTA 协作一个 head

K2 源码已经让四个 compute warp 各管 32 个 value 列。可以沿同一个维度把 `128 x 128` state 切成两个 `128 x 64` 半块，每个 CTA 保留完整的 key 行，只推进 64 个 value 列。`k @ state`、`q @ state`、value 修正、输出和 state 外积更新都能按列独立完成，两 CTA 不需要在每个 chunk 交换部分和。cluster multicast 或 DSM 只负责减少公共输入的重复读取，第一版无需依赖它们。

这会把 K2 CTA 数翻倍。H96 fixed 从 96 个 block 增到 192 个，有机会越过 148 SM 的单波门槛。TP8 fixed 只从 12 增到 24，覆盖率仍只有 16.22%，生产部署的并行度缺口仍然很大。每个 CTA 只保存一半 state、v 和 output，但 `k_decayed`、`q_decayed`、`k_restored`、INV 与 Mqk 仍要各读一份。当前 fixed H96 K2 已读取 885.59 MB、写入 186.41 MB，拆分后的重复读取可能吃掉并行收益。按 chunk 拆成奇偶 CTA 不可行，后一 chunk 需要前一 chunk 的 state。若改成前缀组合或扫描，算法和 workspace 都已超出两 CTA 的局部改造范围。

实验需要先做 K2 only 原型，比较一 CTA 与两 CTA 的 fixed H96、fixed H12、mixed H12 和 equal H12。检查输出与 PyTorch reference 的误差，再量 K2 latency、Tensor 指令、SM throughput、DRAM throughput、DRAM bytes 和每 SM active blocks。H96 至少快 10%，H12 不能变慢，DRAM bytes 的增长也要与公共输入重复量吻合，这组结果才足以覆盖实现与维护成本。

## 当前判断

输入投影在 M 大于等于 1024 时已经落到 compute roof 一侧，说明大型 prefill GEMM 能从 Blackwell Tensor Core 路线继续获益。FlashKDA K2 面对的是 `CHUNK=16` 的细粒度矩阵和跨 chunk 递推，两者的形状与并行度都不同。

当前证据最支持先测两 CTA 的 state 分片。单次 forward 的 persistent 缺少收益依据，跨 decode step 保持 state 常驻可以另立原型。多 head CTA 与 TP8 的目标直接冲突，应放在最后。任何路线都要用 H12 fixed 做淘汰测试，因为 H96 aggregate 的改进很容易掩盖部署时每卡 head 数骤减的问题。

## 证据位置

* assignment02 4.5 题面位于 `assignment02/handout/src/assignment02.md`
* 原始实验位于 `assignment02/cuda/m4_gemm/05_thin_gemm.cu`
* launch 位于 `FlashKDA/csrc/smxx/fwd_launch.cu`
* K2 chunk 递推位于 `FlashKDA/csrc/smxx/fwd_kernel2.cuh`
* 旧 NCU 文本位于 `FlashKDA/TASK1_assignment_artifacts/*_details.txt`
