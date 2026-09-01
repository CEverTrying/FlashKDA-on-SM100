# FlashKDA 为什么在 B300 上仍然使用 SM80 MMA

我把题面、官方设计文档、CUDA 源码、旧 Task 1 材料和这次新增的 B300 原始结果重新对了一遍。这个项目要回答的问题很具体。FlashKDA 已经能在 Blackwell 上运行，也比 FLA 的 Triton `chunk_kda` 快，矩阵乘主路径却仍然使用 `mma.sync`。我们需要判断这是一次保守选择，还是遗漏了 SM100 的性能机会。

这份报告从 KDA 的计算过程讲起。后面的每个性能判断都标明证据属于纸面估算、microbenchmark、真实 kernel benchmark 还是 NCU。四种证据回答的问题不同，不能互相替代。

## 先认识 KDA 和 FlashKDA

普通注意力要显式处理 token 两两之间的关系，序列变长后，注意力矩阵随长度平方增长。线性注意力把历史压进一个固定大小的 state。每个新 token 先更新 state，再用 query 从 state 中取结果。KDA 在这条递推上加入 delta update、逐通道 gate 和 beta，让模型能控制旧信息保留多少、新信息写入多少。

K3 的生产配置有 96 个 KDA head，head dimension 为 128。TP8 部署以后，每张卡只负责 12 个 head。这个变化对大 GEMM 影响有限，对每个 head 各跑一条递推链的 K2 很要紧。

FlashKDA 把 forward 拆成两个 CUDA kernel。

K1 prepare 按 16 token 一个 chunk 处理 q、k、gate 和 beta。它构造 chunk 内的三角矩阵，用有限 Neumann 级数求逆，再把 K2 所需的中间量写到 workspace。不同 chunk 和 head 之间互相独立，所以 K1 的 grid 很大。

K2 recurrence 让一个 CTA 负责一条 sequence 的一个 head。它按 chunk 顺序读 workspace，更新 `128 x 128` state，并写 output。后一块 state 依赖前一块，单条链不能随意并行。固定长度时，K2 的 CTA 数只有 sequence 数乘 head 数。

源码把 `CHUNK=16` 固定在 `csrc/smxx/fwd_launch.cu`。K1 和 K2 的矩阵原子都采用 SM80 `m16n8k16` MMA。TMA 已经用于 global memory 与 shared memory 之间的搬运，矩阵计算没有换成 WGMMA 或 TCGEN05。

## 当前复现结果

实验运行在 NVIDIA B300 SXM6 AC，compute capability 10.3，CUDA 13.0，驱动 580.126.09，PyTorch 2.13.0+cu130。最终扩展 sha256 为 `d2cd7a499664f918dd2ebac8533e62d8f65ad91905373a4f4324b01fbfaa3dc7`。

官方 `tests/test_fwd.py` 已通过。固定 H96 和 mixed-varlen 都与项目自带 `torch_ref` 逐元素相等。20 组 FLA fp64 gold 的 output 检查通过，state 检查产生 8 条 warning。FlashKDA state ratio 的最大值在 fixed 为 `0.009628`，varlen 为 `0.008926`。测试源码对 state 使用 `warning=True`，这些告警没有触发失败。扩展测试一共 580 个 case，覆盖 fixed、varlen、batch、bf16/fp32 API state、四种 state 输入输出组合，以及最长 1048576 token。结果为 `580 passed in 701.59s`，JUnit 文件位于 `experiments/baseline/results/full_tests.xml`。

当前正式 benchmark 显式设置 `FLA_FLASH_KDA=0`，防止 FLA 对照自动分发回 FlashKDA。

| H | 输入 | FlashKDA fp32 API state | FLA chunk KDA | 加速 |
|---|---|---|---|---|
| 96 | fixed 8192 | 0.9990 ms | 2.3490 ms | 2.35x |
| 96 | mixed varlen | 0.8705 ms | 2.3734 ms | 2.73x |
| 96 | equal varlen | 0.7163 ms | 2.3264 ms | 3.25x |
| 64 | fixed 8192 | 0.9105 ms | 1.6276 ms | 1.79x |
| 64 | mixed varlen | 0.6595 ms | 1.6690 ms | 2.53x |
| 64 | equal varlen | 0.4848 ms | 1.5661 ms | 3.23x |

同日较早的一次复跑整体慢了 45% 到 73%，FLA 两条对照也一起变慢。当前表保留了 raw stdout、GPU 信息和扩展哈希。跨 allocation 的绝对值不用于 A/B，只看同一次作业内的对照。

最终 `.so` 导出的 sm_103a SASS 有 1544 条 HMMA、1118 条 TMA load 和 458 条 TMA store。HGMMA 与 TCGEN 计数均为零。这证明当前 B300 二进制确实采用 HMMA 与 TMA 的组合。

## CHUNK 16 为什么合理

第一个理由来自 gate 的数值范围。激活后的 log gate 位于 `(-5, 0)`。一个长度为 C 的 chunk 在最坏情况下会产生 `exp(-5C)` 的累计 decay，以及 `exp(5C)` 的恢复因子。C 为 16 时，两者约为 `1.805e-35` 和 `5.541e34`，仍在 bf16 正常有限范围内。C 为 32 时变成 `3.257e-70` 和 `3.069e69`，已经发生下溢与上溢。没有 rescale 时，动态范围允许的整数上限约为 17。

第二个理由来自 Neumann 求逆。严格下三角的 C 阶矩阵是幂零矩阵，有限级数可以因式分解。C 为 16、32、64 时需要 6、8、10 次矩阵乘。现有 16 方阵 tile 下，MMA 指令数从 12 增到 128 和 1280，求逆 FLOP 分别为 49152、524288 和 5242880。C 从 16 增到 32，求逆工作已经增加 10.7 倍。

K1 主要数组的 shared-memory 下界也会增长。把 `k_restored` 算进去以后，C 为 16、32、64 时至少需要 17920、38912 和 90112 bytes。这里还没有算 beta、`g_total`、barrier、padding 和 spill。

第三个理由来自指令形状。SM80 `m16n8k16` 可以用两条指令完整覆盖 `16 x 16 x 16`，有效算术占比为 100%。三个约束中，bf16 range 在 C 为 32 时先失效，求逆代价紧随其后。形状匹配支持 C 为 16，却没有能力挽救更大的 chunk。

## TCGEN05 直接替换为什么失败

SM100 dense BF16 SS TCGEN05 的最小 M 为 64，N 按 8 递增，K 为 16。逻辑 `16 x 16 x 16` 只能映射到 `m64n16k16`，四分之三的 M 行没有实际工作。C 为 32 时有效率为 50%，C 为 64 才能填满 M。

`experiments/tcgen05/` 实现了真实 HMMA 与 TCGEN05 指令的 B300 microbenchmark。两条路径使用相同 block 数、累计次数和逻辑输入，各自采用合法的 staging 与物理 tile。校验覆盖第一、中间和最后一个 CTA，使用与计时相同的 64 次累加，结果与整数 CPU reference 精确相等。

| 路径 | C | 平均 launch | useful TFLOPS | physical TFLOPS |
|---|---|---|---|---|
| HMMA exact | 16 | 4.371 us | 71.02 | 71.02 |
| TCGEN05 padded | 16 | 12.319 us | 25.19 | 100.78 |
| HMMA exact | 32 | 8.223 us | 301.97 | 301.97 |
| TCGEN05 padded | 32 | 16.414 us | 151.27 | 302.54 |

TCGEN05 的 physical throughput 并不差，C16 甚至高于 HMMA。浪费的 48/64 行把 useful throughput 压到 HMMA 的 35.5%，延迟变成 2.82 倍。C32 仍慢 2.00 倍。另一轮作业的绝对延迟不同，方向和差距都保持稳定。

这个实验否定的是一个 C16 工作独占一条 M64 指令的直接替换。它没有否定把四个独立 head 或 chunk 打包进 M64，也没有替代端到端 K1/K2 原型。直接替换已经跨不过 microbenchmark gate，所以这次没有把更慢的路径接进生产 kernel。

## 并行度还能从哪里来

当前 focused NCU 保存了 fixed H96、fixed H12、mixed H96 和 equal H96 四份 `.ncu-rep`。它们采集 benchmark 的 bf16-state 分支，正式延迟表使用 fp32 API state，不能把两者当成同一 state 配置的联合测量。K1 在 H12 fixed 时仍有 6144 CTA 和 `5.19 waves/SM`，工作量够用。K2 的情况完全不同。

| BF16-state K2 case | CTA | waves/SM | SM throughput | DRAM throughput | achieved occupancy |
|---|---|---|---|---|---|
| fixed H96 | 96 | 0.32 | 20.62% | 18.72% | 9.37% |
| fixed H12 | 12 | 0.04 | 2.51% | 2.16% | 9.37% |
| mixed H96 | 576 | 1.95 | 27.25% | 25.38% | 16.67% |
| equal H96 | 768 | 2.59 | 38.63% | 36.19% | 16.82% |

TP8 后的 fixed K2 只让 12 个 CTA 工作，B300 有 148 个 SM。多数 SM 从一开始就没有任务。此时计算单元和 HBM 都接近空闲，优先问题是 grid 太小。

多 head 放进一个 CTA 能填宽 TCGEN05 的 M，却会把 H12 的 12 个 CTA继续减到 6 或 3。每个现有 K2 CTA 已使用 98.432 KiB dynamic shared memory，合并 state 还可能越过每 block 上限。它适合作为指令打包实验，不适合解决生产并行度。

单次 forward 的 persistent kernel也不能拆开一条 head 内的递推依赖。CUDA scheduler 已经会分配独立 CTA，额外工作队列主要增加原子开销。跨 decode step 保持 state 常驻是另一个问题，它会改变调用协议，值得单独研究。

两 CTA 按 value 列切分 state 更有希望。`128 x 128` state 可以拆成两个 `128 x 64` 半块，两边独立计算 `k @ state`、`q @ state`、output 和外积更新。H96 fixed 的 CTA 会从 96 增到 192，能够越过 148 SM 的一波门槛。H12 只会从 12 增到 24，仍然很少，而且公共输入会重复读取。这个方案应当先做 K2-only 原型，用 H96 与 H12 同时淘汰，不能只看聚合头数较多的形状。

## 这个负载落在哪个 roof

assignment 4.5 的 `in_proj_qkvgfab` 使用 TP8 本地 `N=6288, K=7168`。B300 cuBLAS BF16 实测显示，M 小于等于 64 时延迟约 19 us，M 为 1、8、16 时只有 4.70、38.26、73.34 TFLOPS。M 为 256 时仍落在 memory roof 一侧，M 到 1024 后转入 compute roof，M 为 4096 到 65536 时稳定在 1.15 到 1.29 PFLOPS。

这张表适合解释同一个模型里的大输入投影。它不能直接证明 FlashKDA 的 K1 或 K2 属于同一 roof。FlashKDA 有更小的 MMA、TMA workspace 和跨 chunk 递推，执行形态完全不同。

focused NCU 对 KDA 给出的结论更谨慎。K1 的 SM throughput 约 58% 到 70%，DRAM throughput 约 34% 到 59%，没有单一 roof 饱和。K2 fixed H12 的 SM 与 DRAM 分别只有 2.51% 和 2.16%，它处在并行度 roof 下方。equal-varlen 补足 grid 后，两项升到 38.63% 和 36.19%，仍然没有证据把它严格归为 compute-bound 或 HBM-bound。

后续评估至少保留 Tensor 指令数、SM throughput、DRAM throughput、DRAM read/write bytes、waves、register、dynamic shared memory 和 achieved occupancy。通用 Memory Throughput 不能当作 HBM 利用率。

## bf16 state 的精度边界

当前 fp32 state API 会在入口把 state 转成 bf16，片上递推仍使用 bf16，出口再转回 fp32。580 case 的 exact match 证明 API 转换正确，不能证明片上 fp32 没有价值。

专项实验建立三条共享量化输入的路径。fp64 逐 token recurrence 作为 gold。bf16 candidate 在每 16 token 边界回写 bf16，fp32 candidate 始终保留 fp32。完整矩阵中，bf16 candidate 的最大 RMSE ratio 为 `4.125e-3`，fp32 candidate 的最大值为 `2.819e-6`。真实 FlashKDA 相对独立 fp64 recurrence 的最大 final-state ratio 为 `7.155e-3`，最大 output ratio 为 `7.898e-3`。

合成的弱遗忘 case 在 1024 token 后进入约 0.4% 平台。它的每 token log-decay 约为 `-0.00168`，e-fold 只有约 596 token。这个平台不能外推到 e-fold 为 8192 token 的门控。大 value case 还同时改变了 seed 和长度，也不能只归因于 value 幅值。

当前数据说明 chunk 边界的 bf16 回写是可消除的误差源，在已测合成分布里没有继续发散。它没有覆盖真实 K3 权重与激活、端到端生成质量、多个 seed 和 head、训练反向，以及片上 fp32 state kernel 的性能成本。v2 若服务更长门控记忆，可以保留 fp32 state 选项做模型级评估。

## v2 要不要做 sm100a 专版

我的结论是暂不发布覆盖全部 forward 的 sm100a v2，把 SM100 专项原型集中在 K2 并行度与独立工作打包上。

支持专版的理由很明确。B300 有更强的 Tensor Core、TMA 和 cluster 能力，大输入投影已经能稳定达到 1 PFLOPS 以上。TCGEN05 的 physical throughput 也高。两 CTA state 分片可能让 H96 fixed 跨过单波覆盖门槛，跨 decode step 的持久 state 也可能省掉 HBM 流量和 launch。

反对现在发布的证据更直接。C16 与 TCGEN05 最小 M64 不匹配，直接替换慢 2.82 倍。C32 先撞上 gate range，再承担 10.7 倍求逆工作。生产 TP8 的 K2 只有 12 条链，换更强 MMA 无法给空闲 SM 增加任务。当前 HMMA 版本还保持 SM90 以上统一源码，减少了专版的实现、验证和维护成本。

下一版可以设三道门。第一道用 microbenchmark 淘汰形状不合适的指令映射。第二道要求 K2-only 原型在 H96 fixed 至少快 10%，H12 不退化，并解释 DRAM bytes 增量。第三道才做端到端 FlashKDA 与 FLA 对拍，重跑 580 case。跨过三道门再承担 sm100a 专版的维护成本。

## SM100 挑战完成到哪里

这次选择了题面中的只换指令切面。代码实现了真实 TCGEN05 与 HMMA 的相同形状 microbenchmark，B300 上完成了 timed-workload 校验和性能对比。结果否定了逐个替换 C16 原子的方案。生产 kernel保持原样，因为候选在独立 gate 已经明显更慢。

正确性方面，候选 MMA 对 CPU reference 精确相等，原 FlashKDA 对项目 `torch_ref` 的 580 case 精确相等。官方 FLA 的 output 检查通过，state warning 已在前文列出。性能方面，候选与 HMMA 的局部 A/B、原 FlashKDA 与 FLA 的端到端基线都已保留。这里没有一个可声称为 sm100a FlashKDA 的端到端候选，报告也不把 microbenchmark 写成生产加速。若评审把候选对 FLA 与 FlashKDA 的端到端直接 A/B 视为硬条件，这一挑战项仍缺生产级候选；当前完成的是指令替换切面的可行性淘汰实验。

这份负结果能支持一个具体决策。官方继续用 SM80 MMA 有充分依据，逐原子 TCGEN05 替换不值得集成。打包四个独立 C16、K1 完整 Neumann 原型和两 CTA K2 仍然没有被排除，它们是后续工作，不属于这次数据已经证明的部分。

## 怎样复现

所有 GPU 命令都从仓库根目录通过 `srun.sh` 申请 B300。

```bash
./srun.sh 'cd FlashKDA && export PATH=../.venv-b300/bin:$PATH && export TORCH_EXTENSIONS_DIR=../.cache/torch_extensions && python tests/test_fwd.py'
./srun.sh 'cd FlashKDA && export PATH=../.venv-b300/bin:$PATH && export TORCH_EXTENSIONS_DIR=../.cache/torch_extensions && FLASH_KDA_DIST_GPU=1 python -m pytest tests/test_fwd_full.py -q -n 16 --junitxml=experiments/baseline/results/full_tests.xml'
./srun.sh 'bash FlashKDA/experiments/baseline/run_b300.sh'
./srun.sh 'cd FlashKDA/experiments/tcgen05 && bash run.sh 64 100'
./srun.sh 'bash FlashKDA/experiments/ncu/run_b300.sh'
./srun.sh 'bash FlashKDA/experiments/sass/run.sh'
```

roofline 与 state precision 的复现命令分别位于 `experiments/roofline/README.md` 和 `experiments/precision/README.md`。原始 CSV、JUnit、`.ncu-rep`、SASS、hash 和环境信息都留在对应实验目录。

## 最后判断

FlashKDA 停在 SM80 MMA 是当前算法形状下的合理工程选择。CHUNK 16 同时满足 bf16 range、有限求逆成本和 `m16n8k16` 形状。SM100 的强项没有消失，它只是很难从逐个替换小 MMA 这条路兑现。

下一步最值得花时间的是 K2 并行度。先把一条 state 沿 value 列拆成两个 CTA，再用 H12 fixed 检查生产形状。这个原型若不能让空闲 SM 获得工作，换多少峰值更高的矩阵指令都不会改变结论。
