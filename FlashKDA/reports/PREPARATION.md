# FlashKDA PRE 准备稿

这套内容按 12 页、约 13 分钟准备。页面只放听众需要记住的数字和图，细节留在口头说明与答辩。

## 第 1 页 题目和判断

页面内容

* FlashKDA on SM100
* 为什么 B300 仍使用 SM80 MMA
* 当前判断 直接 TCGEN05 替换不成立，K2 并行度更值得做

讲述约 45 秒

Kimi K3 的 KDA forward 已经有很快的 FlashKDA kernel。它在 B300 上运行时，矩阵计算依然使用 SM80 `mma.sync`。我们复现了这个事实，也尝试推翻它。最后得到的结论很明确。小 tile 直接换 TCGEN05 会更慢，生产 TP8 更大的问题是 K2 只有 12 条独立递推链。

## 第 2 页 KDA 在算什么

页面内容

* 历史被压进 `128 x 128` recurrent state
* 每个 token 更新 state，再由 query 读取 output
* gate 控制遗忘，beta 控制写入

讲述约 60 秒

普通注意力会显式处理 token 两两关系。KDA 把历史压进固定大小 state，计算量随序列长度线性增长。这个 state 让 KDA 适合长上下文，也带来一条不能忽略的依赖。后一块 state 要等前一块更新完。

## 第 3 页 FlashKDA 的两段 kernel

页面内容

* K1 prepare 处理 chunk 内计算，grid 大
* K2 recurrence 顺序推进 state，grid 小
* `CHUNK=16`，矩阵原子为 `m16n8k16`

讲述约 70 秒

K1 为每个 chunk 和 head 准备中间量，chunk 之间独立。K2 让一个 CTA 负责一条 sequence 的一个 head，按 chunk 顺序更新 state。后面的性能分歧都来自这个结构。K1 有大量并行工作，K2 的独立工作数取决于 sequence 和本卡 head 数。

## 第 4 页 复现是否可信

页面内容

* B300 SM103，CUDA 13.0，PyTorch 2.13
* 官方 output 检查通过，state 有 8 条 warning
* 580 case 全通过，最长 1048576 token
* 当前 `.so` sha256 与 NCU、SASS 绑定

讲述约 55 秒

我们没有沿用旧报告里的结论。环境重新搭建，官方测试和扩展 580 case 都重跑。项目参考的 fixed 与 varlen 输出逐元素相等，FLA output 检查通过。FLA state 检查有 8 条 warning，最大 ratio 为 0.009628。扩展套件得到 580 passed，用时 701.59 秒。SASS 和四份 NCU 报告都绑定同一个扩展哈希。

## 第 5 页 当前性能与指令证据

页面内容

* H96 fixed 0.9990 ms，FLA 2.3490 ms
* mixed 0.8705 ms，equal 0.7163 ms
* HMMA 1544，HGMMA 0，TCGEN 0

讲述约 65 秒

FlashKDA 在当前作业内比 FLA chunk KDA 快 2.35 到 3.25 倍。equal-varlen 有八条序列，mixed 有六条，两者的速度差支持增加独立链，暂时不能单独证明均匀长度减少长尾。最终 SASS 有 1544 条 HMMA，也大量使用 TMA，没有 HGMMA 或 TCGEN。

## 第 6 页 CHUNK 16 的三重约束

页面内容

| C | gate range | Neumann FLOP | SM80 useful |
|---|---|---|---|
| 16 | 有限 | 49152 | 100% |
| 32 | 溢出 | 524288 | 100% |
| 64 | 溢出 | 5242880 | 100% |

讲述约 75 秒

最先破的是数值范围。log gate 下界为负五，C32 的累计 decay 与恢复因子已经超出 bf16。求逆工作也增加 10.7 倍。C16 还能恰好匹配 SM80 小矩阵原子。这三个理由共同支持 CHUNK 16。

## 第 7 页 TCGEN05 microbenchmark

页面内容

| C16 路径 | 延迟 | useful TFLOPS |
|---|---|---|
| HMMA | 4.371 us | 71.02 |
| TCGEN05 M64 padded | 12.319 us | 25.19 |

* 有效算术只有 25%
* 直接替换慢 2.82 倍

讲述约 75 秒

TCGEN05 dense BF16 的最小 M 是 64。C16 只用到其中 16 行。physical throughput 达到 100.78 TFLOPS，硬件没有闲着，四分之三工作算在 padding 上。这个结果只否定逐原子替换，不否定把四份独立工作打包进 M64。

## 第 8 页 K2 卡在哪里

页面内容

| K2 case | CTA | SM % | DRAM % |
|---|---|---|---|
| fixed H96 | 96 | 20.62 | 18.72 |
| fixed H12 | 12 | 2.51 | 2.16 |
| equal H96 | 768 | 38.63 | 36.19 |

* NCU 采集 benchmark 的 bf16-state 分支

讲述约 80 秒

TP8 后每卡只有 12 个 head，fixed K2 只发 12 个 CTA，B300 有 148 个 SM。SM 和 HBM 都接近空闲，这个 case 还没碰到 compute roof 或 memory roof。focused NCU 采的是 bf16-state 分支，正式延迟表用 fp32 API state，两者用于回答的问题不同。equal-varlen 提供 768 条链以后，两项利用率一起上升。增加独立工作比替换 MMA 更优先。

## 第 9 页 三个并行方案

页面内容

* multi-head CTA 能填宽 tile，会继续缩小 grid
* persistent 单次 forward 拆不开递推链
* 2 CTA 按 value 列切 state，优先做原型

讲述约 75 秒

multi-head CTA 对 TCGEN 打包有帮助，对 H12 并行度有害。persistent work queue 不能拆开单条依赖，跨 decode step 常驻可以另做。两 CTA 切成两个 `128 x 64` state 半块，计算可以按 value 列独立。它能让 H96 从 96 CTA 增到 192，H12 仍只有 24，所以验收必须包含 H12。

## 第 10 页 bf16 state 精度

页面内容

* bf16 chunk 回写最大 RMSE ratio 0.4125%
* fp32 candidate 最大值低于 `3e-6`
* 当前 fp32 API 内部仍是 bf16
* 数据来自合成分布，不能替代模型质量

讲述约 70 秒

专项实验把输入量化、近似函数和 state 回写分开。bf16 回写误差在已测 gate 下稳定，fp32 可以显著降低这部分误差。弱遗忘 case 的 e-fold 只有约 596 token，8192 token 的平台不能外推到更长记忆。生产决策仍要用真实 K3 激活和生成质量验证。

## 第 11 页 v2 决策

页面内容

* 暂不发布全量 sm100a v2
* 先做 K2 两 CTA 与独立 C16 打包
* 三道门 形状 microbench、H96/H12 K2、580 case 端到端

讲述约 65 秒

Blackwell 专版有机会，当前证据不支持全面重写。直接 TCGEN 慢，C32 数值先坏，H12 grid 又太小。下一步只投最可能改善 K2 并行度的原型。局部结果过关以后，再承担端到端集成和维护成本。

## 第 12 页 收束

页面内容

* SM80 MMA 是当前形状下的合理选择
* SM100 的机会在组织独立工作
* 下一实验 两 CTA state split，H12 是硬门槛

讲述约 40 秒

这次没有得到一个更快的 sm100a kernel，得到了一条清楚的淘汰路径。C16 直接换 TCGEN 不值得做。让 K2 获得更多独立工作，才可能让 Blackwell 的硬件能力用于这条递推。

## 答辩问题

### 为什么 microbenchmark 可以停止生产集成

直接替换路线在相同 CTA 数、累计次数和逻辑输入下已经慢 2.82 倍。两边各自使用合法 staging，TCGEN05 的物理工作是四倍。端到端还要加入递推、workspace 和同步，缺少把差距翻回来的来源。它能淘汰逐原子映射，不能淘汰打包映射。

### 为什么不直接把 CHUNK 改成 64 来适配 TCGEN05

C32 已经触发 bf16 下溢与上溢，C64 更严重。C64 的 Neumann 求逆工作是 C16 的 106.7 倍，shared-memory 下界也增到 90112 bytes。先要设计 rescale 和新求逆方案，已经超出只换指令。

### 你怎样证明当前 kernel 用的是 SM80 MMA

最终 sm_103a cubin 的 SASS 有 1544 条 HMMA，HGMMA 和 TCGEN 都为零。SASS、NCU 和测试使用的 `.so` sha256 相同。

### H96 只有 96 CTA，为什么 K2 还能达到 20% SM throughput

一个 CTA 可以持续运行很久，活跃 SM 内仍有计算。96 CTA 少于 148 SM，第一波覆盖不全，dynamic shared memory 又限制每 SM 驻留。平均利用率不会归零，却明显受 grid 限制。

### 为什么 TP8 的 H12 是必须测试的形状

K3 有 96 个 head，TP8 后每卡是 12 个。H96 聚合 benchmark 会掩盖部署时 grid 缩小八倍的问题。fixed H12 的 K2 实测只有 2.51% SM throughput。

### equal-varlen 为什么比 mixed 快

现有数据只能确认两种因素的合并效果。两者总 token 都是 8192，equal 有八条等长链，mixed 有六条不等长链。序列数和长度分布同时改变，当前证据支持增加独立链，单独判断长尾影响还需要 N=8 mixed control。

### K2 是 compute-bound 还是 memory-bound

fixed H12 两边都远未饱和，首先是 parallelism-bound。equal H96 的 SM 和 DRAM 分别为 38.63% 与 36.19%，仍不足以严格归到单一 roof。

### thin GEMM roofline 为什么还要做

它给 assignment 4.5 的同卡参照，说明大 M 输入投影可以进入 compute roof。它也提醒我们，同一个模型里的大 GEMM 和小递推 kernel 不能用同一结论概括。

### 两 CTA 为什么可以按 value 列独立

state 的 value 维列块互不需要归约。`k @ state`、`q @ state`、output 和 state 外积更新都能在两个 `128 x 64` 半块上独立完成。公共 q、k 和 gate 需要重复读，NCU 要检查这部分代价。

### persistent kernel 为什么没有优先做

单次 forward 内，scheduler 已经会分派独立 CTA。persistent queue 不能拆开最长的递推链。跨 decode step 保持 state 常驻可能有收益，它属于调用协议级改造。

### fp32 API 为什么和 bf16 API 精确一致

当前实现入口把 fp32 state 转成 bf16，片上递推相同，出口再转回 fp32。这个 parity 证明 API 转换，不代表片上 fp32 state 没收益。

### 0.41% 的 state 误差能接受吗

它是合成 recurrence 的数值指标，不能直接对应 perplexity 或生成质量。当前数据说明误差源可控，生产接受与否需要真实激活和端到端模型评估。

### benchmark 为什么出现过 70% 漂移

较慢作业里 FlashKDA、chunk KDA 和 GDN 同时退化，说明环境、时钟或节点负载共同变化。当前结果保存了 raw stdout 和 provenance，任何 candidate A/B 都要求同一 allocation 交错测量。

### 这份负结果交付了什么

题面允许用扎实负结果论证官方选择。这里交付了真实 TCGEN05 代码、timed-workload correctness、两轮方向一致的 B300 数据、当前 FlashKDA 正确性与性能基线，也清楚保留了尚未排除的打包和两 CTA 路线。严格要求端到端 candidate A/B 的评审仍会把 challenge 判为未完全完成，答辩时要主动说明这个边界。

### 下一步实验的验收线是什么

两 CTA K2 在 H96 fixed 至少快 10%，H12 不退化，DRAM bytes 增量能够由公共输入重复解释。随后与 FLA reference 对拍并重跑 580 case，三项同时满足才进入生产集成。
