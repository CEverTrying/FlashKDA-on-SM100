# B300 thin GEMM 与 KDA 并行度实验

这个目录保存 assignment02 4.5 中 `in_proj_qkvgfab` 的独立移植，以及 FlashKDA K1 和 K2 的 grid 分析。原 assignment02 仓库不会被修改。

## 运行 thin GEMM

从当前目录执行下面的命令。脚本会通过仓库根目录的 `srun.sh` 申请一块 GPU，编译 sm_103a cubin，并写出文本与 CSV。

```bash
bash run_b300.sh
```

B300 默认 roof 参数来自 assignment02 的 0.2 推导，dense BF16 为 `2250 TFLOPS`，HBM3e 为 `8000 GB/s`。也可以在已经拿到 GPU 的 shell 中手动传参。

```bash
make clean all
./thin_gemm_bench 2250 8000 results_b300.csv
```

程序固定使用 KDA 输入投影的 TP8 本地形状 `N=6288, K=7168`，M 取 `1, 8, 16, 64, 256, 1024, 4096, 16384, 65536`。每个形状预热十次，随后自适应选择迭代数，做五轮并取单次延迟的中位数。

CSV 中的 `effective_GBps` 按题面逻辑流量计算，即各读一次 A 和 W，各写一次 D。它不等于硬件计数器读出的 HBM 实际流量。`compute_roof_pct` 是实测 TFLOPS 除以 dense BF16 峰值，`memory_roof_pct` 是实测 TFLOPS 除以 `AI * HBM bandwidth`，`active_roof_pct` 使用两者中更低的 roof。

## 复算 grid

```bash
python3 grid_model.py results_grid.csv
```

脚本使用 B300 的 148 个 SM。K1 每 SM 最多 8 blocks、K2 每 SM 最多 2 blocks，取自 Task 1 保存的 NCU details。计算结果用于比较完整 H96 和 TP8 后 H12 的静态并行度。

完整数据解释、候选优化和反例见 [ANALYSIS.md](ANALYSIS.md)。
