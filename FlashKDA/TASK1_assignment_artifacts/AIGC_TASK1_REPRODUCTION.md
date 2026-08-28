# FlashKDA 第一层复现：B300 构建、SASS 与 NCU 证据

- 复现日期：2026-08-28（UTC）
- 仓库提交：`48c4b7aa710f03b96a587cbe6eca19944565fb1e`
- 工作目录：`/home/lcpu/35673796/FlashKDA-on-SM100/FlashKDA`
- Python 环境：`.venv-b300-managed`
- 范围：只验证 B300 构建/运行、正式 benchmark、sm_103a SASS 和 K1/K2 NCU 执行。
- 明确不包含：CHUNK 分析、TCGEN05 原型、精度实验、并行度改造。

## 1. 结论

FlashKDA 在 NVIDIA B300 SXM6 AC（compute capability 10.3）上已正确构建并运行。正式 H96/H64 benchmark 完成；扩展 `.so` 中存在原生 `fwd_launch.sm_103a.cubin`；NCU 对 Fixed、Mixed Varlen 和 1024x8 Varlen 的 H96 FP32-state 路径均实际捕获到 K1 prepare 和 K2 recurrence。

> FlashKDA扩展包含原生sm_103a cubin，但K1和K2仍生成HMMA.16816.F32.BF16，即SM80风格的M16N8K16 BF16输入、FP32累加MMA。未发现HGMMA、UTCMMA或TCGEN。函数符号同时显示使用SM90 TMA load/store，因此当前路径是“SM80 HMMA + SM90 TMA，编译为SM103a”，而不是Blackwell TCGEN05/TMEM路径。

SASS 中的 1544 条 HMMA 是 cubin 内所有已编译 kernel 变体的静态指令条数，不能解释为一次 forward 的动态 HMMA 指令数。NCU full-set replay 的 Duration 也不是正式 benchmark 延迟。

## 2. 环境与构建核验

环境记录见 `assignment_artifacts/B300_ENV.txt`：

| 项目 | 值 |
|---|---|
| GPU | NVIDIA B300 SXM6 AC，275040 MiB |
| Compute capability | 10.3 |
| Driver | 580.126.09 |
| CUDA toolkit | 13.0 / nvcc 13.0.88 |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu130 |
| Nsight Compute | 2025.3.1.0 |
| `flash_kda` | 0.0.1+48c4b7a，可编辑安装 |
| `flash-linear-attention` | 0.5.2 |

可复现的单架构构建/安装命令为：

```bash
cd /home/lcpu/35673796/FlashKDA-on-SM100/FlashKDA
source .venv-b300-managed/bin/activate
FLASH_KDA_CUDA_ARCHS=103a pip install -v --no-build-isolation -e .
```

`setup.py` 将 `103a` 列为支持架构，并生成：

```text
-gencode arch=compute_103a,code=sm_103a
```

本次对已有成功构建产物做了以下只读核验：

```bash
source .venv-b300-managed/bin/activate
python -c 'import sys, torch, flash_kda, flash_kda_C; print(sys.version); print(torch.__version__, torch.version.cuda); print(flash_kda_C.__file__)'
/usr/local/cuda/bin/cuobjdump -lelf flash_kda_C.cpython-312-x86_64-linux-gnu.so
```

关键输出：

```text
ELF file    1: fwd_launch.sm_103a.cubin
```

扩展大小为 16,257,760 bytes，SHA-256 为 `9494cd2338129efe9c198898d326c38a0bf59d64e8ee97ef57215a53533ee465`。后续正式 benchmark 和 NCU 对该扩展的实际 kernel 执行共同证明运行成功。

## 3. 正式 B300 benchmark

完整、未删减的正式结果保存在 `BENCHMARK_B300.md`。生成命令为：

```bash
source .venv-b300-managed/bin/activate
python benchmarks/generate_benchmark_md.py \
  -o BENCHMARK_B300.md \
  --device-label 'Blackwell / B300' \
  --mode all --D 128 --warmup 30 --iters 200 --repeats 5
```

正式均值如下；Flash 列使用 FP32 state：

| H | Case | FlashKDA (ms) | chunk_kda (ms) | GDN (ms) |
|---:|---|---:|---:|---:|
| 96 | Fixed | 0.9987 | 2.3778 | 1.2915 |
| 96 | Mixed Varlen | 0.8716 | 2.3953 | 1.3360 |
| 96 | Varlen 1024x8 | 0.7204 | 2.3414 | 1.2853 |
| 64 | Fixed | 0.9098 | 1.6648 | 0.8856 |
| 64 | Mixed Varlen | 0.6589 | 1.6747 | 0.9490 |
| 64 | Varlen 1024x8 | 0.4873 | 1.5706 | 0.8625 |

`BENCHMARK_B300.md` 的 SHA-256 为 `bdd64f551074eed36816a133909d659b6de43fefef27ce43d4af4083bd7ffed1`。

## 4. sm_103a SASS 证据

### 4.1 导出与架构确认

```bash
source .venv-b300-managed/bin/activate
EXT=$(python -c 'import torch, flash_kda_C; print(flash_kda_C.__file__)')
/usr/local/cuda/bin/cuobjdump -lelf "$EXT"
/usr/local/cuda/bin/cuobjdump -sass "$EXT" \
  > assignment_artifacts/FLASHKDA_B300.sass
```

`FLASHKDA_B300.sass` 文件头为：

```text
Fatbin elf code:
================
arch = sm_103a
code version = [1,8]
host = linux
compile_size = 64bit
code for sm_103a
```

因此下面的反汇编来自原生 sm_103a ELF，而不是用 PTX 在运行时临时生成的其他架构代码。

### 4.2 指令类别计数

计数命令：

```bash
awk 'BEGIN { h=0; g=0; u=0; t=0 }
     /HMMA/   { h++ }
     /HGMMA/  { g++ }
     /UTCMMA/ { u++ }
     /TCGEN/  { t++ }
     END { printf "HMMA %d\nHGMMA %d\nUTCMMA %d\nTCGEN %d\n", h, g, u, t }' \
  assignment_artifacts/FLASHKDA_B300.sass
```

结果：

| 指令类别 | 静态条数 |
|---|---:|
| HMMA | 1544 |
| HGMMA | 0 |
| UTCMMA | 0 |
| TCGEN | 0 |

合并证据保存在 `assignment_artifacts/SASS_HMMA_EVIDENCE.txt`。第一部分是按函数归类的静态计数：

```text
prepare_K1           88
recurrence_K2      1456
```

第二部分是 20 条代表性指令，例如：

```text
HMMA.16816.F32.BF16 R12, R28.reuse, R40, RZ ;
HMMA.16816.F32.BF16 R8, R28, R42, RZ ;
```

`16816` 表示 M16N8K16，输入为 BF16，累加/输出为 FP32。反汇编函数模板符号同时含 `SM90_TMA_LOAD` 和 `SM90_TMA_STORE`。这支持“SM80 风格 HMMA + SM90 TMA，编译为 sm_103a”的结论；SASS 中未发现 Blackwell TCGEN05/TMEM 指令族。

## 5. NCU：K1/K2 实际执行证据

### 5.1 为什么 `--launch-skip` 落在 FP32 state

`benchmarks/bench_fwd.py` 依次运行 bf16-state、no-state、fp32-state，每次配置执行一次强制 warmup 和一次计时迭代；每次 `flash_kda.fwd` 启动 K1/K2 两个目标 kernel。因此：

- `--launch-skip 8 --launch-count 2` 跳过前两种 state 的 8 个 K1/K2 launch，捕获当前 case 的 FP32-state K1/K2。
- `--mode varlen` 依次运行 Mixed 和 1024x8；`--launch-skip 20` 再跳过第一个 case 的 12 个目标 launch 以及第二个 case 的 bf16/no-state 8 个 launch，捕获 1024x8 FP32-state。
- `FLA_FLASH_KDA=0` 只关闭 FLA 的 FlashKDA 自动分派；这里被 profile 的目标是脚本直接调用的 `flash_kda.fwd`。

### 5.2 采集命令

Fixed H96 FP32：

```bash
srun -G 1 --time=00:30:00 \
  env FLA_FLASH_KDA=0 \
  /usr/local/cuda/bin/ncu \
  --set full \
  --target-processes all \
  --kernel-name-base function \
  -k 'regex:_flash_kda_fwd_(prepare|recurrence)' \
  --launch-skip 8 \
  --launch-count 2 \
  --clock-control none \
  --import-source yes \
  --source-folders . \
  --force-overwrite \
  --export assignment_artifacts/B300_FIXED_H96_FP32 \
  .venv-b300-managed/bin/python benchmarks/bench_fwd.py \
    --mode fixed --H 96 --D 128 \
    --warmup 0 --iters 1 --repeats 1
```

Mixed Varlen H96 FP32 使用相同命令，只改：

```text
--launch-skip 8
--export assignment_artifacts/B300_VARLEN_MIXED_H96_FP32
--mode varlen --H 96 --D 128
```

1024x8 Varlen H96 FP32 使用相同命令，只改：

```text
--launch-skip 20
--export assignment_artifacts/B300_VARLEN_1024x8_H96_FP32
--mode varlen --H 96 --D 128
```

三份 FP32 报告都能正常导入；每份各包含一次 `_flash_kda_fwd_prepare<...>` 和一次 `_flash_kda_fwd_recurrence<...>` invocation，设备均为 CC 10.3。NCU 采集日志对每个 kernel 显示 40 passes。完整模板实例名称很长，保留在对应 details 文件；其中 state 的 TMA/指针类型为 `float`，与 FP32-state 选择一致。

### 5.3 导出 details

```bash
for report in assignment_artifacts/*.ncu-rep; do
  base="${report%.ncu-rep}"
  /usr/local/cuda/bin/ncu \
    --import "$report" \
    --page details \
    --print-summary per-kernel \
    --log-file "${base}_details.txt" \
    --force-overwrite
done
```

Tensor Pipe 列使用 raw metric `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`；SM 和 Memory 列来自 details 的 Speed Of Light 百分比。Tensor Pipe 指标表示 tensor pipe active cycles 占峰值持续周期的比例，不是 FLOP/s。

### 5.4 Fixed H96 FP32

报告：`assignment_artifacts/B300_FIXED_H96_FP32.ncu-rep`

| Kernel | Grid x Block | Duration | Reg/thread | Shared memory/block | Achieved occupancy | Waves/SM | Tensor Pipe | SM | Memory |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| K1 `_flash_kda_fwd_prepare` | `(512,96,1)x(256,1,1)` | 272.54 us | 32 | 21.25 KiB dynamic + 1.02 KiB driver | 96.60% | 41.51 | 5.62% | 70.13% | 72.37% |
| K2 `_flash_kda_fwd_recurrence` | `(1,96,1)x(192,1,1)` | 718.11 us | 74 | 98.43 KiB dynamic + 1.02 KiB driver | 9.37% | 0.32 | 20.16% | 21.87% | 42.46% |

两者 static shared memory 均为 0，shared-memory configuration size 均为 200.70 KiB。

### 5.5 Mixed Varlen H96 FP32

报告：`assignment_artifacts/B300_VARLEN_MIXED_H96_FP32.ncu-rep`

| Kernel | Grid x Block | Duration | Reg/thread | Shared memory/block | Achieved occupancy | Waves/SM | Tensor Pipe | SM | Memory |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| K1 `_flash_kda_fwd_prepare` | `(518,96,1)x(256,1,1)` | 281.95 us | 32 | 21.25 KiB dynamic + 1.02 KiB driver | 96.71% | 42.00 | 5.47% | 69.16% | 70.40% |
| K2 `_flash_kda_fwd_recurrence` | `(6,96,1)x(192,1,1)` | 580.06 us | 74 | 98.43 KiB dynamic + 1.02 KiB driver | 16.90% | 1.95 | 23.79% | 27.18% | 47.49% |

### 5.6 Varlen 1024x8 H96 FP32

报告：`assignment_artifacts/B300_VARLEN_1024x8_H96_FP32.ncu-rep`

| Kernel | Grid x Block | Duration | Reg/thread | Shared memory/block | Achieved occupancy | Waves/SM | Tensor Pipe | SM | Memory |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| K1 `_flash_kda_fwd_prepare` | `(520,96,1)x(256,1,1)` | 281.15 us | 32 | 21.25 KiB dynamic + 1.02 KiB driver | 96.87% | 42.16 | 5.47% | 69.65% | 70.27% |
| K2 `_flash_kda_fwd_recurrence` | `(8,96,1)x(192,1,1)` | 417.60 us | 74 | 98.43 KiB dynamic + 1.02 KiB driver | 16.79% | 2.59 | 34.80% | 40.34% | 70.48% |

### 5.7 NCU warning 与实验限制

- 命令显式使用 `--clock-control none`。NCU 警告 GPU clocks 未修改，profile 指标可能受时钟变化影响。
- NCU 警告 6 个 CTC 用户数据收发指标不可访问：`ctc__rx_bytes_data_user.sum`、其峰值百分比/每秒派生项，以及对应的三个 `ctc__tx_bytes_data_user` 指标。这不影响本报告列出的 kernel launch、occupancy、Tensor/SM/Memory 指标。
- `--set full` 对每个目标 kernel 做 40-pass replay。表中 Duration 是 replay 报告内的一次目标 invocation 汇总，不可与正式 benchmark 的 event timing 混用，也不应把 K1/K2 Duration 简单相加后当成正式 forward 延迟。
- 每份报告使用 `--launch-count 2`，只采 FP32-state 的一对 K1/K2；没有做跨运行统计或锁频实验。
- NCU 证明 K1/K2 被动态执行；SASS 证明所编译 sm_103a 代码采用的 MMA 指令类别。二者用途不同，静态 HMMA 数量不是动态执行次数。

## 6. 交付物与校验

| 文件 | SHA-256 |
|---|---|
| `BENCHMARK_B300.md` | `bdd64f551074eed36816a133909d659b6de43fefef27ce43d4af4083bd7ffed1` |
| `assignment_artifacts/FLASHKDA_B300.sass` | `5eae99eedd4e662c40df01a12d6e20a4245d400efd8ec373c6064206e4c9e836` |
| `assignment_artifacts/SASS_HMMA_EVIDENCE.txt` | `81cbda478ba5e6aea6a2df120b270dfcbce20872dc0401bfa85a86087307386c` |
| `assignment_artifacts/B300_FIXED_H96_FP32.ncu-rep` | `7a57d16ea22f8588428ec542294f80a896588c698bd6386bfd72ab6a241cb0df` |
| `assignment_artifacts/B300_VARLEN_MIXED_H96_FP32.ncu-rep` | `98d1de1ca7432f25ee250f8acab315071a02b27f14cec4702f4561084db7c81c` |
| `assignment_artifacts/B300_VARLEN_1024x8_H96_FP32.ncu-rep` | `b7f5cc8015fe15404203af00ffc25a0f3478e18ea4afbb85f27c91c6ecea546d` |

对应文本已生成：

```text
assignment_artifacts/B300_FIXED_H96_FP32_details.txt
assignment_artifacts/B300_VARLEN_MIXED_H96_FP32_details.txt
assignment_artifacts/B300_VARLEN_1024x8_H96_FP32_details.txt
```
