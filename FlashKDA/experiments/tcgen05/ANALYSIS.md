# FlashKDA CHUNK and TCGEN05 shape analysis

This note distinguishes paper estimates from measurements. The only measured
numbers are the CSV rows produced by `kda_mma_microbench` on the named GPU.

## Source facts

- `csrc/smxx/fwd_launch.cu` fixes `CHUNK=16`.
- K1 forms `L` and `Mqk` with an SM80 `m16n8k16` BF16 MMA, then computes the
  inverse with the FP16 factorization `(I-L)(I+L^2)(I+L^4)(I+L^8)`.
- K2 tiles every matrix operation as `16x16x16` and uses the same SM80 MMA.
- CUTLASS `mma_sm100_umma.hpp` constrains dense BF16 SS TCGEN05 to M=64 or
  M=128, N in multiples of 8, and K=16 per instruction. Its smallest legal
  instruction is therefore `m64n8k16`.

## Gate range

With `lower_bound=-5`, each activated log gate lies in `(-5, 0)`. A chunk's
worst-case cumulative decay and reciprocal are bounded by `exp(-5C)` and
`exp(5C)`. BF16 has maximum finite magnitude about `3.39e38`, minimum normal
about `1.18e-38`, and minimum subnormal about `9.18e-41`.

| C | exp(-5C) | exp(5C) | BF16 consequence without rescaling |
|---:|---:|---:|---|
| 16 | 1.805e-35 | 5.541e34 | finite and normal |
| 32 | 3.257e-70 | 3.069e69 | underflow / overflow |
| 64 | 1.061e-139 | 9.424e138 | underflow / overflow |

The finite-range threshold is `C <= floor(log(max_bf16)/5) = 17`. Thus the
range constraint fails before tensor-core shape matching at C=32.

## Neumann work and storage

For a strictly triangular C-by-C matrix, the factored finite series needs
`2*log2(C)-2` matrix products. Tiling a C-by-C product with the current
`16x16x16` atom takes `(C/16)^3` logical 16x16 tile products and two
`m16n8k16` instructions per tile product.

| C | matrix products | SM80 MMA instructions | FP operations | K1 main arrays at D=128 |
|---:|---:|---:|---:|---:|
| 16 | 6 | 12 | 49,152 | 17,920 B |
| 32 | 8 | 128 | 524,288 | 38,912 B |
| 64 | 10 | 1,280 | 5,242,880 | 90,112 B |

The storage column counts four C-by-D BF16 arrays (`k_decayed`, `q_decayed`,
`k_inv`, and the separately unioned `k_restored`) and three C-by-C BF16
arrays. It excludes beta, `g_total`, barriers, alignment gaps, and compiler
spill space, so it remains a lower bound for the full CTA allocation.
Relative to C=16, inversion FP work grows by 10.7x at C=32 and 106.7x at
C=64.

## MMA utilization

The current SM80 path maps C=16 exactly to two `m16n8k16` instructions and
has 100% useful arithmetic. A direct TCGEN05 replacement must cover a logical
`16x16x16` product with `m64n16k16`, wasting 48 of 64 M rows. Useful
arithmetic is 25%. C=32 reaches 50%; C=64 is the first square chunk that fills
the minimum M dimension. This shape mismatch is independent of TCGEN05 peak
throughput and remains even when staging and launch overhead are ignored.

The microbenchmark reports both useful TFLOP/s and issued physical TFLOP/s.
Both paths stage A and B in shared memory, using the physical layouts and
capacities required by their legal instructions. They repeat the MMA to make
timing stable, validate the logical C-by-C result with exact small-integer
inputs, and write CSV to stdout. It is a shape experiment, not an end-to-end
FlashKDA kernel. It does not model K2's register-resident intermediates,
recurrence, TMA pipeline, or state update.

## B300 measurement

`results/mma_microbench.csv` was produced on an NVIDIA B300 SXM6 AC
(compute capability 10.3, 148 SMs). Each launch used 592 CTAs, each CTA
repeated its GEMM 64 times, and each reported latency is the mean of 100
timed launches. The first, middle, and last CTA outputs matched the integer
CPU reference after the same 64 accumulated MMA iterations used by the timed
launch.

| path | C | mean launch | useful TFLOP/s | issued physical TFLOP/s |
|---|---:|---:|---:|---:|
| HMMA exact | 16 | 4.371 us | 71.02 | 71.02 |
| TCGEN05 M-padded | 16 | 12.319 us | 25.19 | 100.78 |
| HMMA exact | 32 | 8.223 us | 301.97 | 301.97 |
| TCGEN05 M-padded | 32 | 16.414 us | 151.27 | 302.54 |

For C=16, TCGEN05's physical throughput exceeded HMMA's in this test, yet
75% of the issued arithmetic belonged to padded rows. Its useful throughput
was 35.5% of HMMA and its launch took 2.82x as long. C=32 improved utilization
to 50%, but useful throughput remained 50.1% of HMMA and latency was 2.00x.
These measurements reject a direct instruction-only replacement for the
small square KDA operations tested here. A viable TCGEN05 design would have
to combine independent work across the M dimension or change the algorithmic
tile; replacing each 16x16 atom independently is not enough.

An earlier run on the same date measured 6.178/20.502 us for C=16 and
14.369/26.655 us for C=32. Absolute clocks varied across allocations, while
both runs kept the direct TCGEN05 replacement slower by a large margin. This
experiment reports one aggregate per variant and therefore supports the
direction of the result, not a precise production speedup estimate.

## Reproduction

From this directory on a B300 allocation:

```bash
make clean all ARCH=103a
./build/kda_mma_microbench 64 100 > results/mma_microbench.csv
```

From the assignment root, using the required launcher:

```bash
./srun.sh 'cd FlashKDA/experiments/tcgen05 && ./run.sh 64 100'
```

Do not compare an old CSV after changing the binary. The program prints GPU,
SM count, block count, and timing parameters to stderr for provenance.
