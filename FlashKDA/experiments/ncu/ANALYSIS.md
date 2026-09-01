# Current B300 NCU evidence

These reports bind the final FlashKDA extension to focused NCU measurements.
Each `.ncu-rep` contains one K1 prepare kernel and one K2 recurrence kernel.
The CSV exports use Mbyte for DRAM read and write values.
The launch skips select the benchmark's timed BF16-state branch after its
mandatory warmup call. The headline latency benchmark uses FP32 API state, so
the NCU counters and headline latency are different state-API configurations.
The grid and parallelism conclusions below do not depend on that API dtype.

| Case | Kernel | CTA | waves/SM | DRAM read MB | DRAM write MB | DRAM % | Tensor inst | SM % | achieved occupancy % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed H96 | K1 | 49152 | 41.51 | 605.70 | 627.64 | 58.85 | 2,162,688 | 70.15 | 96.58 |
| fixed H96 | K2 | 96 | 0.32 | 885.59 | 186.41 | 18.72 | 20,447,232 | 20.62 | 9.37 |
| fixed H12 | K1 | 6144 | 5.19 | 75.75 | 33.36 | 34.09 | 270,336 | 57.72 | 91.94 |
| fixed H12 | K2 | 12 | 0.04 | 110.73 | 11.58 | 2.16 | 2,555,904 | 2.51 | 9.37 |
| mixed H96 | K1 | 49728 | 42.00 | 605.78 | 631.51 | 57.14 | 2,175,360 | 69.23 | 96.66 |
| mixed H96 | K2 | 576 | 1.95 | 906.73 | 202.16 | 25.38 | 20,567,040 | 27.25 | 16.67 |
| equal H96 | K1 | 49920 | 42.16 | 605.74 | 627.51 | 57.23 | 2,162,688 | 69.70 | 96.61 |
| equal H96 | K2 | 768 | 2.59 | 907.70 | 206.19 | 36.19 | 20,447,232 | 38.63 | 16.82 |

K1 has enough grid work in every case. K2 fixed H96 launches fewer CTAs than
the B300's 148 SMs. TP8's H12 case reduces that grid to 12 CTAs, leaving both
SM and DRAM throughput near idle even though each active CTA executes the
same recurrence. This is a parallelism limit before it is a compute or HBM
roof limit.

Independent sequences increase K2 from 96 to 576 or 768 CTAs. Achieved
occupancy, SM throughput, and DRAM throughput rise together. The mixed case
has six sequences while the equal case has eight, so this evidence supports
the benefit of more independent recurrence chains. It does not isolate the
effect of length balance; that requires an eight-sequence mixed control.

The profiles used uncontrolled clocks because cluster policy disallows NCU
clock control. NCU replay inflates the event timing printed by the benchmark,
so those printed seconds are not forward latency. Use the separate CUDA event
benchmark for latency and these reports for counters.
