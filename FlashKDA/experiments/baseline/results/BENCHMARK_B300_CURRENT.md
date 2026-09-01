# KDA forward benchmark (Blackwell / B300 current)

- Generated: 2026-08-30

- Command: `python benchmarks/generate_benchmark_md.py -o experiments/baseline/results/BENCHMARK_B300_CURRENT.md --device-label Blackwell / B300 current --mode all --D 128 --warmup 30 --iters 200 --repeats 5`

- Benchmark settings: `warmup=30`, `iters=200`, `repeats=5`

- `fla_chunk_kda` configuration: `use_gate_in_kernel=True`, `use_qk_l2norm_in_kernel=True`, `use_beta_sigmoid_in_kernel=True`, `lower_bound=-5`, `transpose_state_layout=True`
- `fla_chunk_gated_delta_rule` configuration: scalar per-head gate `g` of shape `(1, T, H)`, `use_qk_l2norm_in_kernel=True`, `transpose_state_layout=True`

### `T=8192`, `H=96`, `D=128`

| Case | `flash_kda` mean (ms) | `fla_chunk_kda` mean (ms) | Speedup vs `chunk_kda` | `fla_chunk_gdn` mean (ms) | Speedup vs `gdn` |
|------|----------------------:|----------------------:|--------:|----------------------:|--------:|
| Fixed | 0.9990 | 2.3490 | 2.35× | 1.2949 | 1.30× |
| Varlen, `seq_lens`=[1300, 547, 2048, 963, 271, 3063] | 0.8705 | 2.3734 | 2.73× | 1.3021 | 1.50× |
| Varlen, `seq_lens`=`1024 x 8` | 0.7163 | 2.3264 | 3.25× | 1.2557 | 1.75× |

### `T=8192`, `H=64`, `D=128`

| Case | `flash_kda` mean (ms) | `fla_chunk_kda` mean (ms) | Speedup vs `chunk_kda` | `fla_chunk_gdn` mean (ms) | Speedup vs `gdn` |
|------|----------------------:|----------------------:|--------:|----------------------:|--------:|
| Fixed | 0.9105 | 1.6276 | 1.79× | 0.8897 | 0.98× |
| Varlen, `seq_lens`=[1300, 547, 2048, 963, 271, 3063] | 0.6595 | 1.6690 | 2.53× | 0.9448 | 1.43× |
| Varlen, `seq_lens`=`1024 x 8` | 0.4848 | 1.5661 | 3.23× | 0.8550 | 1.76× |

Raw stdout and environment provenance are stored beside this report. A prior
run on the same date was 45-73% slower across FlashKDA and both FLA controls;
the shared slowdown indicates run-to-run environment or clock variation, not
a FlashKDA-only regression. Absolute latencies from different allocations are
not used as an A/B comparison.
