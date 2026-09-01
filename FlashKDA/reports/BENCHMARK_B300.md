# KDA forward benchmark (Blackwell / B300)

- Generated: 2026-08-28

- Command: `python benchmarks/generate_benchmark_md.py -o BENCHMARK_B300.md --device-label Blackwell / B300 --mode all --D 128 --warmup 30 --iters 200 --repeats 5`

- Benchmark settings: `warmup=30`, `iters=200`, `repeats=5`

- `fla_chunk_kda` configuration: `use_gate_in_kernel=True`, `use_qk_l2norm_in_kernel=True`, `use_beta_sigmoid_in_kernel=True`, `lower_bound=-5`, `transpose_state_layout=True`
- `fla_chunk_gated_delta_rule` configuration: scalar per-head gate `g` of shape `(1, T, H)`, `use_qk_l2norm_in_kernel=True`, `transpose_state_layout=True`

### `T=8192`, `H=96`, `D=128`

| Case | `flash_kda` mean (ms) | `fla_chunk_kda` mean (ms) | Speedup vs `chunk_kda` | `fla_chunk_gdn` mean (ms) | Speedup vs `gdn` |
|------|----------------------:|----------------------:|--------:|----------------------:|--------:|
| Fixed | 0.9987 | 2.3778 | 2.38× | 1.2915 | 1.29× |
| Varlen, `seq_lens`=[1300, 547, 2048, 963, 271, 3063] | 0.8716 | 2.3953 | 2.75× | 1.3360 | 1.53× |
| Varlen, `seq_lens`=`1024 x 8` | 0.7204 | 2.3414 | 3.25× | 1.2853 | 1.78× |

### `T=8192`, `H=64`, `D=128`

| Case | `flash_kda` mean (ms) | `fla_chunk_kda` mean (ms) | Speedup vs `chunk_kda` | `fla_chunk_gdn` mean (ms) | Speedup vs `gdn` |
|------|----------------------:|----------------------:|--------:|----------------------:|--------:|
| Fixed | 0.9098 | 1.6648 | 1.83× | 0.8856 | 0.97× |
| Varlen, `seq_lens`=[1300, 547, 2048, 963, 271, 3063] | 0.6589 | 1.6747 | 2.54× | 0.9490 | 1.44× |
| Varlen, `seq_lens`=`1024 x 8` | 0.4873 | 1.5706 | 3.22× | 0.8625 | 1.77× |
