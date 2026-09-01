# Experiment index

| Requirement | Evidence | Reproduction |
|---|---|---|
| Current B300 baseline | `baseline/results/` | `baseline/run_b300.sh` |
| Official fixed/varlen and FLA checks | `baseline/results/official_test.log` | command in `../reports/FINAL_REPORT.md` |
| 580-case regression | `baseline/results/full_tests.xml` | command in `../reports/FINAL_REPORT.md` |
| CHUNK and TCGEN05 | `tcgen05/ANALYSIS.md`, source and CSV | `tcgen05/run.sh` |
| Thin GEMM roofline and grid model | `roofline/ANALYSIS.md`, CSV files | `roofline/run_b300.sh` |
| Parallelism candidates and counterexamples | `roofline/ANALYSIS.md`, `ncu/ANALYSIS.md` | `ncu/run_b300.sh` |
| bf16 state precision | `precision/RESULTS.md`, CSV and JSON files | `precision/README.md` |
| Current focused NCU | `ncu/ANALYSIS.md`, `.ncu-rep` and CSV files | `ncu/run_b300.sh` |
| Current SASS | `sass/final.sass`, counts and hashes | `sass/run.sh` |
| v2 decision and full report | `../reports/FINAL_REPORT.md` | evidence paths above |
| 12-page talk and defense questions | `../reports/PREPARATION.md` | rehearse at 12-15 minutes |

The final report distinguishes paper estimates, microbenchmarks, kernel
benchmarks, and NCU counters. Do not use a result from one category as a
replacement for another.
