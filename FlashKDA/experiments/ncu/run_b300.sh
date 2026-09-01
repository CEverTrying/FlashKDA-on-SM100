#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
project_root="$repo_root/FlashKDA"
result_dir="$project_root/experiments/ncu/results"
extension=$(find "$project_root" -maxdepth 1 -name 'flash_kda_C*.so' -print -quit)
metrics="sm__inst_executed_pipe_tensor.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum"

cd "$project_root"
export PATH="$repo_root/.venv-b300/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$repo_root/.cache/torch_extensions"
export FLA_FLASH_KDA=0
mkdir -p "$result_dir"

profile_case() {
    local name=$1
    local skip=$2
    shift 2
    ncu --force-overwrite --kernel-name-base function \
        -k 'regex:_flash_kda_fwd_(prepare|recurrence)' \
        --launch-skip "$skip" --launch-count 2 \
        --section LaunchStats --section Occupancy --metrics "$metrics" \
        --clock-control none --import-source yes --source-folders . \
        --export "$result_dir/$name.ncu-rep" \
        python benchmarks/bench_fwd.py "$@" --warmup 0 --iters 1 --repeats 1
    ncu --import "$result_dir/$name.ncu-rep" --page raw --csv \
        > "$result_dir/$name.csv"
}

# bench_fn always makes one warmup call. These skips select the timed
# BF16-state K1/K2 pair; update them if benchmarks/bench_fwd.py changes order.
profile_case fixed_h96 2 --mode fixed --H 96 --D 128
profile_case fixed_h12 2 --mode fixed --H 12 --D 128
profile_case mixed_h96 2 --mode varlen --H 96 --D 128
profile_case equal_h96 14 --mode varlen --H 96 --D 128

(
    cd "$result_dir"
    sha256sum "../../../$(basename "$extension")" > extension.sha256
    sha256sum ./*.ncu-rep > reports.sha256
)
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv \
    > "$result_dir/gpu.csv"
