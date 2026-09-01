#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
project_root="$repo_root/FlashKDA"
results_dir="$project_root/experiments/baseline/results"

cd "$project_root"
export PATH="$repo_root/.venv-b300/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$repo_root/.cache/torch_extensions"
export FLA_FLASH_KDA=0

mkdir -p "$results_dir"
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv \
    > "$results_dir/provenance_gpu.csv"
python -c 'import platform, torch; print(platform.platform()); print(torch.__version__); print(torch.version.cuda)' \
    > "$results_dir/provenance_python.txt"
sha256sum flash_kda_C*.so \
    > "$results_dir/provenance_extension.sha256"

python benchmarks/bench_fwd.py --mode all --H 96 --D 128 \
    --warmup 30 --iters 200 --repeats 5 \
    | tee "$results_dir/b300_current_h96.raw.txt"
python benchmarks/bench_fwd.py --mode all --H 64 --D 128 \
    --warmup 30 --iters 200 --repeats 5 \
    | tee "$results_dir/b300_current_h64.raw.txt"
