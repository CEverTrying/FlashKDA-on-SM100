#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
result_dir="${script_dir}/results"
mkdir -p "${result_dir}"

make -C "${script_dir}" all
{
    nvcc --version
    (
        cd "${result_dir}"
        sha256sum ../kda_mma_microbench.cu ../build/kda_mma_microbench
    )
} > "${result_dir}/build_provenance.txt"
"${script_dir}/build/kda_mma_microbench" "$@" \
    2> >(tee "${result_dir}/provenance.txt" >&2) \
    | tee "${result_dir}/mma_microbench.csv"
