#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd "$script_dir/../.." && pwd)
extension=$(find "$project_root" -maxdepth 1 -name 'flash_kda_C*.so' -print -quit)

cuobjdump --list-elf "$extension" > "$script_dir/elf.txt"
cuobjdump --dump-sass "$extension" > "$script_dir/final.sass"
test -s "$script_dir/final.sass"
(
    cd "$script_dir"
    sha256sum "../../$(basename "$extension")" final.sass > hashes.sha256
)

{
    awk '/HMMA/{n++} END{print "HMMA", n+0}' "$script_dir/final.sass"
    awk '/UTMALDG/{n++} END{print "UTMALDG", n+0}' "$script_dir/final.sass"
    awk '/UTMASTG/{n++} END{print "UTMASTG", n+0}' "$script_dir/final.sass"
    awk '/HGMMA/{n++} END{print "HGMMA", n+0}' "$script_dir/final.sass"
    awk '/TCGEN|UTCMMA/{n++} END{print "TCGEN", n+0}' "$script_dir/final.sass"
} > "$script_dir/instruction_counts.txt"
