#!/usr/bin/env python3
import csv
import sys


SMS = 148
CASES = (
    ("fixed_8192", 1, 512),
    ("mixed_varlen", 6, 518),
    ("varlen_1024x8", 8, 520),
)
HEADS = (96, 12)


def rows():
    for case, sequences, total_tiles in CASES:
        for heads in HEADS:
            for kernel, ctas, resident_blocks_per_sm in (
                ("K1", total_tiles * heads, 8),
                ("K2", sequences * heads, 2),
            ):
                yield {
                    "case": case,
                    "heads": heads,
                    "kernel": kernel,
                    "ctas": ctas,
                    "resident_blocks_per_sm": resident_blocks_per_sm,
                    "waves_per_sm": ctas / (SMS * resident_blocks_per_sm),
                    "first_wave_sm_coverage_pct": 100.0 * min(ctas, SMS) / SMS,
                }


def main():
    fieldnames = (
        "case",
        "heads",
        "kernel",
        "ctas",
        "resident_blocks_per_sm",
        "waves_per_sm",
        "first_wave_sm_coverage_pct",
    )
    output = open(sys.argv[1], "w", newline="") if len(sys.argv) > 1 else sys.stdout
    try:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows():
            writer.writerow(row)
    finally:
        if output is not sys.stdout:
            output.close()


if __name__ == "__main__":
    main()
