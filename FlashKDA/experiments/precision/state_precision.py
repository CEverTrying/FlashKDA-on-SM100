#!/usr/bin/env python3
"""Measure the error caused by storing FlashKDA recurrent state in bf16.

The numerical model deliberately has a small, auditable surface.  All three
paths consume the same bf16-quantized q/k/v/g/beta inputs.  The gold path runs
the token recurrence in fp64.  The two candidate paths run the same recurrence
in fp32 and differ only at chunk boundaries: one rounds the state to bf16 and
the other keeps it in fp32.

An optional FlashKDA check runs the real kernel twice with bf16 and fp32 state
API tensors.  This is useful because the current kernel converts fp32 state to
bf16 on entry and converts it back on exit; fp32 at the API is not an fp32
on-chip accumulator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F


CHUNK = 16
LOWER_BOUND = -5.0


@dataclass(frozen=True)
class Case:
    name: str
    layout: str
    seq_lens: tuple[int, ...]
    input_distribution: str
    gate_mode: str
    beta_mode: str
    value_scale: float = 1.0


SMOKE_CASES = (
    Case("fixed_baseline", "fixed", (256,), "normal", "random", "random"),
    Case("fixed_long_memory", "fixed", (256,), "normal", "long_memory", "high"),
    Case("varlen_stress", "varlen", (17, 65, 130), "positive", "alternating", "high"),
)


FULL_CASES = (
    Case("fixed_baseline", "fixed", (8192,), "normal", "random", "random"),
    Case("fixed_positive", "fixed", (2048,), "positive", "random", "random"),
    Case("fixed_long_memory", "fixed", (8192,), "normal", "long_memory", "high"),
    Case("fixed_strong_decay", "fixed", (2048,), "normal", "strong_decay", "high"),
    Case("fixed_low_update", "fixed", (2048,), "normal", "random", "low"),
    Case("fixed_large_value", "fixed", (2048,), "normal", "long_memory", "high", 8.0),
    Case("varlen_mixed", "varlen", (17, 65, 257, 1023), "normal", "random", "random"),
    Case("varlen_stress", "varlen", (31, 128, 511, 2048), "positive", "alternating", "high"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", default="cuda", help="Torch device for the reference model")
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=Path("results/state_precision"))
    parser.add_argument(
        "--flash-kda",
        action="store_true",
        help="Also compare the real kernel's bf16-state and fp32-state API paths",
    )
    return parser.parse_args()


def _quantize_bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(x.dtype)


def _sample(
    shape: tuple[int, ...], distribution: str, generator: torch.Generator, device: torch.device
) -> torch.Tensor:
    if distribution == "normal":
        return torch.randn(shape, generator=generator, dtype=torch.float32, device=device)
    if distribution == "positive":
        return torch.rand(shape, generator=generator, dtype=torch.float32, device=device)
    raise ValueError(f"unknown input distribution: {distribution}")


def _gate_raw(
    shape: tuple[int, ...], mode: str, generator: torch.Generator, device: torch.device
) -> torch.Tensor:
    if mode == "random":
        return torch.randn(shape, generator=generator, dtype=torch.float32, device=device)
    if mode == "long_memory":
        return torch.full(shape, -8.0, dtype=torch.float32, device=device)
    if mode == "strong_decay":
        return torch.full(shape, 8.0, dtype=torch.float32, device=device)
    if mode == "alternating":
        raw = torch.empty(shape, dtype=torch.float32, device=device)
        raw[..., 0::2, :, :] = -8.0
        raw[..., 1::2, :, :] = 8.0
        return raw
    raise ValueError(f"unknown gate mode: {mode}")


def _beta_logits(
    shape: tuple[int, ...], mode: str, generator: torch.Generator, device: torch.device
) -> torch.Tensor:
    if mode == "random":
        return torch.randn(shape, generator=generator, dtype=torch.float32, device=device)
    value = {"high": 8.0, "low": -8.0}.get(mode)
    if value is None:
        raise ValueError(f"unknown beta mode: {mode}")
    return torch.full(shape, value, dtype=torch.float32, device=device)


def make_inputs(case: Case, heads: int, dim: int, seed: int, device: torch.device) -> dict[str, torch.Tensor]:
    total = sum(case.seq_lens)
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (1, total, heads, dim)
    q = _quantize_bf16(_sample(shape, case.input_distribution, generator, device))
    k = _quantize_bf16(_sample(shape, case.input_distribution, generator, device))
    v = _quantize_bf16(_sample(shape, case.input_distribution, generator, device) * case.value_scale)
    raw_g = _quantize_bf16(_gate_raw(shape, case.gate_mode, generator, device))
    beta_logits = _quantize_bf16(_beta_logits((1, total, heads), case.beta_mode, generator, device))

    # Keep the initial state common and exactly representable in bf16.  This
    # isolates recurrent rounding rather than initial-state conversion.
    initial = _quantize_bf16(
        0.25 * torch.randn(
            (len(case.seq_lens), heads, dim, dim),
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
    )
    a_log = torch.zeros(heads, dtype=torch.float32, device=device)
    dt_bias = torch.zeros(heads, dim, dtype=torch.float32, device=device)
    return {
        "q": q,
        "k": k,
        "v": v,
        "raw_g": raw_g,
        "beta_logits": beta_logits,
        "initial": initial,
        "a_log": a_log,
        "dt_bias": dt_bias,
    }


def activate_inputs(data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Normalize in fp32 and round to bf16, matching FlashKDA's input contract.
    q = _quantize_bf16(F.normalize(data["q"].float(), p=2, dim=-1, eps=1e-6))
    k = _quantize_bf16(F.normalize(data["k"].float(), p=2, dim=-1, eps=1e-6))
    gate_arg = data["raw_g"].float() + data["dt_bias"].view(1, 1, *data["dt_bias"].shape)
    gate = LOWER_BOUND * torch.sigmoid(torch.exp(data["a_log"]).view(1, 1, -1, 1) * gate_arg)
    beta = torch.sigmoid(data["beta_logits"].float())
    return {"q": q, "k": k, "v": data["v"], "gate": gate, "beta": beta}


def _metric_row(
    case: Case,
    seq_index: int,
    start: int,
    end: int,
    target: str,
    candidate: str,
    gold: torch.Tensor,
    pred: torch.Tensor,
) -> dict[str, object]:
    gold64 = gold.double()
    diff = pred.double() - gold64
    ref_rms = gold64.square().mean().sqrt()
    rmse = diff.square().mean().sqrt()
    return {
        "case": case.name,
        "layout": case.layout,
        "seq_index": seq_index,
        "start": start,
        "end": end,
        "target": target,
        "candidate": candidate,
        "max_abs": diff.abs().max().item(),
        "mean_abs": diff.abs().mean().item(),
        "rmse": rmse.item(),
        "ref_rms": ref_rms.item(),
        "rmse_ratio": (rmse / (ref_rms + 1e-30)).item(),
    }


def _checkpoints(length: int) -> set[int]:
    values = {length}
    point = CHUNK
    while point < length:
        values.add(point)
        point *= 4
    return values


@torch.inference_mode()
def recurrent_reference(
    case: Case,
    active: dict[str, torch.Tensor],
    initial: torch.Tensor,
    scale: float,
    window: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, torch.Tensor]]:
    output_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    outputs = {
        "gold_fp64": torch.empty_like(active["v"], dtype=torch.float64),
        "bf16_state": torch.empty_like(active["v"], dtype=torch.float32),
        "fp32_state": torch.empty_like(active["v"], dtype=torch.float32),
    }
    final_states: dict[str, list[torch.Tensor]] = {name: [] for name in outputs}

    base = 0
    for seq_index, seq_len in enumerate(case.seq_lens):
        q = active["q"][0, base : base + seq_len]
        k = active["k"][0, base : base + seq_len]
        v = active["v"][0, base : base + seq_len]
        gate = active["gate"][0, base : base + seq_len]
        beta = active["beta"][0, base : base + seq_len]
        states = {
            "gold_fp64": initial[seq_index].double().clone(),
            "bf16_state": initial[seq_index].float().clone(),
            "fp32_state": initial[seq_index].float().clone(),
        }
        checkpoints = _checkpoints(seq_len)

        for token in range(seq_len):
            for name, state in states.items():
                compute_dtype = torch.float64 if name == "gold_fp64" else torch.float32
                q_t = q[token].to(compute_dtype)
                k_t = k[token].to(compute_dtype)
                v_t = v[token].to(compute_dtype)
                decay_t = gate[token].to(compute_dtype).exp()
                beta_t = beta[token].to(compute_dtype)
                state.mul_(decay_t.unsqueeze(-1))
                residual = v_t - torch.einsum("hk,hkv->hv", k_t, state)
                state.add_(torch.einsum("h,hk,hv->hkv", beta_t, k_t, residual))
                outputs[name][0, base + token] = torch.einsum("hk,hkv->hv", q_t * scale, state)

            at_chunk_boundary = (token + 1) % CHUNK == 0 or token + 1 == seq_len
            if at_chunk_boundary:
                states["bf16_state"] = states["bf16_state"].to(torch.bfloat16).float()

            if token + 1 in checkpoints:
                for candidate in ("bf16_state", "fp32_state"):
                    state_rows.append(
                        _metric_row(
                            case,
                            seq_index,
                            0,
                            token + 1,
                            "final_state" if token + 1 == seq_len else "state_checkpoint",
                            candidate,
                            states["gold_fp64"],
                            states[candidate],
                        )
                    )

        for name, state in states.items():
            final_states[name].append(state)
        for start in range(0, seq_len, window):
            end = min(start + window, seq_len)
            gold = outputs["gold_fp64"][:, base + start : base + end]
            for candidate in ("bf16_state", "fp32_state"):
                output_rows.append(
                    _metric_row(
                        case,
                        seq_index,
                        start,
                        end,
                        "output_window",
                        candidate,
                        gold,
                        outputs[candidate][:, base + start : base + end],
                    )
                )
        base += seq_len

    stacked_states = {name: torch.stack(values) for name, values in final_states.items()}
    return output_rows, state_rows, {**outputs, **{f"final_{k}": v for k, v in stacked_states.items()}}


@torch.inference_mode()
def run_flash_kda(
    case: Case,
    data: dict[str, torch.Tensor],
    model: dict[str, torch.Tensor],
    scale: float,
) -> list[dict[str, object]]:
    try:
        import flash_kda
    except ImportError as exc:
        raise RuntimeError("--flash-kda requested, but flash_kda cannot be imported") from exc

    cu_seqlens = None
    if case.layout == "varlen":
        offsets = [0]
        for length in case.seq_lens:
            offsets.append(offsets[-1] + length)
        cu_seqlens = torch.tensor(offsets, dtype=torch.long, device=data["q"].device)

    results: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for api_dtype, name in ((torch.bfloat16, "flash_bf16_api"), (torch.float32, "flash_fp32_api")):
        # The auditable recurrence uses FLA's [K, V] layout.  FlashKDA's
        # state_v_first API uses [V, K].
        initial = data["initial"].transpose(-2, -1).contiguous().to(api_dtype)
        final = torch.zeros_like(initial)
        out = torch.zeros_like(data["q"], dtype=torch.bfloat16)
        flash_kda.fwd(
            data["q"].to(torch.bfloat16),
            data["k"].to(torch.bfloat16),
            data["v"].to(torch.bfloat16),
            data["raw_g"].to(torch.bfloat16),
            data["beta_logits"].to(torch.bfloat16),
            scale,
            out,
            A_log=data["a_log"],
            dt_bias=data["dt_bias"],
            lower_bound=LOWER_BOUND,
            initial_state=initial,
            final_state=final,
            cu_seqlens=cu_seqlens,
        )
        results[name] = (out.float(), final.float())
    torch.cuda.synchronize()

    rows: list[dict[str, object]] = []
    gold_out = model["gold_fp64"]
    gold_state = model["final_gold_fp64"].transpose(-2, -1)
    for name, (out, state) in results.items():
        rows.append(_metric_row(case, -1, 0, out.shape[1], "flash_output_all", name, gold_out, out))
        rows.append(_metric_row(case, -1, 0, out.shape[1], "flash_final_state", name, gold_state, state))

    bf_out, bf_state = results["flash_bf16_api"]
    fp_out, fp_state = results["flash_fp32_api"]
    rows.append(_metric_row(case, -1, 0, bf_out.shape[1], "api_parity_output", "fp32_vs_bf16_api", bf_out, fp_out))
    rows.append(_metric_row(case, -1, 0, bf_out.shape[1], "api_parity_state", "fp32_vs_bf16_api", bf_state, fp_state))
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    model_rows = [row for row in rows if row["candidate"] in {"bf16_state", "fp32_state"}]
    by_candidate: dict[str, dict[str, float]] = {}
    for candidate in ("bf16_state", "fp32_state"):
        selected = [row for row in model_rows if row["candidate"] == candidate]
        ratios = [float(row["rmse_ratio"]) for row in selected]
        by_candidate[candidate] = {
            "max_rmse_ratio": max(ratios),
            "mean_rmse_ratio": sum(ratios) / len(ratios),
        }
    bf16_max = by_candidate["bf16_state"]["max_rmse_ratio"]
    fp32_max = by_candidate["fp32_state"]["max_rmse_ratio"]
    return {
        "aggregate_scope": "all output-window and state-checkpoint rows; not case-balanced",
        "aggregate": by_candidate,
        "ratio_of_independent_maxima": bf16_max / max(fp32_max, 1e-30),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if args.head_dim <= 0 or args.heads <= 0 or args.window <= 0:
        raise ValueError("head-dim, heads, and window must be positive")
    if args.flash_kda and args.head_dim != 128:
        raise ValueError("the current FlashKDA kernel requires --head-dim 128")
    cases = SMOKE_CASES if args.profile == "smoke" else FULL_CASES
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scale = 1.0 / math.sqrt(args.head_dim)

    output_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    flash_rows: list[dict[str, object]] = []
    for case_index, case in enumerate(cases):
        print(f"[{case_index + 1}/{len(cases)}] {case.name} seq_lens={case.seq_lens}", flush=True)
        data = make_inputs(case, args.heads, args.head_dim, args.seed + case_index, device)
        active = activate_inputs(data)
        out_case, state_case, model = recurrent_reference(
            case, active, data["initial"], scale, args.window
        )
        output_rows.extend(out_case)
        state_rows.extend(state_case)
        if args.flash_kda:
            if device.type != "cuda":
                raise ValueError("--flash-kda requires --device cuda")
            flash_rows.extend(run_flash_kda(case, data, model, scale))

    write_csv(args.output_dir / "output_window_metrics.csv", output_rows)
    write_csv(args.output_dir / "state_checkpoint_metrics.csv", state_rows)
    write_csv(args.output_dir / "flash_kernel_metrics.csv", flash_rows)
    all_rows = output_rows + state_rows
    metadata = {
        "profile": args.profile,
        "seed": args.seed,
        "device": str(device),
        "head_dim": args.head_dim,
        "heads": args.heads,
        "chunk": CHUNK,
        "window": args.window,
        "lower_bound": LOWER_BOUND,
        "flash_kda_enabled": args.flash_kda,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cases": [asdict(case) for case in cases],
        **summarize(all_rows),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    print(json.dumps(metadata["aggregate"], indent=2))
    print(f"wrote results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
