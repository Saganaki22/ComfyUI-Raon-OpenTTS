"""Unit-test representative INT8 ConvRot layers through comfy_kitchen.int8_linear.

For each tested layer:
  reference  : F.linear(x, bf16(original fp32 weight), bias)
  quantized  : comfy_kitchen.int8_linear(x, int8_weight, weight_scale, bias,
                                         out_dtype=bf16, convrot=True, convrot_groupsize=gs)
  recon check: F.linear(x, dequant+unrotate(int8_weight, scale)) -- isolates kernel math
               from quantization error (should be ~identical to the quantized output).

Reports max/mean abs error, relative L2, cosine similarity, NaN/Inf counts and the
backend comfy-kitchen selected.

Usage:
    python tools/test_int8_kernels.py <int8.safetensors> <fp32.safetensors> [--block 0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

import comfy_kitchen
from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_weight

LAYERS = [
    "transformer.transformer_blocks.{b}.attn.to_q",
    "transformer.transformer_blocks.{b}.attn.to_k",
    "transformer.transformer_blocks.{b}.attn.to_v",
    "transformer.transformer_blocks.{b}.attn.to_out.0",
    "transformer.transformer_blocks.{b}.ff.ff.0.0",
    "transformer.transformer_blocks.{b}.ff.ff.2",
]


def stats(name: str, got: torch.Tensor, ref: torch.Tensor) -> dict:
    got_f, ref_f = got.float(), ref.float()
    err = (got_f - ref_f).abs()
    rel_l2 = ((got_f - ref_f).norm() / ref_f.norm().clamp(min=1e-30)).item()
    cos = F.cosine_similarity(got_f.flatten(), ref_f.flatten(), dim=0).item()
    row = {
        "layer": name,
        "max_abs": err.max().item(),
        "mean_abs": err.mean().item(),
        "rel_l2": rel_l2,
        "cosine": cos,
        "nan": int(torch.isnan(got).sum()),
        "inf": int(torch.isinf(got).sum()),
    }
    print(f"  {name}")
    print(f"    max_abs={row['max_abs']:.5f} mean_abs={row['mean_abs']:.6f} "
          f"rel_l2={row['rel_l2']*100:.3f}% cos={row['cosine']:.6f} "
          f"nan={row['nan']} inf={row['inf']}")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("int8")
    ap.add_argument("fp32")
    ap.add_argument("--block", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda", torch.cuda.current_device())
    print(f"device: {device}")
    backends = comfy_kitchen.list_backends()
    avail = ", ".join(f"{k}: {v['available']}" for k, v in backends.items())
    print(f"backends: {avail}")

    torch.manual_seed(1234)
    all_rows = []
    with safe_open(args.int8, framework="pt", device="cpu") as qi, \
         safe_open(args.fp32, framework="pt", device="cpu") as qf:
        for pattern in LAYERS:
            base = pattern.format(b=args.block)
            meta = json.loads(qi.get_tensor(f"{base}.comfy_quant").numpy().tobytes())
            gs = int(meta["convrot_groupsize"])
            w_q = qi.get_tensor(f"{base}.weight").to(device)          # int8 [out, in]
            w_s = qi.get_tensor(f"{base}.weight_scale").to(device)    # fp32 [out, 1]
            bias = qi.get_tensor(f"{base}.bias").to(device)           # bf16 passthrough
            w_ref = qf.get_tensor(f"{base}.weight")                   # fp32 original
            b_ref = qf.get_tensor(f"{base}.bias")

            in_f = w_q.shape[1]
            x = torch.randn(2, 257, in_f, device=device, dtype=torch.bfloat16) * 0.5

            # reference: original weight at bf16 compute
            ref = F.linear(x, w_ref.to(device, torch.bfloat16), b_ref.to(device, torch.bfloat16))
            # quantized kernel path
            got = comfy_kitchen.int8_linear(
                x, w_q, w_s, bias, out_dtype=torch.bfloat16, convrot=True, convrot_groupsize=gs,
            )
            # reconstruction check: dequant + un-rotate the int8 weight, then plain GEMM
            h = _build_hadamard(gs, device=device, dtype=torch.float32)
            w_deq = _rotate_weight(w_q.float() * w_s, h, gs)
            recon = F.linear(x, w_deq.to(torch.bfloat16), b_ref.to(device, torch.bfloat16))

            print(f"{base}  gs={gs} shape={tuple(w_q.shape)}")
            all_rows.append(stats("int8 vs original-bf16", got, ref) | {"gs": gs})
            all_rows.append(stats("int8 vs recon(bf16 GEMM)", got, recon) | {"gs": gs})

    worst = max(all_rows, key=lambda r: r["rel_l2"])
    bad = [r for r in all_rows if r["nan"] or r["inf"]]
    print(f"\nworst rel_l2: {worst['rel_l2']*100:.3f}% ({worst['layer']} @ {worst['gs'] and 'gs'+str(worst['gs'])})")
    if bad:
        raise SystemExit(f"FAIL: non-finite outputs in {bad}")
    print("PASS (all finite)")


if __name__ == "__main__":
    main()
