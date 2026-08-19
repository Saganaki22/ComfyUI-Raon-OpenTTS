"""Validate a clean Raon-OpenTTS safetensors against the official model + source checkpoint.

Checks (ALL tensors, not a sample):
  1. strict load_state_dict into a freshly instantiated official CFM(DiT)
     (text_num_embeds derived from the checkpoint embedding rows).
  2. against the source .pt inference state dict (ema_model_state_dict with
     "ema_model." stripped): key exists, shape identical, dtype identical,
     torch.equal bit-exact.

Reports tensor/param counts, missing/unexpected, shape/dtype/value mismatches
and the global max_abs_diff.

Usage:
    python tools/validate_raon_safetensors.py ^
        --safetensors models/raon_opentts/Raon-OpenTTS-1B/Raon-OpenTTS-1B-fp32.safetensors ^
        --config models/raon_opentts/Raon-OpenTTS-1B/config.yaml ^
        --source models/raon_opentts/Raon-OpenTTS-1B/model_520000.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_raon_to_safetensors import (  # noqa: E402
    EMA_SKIP,
    STALE_KEYS,
    build_reference_model,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--safetensors", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--source", default=None, help="original .pt for a bit-exact comparison")
    args = ap.parse_args()

    st_path = Path(args.safetensors)
    print(f"[1/3] loading safetensors {st_path} ...", flush=True)
    sd = load_file(str(st_path), device="cpu")
    n_params = sum(t.numel() for t in sd.values())
    dtypes = {str(t.dtype) for t in sd.values()}
    print(f"  tensor count : {len(sd)}")
    print(f"  param count  : {n_params/1e6:.1f}M")
    print(f"  dtypes       : {dtypes}")

    print("[2/3] strict load into a fresh official model ...", flush=True)
    text_num_embeds = sd["transformer.text_embed.text_embed.weight"].shape[0] - 1
    print(f"  text_num_embeds = {text_num_embeds}")
    model = build_reference_model(Path(args.config), text_num_embeds)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    shape_bad = [
        k for k, v in model.state_dict().items()
        if k in sd and sd[k].shape != v.shape
    ]
    print(f"  missing={len(missing)} unexpected={len(unexpected)} shape_mismatch={len(shape_bad)}")
    if missing or unexpected or shape_bad:
        print(f"  missing: {missing[:8]}")
        print(f"  unexpected: {unexpected[:8]}")
        print(f"  shape mismatches: {shape_bad[:8]}")
        raise SystemExit("FAIL: safetensors does not reproduce the official model")
    model.load_state_dict(sd, strict=True)
    print("  model.load_state_dict(state_dict, strict=True): OK")
    del model

    if args.source:
        print("[3/3] full bit-exact comparison against the source checkpoint EMA ...", flush=True)
        ckpt = torch.load(args.source, map_location="cpu", weights_only=True)
        ema = {
            k.replace("ema_model.", ""): v
            for k, v in ckpt["ema_model_state_dict"].items()
            if k not in EMA_SKIP and isinstance(v, torch.Tensor)
        }
        for stale in STALE_KEYS:
            ema.pop(stale, None)

        missing_keys = [k for k in ema if k not in sd]
        extra_keys = [k for k in sd if k not in ema]
        shape_mismatch = []
        dtype_mismatch = []
        value_mismatch = []
        max_abs_diff = 0.0
        for key, ref in ema.items():
            if key not in sd:
                continue
            tensor = sd[key]
            if tensor.shape != ref.shape:
                shape_mismatch.append(key)
                continue
            if tensor.dtype != ref.dtype:
                dtype_mismatch.append(key)
            if not torch.equal(tensor, ref):
                value_mismatch.append(key)
                diff = (tensor.float() - ref.float()).abs().max().item() if tensor.numel() else 0.0
                max_abs_diff = max(max_abs_diff, diff)

        print(f"  compared tensors   : {len(ema)}")
        print(f"  missing keys       : {len(missing_keys)} {missing_keys[:5]}")
        print(f"  unexpected keys    : {len(extra_keys)} {extra_keys[:5]}")
        print(f"  shape mismatches   : {len(shape_mismatch)} {shape_mismatch[:5]}")
        print(f"  dtype mismatches   : {len(dtype_mismatch)} {dtype_mismatch[:5]}")
        print(f"  value mismatches   : {len(value_mismatch)} {value_mismatch[:5]}")
        print(f"  global max_abs_diff: {max_abs_diff}")
        if missing_keys or extra_keys or shape_mismatch or value_mismatch:
            raise SystemExit("FAIL: safetensors is not a lossless extraction of the EMA weights")
        if dtype_mismatch:
            raise SystemExit("FAIL: dtype mismatches (expected the source dtype for the reference build)")
        print("  bit-exact (torch.equal on every tensor): OK")
    print("PASS")


if __name__ == "__main__":
    main()
