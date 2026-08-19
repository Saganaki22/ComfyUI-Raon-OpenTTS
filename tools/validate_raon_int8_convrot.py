"""Validate a Raon-OpenTTS INT8 ConvRot safetensors (Comfy-native int8_tensorwise + ConvRot).

For every `<base>.comfy_quant` marker:
  * `<base>.weight` exists, dtype torch.int8, 2-D
  * `<base>.weight_scale` exists, dtype torch.float32, shape [out_features, 1]
  * marker JSON: format == "int8_tensorwise", convrot == true
  * in_features % convrot_groupsize == 0, group size in {256, 64, 16}
  * `<base>.bias` (if present) is NOT quantized

Also detects orphan weight_scale / comfy_quant entries and confirms the
sensitive paths (text embedding, ConvNeXt text blocks, attn_norm modulation,
time MLP, input projection, norm_out, proj_out) were NOT quantized.

Usage:
    python tools/validate_raon_int8_convrot.py <int8.safetensors> [--expect-blocks 28]
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

VALID_GS = (256, 64, 16)
# paths that must stay full precision in the V1 conservative recipe
SENSITIVE = re.compile(r"text_embed|text_blocks|attn_norm|time_embed|input_embed|norm_out|proj_out|mel_spec")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("safetensors")
    ap.add_argument("--expect-blocks", type=int, default=None,
                    help="expected transformer block count (28 for 1B, 22 for 0.3B)")
    args = ap.parse_args()

    path = Path(args.safetensors)
    errors: list[str] = []
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        key_set = set(keys)
        shapes = {k: tuple(f.get_slice(k).get_shape()) for k in keys}
        dtypes = {k: f.get_slice(k).get_dtype() for k in keys}

        markers = [k for k in keys if k.endswith(".comfy_quant")]
        scales = [k for k in keys if k.endswith(".weight_scale")]

        gs_hist = collections.Counter()
        q_params = 0
        q_layers = []
        for marker in markers:
            base = marker[: -len(".comfy_quant")]
            meta = json.loads(f.get_tensor(marker).numpy().tobytes())
            if meta.get("format") != "int8_tensorwise":
                errors.append(f"{base}: format={meta.get('format')!r} != 'int8_tensorwise'")
                continue
            if meta.get("convrot") is not True:
                errors.append(f"{base}: convrot is not true: {meta}")
            gs = int(meta.get("convrot_groupsize", 0))
            if gs not in VALID_GS:
                errors.append(f"{base}: unsupported convrot_groupsize {gs}")

            w_key, s_key = f"{base}.weight", f"{base}.weight_scale"
            if w_key not in key_set:
                errors.append(f"{base}: missing {w_key}")
                continue
            if s_key not in key_set:
                errors.append(f"{base}: missing {s_key}")
                continue
            if dtypes[w_key] != "I8":
                errors.append(f"{base}: weight dtype {dtypes[w_key]} != I8")
            if dtypes[s_key] != "F32":
                errors.append(f"{base}: weight_scale dtype {dtypes[s_key]} != F32")
            w_shape, s_shape = shapes[w_key], shapes[s_key]
            if len(w_shape) != 2:
                errors.append(f"{base}: weight not 2-D: {w_shape}")
                continue
            out_f, in_f = w_shape
            if s_shape != (out_f, 1):
                errors.append(f"{base}: weight_scale shape {s_shape} != ({out_f}, 1)")
            if gs and in_f % gs != 0:
                errors.append(f"{base}: in_features {in_f} not divisible by gs {gs}")
            if SENSITIVE.search(base):
                errors.append(f"{base}: SENSITIVE path was quantized")
            bias_key = f"{base}.bias"
            if bias_key in key_set and dtypes[bias_key] == "I8":
                errors.append(f"{base}: bias was quantized")
            gs_hist[gs] += 1
            q_params += out_f * in_f
            q_layers.append(base)

        # orphans
        orphan_scales = [s for s in scales if f"{s[:-len('.weight_scale')]}.comfy_quant" not in key_set]
        orphan_markers = [m for m in markers if f"{m[:-len('.comfy_quant')]}.weight_scale" not in key_set]
        if orphan_scales:
            errors.append(f"orphan weight_scale entries: {orphan_scales[:5]}")
        if orphan_markers:
            errors.append(f"orphan comfy_quant entries: {orphan_markers[:5]}")

        # full-precision inventory
        fp_weights = [
            k for k in keys
            if k.endswith(".weight")
            and f"{k[:-len('.weight')]}.comfy_quant" not in key_set
        ]
        fp_params = sum(int(torch.Size(shapes[k]).numel()) for k in fp_weights)
        total = q_params + fp_params

        print(f"file: {path.name} ({path.stat().st_size/1e9:.2f} GB)")
        print(f"quantized matrices      : {len(q_layers)}")
        print(f"  GS256                 : {gs_hist.get(256, 0)}")
        print(f"  GS64                  : {gs_hist.get(64, 0)}")
        print(f"  GS16                  : {gs_hist.get(16, 0)}")
        print(f"full-precision weights  : {len(fp_weights)}")
        print(f"quantized params        : {q_params/1e6:.1f}M")
        print(f"full-precision params   : {fp_params/1e6:.1f}M")
        print(f"quantized share         : {100.0*q_params/max(total,1):.2f}%")

        if args.expect_blocks is not None:
            n = args.expect_blocks
            expect = set()
            for i in range(n):
                p = f"transformer.transformer_blocks.{i}"
                expect |= {
                    f"{p}.attn.to_q", f"{p}.attn.to_k", f"{p}.attn.to_v",
                    f"{p}.attn.to_out.0", f"{p}.ff.ff.0.0", f"{p}.ff.ff.2",
                }
            missing = sorted(expect - set(q_layers))
            extra = sorted(set(q_layers) - expect)
            if missing:
                errors.append(f"expected block layers not quantized: {missing[:6]}")
            if extra:
                errors.append(f"unexpected quantized layers: {extra[:6]}")

    if errors:
        print(f"\nFAIL ({len(errors)} problem(s)):")
        for e in errors[:30]:
            print(f"  - {e}")
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()
