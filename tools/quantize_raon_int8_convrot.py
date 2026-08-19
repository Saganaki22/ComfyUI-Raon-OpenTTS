"""Quantize a clean Raon-OpenTTS safetensors to INT8 ConvRot (comfy-native int8_tensorwise).

Thin wrapper around the current Comfy-Org/comfy-model-tools quant_int8_convrot.py with
the verified Raon recipe:

  * quantizes exactly the 6 repeated block GEMMs per transformer block
    (attn to_q/to_k/to_v/to_out.0, ff up/down) -- 168 layers (1B) / 132 layers (0.3B)
  * ConvRot group size is picked per layer by the quantizer (largest of 256/64/16 that
    divides in_features): 1B gets 112x GS64 (K=1408) + 56x GS256 (K=1536/5632);
    0.3B gets 132x GS256 (K=1024/2048)
  * everything else (text embedding, ConvNeXt text blocks, attn_norm/time/input/final
    projections, norms, convs, inv_freq) passes through; fp32 floats are downcast to bf16
  * standard absmax (no --mseclip)

Usage:
    python tools/quantize_raon_int8_convrot.py ^
        --src models/raon_opentts/Raon-OpenTTS-1B/Raon-OpenTTS-1B-fp32.safetensors ^
        [--dst ...] [--dry-run]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

QUANTIZER = Path(__file__).resolve().parent.parent / "upstream" / "comfy-model-tools" / "quant_int8_convrot.py"

# conservative V1 recipe: keep the AdaLN modulation and the whole text-conditioning
# path (embedding + ConvNeXt blocks) full precision
EXCLUDE = r"attn_norm|text_embed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="clean fp32/bf16 safetensors from convert_raon_to_safetensors.py")
    ap.add_argument("--dst", default=None, help="output path (default: <src dir>/<model>-int8-convrot.safetensors)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    if args.dst:
        dst = Path(args.dst)
    else:
        dst = src.with_name(src.name.replace("-fp32.safetensors", "-int8-convrot.safetensors")
                              .replace("-bf16.safetensors", "-int8-convrot.safetensors"))
        if dst == src:
            dst = src.with_name(src.stem + "-int8-convrot.safetensors")
    verify = dst.with_name(dst.stem + "-verify.json")

    if not QUANTIZER.is_file():
        raise SystemExit(
            f"quantizer not found at {QUANTIZER}. Clone https://github.com/Comfy-Org/comfy-model-tools "
            f"into upstream/comfy-model-tools."
        )

    cmd = [
        sys.executable, str(QUANTIZER), str(src), str(dst),
        "--exclude", EXCLUDE,
        "--downcast-fp32",
        "--verify-report", str(verify),
    ]
    if args.dry_run:
        cmd = [sys.executable, str(QUANTIZER), str(src), "--dry-run", "--exclude", EXCLUDE]
    print("+", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
