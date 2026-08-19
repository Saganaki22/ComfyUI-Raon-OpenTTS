"""INT8 ConvRot runtime for Raon-OpenTTS.

Executes checkpoints quantized by comfy-model-tools/quant_int8_convrot.py through
comfy-kitchen's INT8 ConvRot path:

    weight   : torch.int8, offline-rotated per group (W_rot = W @ H^T)
    scale    : float32 [out_features, 1] per-output-row scale
    comfy_quant: uint8 JSON {"format": "int8_tensorwise", "convrot": true,
                             "convrot_groupsize": G}

At inference the activation is rotated online by comfy_kitchen.int8_linear
(X_rot = X @ H), dynamically row-quantized, and the INT8 GEMM output is rescaled
by scale_x * scale_w + bias. Weights stay INT8 resident; no dequantize-the-whole-
weight fallback exists on the production path.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger("RaonOpenTTS")

QUANT_META_SUFFIX = "comfy_quant"
SUPPORTED_FORMAT = "int8_tensorwise"


@dataclass(frozen=True)
class QuantLayerInfo:
    prefix: str
    group_size: int
    in_features: int = 0   # 0 = not recorded; skip strict shape check
    out_features: int = 0
    has_bias: bool = False


class ConvRotInt8Linear(nn.Module):
    """Drop-in nn.Linear replacement executing the comfy-kitchen INT8 ConvRot path."""

    def __init__(self, in_features: int, out_features: int, bias: bool, group_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.convrot_groupsize = group_size
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.int8), requires_grad=False)
        self.weight_scale = nn.Parameter(torch.empty(out_features, 1, dtype=torch.float32), requires_grad=False)
        self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False) if bias else None
        self.quant_format = SUPPORTED_FORMAT

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # comfy_quant is metadata, not a tensor parameter; consume it here.
        state_dict.pop(f"{prefix}{QUANT_META_SUFFIX}", None)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        import comfy_kitchen

        RUNTIME_STATS["calls"] += 1
        RUNTIME_STATS["weight_dtype"] = self.weight.dtype
        RUNTIME_STATS["groupsize"] = self.convrot_groupsize
        return comfy_kitchen.int8_linear(
            x.contiguous(),
            self.weight,
            self.weight_scale,
            self.bias,
            out_dtype=x.dtype,
            convrot=True,
            convrot_groupsize=self.convrot_groupsize,
        )

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, convrot_groupsize={self.convrot_groupsize}")


RUNTIME_STATS = {"calls": 0, "modules": 0, "weight_dtype": None, "groupsize": None}


def reset_runtime_stats() -> None:
    RUNTIME_STATS.update({"calls": 0, "modules": 0, "weight_dtype": None, "groupsize": None})


def scan_checkpoint_quantization(weights_path: Path) -> dict[str, QuantLayerInfo]:
    """Read every *.comfy_quant key from a safetensors file and parse its JSON.

    Returns {} for a plain float checkpoint. Raises on quant formats this
    nodepack cannot execute (never silently misreads INT8 weights as floats).
    """
    from safetensors import safe_open

    quant_map: dict[str, QuantLayerInfo] = {}
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        return quant_map
    with safe_open(str(weights_path), framework="pt", device="cpu") as f:
        key_names = list(f.keys())
        meta_keys = [k for k in key_names if k.endswith(f".{QUANT_META_SUFFIX}")]
        for key in meta_keys:
            meta = json.loads(f.get_tensor(key).numpy().tobytes())
            fmt = meta.get("format")
            convrot = meta.get("convrot")
            if fmt != SUPPORTED_FORMAT or convrot is not True:
                raise RuntimeError(
                    f"{weights_path.name}:{key} uses quant format {meta!r}, which this nodepack cannot run. "
                    f"Only '{SUPPORTED_FORMAT}' with convrot=true is supported."
                )
            prefix = key[: -len(f".{QUANT_META_SUFFIX}")]
            w_key = f"{prefix}.weight"
            in_f = int(meta.get("in_features", 0))
            out_f = int(meta.get("out_features", 0))
            if (not in_f or not out_f) and w_key in key_names:
                shape = f.get_slice(w_key).get_shape()
                out_f, in_f = int(shape[0]), int(shape[1])
            quant_map[prefix] = QuantLayerInfo(
                prefix=prefix,
                group_size=int(meta["convrot_groupsize"]),
                in_features=in_f,
                out_features=out_f,
                has_bias=bool(meta.get("has_bias", f"{prefix}.bias" in key_names)),
            )
    return quant_map


def validate_group_size(group_size: int, in_features: int) -> None:
    import math

    if group_size < 4 or group_size & (group_size - 1) != 0 or math.log(group_size, 4) % 1 != 0:
        raise ValueError(f"ConvRot group size must be a power of four (4/16/64/256/...), got {group_size}")
    if in_features % group_size != 0:
        raise ValueError(f"in_features {in_features} is not divisible by convrot_groupsize {group_size}")


def replace_quantized_linears(model: nn.Module, quant_map: dict[str, QuantLayerInfo]) -> list[str]:
    """Swap every nn.Linear named in quant_map for a ConvRotInt8Linear.

    Must run BEFORE weights are assigned. Raises if a target is missing, is not
    an nn.Linear, or its shape disagrees with the recorded metadata.
    """
    modules = dict(model.named_modules())
    replaced: list[str] = []
    for prefix, info in quant_map.items():
        if prefix not in modules:
            raise RuntimeError(f"Quantized layer {prefix} not found in model")
        target = modules[prefix]
        if not isinstance(target, nn.Linear) or isinstance(target, ConvRotInt8Linear):
            raise RuntimeError(f"Quantized layer {prefix} is {type(target).__name__}, expected nn.Linear")
        if (info.in_features and target.in_features != info.in_features) or (
            info.out_features and target.out_features != info.out_features
        ):
            raise RuntimeError(
                f"Quantized layer {prefix} shape mismatch: checkpoint "
                f"[{info.out_features}, {info.in_features}] vs model [{target.out_features}, {target.in_features}]"
            )
        validate_group_size(info.group_size, info.in_features or target.in_features)
        parent_name, _, child_name = prefix.rpartition(".")
        parent = modules[parent_name] if parent_name else model
        setattr(parent, child_name, ConvRotInt8Linear(
            info.in_features or target.in_features,
            info.out_features or target.out_features,
            target.bias is not None,
            info.group_size,
        ))
        replaced.append(prefix)
    RUNTIME_STATS["modules"] = len(replaced)
    return replaced


def quantized_parameter_count(model: nn.Module) -> tuple[int, int]:
    """Returns (int8 parameter elements, number of ConvRotInt8Linear modules)."""
    total = 0
    count = 0
    for module in model.modules():
        if isinstance(module, ConvRotInt8Linear):
            count += 1
            total += module.weight.numel()
    return total, count
