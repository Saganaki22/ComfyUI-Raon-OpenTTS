"""Verify the vendored rotary embedding is bit-exact vs the installed x_transformers.

Also proves the node pack imports and runs with x_transformers completely blocked.

Run from the ComfyUI root:
    venv/Scripts/python.exe custom_nodes/ComfyUI-Raon-OpenTTS/tools/test_rotary_vendored.py
"""

from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent.parent
COMFY_ROOT = PKG_DIR.parent.parent
sys.path.insert(0, str(COMFY_ROOT))

# block x_transformers BEFORE importing the pack, to prove independence
import types

block = types.ModuleType("x_transformers")
block.__path__ = []  # mark as package so submodule imports fail loudly


def _blocked(*a, **k):
    raise AssertionError("x_transformers was imported but should not be needed")


sys.modules.setdefault("x_transformers", block)

spec = importlib.util.spec_from_file_location(
    "pkgrot", PKG_DIR / "__init__.py", submodule_search_locations=[str(PKG_DIR)]
)
mod = importlib.util.module_from_spec(spec)
sys.modules["pkgrot"] = mod
spec.loader.exec_module(mod)
native = sys.modules["pkgrot.native"]
print("package imported with x_transformers blocked: OK")

import torch

with open(PKG_DIR / "tools" / "outputs" / "golden_rotary.pkl", "rb") as f:
    cases = pickle.load(f)

failures = 0
for case in cases:
    kind = case[0]
    if kind == "freqs":
        _, head_dim, seq_len, ref_freqs, ref_scale = case
        freqs, scale = native.RotaryEmbedding(head_dim).forward_from_seq_len(seq_len)
        ok_shape = freqs.shape == ref_freqs.shape
        ok_val = torch.equal(freqs, ref_freqs)
        print(f"freqs dim={head_dim} seq={seq_len}: shape={ok_shape} bit-exact={ok_val}")
        failures += 0 if (ok_shape and ok_val) else 1
    elif kind == "apply":
        _, head_dim, seq_len, dtype, q, ref_freqs, ref_out, ref_out2 = case
        out = native.apply_rotary_pos_emb(q, ref_freqs, 1.0)
        out2 = native.apply_rotary_pos_emb(q, ref_freqs)
        ok1 = torch.equal(out, ref_out)
        ok2 = torch.equal(out2, ref_out2)
        print(f"apply dim={head_dim} seq={seq_len} {str(dtype).split('.')[-1]:8s}: "
              f"scale-arg={ok1} default-scale={ok2}")
        failures += 0 if (ok1 and ok2) else 1
    elif kind == "inv_freq":
        ref = case[1]
        got = native.RotaryEmbedding(64).inv_freq
        ok = torch.equal(got, ref)
        print(f"inv_freq: bit-exact={ok}")
        failures += 0 if ok else 1

if failures:
    raise SystemExit(f"FAIL: {failures} case(s) not bit-exact")
print("VENDORED ROTARY PASS")
