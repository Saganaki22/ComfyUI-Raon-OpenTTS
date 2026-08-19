"""Capture the installed x_transformers rotary behavior as a golden reference.

Dumps freqs/rotary outputs for several shapes/dtypes/seq lens to golden files so a
vendored replacement can be verified bit-exact against them.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import torch
from x_transformers.x_transformers import RotaryEmbedding, apply_rotary_pos_emb

OUT = Path(__file__).resolve().parent.parent / "tests" / "golden_rotary.pkl"

cases = []
for head_dim in (64,):
    rot = RotaryEmbedding(head_dim)
    for seq_len in (1, 813, 4096):
        freqs, scale = rot.forward_from_seq_len(seq_len)
        cases.append(("freqs", head_dim, seq_len, freqs.clone(), scale))
for head_dim in (64,):
    rot = RotaryEmbedding(head_dim)
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        for seq_len in (37, 512):
            torch.manual_seed(7)
            q = torch.randn(2, 24, seq_len, head_dim, dtype=dtype)
            freqs, scale = rot.forward_from_seq_len(seq_len)
            out = apply_rotary_pos_emb(q, freqs, 1.0)
            out2 = apply_rotary_pos_emb(q.clone(), freqs)  # scale default
            cases.append(("apply", head_dim, seq_len, dtype, q.clone(), freqs.clone(), out.clone(), out2.clone()))

# also inv_freq formula check vs checkpoint values
inv_freq = RotaryEmbedding(64).inv_freq.clone()
cases.append(("inv_freq", inv_freq))

with open(OUT, "wb") as f:
    pickle.dump(cases, f)
print(f"golden saved: {OUT} ({len(cases)} cases)")
print("freqs shape for seq 813:", cases[1][3].shape, "scale:", cases[1][4])
print("inv_freq[:4]:", inv_freq[:4].tolist())
