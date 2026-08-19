"""Convert a Raon-OpenTTS training checkpoint (.pt) to a clean inference-only safetensors.

The official checkpoint (model_520000.pt / model_225000.pt) contains:
    model_state_dict        raw online CFM weights (fp32)
    ema_model_state_dict    EMA module state: "ema_model.<cfm key>" + "initted"/"step"
    optimizer_state_dict    AdamW moments (training only)
    scheduler_state_dict    (training only)
    update                  (training counter)

The official inference path (f5_tts.infer.utils_infer.load_checkpoint, use_ema=True)
loads the EMA weights with the "ema_model." prefix stripped. This script verifies that
candidate against a freshly instantiated official model (exact key + shape match) and
saves only those inference tensors.

Usage:
    python tools/convert_raon_to_safetensors.py ^
        --ckpt  models/raon_opentts/Raon-OpenTTS-1B/model_520000.pt ^
        --config models/raon_opentts/Raon-OpenTTS-1B/config.yaml ^
        --vocab models/raon_opentts/Raon-OpenTTS-1B/vocab.txt ^
        --out   models/raon_opentts/Raon-OpenTTS-1B/Raon-OpenTTS-1B-model_520000-fp32.safetensors
    # add --dtype bf16 for the bf16 build
"""

from __future__ import annotations

import argparse
import collections
import gc
import sys
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file

UPSTREAM_SRC = Path(__file__).resolve().parent.parent / "upstream" / "Raon-OpenTTS" / "src"
sys.path.insert(0, str(UPSTREAM_SRC))

# f5_tts.model.__init__ pulls in Trainer (wandb etc.) which we do not need for
# inference-shaped model construction; stub the training-only imports.
import importlib.machinery

for _stub in ("wandb",):
    if _stub not in sys.modules:
        try:
            __import__(_stub)
        except ImportError:
            _mod = type(sys)(_stub)
            _mod.__spec__ = importlib.machinery.ModuleSpec(_stub, loader=None)
            sys.modules[_stub] = _mod

# Keys the official inference loader explicitly drops.
EMA_SKIP = {"initted", "step", "update"}
STALE_KEYS = {
    "mel_spec.mel_stft.mel_scale.fb",
    "mel_spec.mel_stft.spectrogram.window",
}


def read_vocab_size(vocab_path: Path) -> int:
    """Mirror f5_tts.model.utils.get_tokenizer(..., 'custom'): one token per line."""
    vocab_char_map = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for i, char in enumerate(f):
            vocab_char_map[char[:-1]] = i
    return len(vocab_char_map)


def build_reference_model(config_path: Path, vocab_size: int) -> torch.nn.Module:
    """Instantiate the official CFM(DiT) exactly as f5_tts.infer does."""
    from f5_tts.model import CFM, DiT

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    arch = dict(cfg["model"]["arch"])
    mel = dict(cfg["model"]["mel_spec"])
    model = CFM(
        transformer=DiT(
            **arch,
            text_num_embeds=vocab_size,
            mel_dim=mel["n_mel_channels"],
        ),
        mel_spec_kwargs=dict(
            n_fft=mel["n_fft"],
            hop_length=mel["hop_length"],
            win_length=mel["win_length"],
            n_mel_channels=mel["n_mel_channels"],
            target_sample_rate=mel["target_sample_rate"],
            mel_spec_type=mel["mel_spec_type"],
        ),
        odeint_kwargs=dict(method="euler"),
        vocab_char_map=None,
    )
    return model


def describe_dict(name: str, sd: dict) -> None:
    n_tensors = sum(isinstance(v, torch.Tensor) for v in sd.values())
    dtypes = collections.Counter(str(v.dtype) for v in sd.values() if isinstance(v, torch.Tensor))
    nbytes = sum(v.numel() * v.element_size() for v in sd.values() if isinstance(v, torch.Tensor))
    prefixes = collections.Counter(
        k.split(".")[0] for k, v in sd.items() if isinstance(v, torch.Tensor)
    )
    print(f"  [{name}] entries={len(sd)} tensors={n_tensors} bytes={nbytes/1e9:.2f}GB")
    print(f"    dtypes: {dict(dtypes)}")
    print(f"    top prefixes: {dict(prefixes.most_common(6))}")


def candidate_state_dicts(ckpt: dict) -> dict[str, dict[str, torch.Tensor]]:
    """All plausible inference state dicts in the checkpoint, normalized form."""
    candidates = {}
    if isinstance(ckpt.get("ema_model_state_dict"), dict):
        ema = {
            k.replace("ema_model.", ""): v
            for k, v in ckpt["ema_model_state_dict"].items()
            if k not in EMA_SKIP and isinstance(v, torch.Tensor)
        }
        for stale in STALE_KEYS:
            ema.pop(stale, None)
        candidates["ema_model_state_dict(stripped)"] = ema
    if isinstance(ckpt.get("model_state_dict"), dict):
        raw = {k: v for k, v in ckpt["model_state_dict"].items() if isinstance(v, torch.Tensor)}
        for stale in STALE_KEYS:
            raw.pop(stale, None)
        candidates["model_state_dict(raw)"] = raw
    flat = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
    if flat:
        candidates["toplevel-flat"] = flat
    return candidates


def score_candidate(ref_sd: dict[str, torch.Tensor], cand: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    """(shape-matching keys, missing keys, unexpected keys) against the reference model."""
    match = sum(1 for k, v in ref_sd.items() if k in cand and cand[k].shape == v.shape)
    missing = sum(1 for k in ref_sd if k not in cand)
    unexpected = sum(1 for k in cand if k not in ref_sd)
    return match, missing, unexpected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", choices=["keep", "bf16"], default="keep",
                    help="'keep' preserves the source dtype (clean reference); 'bf16' casts floats to bfloat16.")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    out_path = Path(args.out)

    print(f"[1/5] loading {ckpt_path} ({ckpt_path.stat().st_size/1e9:.2f} GB) ...", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    print(f"  top-level keys: {list(ckpt.keys())}")
    for key, value in ckpt.items():
        if isinstance(value, dict):
            describe_dict(key, value)
        else:
            print(f"  [{key}] {type(value).__name__} = {value if not isinstance(value, torch.Tensor) else value.shape}")

    # The checkpoint embedding rows are ground truth for text_num_embeds. The shipped
    # vocab.txt can carry a few extra sorted-tail tokens the model was not trained with
    # (1B/0.3B: 5559 tokens vs 5555 embedding rows); the runtime tokenizer clamps them.
    ema_key = "ema_model.transformer.text_embed.text_embed.weight"
    raw_key = "transformer.text_embed.text_embed.weight"
    if ema_key in ckpt.get("ema_model_state_dict", {}):
        ckpt_num_embeds = ckpt["ema_model_state_dict"][ema_key].shape[0] - 1
    else:
        ckpt_num_embeds = ckpt["model_state_dict"][raw_key].shape[0] - 1

    print("[2/5] instantiating official reference model ...", flush=True)
    vocab_size = read_vocab_size(Path(args.vocab))
    print(f"  vocab.txt tokens = {vocab_size}, checkpoint text_num_embeds = {ckpt_num_embeds}")
    if vocab_size != ckpt_num_embeds:
        print(f"  note: vocab.txt has {vocab_size - ckpt_num_embeds} extra tail token(s); "
              f"building the reference with text_num_embeds={ckpt_num_embeds} (checkpoint is ground truth)")
    model = build_reference_model(Path(args.config), ckpt_num_embeds)
    ref_sd = model.state_dict()
    n_params = sum(v.numel() for v in ref_sd.values())
    print(f"  reference model: {len(ref_sd)} state tensors, {n_params/1e6:.1f}M params")

    print("[3/5] scoring checkpoint candidates against the reference ...", flush=True)
    candidates = candidate_state_dicts(ckpt)
    best_name, best_score = None, None
    for name, cand in candidates.items():
        match, missing, unexpected = score_candidate(ref_sd, cand)
        print(f"  {name}: keys={len(cand)} shape-match={match}/{len(ref_sd)} missing={missing} unexpected={unexpected}")
        score = (match, -missing, -unexpected)
        if best_score is None or score > best_score:
            best_name, best_score = name, score
    print(f"  -> selected: {best_name}")
    match, missing, unexpected = score_candidate(ref_sd, candidates[best_name])
    if missing or unexpected or match != len(ref_sd):
        missing_keys = [k for k in ref_sd if k not in candidates[best_name]][:10]
        unexpected_keys = [k for k in candidates[best_name] if k not in ref_sd][:10]
        shape_bad = [k for k, v in ref_sd.items()
                     if k in candidates[best_name] and candidates[best_name][k].shape != v.shape][:10]
        raise SystemExit(
            f"FATAL: candidate does not reproduce the inference model exactly "
            f"(missing={missing} {missing_keys}, unexpected={unexpected} {unexpected_keys}, "
            f"shape mismatches={shape_bad})"
        )

    print("[4/5] extracting inference state dict ...", flush=True)
    sd = candidates[best_name]
    del ckpt, candidates
    gc.collect()
    out = {}
    cast = 0
    for key in ref_sd:  # reference order, exact reference key set
        tensor = sd[key].detach().cpu().contiguous()
        # inv_freq stays fp32 even in the bf16 build: bf16 rounding of the rotary base
        # frequencies corrupts the phase over long sequences (matches/exceeds fp16 official
        # behaviour otherwise).
        if args.dtype == "bf16" and tensor.is_floating_point() and not key.endswith("inv_freq"):
            tensor = tensor.to(torch.bfloat16)
            cast += 1
        out[key] = tensor
    del sd
    gc.collect()
    total = sum(t.numel() for t in out.values())
    nbytes = sum(t.numel() * t.element_size() for t in out.values())
    print(f"  {len(out)} tensors, {total/1e6:.1f}M params, {nbytes/1e9:.2f} GB "
          f"({args.dtype}; cast {cast} tensors)")

    print(f"[5/5] saving {out_path} ...", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_path), metadata={"format": "pt", "source": ckpt_path.name,
                                            "state_dict": best_name, "dtype": args.dtype})
    print(f"DONE -> {out_path} ({out_path.stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
