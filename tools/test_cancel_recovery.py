"""Reproduce cancel-mid-ODE then new generation (different and same durations).

Cancels a sample() via an exception raised from the step callback (what ComfyUI's
interrupt does between steps), then runs new samples that must succeed with correct
text conditioning.

Run from the ComfyUI root:
    venv/Scripts/python.exe custom_nodes/ComfyUI-Raon-OpenTTS/tools/test_cancel_recovery.py [weights]
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent.parent
COMFY_ROOT = PKG_DIR.parent.parent
sys.path.insert(0, str(COMFY_ROOT))

spec = importlib.util.spec_from_file_location(
    "pkgcancel", PKG_DIR / "__init__.py", submodule_search_locations=[str(PKG_DIR)]
)
mod = importlib.util.module_from_spec(spec)
sys.modules["pkgcancel"] = mod
spec.loader.exec_module(mod)

import torch
import torchaudio

loader = sys.modules["pkgcancel.loader"]
native = sys.modules["pkgcancel.native"]


def main() -> None:
    weights = sys.argv[1] if len(sys.argv) > 1 else "Raon-OpenTTS-1B/Raon-OpenTTS-1B-bf16.safetensors"
    bundle = loader.load_raon_bundle(
        weights_selection=weights, dtype_name="auto", device_name="auto",
        attention="auto", download_if_missing=True,
    )
    model = bundle.model
    audio, sr = torchaudio.load(str(PKG_DIR / "example_workflows" / "basic_ref_en.wav"))
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    audio = native.normalize_peak(audio)
    if sr != 16000:
        audio = torchaudio.transforms.Resample(sr, 16000)(audio)
    cond = audio.to(bundle.device)
    ref_len = cond.shape[-1] // 256

    text_a = native.convert_char_to_pinyin(["Some call me nature, others call me mother nature. " + "The ocean is deep and full of mysteries. " * 20])
    text_b = native.convert_char_to_pinyin(["Some call me nature, others call me mother nature. " + "A short one. "])

    # 1) cancel a long generation at step 5 of 32 (mimics ComfyUI interrupt)
    class Cancel(Exception):
        pass

    def cancel_at_5(step):
        if step >= 5:
            raise Cancel()

    try:
        with torch.inference_mode():
            model.sample(cond=cond, text=text_a, duration=ref_len + 1400, steps=32,
                         cfg_strength=2.0, sway_sampling_coef=-1.0, seed=1,
                         step_callback=cancel_at_5)
        raise SystemExit("FAIL: expected cancellation")
    except Cancel:
        print("cancelled mid-ODE at step 5; stale cache now:",
              model.transformer.text_cond is not None)

    # 2) new run, SHORTER duration (the reported crash: 1664 vs 1360)
    with torch.inference_mode():
        out, _ = model.sample(cond=cond, text=text_b, duration=ref_len + 300, steps=4,
                              cfg_strength=2.0, sway_sampling_coef=-1.0, seed=2)
    assert torch.isfinite(out).all()
    print(f"shorter-duration run after cancel: OK ({tuple(out.shape)})")

    # 3) new run, same duration as the cancelled one, different text
    with torch.inference_mode():
        out, _ = model.sample(cond=cond, text=text_b, duration=ref_len + 1400, steps=4,
                              cfg_strength=2.0, sway_sampling_coef=-1.0, seed=3)
    assert torch.isfinite(out).all()
    print(f"same-duration run after cancel: OK ({tuple(out.shape)})")

    assert model.transformer.text_cond is None and model.transformer.text_uncond is None
    print("CANCEL RECOVERY PASS")


if __name__ == "__main__":
    main()
