"""Node-level test: call the actual node classes the way ComfyUI execution does.

Covers RaonOpenTTSLoadModel.load -> RaonOpenTTSGenerate.generate with a real
ComfyUI AUDIO dict (the path the UI exercises), for a given weights file.

Run from the ComfyUI root:
    venv/Scripts/python.exe custom_nodes/ComfyUI-Raon-OpenTTS/tools/test_nodes_e2e.py ^
        "Raon-OpenTTS-1B/Raon-OpenTTS-1B-bf16.safetensors"
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent.parent
COMFY_ROOT = PKG_DIR.parent.parent
sys.path.insert(0, str(COMFY_ROOT))


def import_package():
    spec = importlib.util.spec_from_file_location(
        "ComfyUI_Raon_OpenTTS_nodes", PKG_DIR / "__init__.py",
        submodule_search_locations=[str(PKG_DIR)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ComfyUI_Raon_OpenTTS_nodes"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    weights = sys.argv[1] if len(sys.argv) > 1 else "Raon-OpenTTS-1B/Raon-OpenTTS-1B-bf16.safetensors"
    use_aimdo = len(sys.argv) > 2 and "aimdo" in sys.argv
    block_xtransformers = "noxtrans" in sys.argv
    if block_xtransformers:
        # prove the vendored rotary runs without x_transformers installed
        import types

        stub = types.ModuleType("x_transformers")
        stub.__path__ = []
        sys.modules["x_transformers"] = stub
        print("x_transformers blocked for this test")
    if use_aimdo:
        import comfy_aimdo.control
        import comfy.memory_management

        comfy_aimdo.control.init(simple_vram_headroom=None, nvml_pressure=False)
        comfy.memory_management.aimdo_enabled = True
        print("AIMDO DynamicVRAM force-enabled for this test")
    import_package()
    nodes = sys.modules["ComfyUI_Raon_OpenTTS_nodes.nodes"]

    import torchaudio

    (bundle,) = nodes.RaonOpenTTSLoadModel().load(
        weights=weights, dtype="auto", device="auto", attention="auto", download_if_missing=True,
    )
    print(f"loaded {bundle.weights_file.name} on {bundle.device} ({bundle.dtype_name}, {bundle.attention})")

    audio, sr = torchaudio.load(str(PKG_DIR / "example_workflows" / "basic_ref_en.wav"))
    ref_audio = {"waveform": audio.unsqueeze(0), "sample_rate": sr}  # ComfyUI AUDIO: [b, c, n]

    (out,) = nodes.RaonOpenTTSGenerate().generate(
        raon_model=bundle,
        text="This is a node level test of Raon OpenTTS inside ComfyUI.",
        ref_audio=ref_audio,
        ref_text="Some call me nature, others call me mother nature.",
        steps=8,  # fast smoke; UI default is 32
        cfg_strength=2.0,
        sway_sampling_coef=-1.0,
        speed=1.0,
        seed=42,
        fix_duration_seconds=0.0,
        target_rms=0.1,
        use_vad_duration=True,
        cross_fade_ms=150.0,
        do_split=True,
        max_chars=0,
    )
    wave = out["waveform"]
    import torch

    print(f"output AUDIO: waveform {tuple(wave.shape)} @ {out['sample_rate']} Hz, "
          f"finite={bool(torch.isfinite(wave).all())}, peak={wave.abs().max().item():.3f}")
    assert wave.ndim == 3 and wave.shape[0] == 1 and wave.shape[1] == 1
    assert torch.isfinite(wave).all()
    assert wave.shape[-1] > 16000  # at least ~1s of audio

    import soundfile as sf

    out_path = PKG_DIR / "tools" / "outputs" / "node_level_test.wav"
    sf.write(str(out_path), wave[0, 0].cpu().numpy(), out["sample_rate"])
    print(f"saved {out_path}")
    print("NODE E2E PASS")


if __name__ == "__main__":
    main()
