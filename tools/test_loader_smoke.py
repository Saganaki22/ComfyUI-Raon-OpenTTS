"""Smoke test: import the node package like ComfyUI does and load a bundle.

Run from the ComfyUI root so comfy/folder_paths resolve:
    venv/Scripts/python.exe custom_nodes/ComfyUI-Raon-OpenTTS/tools/test_loader_smoke.py [weights]

Example:
    venv/Scripts/python.exe custom_nodes/ComfyUI-Raon-OpenTTS/tools/test_loader_smoke.py \
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
        "ComfyUI_Raon_OpenTTS", PKG_DIR / "__init__.py",
        submodule_search_locations=[str(PKG_DIR)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ComfyUI_Raon_OpenTTS"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    weights = sys.argv[1] if len(sys.argv) > 1 else "Raon-OpenTTS-1B/Raon-OpenTTS-1B-bf16.safetensors"
    device_name = sys.argv[2] if len(sys.argv) > 2 else "auto"
    mod = import_package()
    loader = sys.modules["ComfyUI_Raon_OpenTTS.loader"]
    nodes = sys.modules["ComfyUI_Raon_OpenTTS.nodes"]

    print("NODE_CLASS_MAPPINGS:", list(mod.NODE_CLASS_MAPPINGS))
    print("available_models:", loader.available_models())
    print("weight choices:", nodes._weight_choices())

    model_name, _, weights_name = weights.partition("/")
    if not weights_name:
        weights_name = model_name  # accept a bare filename too
    import torch

    torch.cuda.reset_peak_memory_stats()
    bundle = loader.load_raon_bundle(
        weights_selection=weights,
        dtype_name="auto",
        device_name=device_name,
        attention="auto",
        download_if_missing=True,
    )
    print(f"loaded: {bundle.model_name} weights={bundle.weights_file.name} "
          f"device={bundle.device} dtype={bundle.dtype_name} attention={bundle.attention} "
          f"quantized={bundle.quantized}")
    n_params = sum(p.numel() for p in bundle.model.parameters())
    print(f"model params: {n_params/1e6:.1f}M, text_num_embeds={bundle.text_num_embeds}")
    if bundle.quantized:
        int8 = sys.modules["ComfyUI_Raon_OpenTTS.int8"]
        qp, qc = int8.quantized_parameter_count(bundle.model)
        gs = sorted({m.convrot_groupsize for m in bundle.model.modules() if isinstance(m, int8.ConvRotInt8Linear)})
        dtypes = {str(p.dtype) for p in bundle.model.parameters()}
        print(f"int8: {qc} layers / {qp/1e6:.1f}M params / gs={gs}; param dtypes={dtypes}")
    print(f"peak VRAM during load: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    if len(sys.argv) > 3 and sys.argv[3] == "gen":
        native = sys.modules["ComfyUI_Raon_OpenTTS.native"]
        import torchaudio

        audio, sr = torchaudio.load(str(PKG_DIR / "tests" / "assets" / "basic_ref_en.wav"))
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        audio = native.normalize_peak(audio)
        if sr != 16000:
            audio = torchaudio.transforms.Resample(sr, 16000)(audio)
        cond = audio.to(bundle.device)
        ref_len = cond.shape[-1] // 256
        text = native.convert_char_to_pinyin(
            ["Some call me nature, others call me mother nature. Hello from the device dropdown test."])
        with torch.inference_mode():
            out, _ = bundle.model.sample(
                cond=cond, text=text, duration=ref_len + 120, steps=4,
                cfg_strength=2.0, sway_sampling_coef=-1.0, seed=42,
            )
        mel = out.to(torch.float32)[:, ref_len:, :]
        with torch.inference_mode():
            wave = bundle.vocoder(mel.permute(0, 2, 1))
        finite = bool(torch.isfinite(wave).all())
        print(f"gen on {bundle.device}: mel {tuple(mel.shape)}, wave {tuple(wave.shape)}, finite={finite}")
        if not finite:
            raise SystemExit("FAIL: non-finite audio")
    print("LOADER SMOKE PASS")


if __name__ == "__main__":
    main()
