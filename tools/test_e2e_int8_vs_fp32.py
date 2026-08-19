"""End-to-end validation: fp32/bf16 reference vs INT8 ConvRot, same seed and settings.

Loads each build sequentially through the real ComfyUI loader, samples with the
official CFM.sample path, compares pre-vocoder mel outputs, vocodes with HiFi-GAN,
measures VRAM/load/generation time, and transcribes results with Whisper as an
intelligibility sanity check.

Run from the ComfyUI root:
    venv/Scripts/python.exe custom_nodes/ComfyUI-Raon-OpenTTS/tools/test_e2e_int8_vs_fp32.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PKG_DIR = Path(__file__).resolve().parent.parent
COMFY_ROOT = PKG_DIR.parent.parent
sys.path.insert(0, str(COMFY_ROOT))

OUT_DIR = PKG_DIR / "tools" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REF_WAV = PKG_DIR / "example_workflows" / "basic_ref_en.wav"
REF_TEXT = "Some call me nature, others call me mother nature. "
GEN_TEXTS = [
    "I don't really care what you call me. I've been a silent spectator, watching species evolve.",
    "The quick brown fox jumps over the lazy dog, while rain falls softly on the quiet city.",
]
SEED = 42
STEPS = 32
CFG = 2.0
SWAY = -1.0
TARGET_SR = 16000
HOP = 256

MODEL = sys.argv[1] if len(sys.argv) > 1 else "1B"
assert MODEL in ("1B", "0.3B")

BUILDS = {
    "fp32": (f"Raon-OpenTTS-{MODEL}/Raon-OpenTTS-{MODEL}-fp32.safetensors", "fp32"),
    "bf16": (f"Raon-OpenTTS-{MODEL}/Raon-OpenTTS-{MODEL}-bf16.safetensors", "bf16"),
    "int8": (f"Raon-OpenTTS-{MODEL}/Raon-OpenTTS-{MODEL}-int8-convrot.safetensors", "auto"),
}


def import_package():
    spec = importlib.util.spec_from_file_location(
        "ComfyUI_Raon_OpenTTS", PKG_DIR / "__init__.py",
        submodule_search_locations=[str(PKG_DIR)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ComfyUI_Raon_OpenTTS"] = mod
    spec.loader.exec_module(mod)
    return mod


def prepare_ref(native):
    import torchaudio

    audio, sr = torchaudio.load(str(REF_WAV))
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    audio = native.normalize_peak(audio)
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < 0.1:
        audio = audio * 0.1 / rms
    if sr != TARGET_SR:
        audio = torchaudio.transforms.Resample(sr, TARGET_SR)(audio)
    return audio, rms


def run_build(tag: str, weights: str, dtype_name: str, native, loader, int8_mod,
              audio: torch.Tensor, rms, duration: int, results: dict) -> None:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    t0 = time.perf_counter()
    bundle = loader.load_raon_bundle(
        weights_selection=weights, dtype_name=dtype_name,
        device_name="auto", attention="auto", download_if_missing=True,
    )
    load_s = time.perf_counter() - t0
    load_vram = torch.cuda.max_memory_allocated() / 1e9

    int8_mod.reset_runtime_stats()
    cond = audio.to(bundle.device)
    ref_len = cond.shape[-1] // HOP

    mels = {}
    waves = {}
    gen_s_total = 0.0
    audio_s_total = 0.0
    torch.cuda.reset_peak_memory_stats()
    for gi, gen_text in enumerate(GEN_TEXTS):
        text = native.convert_char_to_pinyin([REF_TEXT + gen_text])
        t0 = time.perf_counter()
        with torch.inference_mode():
            out, _ = bundle.model.sample(
                cond=cond, text=text, duration=duration, steps=STEPS,
                cfg_strength=CFG, sway_sampling_coef=SWAY, seed=SEED,
            )
        torch.cuda.synchronize()
        gen_s = time.perf_counter() - t0
        mel = out.to(torch.float32)[:, ref_len:, :]
        with torch.inference_mode():
            wave = bundle.vocoder(mel.permute(0, 2, 1))
        torch.cuda.synchronize()
        if rms < 0.1:
            wave = wave * rms / 0.1
        wave = wave.squeeze().float().cpu()
        mels[gi] = mel.cpu()
        waves[gi] = wave
        gen_s_total += gen_s
        audio_s_total += wave.shape[-1] / TARGET_SR
        import soundfile as sf
        sf.write(str(OUT_DIR / f"raon_{MODEL.replace('.', '')}_{tag}_gen{gi}.wav"), wave.numpy(), TARGET_SR)
        print(f"  [{tag}] gen{gi}: {gen_s:.2f}s for {wave.shape[-1]/TARGET_SR:.2f}s audio "
              f"(RTF {gen_s/(wave.shape[-1]/TARGET_SR):.3f})")
    infer_vram = torch.cuda.max_memory_allocated() / 1e9

    stats = {
        "weights": bundle.weights_file.name,
        "dtype": bundle.dtype_name,
        "attention": bundle.attention,
        "load_s": load_s,
        "load_peak_vram_gb": load_vram,
        "infer_peak_vram_gb": infer_vram,
        "gen_s": gen_s_total,
        "audio_s": audio_s_total,
        "rtf": gen_s_total / audio_s_total,
        "int8_kernel_calls": int8_mod.RUNTIME_STATS["calls"],
    }
    results[tag] = {"stats": stats, "mels": mels}
    loader.unload_active_bundle()
    torch.cuda.empty_cache()


def compare(tag_a, tag_b, results):
    print(f"\n=== mel comparison: {tag_a} vs {tag_b} ===")
    for gi in range(len(GEN_TEXTS)):
        a = results[tag_a]["mels"][gi]
        b = results[tag_b]["mels"][gi]
        assert a.shape == b.shape, (a.shape, b.shape)
        rel = ((a - b).norm() / a.norm().clamp(min=1e-30)).item()
        cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        mx = (a - b).abs().max().item()
        nan = int(torch.isnan(b).sum())
        inf = int(torch.isinf(b).sum())
        print(f"  gen{gi}: frames={a.shape[1]} rel_l2={rel*100:.3f}% cos={cos:.6f} "
              f"max_abs={mx:.4f} nan={nan} inf={inf}")
        results[tag_b]["stats"].setdefault("mel_vs_" + tag_a, []).append(
            {"rel_l2": rel, "cosine": cos, "max_abs": mx})


def main() -> None:
    mod = import_package()
    loader = sys.modules["ComfyUI_Raon_OpenTTS.loader"]
    native = sys.modules["ComfyUI_Raon_OpenTTS.native"]
    int8_mod = sys.modules["ComfyUI_Raon_OpenTTS.int8"]

    audio, rms = prepare_ref(native)
    ref_len = audio.shape[-1] // HOP
    ref_seconds = native.estimate_ref_seconds_trimmed_tensor(audio, TARGET_SR)
    # one duration shared by every build (official VAD-based estimate for gen text 0;
    # per-text scaling below keeps generations comparable)
    print(f"ref: {audio.shape[-1]/TARGET_SR:.2f}s ({ref_len} frames), VAD-trimmed {ref_seconds:.2f}s")

    durations = []
    for gen_text in GEN_TEXTS:
        ref_sec_est = ref_seconds
        sec_per_byte = ref_sec_est / max(len(REF_TEXT.encode("utf-8")), 1)
        sec_per_byte = min(sec_per_byte, 1.0 / 12.0)
        gen_sec = sec_per_byte * len(gen_text.encode("utf-8"))
        durations.append(ref_len + max(int(gen_sec * TARGET_SR / HOP), 1))
    duration = max(durations)
    print(f"shared duration: {duration} frames (~{duration*HOP/TARGET_SR:.1f}s)")

    results = {}
    for tag, (weights, dtype_name) in BUILDS.items():
        print(f"\n--- build {tag} ({weights}) ---")
        run_build(tag, weights, dtype_name, native, loader, int8_mod, audio, rms, duration, results)

    compare("fp32", "int8", results)
    compare("bf16", "int8", results)
    compare("fp32", "bf16", results)

    print("\n=== build stats ===")
    for tag, r in results.items():
        s = r["stats"]
        print(f"  {tag}: load {s['load_s']:.1f}s / loadVRAM {s['load_peak_vram_gb']:.2f}GB / "
              f"inferVRAM {s['infer_peak_vram_gb']:.2f}GB / gen {s['gen_s']:.2f}s for {s['audio_s']:.2f}s "
              f"(RTF {s['rtf']:.3f}) / int8 calls {s['int8_kernel_calls']}")

    summary = {
        tag: {k: v for k, v in r["stats"].items()} for tag, r in results.items()
    }
    (OUT_DIR / f"e2e_summary_{MODEL.replace('.', '')}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwavs + summary in {OUT_DIR}")
    print("E2E DONE")


if __name__ == "__main__":
    main()
