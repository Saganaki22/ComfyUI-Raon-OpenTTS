"""Transcribe generated test wavs with Whisper (intelligibility sanity check).

Run from the ComfyUI root:
    venv/Scripts/python.exe custom_nodes/ComfyUI-Raon-OpenTTS/tools/transcribe_outputs.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent.parent
COMFY_ROOT = PKG_DIR.parent.parent
sys.path.insert(0, str(COMFY_ROOT))

GEN_TEXTS = [
    "I don't really care what you call me. I've been a silent spectator, watching species evolve.",
    "The quick brown fox jumps over the lazy dog, while rain falls softly on the quiet city.",
]


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
    import_package()
    whisper = sys.modules["ComfyUI_Raon_OpenTTS.whisper"]

    wavs = sorted((PKG_DIR / "tools" / "outputs").glob("*.wav"))
    if not wavs:
        raise SystemExit("no wavs found; run test_e2e_int8_vs_fp32.py first")
    for wav in wavs:
        import torchaudio

        audio, sr = torchaudio.load(str(wav))
        audio_dict = {"waveform": audio.unsqueeze(0), "sample_rate": sr}
        text = whisper.transcribe_audio(
            audio_dict, "whisper-large-v3-turbo", "auto", "english", "transcribe", 30, True,
        )
        print(f"{wav.name}\n  -> {text}")
    for gi, gt in enumerate(GEN_TEXTS):
        print(f"expected gen{gi}: {gt}")


if __name__ == "__main__":
    main()
