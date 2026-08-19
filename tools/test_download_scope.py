"""Verify download_if_missing fetches ONLY the selected build (+config/vocab/vocoder), not the whole repo.

Mocks huggingface_hub.snapshot_download, inspects allow_patterns, and simulates only
those files landing. Run from the ComfyUI root:
    venv/Scripts/python.exe custom_nodes/ComfyUI-Raon-OpenTTS/tools/test_download_scope.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent.parent
COMFY_ROOT = PKG_DIR.parent.parent
sys.path.insert(0, str(COMFY_ROOT))

spec = importlib.util.spec_from_file_location(
    "pkgdl", PKG_DIR / "__init__.py", submodule_search_locations=[str(PKG_DIR)]
)
mod = importlib.util.module_from_spec(spec)
sys.modules["pkgdl"] = mod
spec.loader.exec_module(mod)
loader = sys.modules["pkgdl.loader"]

import huggingface_hub

captured: list[list[str]] = []


def fake_snapshot_download(**kwargs):
    captured.append(list(kwargs["allow_patterns"]))
    # simulate only the allowed files landing on disk
    dest = Path(kwargs["local_dir"])
    for pattern in kwargs["allow_patterns"]:
        folder, _, fname = pattern.rpartition("/")
        if fname in ("config.yaml", "vocab.txt", "*.ckpt") or fname.endswith(".safetensors"):
            if fname == "*.ckpt":
                target = dest / folder / "generator.ckpt"
            else:
                target = dest / folder / fname
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"0")
    return "fake"


huggingface_hub.snapshot_download = fake_snapshot_download


def main() -> None:
    import tempfile

    # redirect the model folders to an empty temp base so nothing is found locally
    # and the download path is forced
    tmp = Path(tempfile.mkdtemp(prefix="raon_dl_test_"))
    original_model_dirs = loader.model_dirs
    loader.model_dirs = lambda: [tmp]
    loader.model_dir = lambda: (tmp.mkdir(parents=True, exist_ok=True) or tmp)

    # case 1: bare filename dropdown value, no local files at all -> only that build downloads
    captured.clear()
    try:
        got = loader.resolve_weights("Raon-OpenTTS-1B-int8-convrot.safetensors", download_if_missing=True)
        print("case1 resolved:", got[1])
    except FileNotFoundError as exc:
        raise SystemExit(f"case1 failed: {exc}")
    patterns = captured[-1]
    print("case1 patterns:", patterns)
    assert patterns == [
        "Raon-OpenTTS-1B/Raon-OpenTTS-1B-int8-convrot.safetensors",
        "Raon-OpenTTS-1B/config.yaml",
        "Raon-OpenTTS-1B/vocab.txt",
        "tts-hifigan-libritts-16kHz/*",
    ], patterns

    # case 2: legacy 'dir/file' form downloads only that file too
    captured.clear()
    try:
        loader.resolve_weights("Raon-OpenTTS-0.3B/Raon-OpenTTS-0.3B-bf16.safetensors", download_if_missing=True)
    except FileNotFoundError:
        pass  # patterns are what matter here
    patterns = captured[-1]
    print("case2 patterns:", patterns)
    assert patterns == [
        "Raon-OpenTTS-0.3B/Raon-OpenTTS-0.3B-bf16.safetensors",
        "Raon-OpenTTS-0.3B/config.yaml",
        "Raon-OpenTTS-0.3B/vocab.txt",
        "tts-hifigan-libritts-16kHz/*",
    ], patterns

    loader.model_dirs = original_model_dirs
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)
    print("DOWNLOAD SCOPE PASS")


if __name__ == "__main__":
    main()
