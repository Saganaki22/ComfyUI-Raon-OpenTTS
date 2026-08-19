"""Create the drbaph/Raon-OpenTTS-comfyui HF repo and upload the converted models.

Uploads, for each model size: config.yaml, vocab.txt, the fp32/bf16/int8-convrot
safetensors and the int8 verify report, plus the HiFi-GAN vocoder and the model
card (huggingface/README.md). The original .pt training checkpoints are NOT uploaded.

Requires `huggingface-cli login` (or HF_TOKEN) with write access to the drbaph account.

    python tools/upload_to_hf.py [--repo drbaph/Raon-OpenTTS-comfyui] [--dry-run]
"""

from __future__ import annotations

import argparse
from pathlib import Path

MODELS_ROOT = Path(r"C:\Users\drbaph\Documents\ComfyUI\models\raon_opentts")
CARD = Path(__file__).resolve().parent.parent / "huggingface" / "README.md"
DEFAULT_REPO = "drbaph/Raon-OpenTTS-comfyui"

PER_MODEL = [
    "config.yaml",
    "vocab.txt",
    "{name}-fp32.safetensors",
    "{name}-bf16.safetensors",
    "{name}-int8-convrot.safetensors",
    "{name}-int8-convrot-verify.json",
]
MODEL_DIRS = ["Raon-OpenTTS-1B", "Raon-OpenTTS-0.3B"]
VOCODER = ("tts-hifigan-libritts-16kHz", "generator.ckpt")


def collect_uploads() -> list[tuple[Path, str]]:
    uploads = []
    for model_dir in MODEL_DIRS:
        for pattern in PER_MODEL:
            path = MODELS_ROOT / model_dir / pattern.format(name=model_dir)
            if not path.is_file():
                raise FileNotFoundError(f"expected upload file missing: {path}")
            uploads.append((path, f"{model_dir}/{path.name}"))
    vocoder_path = MODELS_ROOT / VOCODER[0] / VOCODER[1]
    if vocoder_path.is_file():
        uploads.append((vocoder_path, f"{VOCODER[0]}/{VOCODER[1]}"))
    if CARD.is_file():
        uploads.append((CARD, "README.md"))
    else:
        print(f"note: model card not found at {CARD}; skipping (edit the card on the HF repo instead)")
    return uploads


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    uploads = collect_uploads()
    total = sum(p.stat().st_size for p, _ in uploads)
    print(f"repo: {args.repo}  files: {len(uploads)}  total: {total/1e9:.2f} GB")
    for path, arcname in uploads:
        print(f"  {arcname:70s} {path.stat().st_size/1e9:7.2f} GB")
    if args.dry_run:
        print("[dry-run] nothing uploaded.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    for path, arcname in uploads:
        print(f"uploading {arcname} ...", flush=True)
        api.upload_file(path_or_fileobj=str(path), path_in_repo=arcname, repo_id=args.repo)
    print(f"DONE -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
