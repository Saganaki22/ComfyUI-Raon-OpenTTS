import sys
from huggingface_hub import hf_hub_download

JOBS = [
    ("KRAFTON/Raon-OpenTTS-1B", "model_520000.pt", r"C:\Users\drbaph\Documents\ComfyUI\models\raon_opentts\Raon-OpenTTS-1B"),
    ("KRAFTON/Raon-OpenTTS-1B", "config.yaml", r"C:\Users\drbaph\Documents\ComfyUI\models\raon_opentts\Raon-OpenTTS-1B"),
    ("KRAFTON/Raon-OpenTTS-1B", "vocab.txt", r"C:\Users\drbaph\Documents\ComfyUI\models\raon_opentts\Raon-OpenTTS-1B"),
    ("KRAFTON/Raon-OpenTTS-0.3B", "model_225000.pt", r"C:\Users\drbaph\Documents\ComfyUI\models\raon_opentts\Raon-OpenTTS-0.3B"),
    ("KRAFTON/Raon-OpenTTS-0.3B", "config.yaml", r"C:\Users\drbaph\Documents\ComfyUI\models\raon_opentts\Raon-OpenTTS-0.3B"),
    ("KRAFTON/Raon-OpenTTS-0.3B", "vocab.txt", r"C:\Users\drbaph\Documents\ComfyUI\models\raon_opentts\Raon-OpenTTS-0.3B"),
    ("speechbrain/tts-hifigan-libritts-16kHz", "generator.ckpt", r"C:\Users\drbaph\Documents\ComfyUI\models\raon_opentts\tts-hifigan-libritts-16kHz"),
]

for repo, fname, dest in JOBS:
    print(f"[download] {repo}/{fname} -> {dest}", flush=True)
    path = hf_hub_download(repo_id=repo, filename=fname, local_dir=dest)
    print(f"[done] {path}", flush=True)

print("ALL DOWNLOADS COMPLETE", flush=True)
