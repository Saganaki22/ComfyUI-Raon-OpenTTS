# ComfyUI-Raon-OpenTTS

**[English](./README.md)** | **[中文](./README_zh.md)**

Native ComfyUI nodes for [KRAFTON Raon-OpenTTS](https://github.com/krafton-ai/Raon-OpenTTS) — open-weight, open-data zero-shot voice cloning (F5-TTS-style CFM/DiT + HiFi-GAN, 16 kHz English).

Supports the official checkpoints repacked to safetensors in three flavors, including an **INT8 ConvRot** build that executes through [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) quantized kernels (weights stay INT8-resident; no dequantize-to-bf16 at load).

**Pre-converted weights:** [drbaph/Raon-OpenTTS-comfyui](https://huggingface.co/drbaph/Raon-OpenTTS-comfyui) — the loader downloads from there automatically when files are missing.

**Upstream:** [KRAFTON/Raon-OpenTTS-1B](https://huggingface.co/KRAFTON/Raon-OpenTTS-1B) · [KRAFTON/Raon-OpenTTS-0.3B](https://huggingface.co/KRAFTON/Raon-OpenTTS-0.3B) · [arXiv:2605.20830](https://arxiv.org/abs/2605.20830)

| Build | 1B file | 0.3B file | Notes |
|---|---|---|---|
| fp32 | 4.17 GB | 1.35 GB | lossless extraction of the EMA weights (reference) |
| bf16 | 2.08 GB | 0.68 GB | recommended full-precision runtime build |
| int8-convrot | 1.41 GB | 0.50 GB | 65.9% (1B) / 54.6% (0.3B) of params quantized, ~1% mel deviation vs bf16 |

## Nodes

- **Raon OpenTTS Load Model** — picks a checkpoint filename from `ComfyUI/models/raon_opentts` (both model sizes' folders are searched; int8-convrot and bf16 are listed, fp32 is conversion reference only and is not listed), dtype (auto/bf16/fp32), device, attention backend (auto/sdpa/flash_attention/sageattention). Registers with ComfyUI/AIMDO memory management. With `download_if_missing` on, downloads **only the selected build** (plus its config/vocab and the HiFi-GAN vocoder) from [drbaph/Raon-OpenTTS-comfyui](https://huggingface.co/drbaph/Raon-OpenTTS-comfyui) — never the whole repo.
- **Raon OpenTTS Generate (Voice Clone)** — zero-shot cloning: `text` + `ref_audio` + `ref_text`, with official defaults (steps 32, cfg 2.0, sway -1.0, VAD-based duration estimation, target RMS 0.1, 150 ms cross-fade). Seed 42 default; 0 = random; incremented per text chunk. Live tqdm progress in the console plus the ComfyUI progress bar. Text chunking controls: `do_split` (on = split long text into chunks; off = always one chunk) and `max_chars` (0 = auto budget from the reference speech rate — `ref_bytes/ref_seconds x (22 - ref_seconds)` — any positive value forces a fixed, speaker-independent budget).
- **Raon Whisper Transcribe** — transcribes the reference clip for `ref_text` (reuses local Whisper copies under `ComfyUI/models/audio_encoders`; downloads only when missing).

## Model layout

```text
ComfyUI/models/raon_opentts/
  Raon-OpenTTS-1B/
    config.yaml
    vocab.txt
    Raon-OpenTTS-1B-fp32.safetensors
    Raon-OpenTTS-1B-bf16.safetensors
    Raon-OpenTTS-1B-int8-convrot.safetensors
  Raon-OpenTTS-0.3B/
    config.yaml
    vocab.txt
    Raon-OpenTTS-0.3B-fp32.safetensors
    Raon-OpenTTS-0.3B-bf16.safetensors
    Raon-OpenTTS-0.3B-int8-convrot.safetensors
  tts-hifigan-libritts-16kHz/generator.ckpt   (auto-downloaded)
```

## Notes

- Attention: `auto` uses the `flash_attn` package (FA2) when installed and compute is half precision, else torch SDPA. The official model defaults to SDPA; FA2 is numerically equivalent within bf16 rounding and helps on long generations. sageattention runs as an SDPA patch.
- The ODE time schedule is computed in fp32 and cast to the compute dtype (upstream computes it directly in the compute dtype; in bf16 that collapses adjacent timesteps, which torchdiffeq rejects).
- INT8 speed: on an RTX 5090 at these GEMM sizes the INT8 ConvRot path is currently about bf16-fp32 speed while using the least VRAM (1B: ~1.7 GB peak vs 2.4 GB bf16 / 4.5 GB fp32 during inference).

## Citation

```bibtex
@article{kim2026raonopentts,
  title     = {Raon-OpenTTS: Open Models and Data for Robust Text-to-Speech},
  author    = {Kim, Semin and Chung, Seungjun and Moon, Taehong and Lee, Sangheon and Ahn, Minyoung and Lee, Keon and Kim, Nam Soo and Cho, Jaewoong and Schmidt, Ludwig and Lee, Kangwook and Park, Dongmin},
  journal   = {arXiv preprint arXiv:2605.20830},
  year      = {2026},
  url       = {https://arxiv.org/abs/2605.20830}
}
```

## License

Code: MIT. The model weights (including the converted checkpoints on [drbaph/Raon-OpenTTS-comfyui](https://huggingface.co/drbaph/Raon-OpenTTS-comfyui)) are licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/), matching the upstream KRAFTON releases. Model code is vendored from [krafton-ai/Raon-OpenTTS](https://github.com/krafton-ai/Raon-OpenTTS) (Apache-2.0), itself based on [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS); HiFi-GAN vocoder adapted from speechbrain (Apache-2.0).
