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

## Reproducing the checkpoints

Requires the official training checkpoint (`model_520000.pt` / `model_225000.pt`), `config.yaml` and `vocab.txt` from the KRAFTON HF repos, plus a clone of [Comfy-Org/comfy-model-tools](https://github.com/Comfy-Org/comfy-model-tools) in `upstream/`.

```powershell
$PY = "C:\path\to\ComfyUI\venv\Scripts\python.exe"
$M  = "C:\path\to\ComfyUI\models\raon_opentts\Raon-OpenTTS-1B"

# 1) clean inference-only extraction (EMA weights, fp32)
& $PY tools/convert_raon_to_safetensors.py --ckpt "$M/model_520000.pt" `
    --config "$M/config.yaml" --vocab "$M/vocab.txt" `
    --out "$M/Raon-OpenTTS-1B-fp32.safetensors" --dtype keep

# 2) verify the extraction is bit-exact (all 448 tensors, torch.equal)
& $PY tools/validate_raon_safetensors.py --safetensors "$M/Raon-OpenTTS-1B-fp32.safetensors" `
    --config "$M/config.yaml" --source "$M/model_520000.pt"

# 3) bf16 runtime build (inv_freq stays fp32)
& $PY tools/convert_raon_to_safetensors.py --ckpt "$M/model_520000.pt" `
    --config "$M/config.yaml" --vocab "$M/vocab.txt" `
    --out "$M/Raon-OpenTTS-1B-bf16.safetensors" --dtype bf16

# 4) INT8 ConvRot (dry-run first, then generate + validate)
& $PY tools/quantize_raon_int8_convrot.py --src "$M/Raon-OpenTTS-1B-fp32.safetensors" --dry-run
& $PY tools/quantize_raon_int8_convrot.py --src "$M/Raon-OpenTTS-1B-fp32.safetensors"
& $PY tools/validate_raon_int8_convrot.py "$M/Raon-OpenTTS-1B-int8-convrot.safetensors" --expect-blocks 28

# 5) kernel unit tests + end-to-end comparison (from the ComfyUI root)
& $PY tools/test_int8_kernels.py "$M/Raon-OpenTTS-1B-int8-convrot.safetensors" "$M/Raon-OpenTTS-1B-fp32.safetensors"
& $PY tools/test_e2e_int8_vs_fp32.py 1B
```

The 0.3B uses `model_225000.pt` and `--expect-blocks 22`.

## Quantization recipe (V1, conservative)

Quantized: exactly the 6 repeated block GEMMs per DiT block — `attn.to_q/to_k/to_v/to_out.0`, `ff.ff.0.0` (up), `ff.ff.2` (down).

- 1B (dim 1408, heads 24, inner 1536, ff 5632): 168 layers = 112x GS64 (K=1408) + 56x GS256 (K=1536/5632). Note 1408 % 256 != 0, so GS64 is what makes the main transformer width ConvRot-eligible.
- 0.3B (dim 1024, inner 1024, ff 2048): 132 layers, all GS256.
- Left full precision: token embedding, ConvNeXt text blocks, per-block AdaLN modulation, time-embedding MLP, input projection, conv positional embedding, final AdaLN + proj_out, norms, `inv_freq`.
- Standard absmax scales (no MSE clip). Per-row fp32 scales, per-layer JSON markers (`int8_tensorwise` + `convrot` + `convrot_groupsize`).

## Vocab note (5559 vs 5555)

The released `vocab.txt` has 5559 tokens; both checkpoints embed 5555 (`embedding rows = 5556` including the filler). The model is therefore built with `text_num_embeds=5555` (checkpoint is ground truth), and the 4 overhanging tokens — the highest-codepoint entries in the sorted file (U+FDFA `ﷺ`, U+FDFB `ﷻ`, U+FFFD `�`, U+1F3B5 `🎵`) — are dropped from the runtime map (they tokenize like any other unknown character). Since the vocab file is sorted and the four extras are its four highest-codepoint entries, every usable character keeps an identical index; end-to-end Whisper transcription of generated audio confirms clean English output. Formally verifying the delta against the training corpus is impractical (the dataset is ~19 TB), so treat those four codepoints as unsupported.

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
