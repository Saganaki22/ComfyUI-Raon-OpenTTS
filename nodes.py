"""ComfyUI node definitions for Raon-OpenTTS."""

from __future__ import annotations

import logging

import torch

from . import native
from .loader import (
    ATTENTION_OPTIONS,
    DEVICE_OPTIONS,
    DTYPE_OPTIONS,
    RaonBundle,
    available_models,
    available_weights,
    load_raon_bundle,
    resume_bundle_to_device,
)
from .whisper import RaonWhisperTranscribe

logger = logging.getLogger("RaonOpenTTS")

try:
    from comfy.utils import ProgressBar
except Exception:
    ProgressBar = None

CATEGORY = "RaonOpenTTS"
TARGET_SAMPLE_RATE = 16000
HOP_LENGTH = 256


def _text_input(default: str, tooltip: str) -> tuple:
    return ("STRING", {"multiline": True, "default": default, "tooltip": tooltip})


def _weight_choices() -> list[str]:
    """Every loadable "model_dir/weights.safetensors" pair, preferred defaults first."""
    def priority(name: str) -> int:
        lowered = name.lower()
        if "int8" in lowered:
            return 0
        if "bf16" in lowered:
            return 1
        if "fp32" in lowered:
            return 2
        return 3

    pairs = []
    for model_name in available_models():
        for weights in available_weights(model_name):
            if "fp32" in weights.lower():
                continue  # fp32 is the archival/conversion reference, not a runtime build
            pairs.append((priority(weights), weights))
    pairs.sort(key=lambda item: (item[0], item[1]))
    choices = [entry for _, entry in pairs]
    return choices or ["Raon-OpenTTS-1B-int8-convrot.safetensors"]


class RaonOpenTTSLoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "weights": (
                    _weight_choices(),
                    {
                        "default": _weight_choices()[0],
                        "tooltip": "Checkpoint file from ComfyUI/models/raon_opentts (0.3B and 1B in separate folders). int8-convrot is the quantized build (smallest, runs through comfy-kitchen INT8 kernels); bf16 is the half-precision reference. The fp32 extraction is conversion reference only and is not listed. With download_if_missing on, only the selected build (plus its config/vocab and the vocoder) is downloaded — never the whole repo.",
                    },
                ),
                "dtype": (
                    DTYPE_OPTIONS,
                    {
                        "default": "auto",
                        "tooltip": "Compute/storage dtype for the DiT. auto picks bf16 on supported GPUs. INT8 weights always stay INT8 and their scales fp32 regardless of this choice.",
                    },
                ),
                "device": (
                    DEVICE_OPTIONS,
                    {
                        "default": "auto",
                        "tooltip": "Device for inference. auto follows ComfyUI's current torch device; cpu is a slow fallback.",
                    },
                ),
                "attention": (
                    ATTENTION_OPTIONS,
                    {
                        "default": "auto",
                        "tooltip": "Attention backend for the DiT blocks. auto uses flash_attn (FA2) when installed with half-precision compute, otherwise torch SDPA. sageattention patches SDPA at generation time.",
                    },
                ),
                "download_if_missing": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Download the HiFi-GAN vocoder into ComfyUI/models/raon_opentts when missing.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("RAON_OPENTTS_MODEL",)
    RETURN_NAMES = ("raon_model",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "Load Raon-OpenTTS (0.3B or 1B) natively with ComfyUI/AIMDO memory registration."

    def load(self, weights: str, dtype: str, device: str, attention: str, download_if_missing: bool):
        bundle = load_raon_bundle(
            weights_selection=weights,
            dtype_name=dtype,
            device_name=device,
            attention=attention,
            download_if_missing=bool(download_if_missing),
        )
        return (bundle,)


class RaonOpenTTSGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raon_model": ("RAON_OPENTTS_MODEL",),
                "text": _text_input(
                    "Hello! This is Raon OpenTTS running natively inside ComfyUI.",
                    "Text to synthesize. Long text is split into chunks automatically and cross-faded together.",
                ),
                "ref_audio": (
                    "AUDIO",
                    {"tooltip": "Reference voice clip for zero-shot cloning. Clean speech with little noise works best."},
                ),
                "ref_text": _text_input(
                    "",
                    "Exact transcript of the reference clip. Strongly improves cloning quality. Use the Whisper Transcribe node to generate it.",
                ),
                "steps": (
                    "INT",
                    {
                        "default": 32,
                        "min": 1,
                        "max": 64,
                        "step": 1,
                        "tooltip": "NFE steps for the flow-matching ODE (euler + EPSS schedule). 32 is the official default.",
                    },
                ),
                "cfg_strength": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.05,
                        "tooltip": "Classifier-free guidance strength. 2.0 is the official default; 0 disables CFG.",
                    },
                ),
                "sway_sampling_coef": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Sway sampling coefficient for the time schedule. -1.0 is the official default.",
                    },
                ),
                "speed": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.5,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Speech-rate multiplier for duration estimation (>1 = faster).",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 2**31 - 1,
                        "tooltip": "0 uses the current random state. A positive value is repeatable (incremented per text chunk).",
                    },
                ),
                "fix_duration_seconds": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 60.0,
                        "step": 0.5,
                        "tooltip": "Force the generated segment length in seconds. 0 estimates it from the reference speech rate (official behaviour).",
                    },
                ),
                "target_rms": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.01,
                        "tooltip": "Loudness normalization target for the reference clip (official: 0.1).",
                    },
                ),
                "use_vad_duration": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Estimate the generation length from the VAD-trimmed reference length (official default, robust for quiet speakers) while conditioning on the untrimmed audio.",
                    },
                ),
                "cross_fade_ms": (
                    "FLOAT",
                    {
                        "default": 150.0,
                        "min": 0.0,
                        "max": 500.0,
                        "step": 10.0,
                        "tooltip": "Cross-fade between generated text chunks in milliseconds (official: 150).",
                    },
                ),
                "do_split": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Split long text into chunks and generate them one by one (cross-faded together). Off = always one chunk, whatever the length.",
                    },
                ),
                "max_chars": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2000,
                        "step": 1,
                        "tooltip": "Chunk size budget in UTF-8 bytes. 0 = auto, estimated from the reference speech rate (official behaviour: ref_bytes/ref_seconds x (22 - ref_seconds)). Any positive value forces that budget, so the split becomes deterministic across speakers.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = "Zero-shot voice cloning with Raon-OpenTTS (F5-TTS-style CFM/DiT + HiFi-GAN)."

    def generate(self, raon_model: RaonBundle, text, ref_audio, ref_text, steps, cfg_strength,
                 sway_sampling_coef, speed, seed, fix_duration_seconds, target_rms,
                 use_vad_duration, cross_fade_ms, do_split, max_chars):
        if not ref_text.strip():
            raise ValueError(
                "ref_text is empty. Raon-OpenTTS needs the reference transcript for conditioning; "
                "run the Raon Whisper Transcribe node on ref_audio and feed its output here."
            )
        resume_bundle_to_device(raon_model)
        model, vocoder, device = raon_model.model, raon_model.vocoder, raon_model.device

        waveform, sample_rate = native.comfy_audio_to_tensor(ref_audio)
        waveform = native.normalize_peak(waveform)
        ref_text = native.fix_ref_text_ending(ref_text)

        if bool(use_vad_duration):
            ref_seconds_for_length = native.estimate_ref_seconds_trimmed_tensor(waveform, sample_rate)
        else:
            ref_seconds_for_length = None

        if not bool(do_split):
            gen_text_batches = [text.strip()] if text.strip() else []
        else:
            if int(max_chars) > 0:
                budget = int(max_chars)
            elif ref_seconds_for_length is not None:
                budget = int(len(ref_text.encode("utf-8")) / max(ref_seconds_for_length, 1e-6)
                             * (22 - ref_seconds_for_length))
            else:
                ref_raw_seconds = waveform.shape[-1] / sample_rate
                budget = int(len(ref_text.encode("utf-8")) / max(ref_raw_seconds, 1e-6)
                             * (22 - ref_raw_seconds))
            budget = max(budget, 20)
            gen_text_batches = native.chunk_text(text, max_chars=budget)
        logger.info("generating %d chunk(s): %s", len(gen_text_batches),
                    [c[:40] for c in gen_text_batches])

        rms = torch.sqrt(torch.mean(torch.square(waveform)))
        if rms < target_rms:
            waveform = waveform * target_rms / rms
        if sample_rate != TARGET_SAMPLE_RATE:
            import torchaudio

            resampler = torchaudio.transforms.Resample(sample_rate, TARGET_SAMPLE_RATE)
            waveform = resampler(waveform)
        audio = waveform.unsqueeze(0).to(device)  # CFM.sample expects cond as [batch, samples]

        ref_audio_len_cond = audio.shape[-1] // HOP_LENGTH
        total_steps = len(gen_text_batches) * int(steps)
        pbar = ProgressBar(total_steps) if ProgressBar is not None else None
        segments: list[torch.Tensor] = []

        with native.attention_runtime(raon_model.attention):
            for index, gen_text in enumerate(gen_text_batches):
                local_speed = speed
                if len(gen_text.encode("utf-8")) < 10:
                    local_speed = 0.3

                final_text_list = native.convert_char_to_pinyin([ref_text + gen_text])

                if fix_duration_seconds and fix_duration_seconds > 0:
                    gen_len = int(fix_duration_seconds * TARGET_SAMPLE_RATE / HOP_LENGTH)
                else:
                    if ref_seconds_for_length is not None:
                        ref_audio_len_est = int(ref_seconds_for_length * TARGET_SAMPLE_RATE / HOP_LENGTH)
                    else:
                        ref_audio_len_est = ref_audio_len_cond
                    ref_sec_est = ref_audio_len_est * HOP_LENGTH / TARGET_SAMPLE_RATE
                    ref_text_len = len(ref_text.encode("utf-8"))
                    gen_text_len = len(gen_text.encode("utf-8"))
                    sec_per_byte = ref_sec_est / max(ref_text_len, 1)
                    if use_vad_duration:
                        sec_per_byte = min(sec_per_byte, 1.0 / 12.0)  # minimum speech rate 12 chars/sec
                    gen_sec = (sec_per_byte * gen_text_len) / max(local_speed, 1e-6)
                    gen_len = int(gen_sec * TARGET_SAMPLE_RATE / HOP_LENGTH)
                gen_len = max(gen_len, 1)
                duration = ref_audio_len_cond + gen_len

                chunk_seed = None if int(seed) == 0 else int(seed) + index

                from tqdm import tqdm

                cli_pbar = tqdm(total=int(steps), desc=f"RaonOpenTTS chunk {index + 1}/{len(gen_text_batches)}",
                                unit="step", leave=False, dynamic_ncols=True)

                def update(step: int, _index=index) -> None:
                    if pbar is not None:
                        pbar.update_absolute(min(_index * int(steps) + step + 1, total_steps), total_steps)
                    cli_pbar.update(1)

                with torch.inference_mode():
                    generated, _ = model.sample(
                        cond=audio,
                        text=final_text_list,
                        duration=duration,
                        steps=int(steps),
                        cfg_strength=float(cfg_strength),
                        sway_sampling_coef=float(sway_sampling_coef),
                        seed=chunk_seed,
                        step_callback=update,
                    )
                    del _
                cli_pbar.close()

                generated = generated.to(torch.float32)
                generated = generated[:, ref_audio_len_cond:, :]
                generated = generated.permute(0, 2, 1)
                generated_wave = vocoder(generated)
                if rms < target_rms:
                    generated_wave = generated_wave * rms / target_rms
                segments.append(generated_wave.squeeze(0).squeeze(0).float().cpu())
                logger.info("chunk %d/%d done (%.2fs)", index + 1, len(gen_text_batches),
                            segments[-1].shape[-1] / TARGET_SAMPLE_RATE)

        if not segments:
            raise RuntimeError("No audio was generated.")
        gen_audio = segments[0].unsqueeze(0)
        for segment in segments[1:]:
            gen_audio = native.cross_fade(gen_audio, segment.unsqueeze(0), int(cross_fade_ms / 1000.0 * TARGET_SAMPLE_RATE))
        if not torch.isfinite(gen_audio).all():
            raise RuntimeError("Raon-OpenTTS generated non-finite audio samples.")
        return (native.tensor_audio_to_comfy(gen_audio.squeeze(0), TARGET_SAMPLE_RATE),)


NODE_CLASS_MAPPINGS = {
    "RaonOpenTTSLoadModel": RaonOpenTTSLoadModel,
    "RaonOpenTTSGenerate": RaonOpenTTSGenerate,
    "RaonWhisperTranscribe": RaonWhisperTranscribe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RaonOpenTTSLoadModel": "Raon OpenTTS Load Model",
    "RaonOpenTTSGenerate": "Raon OpenTTS Generate (Voice Clone)",
    "RaonWhisperTranscribe": "Raon Whisper Transcribe",
}
