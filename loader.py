"""Raon-OpenTTS model loading, downloads, and ComfyUI/AIMDO memory registration."""

from __future__ import annotations

import gc
import importlib.util
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

from . import int8
from . import native
from .vocoder import load_hifigan_vocoder

logger = logging.getLogger("RaonOpenTTS")

MODEL_FOLDER_NAME = "raon_opentts"
VOCODER_DIR_NAME = "tts-hifigan-libritts-16kHz"
VOCODER_REPO_ID = "speechbrain/tts-hifigan-libritts-16kHz"
HF_REPO_ID = "drbaph/Raon-OpenTTS-comfyui"
HF_ENDPOINT = "https://huggingface.co"
MODEL_NAMES = ("Raon-OpenTTS-1B", "Raon-OpenTTS-0.3B")

DTYPE_OPTIONS = ["auto", "bf16", "fp32"]
DEVICE_OPTIONS = ["auto", "cuda", "cpu"]
ATTENTION_OPTIONS = ["auto", "sdpa", "flash_attention", "sageattention"]

_ACTIVE_BUNDLE: "RaonBundle | None" = None
_ACTIVE_LOAD_KEY: tuple[Any, ...] | None = None


@dataclass
class RaonBundle:
    model_name: str
    model: torch.nn.Module           # CFM (DiT backbone)
    vocoder: torch.nn.Module         # HiFi-GAN generator
    vocab_char_map: dict
    text_num_embeds: int
    model_dir: Path
    weights_file: Path
    device: torch.device
    dtype_name: str
    attention: str
    quantized: bool
    patchers: list[Any] = field(default_factory=list)


try:
    import comfy.model_patcher as _model_patcher

    _ComfyCorePatcher = _model_patcher.CoreModelPatcher
    del _model_patcher
except Exception:
    _ComfyCorePatcher = None


def _empty_accelerator_cache() -> None:
    try:
        import comfy.model_management as mm

        mm.soft_empty_cache()
        return
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def model_dirs() -> list[Path]:
    """All registered raon_opentts folders (extra_model_paths.yaml aware), primary first."""
    try:
        import folder_paths

        primary = Path(folder_paths.models_dir) / MODEL_FOLDER_NAME
        paths = [Path(p) for p in folder_paths.get_folder_paths(MODEL_FOLDER_NAME)]
        if primary not in paths:
            paths.insert(0, primary)
        if paths:
            return paths
    except Exception:
        pass
    return [Path(__file__).resolve().parent / "models" / MODEL_FOLDER_NAME]


def model_dir() -> Path:
    base = model_dirs()[0]
    base.mkdir(parents=True, exist_ok=True)
    return base


def register_model_folder() -> None:
    try:
        import folder_paths

        base = str(model_dir())
        if MODEL_FOLDER_NAME not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path(MODEL_FOLDER_NAME, base)
            logger.info("model folder registered: %s", base)
    except Exception:
        pass


def _has_model_files(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "config.yaml").is_file()
        and (path / "vocab.txt").is_file()
        and any(path.glob("*.safetensors"))
    )


def available_models() -> list[str]:
    """Model directories with at least config.yaml, vocab.txt and one safetensors."""
    names: list[str] = []
    for base in model_dirs():
        if not base.is_dir():
            continue
        for child in sorted(base.iterdir()):
            if child.is_dir() and _has_model_files(child) and child.name not in names:
                names.append(child.name)
    return names or ["Raon-OpenTTS-1B", "Raon-OpenTTS-0.3B"]


def available_weights(model_name: str) -> list[str]:
    """Safetensors filenames present for the given model directory."""
    for base in model_dirs():
        candidate = base / model_name
        if _has_model_files(candidate):
            return sorted(p.name for p in candidate.glob("*.safetensors"))
    return []


def resolve_weights(weights_selection: str, download_if_missing: bool) -> tuple[Path, Path]:
    """Resolve a dropdown value to (model_dir, weights_file).

    Accepts a bare filename ('Raon-OpenTTS-1B-int8-convrot.safetensors', searched in
    every model folder) or the legacy 'model_dir/filename.safetensors' form.
    """
    name = str(weights_selection).strip().replace("\\", "/").rstrip("/")
    model_name, slash, fname = name.rpartition("/")
    if not slash:  # bare filename: search every model folder
        for base in model_dirs():
            for candidate in sorted(base.glob(f"*/{name}")):
                if candidate.is_file():
                    return candidate.parent, candidate
        if download_if_missing:
            models = set(available_models()) | set(MODEL_NAMES)
            for model in sorted(models, key=len, reverse=True):
                if name.startswith(model):  # filenames encode their model folder
                    _download_model_files(model, name)
                    for base in model_dirs():
                        candidate = base / model / name
                        if candidate.is_file():
                            return candidate.parent, candidate
        raise FileNotFoundError(
            f"weights '{weights_selection}' not found under any model folder in: "
            f"{', '.join(str(b) for b in model_dirs())}"
        )
    for base in model_dirs():
        candidate = base / model_name / fname
        if candidate.is_file():
            return candidate.parent, candidate
    if download_if_missing:
        _download_model_files(model_name, fname)
        for base in model_dirs():
            candidate = base / model_name / fname
            if candidate.is_file():
                return candidate.parent, candidate
    raise FileNotFoundError(
        f"'{fname}' not found in {model_name} (looked in: {', '.join(str(b) for b in model_dirs())})"
    )


def _download_model_files(model_name: str, weights_name: str | None = None) -> None:
    """Download only what the user selected: one weights build + its config/vocab + the vocoder."""
    from huggingface_hub import snapshot_download

    if weights_name:
        allow_patterns = [
            f"{model_name}/{weights_name}",
            f"{model_name}/config.yaml",
            f"{model_name}/vocab.txt",
        ]
        what = f"{weights_name}"
    else:
        allow_patterns = [f"{model_name}/*"]
        what = f"{model_name} (all builds)"
    allow_patterns.append(f"{VOCODER_DIR_NAME}/*")

    dest = model_dir()
    logger.info("Downloading %s from %s into %s.", what, HF_REPO_ID, dest)
    snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=str(dest),
        allow_patterns=allow_patterns,
        endpoint=HF_ENDPOINT,
    )


def ensure_vocoder(download_if_missing: bool) -> Path:
    for base in model_dirs():
        candidate = base / VOCODER_DIR_NAME / "generator.ckpt"
        if candidate.is_file():
            return candidate
    dest = model_dir() / VOCODER_DIR_NAME / "generator.ckpt"
    if not download_if_missing:
        raise FileNotFoundError(
            f"HiFi-GAN vocoder missing at {dest}. Enable download_if_missing to fetch it."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    for repo_id in (HF_REPO_ID, VOCODER_REPO_ID):
        try:
            logger.info("Downloading HiFi-GAN vocoder from %s to %s", repo_id, dest)
            hf_hub_download(repo_id=repo_id, filename="generator.ckpt",
                            local_dir=str(dest.parent), endpoint=HF_ENDPOINT)
            break
        except Exception as exc:
            logger.warning("vocoder download from %s failed: %s", repo_id, exc)
    if not dest.is_file():
        raise RuntimeError(f"Vocoder download finished but {dest} is still missing.")
    return dest


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        try:
            import comfy.model_management as mm

            device = torch.device(mm.get_torch_device())
        except Exception:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected, but torch.cuda is not available.")
        if device.index is None:
            # Bare "cuda" has index None; comfy_aimdo's get_devctx(int(index)) needs
            # an explicit index, so pin the current device.
            device = torch.device("cuda", torch.cuda.current_device())
    return device


def resolve_dtype_mode(dtype_name: str, device: torch.device) -> str:
    """Returns 'bf16' or 'fp32'."""
    if device.type == "cpu":
        if dtype_name == "bf16":
            logger.warning("bf16 is not practical on CPU; using fp32.")
        return "fp32"
    if dtype_name == "auto":
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                return "bf16" if torch.cuda.is_bf16_supported() else "fp32"
            except Exception:
                return "fp32"
        return "bf16"
    if dtype_name in ("bf16", "fp32"):
        return dtype_name
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def resolve_attention(attention: str, device: torch.device, dtype_mode: str) -> tuple[str, str]:
    """Returns (node_choice, attn_backend) where attn_backend feeds the DiT constructor.

    auto matches what runs best here: the flash_attn package when it is installed and
    the compute is half precision on CUDA, otherwise torch SDPA. sageattention runs as
    an SDPA patch at generation time, so the model itself stays on the torch backend.
    """
    flash_usable = (
        importlib.util.find_spec("flash_attn") is not None
        and torch.device(device).type == "cuda"
        and dtype_mode != "fp32"
    )
    if attention == "auto":
        return ("flash_attention" if flash_usable else "sdpa"), ("flash_attn" if flash_usable else "torch")
    if attention == "sdpa":
        return "sdpa", "torch"
    if attention == "sageattention":
        if importlib.util.find_spec("sageattention") is None:
            raise ImportError("sageattention was selected, but sageattention is not installed.")
        return "sageattention", "torch"
    if attention == "flash_attention":
        if importlib.util.find_spec("flash_attn") is None:
            raise ImportError("flash_attention was selected, but flash_attn is not installed.")
        if dtype_mode == "fp32":
            logger.warning("flash_attention needs bf16/fp16 compute; using sdpa because dtype=fp32 was selected.")
            return "sdpa", "torch"
        return "flash_attention", "flash_attn"
    raise ValueError(f"Unsupported attention mode: {attention}")


def dynamic_vram_active(device: torch.device) -> bool:
    if torch.device(device).type == "cpu":
        return False
    try:
        import comfy.memory_management

        if not bool(comfy.memory_management.aimdo_enabled):
            return False
        try:
            import comfy_aimdo.control
            import comfy_aimdo.host_buffer
            import comfy_aimdo.model_vbar

            return (
                comfy_aimdo.control.lib is not None
                and comfy_aimdo.host_buffer.lib is not None
                and comfy_aimdo.model_vbar.lib is not None
            )
        except Exception:
            return False
    except Exception:
        return False


def _register_many_with_comfy(patchers: list[Any]) -> None:
    patchers = [p for p in patchers if p is not None and p.load_device.type != "cpu"]
    if not patchers:
        return
    try:
        import comfy.model_management as mm

        already_loaded = {
            id(loaded.model) for loaded in mm.current_loaded_models if loaded.model is not None
        }
        to_load = [p for p in patchers if id(p) not in already_loaded]
        if not to_load:
            return
        mm.load_models_gpu(to_load)
        for patcher in to_load:
            logger.info(
                "Loaded %s through ComfyUI%s memory management.",
                patcher.model.__class__.__name__,
                "/AIMDO" if patcher.is_dynamic() else "",
            )
    except Exception as exc:
        raise RuntimeError("Could not load model through ComfyUI memory management.") from exc


def _unregister_from_comfy(patcher: Any) -> None:
    try:
        import comfy.model_management as mm

        survivors = []
        for loaded in mm.current_loaded_models:
            if loaded.model is patcher:
                try:
                    if loaded.model_finalizer is not None:
                        loaded.model_finalizer.detach()
                    loaded.model_finalizer = None
                    loaded.real_model = None
                except Exception:
                    pass
                try:
                    finalizer = getattr(loaded, "_patcher_finalizer", None)
                    if finalizer is not None:
                        finalizer.detach()
                    loaded._patcher_finalizer = None
                except Exception:
                    pass
                continue
            survivors.append(loaded)
        mm.current_loaded_models[:] = survivors
    except Exception:
        pass


def _set_module_device_if_writable(module: torch.nn.Module, device: torch.device) -> None:
    try:
        module.device = torch.device(device)
    except (AttributeError, RuntimeError, TypeError):
        pass


def _ensure_writable_device_property(module: torch.nn.Module) -> None:
    cls = module.__class__
    prop = getattr(cls, "device", None)
    if not isinstance(prop, property) or prop.fset is not None:
        return
    if getattr(module, "_raon_writable_device_property", False):
        return

    def _get_device(self):
        runtime_device = self.__dict__.get("_raon_runtime_device")
        if runtime_device is not None:
            return runtime_device
        return prop.fget(self)

    def _set_device(self, value):
        self.__dict__["_raon_runtime_device"] = torch.device(value)

    writable_cls = type(
        cls.__name__,
        (cls,),
        {
            "device": property(_get_device, _set_device),
            "_raon_device_base_class": cls,
            "__module__": cls.__module__,
        },
    )
    module.__class__ = writable_cls
    module._raon_writable_device_property = True


def register_runtime_module(module: torch.nn.Module, device: torch.device, *, dynamic: bool | None = None) -> Any:
    device = torch.device(device)
    module._raon_runtime_device = torch.device(device)
    _ensure_writable_device_property(module)
    if _ComfyCorePatcher is None or device.type == "cpu":
        module.to(device)
        return None

    import comfy.model_patcher as model_patcher

    use_dynamic = dynamic_vram_active(device) and dynamic is not False
    patcher_class = model_patcher.ModelPatcherDynamic if use_dynamic else model_patcher.ModelPatcher
    patcher = patcher_class(module, load_device=device, offload_device=torch.device("cpu"))
    module.model_loaded_weight_memory = 0
    _register_many_with_comfy([patcher])
    if not patcher.is_dynamic():
        _set_module_device_if_writable(module, device)
    return patcher


def resume_runtime_module(patcher: Any, device: torch.device) -> None:
    del device
    if patcher is not None:
        _register_many_with_comfy([patcher])


def unload_runtime_module(patcher: Any, *, hard: bool = True) -> None:
    if patcher is None:
        return
    _unregister_from_comfy(patcher)
    try:
        patcher.detach()
    except Exception:
        pass


def resume_bundle_to_device(bundle: RaonBundle) -> None:
    for patcher in bundle.patchers:
        resume_runtime_module(patcher, bundle.device)


def unload_raon_bundle(bundle: RaonBundle | None, reason: str = "manual unload", hard: bool = True) -> None:
    global _ACTIVE_BUNDLE, _ACTIVE_LOAD_KEY
    if bundle is None:
        return
    logger.info("Unloading bundle (%s).", reason)
    for patcher in list(bundle.patchers):
        unload_runtime_module(patcher, hard=hard)
    modules = [bundle.model, bundle.vocoder]
    if not hard:
        for module in modules:
            try:
                module.to("cpu")
            except Exception:
                pass
    for module in modules:
        try:
            module.model_loaded_weight_memory = 0
            if hasattr(module, "dynamic_vbars"):
                module.dynamic_vbars.clear()
            if hard and hasattr(module, "to_empty"):
                module.to_empty(device=torch.device("meta"))
        except Exception:
            pass
    bundle.patchers.clear()
    if hard:
        bundle.model = None
        bundle.vocoder = None
        bundle.vocab_char_map = None
    gc.collect()
    _empty_accelerator_cache()
    if _ACTIVE_BUNDLE is bundle:
        _ACTIVE_BUNDLE = None
        _ACTIVE_LOAD_KEY = None


def _dtype_policy(mode: str):
    def policy(name: str) -> torch.dtype | None:
        if name.endswith("weight_scale"):
            return None  # INT8 ConvRot per-row scales must stay fp32
        if name.endswith("inv_freq"):
            return None  # rotary base frequencies stay fp32 (phase precision over long sequences)
        return torch.bfloat16 if mode == "bf16" else torch.float32

    return policy


def read_vocab(vocab_path: Path) -> dict:
    """Official 'custom' tokenizer: one token per line, index = line number."""
    vocab_char_map = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for i, char in enumerate(f):
            vocab_char_map[char[:-1]] = i
    return vocab_char_map


def build_model(config_path: Path, text_num_embeds: int, attn_backend: str) -> torch.nn.Module:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    arch = dict(cfg["model"]["arch"])
    arch["attn_backend"] = attn_backend
    mel = dict(cfg["model"]["mel_spec"])
    model = native.CFM(
        transformer=native.DiT(
            **arch,
            text_num_embeds=text_num_embeds,
            mel_dim=mel["n_mel_channels"],
        ),
        mel_spec_kwargs=dict(
            n_fft=mel["n_fft"],
            hop_length=mel["hop_length"],
            win_length=mel["win_length"],
            n_mel_channels=mel["n_mel_channels"],
            target_sample_rate=mel["target_sample_rate"],
            mel_spec_type=mel["mel_spec_type"],
        ),
        odeint_kwargs=dict(method="euler"),
        vocab_char_map=None,
    )
    return model


def _peek_text_num_embeds(weights_file: Path) -> int:
    from safetensors import safe_open

    with safe_open(str(weights_file), framework="pt", device="cpu") as f:
        shape = f.get_slice("transformer.text_embed.text_embed.weight").get_shape()
    return int(shape[0]) - 1


def _log_int8_banner(model: torch.nn.Module, device: torch.device) -> None:
    """One-time INT8 ConvRot status block, including a live kernel smoke test."""
    import comfy_kitchen
    from comfy_kitchen.tensor import TensorWiseINT8Layout

    qparams, qcount = int8.quantized_parameter_count(model)
    group_sizes = sorted({m.convrot_groupsize for m in model.modules() if isinstance(m, int8.ConvRotInt8Linear)})
    smoke = "SKIPPED (cpu)"
    if device.type in ("cuda", "xpu"):
        try:
            g = group_sizes[0]
            w = torch.randn(64, g * 2, device=device)
            q, p = TensorWiseINT8Layout.quantize(
                w, is_weight=True, per_channel=True, convrot=True, convrot_groupsize=g, stochastic_rounding=0
            )
            y = comfy_kitchen.int8_linear(
                torch.randn(8, g * 2, device=device), q, p.scale, None,
                out_dtype=torch.bfloat16, convrot=True, convrot_groupsize=g,
            )
            smoke = "PASS" if bool(torch.isfinite(y).all()) else "FAIL"
        except Exception as exc:
            smoke = f"FAIL ({exc})"
    backends = ",".join(sorted(k for k, v in comfy_kitchen.list_backends().items() if v["available"]))
    logger.info(
        "INT8 ConvRot: %d layers / %.2fB params / group %s / comfy_kitchen.int8_linear / backends %s / runtime smoke %s",
        qcount, qparams / 1e9, group_sizes, backends, smoke,
    )
    if smoke.startswith("FAIL"):
        raise RuntimeError(f"INT8 ConvRot kernel smoke test failed on {device}: {smoke}")


def load_raon_bundle(
    weights_selection: str,
    dtype_name: str,
    device_name: str,
    attention: str,
    download_if_missing: bool,
) -> RaonBundle:
    global _ACTIVE_BUNDLE, _ACTIVE_LOAD_KEY

    register_model_folder()
    runtime_dir, weights_file = resolve_weights(weights_selection, download_if_missing)
    weights_name = weights_file.name
    device = resolve_device(device_name)
    dtype_mode = resolve_dtype_mode(dtype_name, device)
    attention, attn_backend = resolve_attention(attention, device, dtype_mode)

    load_key = (
        str(runtime_dir.resolve()),
        weights_name,
        weights_file.stat().st_mtime_ns,
        str(device),
        dtype_mode,
        attention,
    )
    if _ACTIVE_BUNDLE is not None and _ACTIVE_LOAD_KEY == load_key:
        resume_bundle_to_device(_ACTIVE_BUNDLE)
        return _ACTIVE_BUNDLE
    if _ACTIVE_BUNDLE is not None:
        unload_raon_bundle(_ACTIVE_BUNDLE, reason="load settings changed")

    logger.info(
        "Loading %s (%s) on %s (%s, %s)",
        runtime_dir.name, weights_name, device, dtype_mode, attention,
    )

    text_num_embeds = _peek_text_num_embeds(weights_file)
    quant_map = int8.scan_checkpoint_quantization(weights_file)

    model = vocoder = None
    patchers: list[Any] = []
    try:
        try:
            from accelerate import init_empty_weights

            with init_empty_weights():
                model = build_model(runtime_dir / "config.yaml", text_num_embeds, attn_backend)
        except ImportError:
            model = build_model(runtime_dir / "config.yaml", text_num_embeds, attn_backend)

        if quant_map:
            int8.replace_quantized_linears(model, quant_map)
        native.load_safetensors_file(model, weights_file, dtype_policy=_dtype_policy(dtype_mode))
        native.convert_modules_for_comfy(model)
        native.set_runtime_dtype(model, torch.bfloat16 if dtype_mode == "bf16" else torch.float32)

        vocab_char_map = read_vocab(runtime_dir / "vocab.txt")
        if len(vocab_char_map) != text_num_embeds:
            # The released vocab.txt carries a few extra tokens beyond the checkpoint's
            # embedding table (1B/0.3B: 5559 tokens vs 5555 rows). The affected entries
            # are the highest-codepoint tail of the sorted file (U+FDFA, U+FDFB, U+FFFD,
            # U+1F3B5); they cannot be embedded, so they are dropped from the runtime map
            # (unknown -> index 0, same as any other out-of-vocab character).
            vocab_char_map = {tok: idx for tok, idx in vocab_char_map.items() if idx < text_num_embeds}
        model.vocab_char_map = vocab_char_map

        vocoder = load_hifigan_vocoder(ensure_vocoder(download_if_missing), device="cpu")
        native.convert_modules_for_comfy(vocoder)
        native.set_runtime_dtype(vocoder, torch.float32)  # vocoder stays fp32
        vocoder.model_loaded_weight_memory = 0

        use_dynamic = dynamic_vram_active(device)
        if use_dynamic:
            logger.info("AIMDO DynamicVRAM is active; using dynamic patchers.")
        else:
            logger.info("AIMDO not active; using static ComfyUI memory management.")
        for module in (model, vocoder):
            patcher = register_runtime_module(module, device, dynamic=use_dynamic)
            if patcher is not None:
                patchers.append(patcher)

        if quant_map:
            _log_int8_banner(model, device)

        bundle = RaonBundle(
            model_name=runtime_dir.name,
            model=model,
            vocoder=vocoder,
            vocab_char_map=vocab_char_map,
            text_num_embeds=text_num_embeds,
            model_dir=runtime_dir,
            weights_file=weights_file,
            device=device,
            dtype_name=dtype_mode,
            attention=attention,
            quantized=bool(quant_map),
            patchers=patchers,
        )
        _ACTIVE_BUNDLE = bundle
        _ACTIVE_LOAD_KEY = load_key
        install_comfy_unload_hook()
        _empty_accelerator_cache()
        return bundle
    except Exception:
        for patcher in list(patchers):
            unload_runtime_module(patcher, hard=True)
        for module in (model, vocoder):
            if module is None:
                continue
            try:
                module.model_loaded_weight_memory = 0
                if hasattr(module, "dynamic_vbars"):
                    module.dynamic_vbars.clear()
                if hasattr(module, "to_empty"):
                    module.to_empty(device=torch.device("meta"))
            except Exception:
                pass
        gc.collect()
        _empty_accelerator_cache()
        raise


def unload_active_bundle() -> None:
    unload_raon_bundle(_ACTIVE_BUNDLE, reason="active unload")


def install_comfy_unload_hook() -> None:
    """Patch ComfyUI unload calls so the active native bundle hard-releases."""
    try:
        import comfy.model_management as mm
    except Exception:
        return

    if getattr(mm, "_raon_opentts_unload_hook_installed", False):
        return

    original_unload_all_models = mm.unload_all_models

    def unload_all_models_with_raon(*args, **kwargs):
        try:
            return original_unload_all_models(*args, **kwargs)
        finally:
            unload_raon_bundle(_ACTIVE_BUNDLE, reason="ComfyUI unload_all_models")

    mm.unload_all_models = unload_all_models_with_raon

    original_unload_model_and_clones = getattr(mm, "unload_model_and_clones", None)
    if original_unload_model_and_clones is not None:
        def unload_model_and_clones_with_raon(model, *args, **kwargs):
            try:
                return original_unload_model_and_clones(model, *args, **kwargs)
            finally:
                if _ACTIVE_BUNDLE is not None and model is not None:
                    owned = list(_ACTIVE_BUNDLE.patchers) + [
                        m for m in (_ACTIVE_BUNDLE.model, _ACTIVE_BUNDLE.vocoder)
                        if m is not None
                    ]
                    if any(existing is model or existing is getattr(model, "model", None)
                           for existing in owned if existing is not None):
                        unload_raon_bundle(_ACTIVE_BUNDLE, reason="ComfyUI unload_model_and_clones")

        mm.unload_model_and_clones = unload_model_and_clones_with_raon

    mm._raon_opentts_unload_hook_installed = True
    logger.debug("Installed Raon-OpenTTS unload hook for ComfyUI native unload.")
