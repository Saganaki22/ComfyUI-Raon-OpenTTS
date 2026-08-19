"""ComfyUI custom nodes for KRAFTON Raon-OpenTTS (F5-TTS-style CFM/DiT voice cloning).

Supports the official fp32 extraction, a bf16 build, and the INT8 ConvRot
(int8_tensorwise) quantized build executed through comfy-kitchen kernels.
"""

import logging

from .loader import register_model_folder
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

logger = logging.getLogger("RaonOpenTTS")

register_model_folder()

WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
