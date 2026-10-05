import logging

from .model_relay_sampler import (
    NODE_CLASS_MAPPINGS as RELAY_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as RELAY_NODE_DISPLAY_NAME_MAPPINGS,
)

NODE_CLASS_MAPPINGS = dict(RELAY_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS = dict(RELAY_NODE_DISPLAY_NAME_MAPPINGS)

WEB_DIRECTORY = "./js"

try:
    from .qwen_image_2_1_speedup import (
        NODE_CLASS_MAPPINGS as QWEN_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as QWEN_NODE_DISPLAY_NAME_MAPPINGS,
    )
except ImportError as exc:
    # Older ComfyUI installs can still use the original relay nodes.
    logging.warning(
        "ComfyUI Model Relay Sampler: Qwen Image 2.1 nodes were not loaded (%s). "
        "A ComfyUI installation with Qwen Image 2.1 support is required.",
        exc,
    )
else:
    NODE_CLASS_MAPPINGS.update(QWEN_NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(QWEN_NODE_DISPLAY_NAME_MAPPINGS)

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
