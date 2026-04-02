from __future__ import annotations

from typing import Dict, List


DEFAULT_SWAP_MODEL = "inswapper_auto"

SWAP_MODELS: Dict[str, Dict[str, object]] = {
    "inswapper_auto": {
        "label": "InSwapper (Auto)",
        "runtime": "alias",
    },
    "inswapper_128": {
        "label": "InSwapper 128",
        "filename": "inswapper_128.onnx",
        "download_url": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.onnx",
        "runtime": "inswapper",
        "size": (128, 128),
        "template": "arcface_128",
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
    },
    "inswapper_128_fp16": {
        "label": "InSwapper 128 FP16",
        "filename": "inswapper_128_fp16.onnx",
        "download_url": "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx",
        "runtime": "inswapper",
        "size": (128, 128),
        "template": "arcface_128",
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
    },
    "hyperswap_1a_256": {
        "label": "HyperSwap 1A 256",
        "filename": "hyperswap_1a_256.onnx",
        "download_url": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/hyperswap_1a_256.onnx",
        "runtime": "hyperswap",
        "size": (256, 256),
        "template": "arcface_128",
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
    },
    "hyperswap_1b_256": {
        "label": "HyperSwap 1B 256",
        "filename": "hyperswap_1b_256.onnx",
        "download_url": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/hyperswap_1b_256.onnx",
        "runtime": "hyperswap",
        "size": (256, 256),
        "template": "arcface_128",
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
    },
}

SWAP_MODEL_CHOICES: List[str] = list(SWAP_MODELS.keys())
SWAP_MODEL_LABELS: Dict[str, str] = {
    model_id: str(model_info["label"]) for model_id, model_info in SWAP_MODELS.items()
}
SWAP_MODEL_LABEL_TO_ID: Dict[str, str] = {
    label: model_id for model_id, label in SWAP_MODEL_LABELS.items()
}


def resolve_swap_model_id(selected_model: str, execution_providers: List[str]) -> str:
    if selected_model == "inswapper_auto":
        if "CUDAExecutionProvider" in execution_providers:
            return "inswapper_128_fp16"
        return "inswapper_128"
    return selected_model


def get_swap_model(selected_model: str, execution_providers: List[str]) -> Dict[str, object]:
    resolved_model = resolve_swap_model_id(selected_model, execution_providers)
    return SWAP_MODELS[resolved_model]
