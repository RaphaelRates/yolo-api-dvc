import os
from pathlib import Path

import torch
from ultralytics import YOLO

# Ajuste para PyTorch 2.6+:
# desativa temporariamente o comportamento weights_only
# durante o carregamento dos pesos do YOLO.
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


MODELS_DIR = Path("./app/models")
_model_cache: dict[str, YOLO] = {}


def load_model(model_name: str) -> YOLO:
    """
    Carrega o modelo YOLO na primeira utilização e mantém
    a instância armazenada em cache para reutilização.
    """
    if model_name not in _model_cache:
        model_path = MODELS_DIR / model_name

        if not model_path.exists():
            available_models = list(MODELS_DIR.glob("*.pt"))

            raise FileNotFoundError(
                f"Modelo '{model_name}' não encontrado em {MODELS_DIR}. "
                f"Arquivos disponíveis: {available_models}"
            )

        _model_cache[model_name] = YOLO(str(model_path))

    return _model_cache[model_name]


def get_default_model_name() -> str:
    """Retorna o nome do modelo definido pela variável de ambiente."""
    return os.getenv("MODEL_NAME", "yolov8n.pt")