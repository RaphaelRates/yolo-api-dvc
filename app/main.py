import base64
import io
import json
import time
import uuid

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from model import get_default_model_name, load_model
from PIL import Image
from schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    Detection,
    HealthResponse,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
)

app = FastAPI(
    title="YOLO Inference API",
    description="API REST para inferência com YOLOv8 no Raspberry Pi 5",
    version="1.0.0",
)


# ─────────────────────────────────────────────────────────────
# Métricas simples em memória
# ─────────────────────────────────────────────────────────────

_metrics = {
    "total": 0,
    "success": 0,
    "total_ms": 0.0,
}


# ─────────────────────────────────────────────────────────────
# Função de log estruturado
# ─────────────────────────────────────────────────────────────

def log_event(event: str, level: str = "INFO", **kwargs):
    """Emite um evento estruturado em JSON para stdout."""
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "level": level,
        "event": event,
        **kwargs,
    }
    print(json.dumps(record, ensure_ascii=False), flush=True)


# ─────────────────────────────────────────────────────────────
# Processamento de imagens
# ─────────────────────────────────────────────────────────────

def _decode_image(image_base64: str) -> np.ndarray:
    """Converte uma imagem Base64 em um array NumPy RGB."""
    raw = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(image)


def _load_image_from_request(request: PredictRequest) -> np.ndarray:
    """Carrega a imagem a partir de Base64 ou URL pública em formato RGB."""
    if not request.image_base64 and not request.image_url:
        raise HTTPException(
            status_code=422,
            detail="Forneça image_base64 ou image_url.",
        )

    if request.image_base64:
        return _decode_image(request.image_base64)

    response = httpx.get(
        request.image_url,
        timeout=15.0,
        follow_redirects=True,
    )
    response.raise_for_status()

    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    return np.array(image)


# ─────────────────────────────────────────────────────────────
# Inferência
# ─────────────────────────────────────────────────────────────

def _run_inference(
    image_np: np.ndarray,
    model_name: str,
    confidence: float,
) -> PredictResponse:
    """Executa a inferência YOLO e retorna as detecções."""
    model = load_model(model_name)

    start_time = time.perf_counter()

    results = model(
        image_np,
        conf=confidence,
        verbose=False,
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    detections = []

    for result in results:
        for box in result.boxes:
            coords = box.xyxy[0].tolist()
            class_id = int(box.cls[0].item())
            confidence_value = float(box.conf[0].item())

            detections.append(
                Detection(
                    label=model.names[class_id],
                    confidence=round(confidence_value, 4),
                    bbox=[
                        round(float(coord), 2)
                        for coord in coords
                    ],
                )
            )

    height, width = image_np.shape[:2]

    return PredictResponse(
        detections=detections,
        inference_ms=round(elapsed_ms, 2),
        model_used=model_name,
        image_width=width,
        image_height=height,
    )


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
)
async def health_check():
    """Verifica o estado da API e do modelo padrão."""
    model_name = get_default_model_name()

    try:
        load_model(model_name)
        model_loaded = True
    except (FileNotFoundError, RuntimeError, ValueError):
        model_loaded = False

    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        model_name=model_name,
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
)
def predict(request: PredictRequest):
    """Executa a inferência e retorna as detecções."""
    request_id = str(uuid.uuid4())[:8]
    _metrics["total"] += 1

    log_event(
        "predict_start",
        request_id=request_id,
        model=request.model_name,
        confidence=request.confidence
    )

    if not request.image_base64 and not request.image_url:
        log_event(
            "predict_error",
            level="WARN",
            request_id=request_id,
            reason="missing_input"
        )
        raise HTTPException(
            status_code=422,
            detail="Forneça image_base64 ou image_url."
        )

    try:
        if request.image_base64:
            img = _decode_image(request.image_base64)
        else:
            response = httpx.get(request.image_url, timeout=10)
            response.raise_for_status()
            img = _decode_image(base64.b64encode(response.content).decode())

        result = _run_inference(
            img,
            request.model_name,
            request.confidence,
        )

        _metrics["success"] += 1
        _metrics["total_ms"] += result.inference_ms

        log_event(
            "predict_complete",
            request_id=request_id,
            model=result.model_used,
            detections=len(result.detections),
            inference_ms=result.inference_ms,
            image_size=f"{result.image_width}x{result.image_height}"
        )

        return result

    except FileNotFoundError as error:
        log_event(
            "predict_error",
            level="ERROR",
            request_id=request_id,
            reason=str(error)
        )
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except Exception as error:
        log_event(
            "predict_error",
            level="ERROR",
            request_id=request_id,
            reason=str(error)
        )
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


@app.post(
    "/predict/image",
    responses={
        200: {
            "content": {
                "image/jpeg": {},
            }
        }
    },
)
def predict_image(request: PredictRequest):
    """Executa a inferência e retorna a imagem anotada em JPEG."""
    _metrics["total"] += 1

    try:
        # 1. Carrega a imagem em RGB.
        image_rgb = _load_image_from_request(request)

        model = load_model(request.model_name)

        start_time = time.perf_counter()

        results = model(
            image_rgb,
            conf=request.confidence,
            verbose=False,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        _metrics["success"] += 1
        _metrics["total_ms"] += elapsed_ms

        # 2. plot() retorna o array da imagem anotada.
        annotated_array = results[0].plot()

        # 3. Converte diretamente para PIL e salva como JPEG.
        annotated_image = Image.fromarray(annotated_array)

        buffer = io.BytesIO()

        annotated_image.save(
            buffer,
            format="JPEG",
            quality=95,
        )

        return Response(
            content=buffer.getvalue(),
            media_type="image/jpeg",
        )

    except HTTPException:
        raise

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
)
def predict_batch(request: BatchPredictRequest):
    """Executa inferência em múltiplas imagens Base64."""
    start_time = time.perf_counter()

    results = []

    for image_base64 in request.images_base64:
        image = _decode_image(image_base64)

        results.append(
            _run_inference(
                image,
                request.model_name,
                request.confidence,
            )
        )

    total_ms = (time.perf_counter() - start_time) * 1000

    return BatchPredictResponse(
        results=results,
        total_inference_ms=round(total_ms, 2),
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
)
async def get_metrics():
    """Retorna métricas básicas de utilização da API."""
    if _metrics["success"] > 0:
        average_ms = _metrics["total_ms"] / _metrics["success"]
    else:
        average_ms = 0.0

    return MetricsResponse(
        total_requests=_metrics["total"],
        successful_requests=_metrics["success"],
        avg_inference_ms=round(average_ms, 2),
    )