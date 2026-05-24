"""HTTP inference service for the recommended breast cancer classifier."""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Response
import joblib
import pandas as pd
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "resultados" / "fase2" / "modelo_serving.joblib"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))

logger = logging.getLogger("diagnostico_api")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

REQUESTS = Counter(
    "diagnostico_requests_total",
    "Total de requisicoes processadas pela API.",
    ["endpoint", "status"],
)
PREDICTIONS = Counter(
    "diagnostico_predictions_total",
    "Predicoes emitidas por classe prevista.",
    ["classe"],
)
PREDICTION_LATENCY = Histogram(
    "diagnostico_prediction_duration_seconds",
    "Latencia de inferencia do endpoint predict.",
)
MODEL_READY = Gauge(
    "diagnostico_model_ready",
    "Indica se o artefato de modelo foi carregado (1) ou nao (0).",
)

_artifact: dict[str, Any] | None = None


class PredictRequest(BaseModel):
    """Feature values in the original Wisconsin dataset column names."""

    features: dict[str, float] = Field(
        ...,
        description="Mapa contendo exatamente as 30 features numericas do modelo.",
    )


class PredictResponse(BaseModel):
    prediction: int
    diagnosis: str
    probability_malignant: float
    probability_benign: float
    model: str


def load_model_artifact(path: Path | None = None) -> dict[str, Any]:
    """Load and validate the pipeline payload created by genetic_optimization."""

    path = path or MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Modelo nao encontrado em {path}. Execute a otimizacao genetica antes da API."
        )
    artifact = joblib.load(path)
    required = {"model", "feature_names", "model_label"}
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"Artefato de modelo invalido; faltam campos: {sorted(missing)}")
    return artifact


def initialize_model() -> None:
    """Attempt model initialization while keeping liveness available on failure."""

    global _artifact
    try:
        _artifact = load_model_artifact()
        MODEL_READY.set(1)
        logger.info(
            json.dumps(
                {
                    "event": "model_loaded",
                    "path": str(MODEL_PATH),
                    "model": _artifact["model_label"],
                }
            )
        )
    except (FileNotFoundError, ValueError) as error:
        _artifact = None
        MODEL_READY.set(0)
        logger.error(json.dumps({"event": "model_load_failed", "error": str(error)}))


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_model()
    yield


app = FastAPI(
    title="API de Diagnostico de Cancer de Mama",
    description="Inferencia do modelo recomendado apos comparacao experimental.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health/live")
def liveness() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/health")
@app.get("/health/ready")
def readiness() -> dict[str, str]:
    if _artifact is None:
        raise HTTPException(status_code=503, detail="Modelo ainda nao esta disponivel.")
    return {"status": "ready", "model": str(_artifact["model_label"])}


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    start = time.perf_counter()
    request_status = "success"
    try:
        if _artifact is None:
            request_status = "model_unavailable"
            raise HTTPException(status_code=503, detail="Modelo ainda nao esta disponivel.")

        expected = list(_artifact["feature_names"])
        received = set(request.features)
        missing = sorted(set(expected).difference(received))
        unexpected = sorted(received.difference(expected))
        if missing or unexpected:
            request_status = "invalid_features"
            raise HTTPException(
                status_code=422,
                detail={"missing_features": missing, "unexpected_features": unexpected},
            )

        sample = pd.DataFrame([{feature: request.features[feature] for feature in expected}])
        model = _artifact["model"]
        prediction = int(model.predict(sample)[0])
        probabilities = model.predict_proba(sample)[0]
        classes = list(model.named_steps["model"].classes_)
        probability_malignant = float(probabilities[classes.index(0)])
        probability_benign = float(probabilities[classes.index(1)])
        diagnosis = "Maligno" if prediction == 0 else "Benigno"

        PREDICTIONS.labels(classe=diagnosis).inc()
        logger.info(
            json.dumps(
                {
                    "event": "prediction",
                    "prediction": prediction,
                    "diagnosis": diagnosis,
                    "probability_malignant": round(probability_malignant, 6),
                    "model": _artifact["model_label"],
                }
            )
        )
        return PredictResponse(
            prediction=prediction,
            diagnosis=diagnosis,
            probability_malignant=probability_malignant,
            probability_benign=probability_benign,
            model=str(_artifact["model_label"]),
        )
    finally:
        REQUESTS.labels(endpoint="/predict", status=request_status).inc()
        PREDICTION_LATENCY.observe(time.perf_counter() - start)


def main() -> None:
    """Run the development API when this file is executed directly."""

    import uvicorn

    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
