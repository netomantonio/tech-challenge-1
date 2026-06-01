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

try:
    from src.llm_interpretation import (
        DEFAULT_LLM_MODEL,
        LLMUnavailableError,
        ModelResult,
        derive_feature_evidence,
        evaluate_interpretation_quality,
        generate_interpretation,
    )
except ModuleNotFoundError:
    from llm_interpretation import (
        DEFAULT_LLM_MODEL,
        LLMUnavailableError,
        ModelResult,
        derive_feature_evidence,
        evaluate_interpretation_quality,
        generate_interpretation,
    )


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
INTERPRETATIONS = Counter(
    "diagnostico_llm_interpretations_total",
    "Interpretacoes solicitadas a LLM por status.",
    ["status", "llm_model"],
)
INTERPRETATION_LATENCY = Histogram(
    "diagnostico_llm_interpretation_duration_seconds",
    "Latencia para gerar explicacao da LLM.",
)
INTERPRETATION_QUALITY = Histogram(
    "diagnostico_llm_quality_score",
    "Pontuacao objetiva das interpretacoes geradas.",
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


class InterpretResponse(PredictResponse):
    explanation: str
    llm_model: str
    prompt_version: str
    disclaimer: str
    evidence: list[dict[str, str | float]]
    quality_checks: dict[str, bool | float]


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


def _run_prediction(features: dict[str, float]) -> PredictResponse:
    if _artifact is None:
        raise HTTPException(status_code=503, detail="Modelo ainda nao esta disponivel.")

    expected = list(_artifact["feature_names"])
    received = set(features)
    missing = sorted(set(expected).difference(received))
    unexpected = sorted(received.difference(expected))
    if missing or unexpected:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing, "unexpected_features": unexpected},
        )

    sample = pd.DataFrame([{feature: features[feature] for feature in expected}])
    model = _artifact["model"]
    prediction = int(model.predict(sample)[0])
    probabilities = model.predict_proba(sample)[0]
    classes = list(model.named_steps["model"].classes_)
    probability_malignant = float(probabilities[classes.index(0)])
    probability_benign = float(probabilities[classes.index(1)])
    diagnosis = "Maligno" if prediction == 0 else "Benigno"
    return PredictResponse(
        prediction=prediction,
        diagnosis=diagnosis,
        probability_malignant=probability_malignant,
        probability_benign=probability_benign,
        model=str(_artifact["model_label"]),
    )


def _record_prediction(result: PredictResponse, endpoint: str) -> None:
    PREDICTIONS.labels(classe=result.diagnosis).inc()
    logger.info(
        json.dumps(
            {
                "event": "prediction",
                "endpoint": endpoint,
                "prediction": result.prediction,
                "diagnosis": result.diagnosis,
                "probability_malignant": round(result.probability_malignant, 6),
                "model": result.model,
            }
        )
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    start = time.perf_counter()
    request_status = "success"
    try:
        result = _run_prediction(request.features)
        _record_prediction(result, "/predict")
        return result
    except HTTPException as error:
        request_status = "invalid_features" if error.status_code == 422 else "model_unavailable"
        raise
    finally:
        REQUESTS.labels(endpoint="/predict", status=request_status).inc()
        PREDICTION_LATENCY.observe(time.perf_counter() - start)


@app.post("/interpret", response_model=InterpretResponse)
def interpret(request: PredictRequest) -> InterpretResponse:
    start = time.perf_counter()
    llm_model = os.getenv("GROQ_LLM_MODEL", DEFAULT_LLM_MODEL)
    status = "success"
    try:
        prediction = _run_prediction(request.features)
        _record_prediction(prediction, "/interpret")
        assert _artifact is not None
        evidence = derive_feature_evidence(_artifact, request.features)
        result = ModelResult(**prediction.model_dump())
        interpretation = generate_interpretation(result, evidence, model_name=llm_model)
        quality = evaluate_interpretation_quality(
            interpretation.explanation, prediction.diagnosis
        )
        INTERPRETATION_QUALITY.observe(float(quality["score_objetivo"]))
        logger.info(
            json.dumps(
                {
                    "event": "llm_interpretation",
                    "llm_model": interpretation.llm_model,
                    "prompt_version": interpretation.prompt_version,
                    "quality_score": quality["score_objetivo"],
                }
            )
        )
        return InterpretResponse(
            **prediction.model_dump(),
            explanation=interpretation.explanation,
            llm_model=interpretation.llm_model,
            prompt_version=interpretation.prompt_version,
            disclaimer=interpretation.disclaimer,
            evidence=[item.__dict__ for item in interpretation.evidence],
            quality_checks=quality,
        )
    except LLMUnavailableError as error:
        status = "llm_unavailable"
        raise HTTPException(status_code=503, detail=str(error)) from error
    except HTTPException as error:
        status = "invalid_features" if error.status_code == 422 else "model_unavailable"
        raise
    finally:
        REQUESTS.labels(endpoint="/interpret", status=status).inc()
        INTERPRETATIONS.labels(status=status, llm_model=llm_model).inc()
        INTERPRETATION_LATENCY.observe(time.perf_counter() - start)


def main() -> None:
    """Run the development API when this file is executed directly."""

    import uvicorn

    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
