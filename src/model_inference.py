"""Inferência portátil do modelo recomendado para ambientes locais e Workers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ServingModel:
    """Parâmetros imutáveis do pipeline de serving."""

    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    classes: tuple[int, int]
    model_label: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        quantidade = len(self.feature_names)
        if quantidade != 30:
            raise ValueError(
                f"O modelo deve declarar 30 features; recebeu {quantidade}."
            )
        if not all(
            len(values) == quantidade
            for values in (self.means, self.scales, self.coefficients)
        ):
            raise ValueError(
                "Médias, escalas e coeficientes devem acompanhar as features."
            )
        if any(scale <= 0 for scale in self.scales):
            raise ValueError("Todas as escalas do modelo devem ser positivas.")
        if len(self.classes) != 2:
            raise ValueError("O classificador publicado deve ser binário.")

    def contributions(self, features: dict[str, float]) -> list[float]:
        """Calcula as contribuições padronizadas usadas pela regressão logística."""

        return [
            ((float(features[name]) - mean) / scale) * coefficient
            for name, mean, scale, coefficient in zip(
                self.feature_names,
                self.means,
                self.scales,
                self.coefficients,
                strict=True,
            )
        ]

    def predict(self, features: dict[str, float]) -> tuple[int, float, float]:
        """Retorna classe, probabilidade maligna e probabilidade benigna."""

        score = self.intercept + sum(self.contributions(features))
        if score >= 0:
            probability_positive = 1.0 / (1.0 + math.exp(-score))
        else:
            exp_score = math.exp(score)
            probability_positive = exp_score / (1.0 + exp_score)

        probabilities = {
            self.classes[0]: 1.0 - probability_positive,
            self.classes[1]: probability_positive,
        }
        prediction = self.classes[1] if score > 0 else self.classes[0]
        return prediction, probabilities[0], probabilities[1]


def _number_tuple(values: list[Any]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def serving_model_from_manifest(payload: dict[str, Any]) -> ServingModel:
    """Valida e converte o manifesto JSON em um modelo executável."""

    if payload.get("schema_version") != 1:
        raise ValueError("Versão de manifesto de modelo não suportada.")
    preprocessing = payload.get("preprocessing", {})
    classifier = payload.get("classifier", {})
    if preprocessing.get("type") != "StandardScaler":
        raise ValueError("O manifesto deve usar StandardScaler.")
    if classifier.get("type") != "LogisticRegression":
        raise ValueError("O manifesto deve usar LogisticRegression.")

    return ServingModel(
        feature_names=tuple(str(name) for name in payload["feature_names"]),
        means=_number_tuple(preprocessing["mean"]),
        scales=_number_tuple(preprocessing["scale"]),
        coefficients=_number_tuple(classifier["coefficients"]),
        intercept=float(classifier["intercept"]),
        classes=tuple(int(value) for value in classifier["classes"]),
        model_label=str(payload["model_label"]),
        metadata={
            key: value
            for key, value in payload.items()
            if key not in {"feature_names", "preprocessing", "classifier"}
        },
    )


def serving_model_from_joblib_artifact(artifact: dict[str, Any]) -> ServingModel:
    """Converte o artefato legado sem alterar seus parâmetros matemáticos."""

    pipeline = artifact["model"]
    scaler = pipeline.named_steps["scaler"]
    classifier = pipeline.named_steps["model"]
    return ServingModel(
        feature_names=tuple(str(name) for name in artifact["feature_names"]),
        means=tuple(float(value) for value in scaler.mean_),
        scales=tuple(float(value) for value in scaler.scale_),
        coefficients=tuple(float(value) for value in classifier.coef_[0]),
        intercept=float(classifier.intercept_[0]),
        classes=tuple(int(value) for value in classifier.classes_),
        model_label=str(artifact["model_label"]),
        metadata={
            "schema_version": 1,
            "model_key": artifact.get("model_key"),
            "source": artifact.get("source"),
            "target_mapping": artifact.get("target_mapping"),
            "parameters": artifact.get("parameters"),
        },
    )


def load_serving_model(path: Path) -> ServingModel:
    """Carrega o manifesto portátil ou, localmente, o joblib legado."""

    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em {path}.")
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as file:
            return serving_model_from_manifest(json.load(file))

    try:
        import joblib
    except ImportError as error:
        raise ValueError("O runtime atual não suporta artefatos joblib.") from error
    return serving_model_from_joblib_artifact(joblib.load(path))
