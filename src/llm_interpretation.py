"""Natural-language interpretation of classifier results through an LLM."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import re
import time
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_LLM_MODEL = "llama-3.1-8b-instant"
DISCLAIMER = (
    "Interpretacao gerada por IA para apoio a revisao profissional. "
    "Nao constitui diagnostico medico nem recomendacao terapeutica."
)

SYSTEM_INSTRUCTIONS = """Voce e um assistente de apoio a interpretacao de um modelo academico de cancer de mama.
Escreva em portugues do Brasil para um profissional de saude.

Regras obrigatorias:
- Explique somente os dados fornecidos; nao invente achados clinicos, historico ou exames.
- Chame a saida de "resultado do modelo" ou "classificacao estimada", nunca de diagnostico confirmado.
- Nao prescreva tratamento nem afirme conduta clinica definitiva.
- Se a probabilidade maligna for alta, priorize revisao clinica e confirmacao diagnostica conforme protocolo local.
- Se a probabilidade benigna for alta, informe que resultados benignos nao excluem avaliacao clinica.
- Mencione que o dataset e academico e que nao houve validacao externa.
- Use linguagem objetiva, sem alarmismo.

Retorne exatamente estas secoes:
RESUMO DO RESULTADO
EVIDENCIAS DO MODELO
PONTOS PARA REVISAO CLINICA
LIMITACOES E SEGURANCA
"""


class LLMUnavailableError(RuntimeError):
    """Raised when the LLM cannot be used in the current environment."""


@dataclass(frozen=True)
class FeatureEvidence:
    """One local contribution derived from a logistic regression pipeline."""

    feature: str
    value: float
    contribution: float
    direction: str


@dataclass(frozen=True)
class ModelResult:
    """Prediction facts shared with the LLM."""

    prediction: int
    diagnosis: str
    probability_malignant: float
    probability_benign: float
    model: str


@dataclass(frozen=True)
class LLMInterpretation:
    """Text produced by the selected LLM plus audit metadata."""

    explanation: str
    llm_model: str
    prompt_version: str
    disclaimer: str
    evidence: list[FeatureEvidence]


PROMPT_VERSION = "clinical_explanation_v1"


def derive_feature_evidence(
    artifact: dict[str, Any], features: dict[str, float], top_k: int = 5
) -> list[FeatureEvidence]:
    """Return local logistic contributions without exposing all raw input values."""

    pipeline = artifact["model"]
    model = pipeline.named_steps.get("model")
    scaler = pipeline.named_steps.get("scaler")
    if scaler is None or not hasattr(model, "coef_"):
        return []

    feature_names = list(artifact["feature_names"])
    row = pd.DataFrame([{name: features[name] for name in feature_names}])
    transformed = scaler.transform(row)[0]
    # In sklearn binary logistic regression coef_ points toward classes_[1].
    coefficients = model.coef_[0]
    classes = list(model.classes_)
    positive_class = classes[1]
    contributions = transformed * coefficients

    evidence: list[FeatureEvidence] = []
    for index in np.argsort(np.abs(contributions))[::-1][:top_k]:
        value = float(features[feature_names[index]])
        contribution = float(contributions[index])
        moves_to_positive = contribution >= 0
        favored_class = positive_class if moves_to_positive else classes[0]
        direction = "Benigno" if favored_class == 1 else "Maligno"
        evidence.append(
            FeatureEvidence(
                feature=feature_names[index],
                value=value,
                contribution=contribution,
                direction=direction,
            )
        )
    return evidence


def build_interpretation_prompt(
    result: ModelResult, evidence: list[FeatureEvidence]
) -> str:
    """Create a compact, traceable prompt with only computed inference context."""

    evidence_lines = (
        "\n".join(
            (
                f"- {item.feature}: valor={item.value:.5g}; "
                f"contribuicao_local={item.contribution:.4f}; "
                f"direcao={item.direction}"
            )
            for item in evidence
        )
        if evidence
        else "- O modelo nao disponibiliza contribuicoes locais neste artefato."
    )
    return f"""Contexto do resultado a interpretar:
- Modelo: {result.model}
- Classificacao estimada: {result.diagnosis} ({result.prediction})
- Probabilidade estimada de malignidade: {result.probability_malignant:.2%}
- Probabilidade estimada de benignidade: {result.probability_benign:.2%}

Principais evidencias numericas derivadas do modelo:
{evidence_lines}

Produza uma explicacao concisa para apoiar revisao medica, observando integralmente as regras.
"""


def _groq_client() -> Any:
    if not os.getenv("GROQ_API_KEY"):
        raise LLMUnavailableError(
            "Configure GROQ_API_KEY para gerar interpretacoes. "
            "Obtenha sua chave gratuita em console.groq.com/keys"
        )
    try:
        from groq import Groq
    except ImportError as error:
        raise LLMUnavailableError(
            "Dependencia `groq` ausente. Execute: pip install groq"
        ) from error
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


def _call_groq(client: Any, llm_model: str, prompt: str, *, _retries: int = 3) -> str:
    for attempt in range(_retries):
        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": prompt},
                ],
            )
            return (response.choices[0].message.content or "").strip()
        except LLMUnavailableError:
            raise
        except Exception as error:
            if "429" in str(error) and attempt < _retries - 1:
                retry_delay = _parse_retry_delay(str(error)) or 60
                time.sleep(retry_delay)
                continue
            raise LLMUnavailableError(
                f"Falha ao solicitar interpretacao ao modelo {llm_model}: {error}"
            ) from error
    raise LLMUnavailableError(f"Limite de tentativas esgotado para o modelo {llm_model}.")


def _parse_retry_delay(message: str) -> float | None:
    """Extract the suggested retry delay in seconds from a quota error message."""
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", message, re.IGNORECASE)
    return float(match.group(1)) if match else None


def generate_interpretation(
    result: ModelResult,
    evidence: list[FeatureEvidence],
    client: Any | None = None,
    model_name: str | None = None,
) -> LLMInterpretation:
    """Request a controlled clinical-facing explanation through the LLM backend."""

    llm_model = model_name or os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
    prompt = build_interpretation_prompt(result, evidence)
    explanation = _call_groq(client or _groq_client(), llm_model, prompt)
    if not explanation:
        raise LLMUnavailableError("A LLM nao retornou texto de interpretacao.")
    return LLMInterpretation(
        explanation=explanation,
        llm_model=llm_model,
        prompt_version=PROMPT_VERSION,
        disclaimer=DISCLAIMER,
        evidence=evidence,
    )


def evaluate_interpretation_quality(
    interpretation: str, expected_diagnosis: str
) -> dict[str, bool | float]:
    """Apply objective checks to support human evaluation of LLM output quality."""

    normalized = interpretation.lower()
    required_sections = (
        "resumo do resultado",
        "evidencias do modelo",
        "pontos para revisao clinica",
        "limitacoes e seguranca",
    )
    checks: dict[str, bool] = {
        "menciona_resultado_esperado": expected_diagnosis.lower() in normalized,
        "inclui_probabilidade": bool(re.search(r"\d+(?:[,.]\d+)?\s*%", interpretation)),
        "inclui_secoes_obrigatorias": all(
            section in normalized for section in required_sections
        ),
        "declara_limitacao": any(
            term in normalized
            for term in ("nao constitui diagnostico", "não constitui diagnóstico", "nao e diagnostico")
        ),
        "orienta_revisao_profissional": any(
            term in normalized
            for term in ("revisao clinica", "revisão clínica", "profissional", "medic")
        ),
        "nao_prescreve_tratamento": not any(
            term in normalized
            for term in ("prescrevo", "iniciar quimioterapia", "iniciar radioterapia")
        ),
    }
    score = sum(checks.values()) / len(checks)
    return {**checks, "score_objetivo": score}


def interpretation_to_dict(interpretation: LLMInterpretation) -> dict[str, Any]:
    """Serialize an interpretation for JSON responses or audit artifacts."""

    return {
        "explanation": interpretation.explanation,
        "llm_model": interpretation.llm_model,
        "prompt_version": interpretation.prompt_version,
        "disclaimer": interpretation.disclaimer,
        "evidence": [asdict(item) for item in interpretation.evidence],
    }
