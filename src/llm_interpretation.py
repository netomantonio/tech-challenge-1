"""Interpretacao em linguagem natural de resultados do classificador via LLM."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import re
import time
from typing import Any
import unicodedata

import numpy as np
import pandas as pd


DEFAULT_LLM_MODEL = "openai/gpt-oss-120b"
DISCLAIMER = (
    "Interpretacao gerada por IA para apoio a revisao profissional. "
    "Nao constitui diagnostico medico nem recomendacao terapeutica."
)

SYSTEM_INSTRUCTIONS = """Voce e um assistente de apoio a interpretacao de um modelo academico de cancer de mama,
um contexto de saude da mulher. Escreva em portugues do Brasil para um profissional de saude.

Regras obrigatorias:
- Explique somente os dados fornecidos; nao invente achados clinicos, historico, exames ou dados pessoais da paciente.
- Chame a saida de "resultado do modelo" ou "classificacao estimada", nunca de diagnostico confirmado.
- Nao prescreva tratamento nem afirme conduta clinica definitiva.
- Transforme os numeros em insights acionaveis para revisao medica, sem extrapolar conduta.
- Se a probabilidade maligna for alta, priorize revisao clinica e confirmacao diagnostica conforme protocolo local.
- Se a probabilidade benigna for alta, informe que resultados benignos nao excluem avaliacao clinica.
- Mencione que o dataset e academico e que nao houve validacao externa.
- Use linguagem objetiva, sem alarmismo, sem termos sentenciosos ou estigmatizantes (evite, por exemplo,
  "sentenca de morte", "fatal", "sem cura", "tragico", "vai morrer").
- Considere o contexto de saude da mulher: trate o caso com sensibilidade de genero e, quando pertinente,
  situe os achados dentro de cuidados tipicos da saude da mulher (ex.: acompanhamento ginecologico ou
  mastologico periodico), sem presumir dados que nao foram fornecidos.
- Preserve privacidade e confidencialidade: nao solicite, nao infira e nao inclua identificadores pessoais
  (nome, idade exata, endereco, numero de prontuario ou qualquer dado que possa identificar a paciente).
- Ao descrever os insights acionaveis, inclua tambem orientacoes praticas de proximos passos (por exemplo,
  agendamento de exames complementares, encaminhamento ou acompanhamento) pensando na realidade de acesso
  da paciente ao sistema de saude, sem nunca prescrever tratamento ou conduta definitiva.

Retorne exatamente estas secoes:
RESUMO DO RESULTADO
EVIDENCIAS DO MODELO
INSIGHTS ACIONAVEIS PARA MEDICOS
LIMITACOES E SEGURANCA
"""


class LLMUnavailableError(RuntimeError):
    """Raised when the LLM cannot be used in the current environment."""


@dataclass(frozen=True)
class FeatureEvidence:
    """Uma contribuicao local derivada do pipeline de regressao logistica."""

    feature: str
    value: float
    contribution: float
    direction: str


@dataclass(frozen=True)
class ModelResult:
    """Fatos da predicao compartilhados com a LLM."""

    prediction: int
    diagnosis: str
    probability_malignant: float
    probability_benign: float
    model: str


@dataclass(frozen=True)
class ActionableInsight:
    """Traducao estruturada de uma evidencia numerica para revisao medica."""

    sinal: str
    evidencia_numerica: str
    implicacao_para_revisao: str
    cautela: str


@dataclass(frozen=True)
class LLMInterpretation:
    """Texto produzido pela LLM selecionada mais metadados de auditoria."""

    explanation: str
    llm_model: str
    prompt_version: str
    disclaimer: str
    evidence: list[FeatureEvidence]
    insights_acionaveis: list[ActionableInsight]


PROMPT_VERSION = "clinical_explanation_v3"


def derive_feature_evidence(
    artifact: dict[str, Any], features: dict[str, float], top_k: int = 5
) -> list[FeatureEvidence]:
    """Retorna contribuicoes locais sem expor todos os valores brutos de entrada."""

    pipeline = artifact["model"]
    model = pipeline.named_steps.get("model")
    scaler = pipeline.named_steps.get("scaler")
    if scaler is None or not hasattr(model, "coef_"):
        return []

    feature_names = list(artifact["feature_names"])
    row = pd.DataFrame([{name: features[name] for name in feature_names}])
    transformed = scaler.transform(row)[0]
    # Em classificacao binaria, coef_ aponta para classes_[1] no scikit-learn.
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


def derive_actionable_insights(
    result: ModelResult, evidence: list[FeatureEvidence], top_k: int = 5
) -> list[ActionableInsight]:
    """Converte evidencias numericas em pontos estruturados de revisao medica."""

    insights: list[ActionableInsight] = []
    for item in evidence[:top_k]:
        magnitude = abs(item.contribution)
        intensidade = (
            "forte"
            if magnitude >= 2.0
            else "moderada"
            if magnitude >= 0.75
            else "baixa"
        )
        sinal = (
            f"{item.feature} apresentou sinal {intensidade} na direcao "
            f"{item.direction}."
        )
        evidencia_numerica = (
            f"valor={item.value:.5g}; contribuicao_local={item.contribution:.4f}; "
            f"probabilidade_maligna={result.probability_malignant:.2%}."
        )
        if item.direction == "Maligno":
            implicacao = (
                "Priorizar revisao deste achado junto aos exames e ao historico "
                "clinico, pois ele aumenta o peso estatistico para malignidade."
            )
        else:
            implicacao = (
                "Verificar se este achado e coerente com sinais de menor risco, "
                "sem descartar investigacao quando outros sinais forem conflitantes."
            )
        cautela = (
            "Nao interpretar esta evidencia isoladamente; a contribuicao e "
            "estatistica, nao causal, e depende do conjunto de variaveis do modelo."
        )
        insights.append(
            ActionableInsight(
                sinal=sinal,
                evidencia_numerica=evidencia_numerica,
                implicacao_para_revisao=implicacao,
                cautela=cautela,
            )
        )
    return insights


def build_interpretation_prompt(
    result: ModelResult, evidence: list[FeatureEvidence]
) -> str:
    """Cria um prompt compacto e rastreavel com contexto numerico calculado."""

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
    insight_lines = "\n".join(
        (
            f"- sinal: {item.sinal}\n"
            f"  evidencia_numerica: {item.evidencia_numerica}\n"
            f"  implicacao_para_revisao: {item.implicacao_para_revisao}\n"
            f"  cautela: {item.cautela}"
        )
        for item in derive_actionable_insights(result, evidence)
    )
    return f"""Contexto do resultado a interpretar:
- Modelo: {result.model}
- Classificacao estimada: {result.diagnosis} ({result.prediction})
- Probabilidade estimada de malignidade: {result.probability_malignant:.2%}
- Probabilidade estimada de benignidade: {result.probability_benign:.2%}

Principais evidencias numericas derivadas do modelo:
{evidence_lines}

Insights acionaveis estruturados que devem orientar a explicacao:
{insight_lines if insight_lines else "- Sem insights estruturados disponiveis para este artefato."}

Produza uma explicacao concisa para apoiar revisao medica, observando integralmente as regras.
Na secao INSIGHTS ACIONAVEIS PARA MEDICOS, transforme as evidencias em acoes de revisao, nao em condutas terapeuticas.
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
    """Extrai o tempo sugerido de nova tentativa em segundos."""
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", message, re.IGNORECASE)
    return float(match.group(1)) if match else None


def generate_interpretation(
    result: ModelResult,
    evidence: list[FeatureEvidence],
    client: Any | None = None,
    model_name: str | None = None,
) -> LLMInterpretation:
    """Solicita uma explicacao clinica controlada ao backend de LLM."""

    llm_model = (
        model_name
        or os.getenv("GROQ_LLM_MODEL")
        or os.getenv("LLM_MODEL")
        or DEFAULT_LLM_MODEL
    )
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
        insights_acionaveis=derive_actionable_insights(result, evidence),
    )


def _normalize_text(text: str) -> str:
    without_accents = "".join(
        char
        for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    )
    normalized = without_accents.lower()
    normalized = re.sub(r"[*_`#|:;\[\]()>-]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def evaluate_interpretation_quality(
    interpretation: str, expected_diagnosis: str
) -> dict[str, bool | float]:
    """Aplica checks objetivos para apoiar avaliacao humana da interpretacao."""

    normalized = _normalize_text(interpretation)
    required_section_groups = (
        ("resumo do resultado",),
        ("evidencias do modelo",),
        ("insights acionaveis para medicos", "pontos para revisao clinica"),
        ("limitacoes e seguranca",),
    )
    checks: dict[str, bool] = {
        "menciona_resultado_esperado": _normalize_text(expected_diagnosis) in normalized,
        "inclui_probabilidade": bool(re.search(r"\d+(?:[,.]\d+)?\s*%", interpretation)),
        "inclui_secoes_obrigatorias": all(
            any(section in normalized for section in group)
            for group in required_section_groups
        ),
        "inclui_insights_acionaveis": any(
            term in normalized
            for term in (
                "insights acionaveis",
                "pontos para revisao clinica",
                "implicacao para revisao",
                "revisao medica",
            )
        ),
        "declara_limitacao": any(
            term in normalized
            for term in (
                "nao constitui diagnostico",
                "nao e diagnostico",
                "nao substitui avaliacao",
                "nao substitui a avaliacao",
                "nao substitui o diagnostico",
                "sem validacao externa",
                "nao validado externamente",
                "nao foi validado externamente",
                "dataset academico",
                "conjunto de dados academico",
                "academico e nao",
            )
        ),
        "orienta_revisao_profissional": any(
            term in normalized
            for term in ("revisao clinica", "revisao medica", "profissional", "medic")
        ),
        "nao_prescreve_tratamento": not any(
            term in normalized
            for term in (
                "prescrevo",
                "iniciar quimioterapia",
                "iniciar radioterapia",
                "deve iniciar tratamento",
                "tratamento indicado e",
            )
        ),
        "sensibilidade_cultural_e_genero": not any(
            term in normalized
            for term in (
                "sentenca de morte",
                "fatal",
                "sem cura",
                "tragico",
                "desesperador",
                "doente terminal",
                "vai morrer",
            )
        ),
    }
    score = sum(checks.values()) / len(checks)
    return {**checks, "score_objetivo": score}


def interpretation_to_dict(interpretation: LLMInterpretation) -> dict[str, Any]:
    """Serializa uma interpretacao para respostas JSON ou artefatos de auditoria."""

    return {
        "explanation": interpretation.explanation,
        "llm_model": interpretation.llm_model,
        "prompt_version": interpretation.prompt_version,
        "disclaimer": interpretation.disclaimer,
        "evidence": [asdict(item) for item in interpretation.evidence],
        "insights_acionaveis": [
            asdict(item) for item in interpretation.insights_acionaveis
        ],
    }
