"""Interpretacao em linguagem natural de resultados do classificador via LLM."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import inspect
import os
import re
from typing import Any
import unicodedata

import httpx

try:
    from src.model_inference import ServingModel, serving_model_from_joblib_artifact
except ModuleNotFoundError:
    from model_inference import ServingModel, serving_model_from_joblib_artifact


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
    """Indica que a LLM não pode ser usada no ambiente atual."""


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
    proximos_passos: str


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
    artifact: ServingModel | dict[str, Any],
    features: dict[str, float],
    top_k: int = 5,
) -> list[FeatureEvidence]:
    """Retorna contribuicoes locais sem expor todos os valores brutos de entrada."""

    serving_model = (
        artifact
        if isinstance(artifact, ServingModel)
        else serving_model_from_joblib_artifact(artifact)
    )
    feature_names = list(serving_model.feature_names)
    classes = list(serving_model.classes)
    positive_class = classes[1]
    contributions = serving_model.contributions(features)

    evidence: list[FeatureEvidence] = []
    strongest_indices = sorted(
        range(len(contributions)),
        key=lambda index: abs(contributions[index]),
        reverse=True,
    )[:top_k]
    for index in strongest_indices:
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
        if item.direction == "Maligno":
            proximos_passos = (
                "Sugerir agendamento de exames de imagem complementares "
                "(mamografia, ultrassonografia mamaria ou ressonancia magnetica "
                "conforme protocolo local) e, se pertinente, encaminhamento "
                "para mastologista. Considerar biopsia quando indicado por "
                "avaliacao clinica integrada. Orientar a paciente sobre a "
                "importancia do acompanhamento regular, respeitando a realidade "
                "de acesso ao sistema de saude."
            )
        else:
            proximos_passos = (
                "Recomendar manutencao do acompanhamento ginecologico ou "
                "mastologico periodico conforme faixa etaria e historico "
                "clinico da paciente. Avaliar necessidade de exames de rotina "
                "adicionais se houver outros fatores de risco nao capturados "
                "por este modelo. Orientar a paciente sobre sinais de alerta "
                "que justifiquem retorno antecipado ao servico de saude."
            )
        insights.append(
            ActionableInsight(
                sinal=sinal,
                evidencia_numerica=evidencia_numerica,
                implicacao_para_revisao=implicacao,
                cautela=cautela,
                proximos_passos=proximos_passos,
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
            f"  cautela: {item.cautela}\n"
            f"  proximos_passos: {item.proximos_passos}"
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


async def _call_groq(
    client: Any | None,
    llm_model: str,
    prompt: str,
    *,
    api_key: str | None = None,
    _retries: int = 3,
) -> str:
    api_key = api_key or os.getenv("GROQ_API_KEY")
    if client is None and not api_key:
        raise LLMUnavailableError(
            "Configure GROQ_API_KEY para gerar interpretacoes. "
            "Obtenha sua chave gratuita em console.groq.com/keys"
        )

    for attempt in range(_retries):
        try:
            messages = [
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ]
            if client is not None:
                response = client.chat.completions.create(
                    model=llm_model,
                    messages=messages,
                )
                if inspect.isawaitable(response):
                    response = await response
                content = response.choices[0].message.content
            else:
                async with httpx.AsyncClient(timeout=60.0) as http_client:
                    response = await http_client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json={"model": llm_model, "messages": messages},
                    )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
            return (content or "").strip()
        except LLMUnavailableError:
            raise
        except Exception as error:
            if "429" in str(error) and attempt < _retries - 1:
                retry_delay = _parse_retry_delay(str(error)) or 60
                await asyncio.sleep(retry_delay)
                continue
            raise LLMUnavailableError(
                f"Falha ao solicitar interpretacao ao modelo {llm_model}: {error}"
            ) from error
    raise LLMUnavailableError(
        f"Limite de tentativas esgotado para o modelo {llm_model}."
    )


def _parse_retry_delay(message: str) -> float | None:
    """Extrai o tempo sugerido de nova tentativa em segundos."""
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", message, re.IGNORECASE)
    return float(match.group(1)) if match else None


async def generate_interpretation(
    result: ModelResult,
    evidence: list[FeatureEvidence],
    client: Any | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
) -> LLMInterpretation:
    """Solicita uma explicacao clinica controlada ao backend de LLM."""

    llm_model = (
        model_name
        or os.getenv("GROQ_LLM_MODEL")
        or os.getenv("LLM_MODEL")
        or DEFAULT_LLM_MODEL
    )
    prompt = build_interpretation_prompt(result, evidence)
    explanation = await _call_groq(
        client,
        llm_model,
        prompt,
        api_key=api_key,
    )
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
        "menciona_resultado_esperado": _normalize_text(expected_diagnosis)
        in normalized,
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
        "preserva_privacidade": not (
            # CPF (xxx.xxx.xxx-xx)
            bool(re.search(r"\d{3}\.\d{3}\.\d{3}-\d{2}", interpretation))
            # RG, CNS (Cartão Nacional de Saúde), número de prontuário
            or bool(re.search(r"\b(rg|cns|prontuario|matricula)\s*[:#n]?\s*\d", normalized))
            # Nome completo presumido (3+ palavras em maiúsculas, típico de identificação pessoal)
            # Usa normalized (sem ** de markdown) para evitar falsos positivos em títulos de seção
            or bool(re.search(r"\b[a-záàâãéèêíìóòôõúùûç]{4,}(?:\s+[a-záàâãéèêíìóòôõúùûç]{4,}){2,}\b", normalized)
                    # Remove o falso positivo se o "nome" for parte de um título de seção
                    and not re.search(r"\b(resumo do resultado|evidencias do modelo|insights acionaveis para medicos|limitacoes e seguranca)\b", normalized))
            # Telefone (padrões brasileiros)
            or bool(re.search(r"\(\d{2}\)\s*\d{4,5}-\d{4}", interpretation))
            or bool(re.search(r"\b\d{4,5}-\d{4}\b", interpretation))
            # Endereço (rua, av., travessa etc.)
            or bool(re.search(r"\b(rua|avenida|travessa|praca|alameda|rodovia|estrada)\s+[\w\s]+[,.]?\s*\d+", normalized))
            # Data de nascimento
            or bool(re.search(r"\b\d{2}/\d{2}/\d{4}\b.*\b(nascimento|nasceu)\b", normalized))
        ),
        "inclui_proximos_passos_praticos": any(
            term in normalized
            for term in (
                "agendamento",
                "encaminhamento",
                "mastologista",
                "ginecologista",
                "exames complementares",
                "exame complementar",
                "mamografia",
                "ultrassonografia",
                "ressonancia magnetica",
                "biopsia",
                "acompanhamento",
                "retorno",
                "proximo passo",
                "proximos passos",
                "seguimento",
                "consulta",
                "protocolo local",
                "sistema de saude",
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
