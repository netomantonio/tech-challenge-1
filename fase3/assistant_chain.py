"""Pipeline LangChain do assistente medico virtual (Fase 3).

Monta uma chain LCEL (``prompt | llm | StrOutputParser``) que integra:

1. Retrieval sobre os protocolos internos (``fase3.retrieval``);
2. Contexto estruturado do paciente vindo do mock de EHR (``fase3.ehr_tools``);
3. O LLM plugavel (``fase3.llm_backend``, modelo local por padrao);
4. Guardrails de seguranca e disclaimer obrigatorio (``fase3.guardrails``);
5. Log de auditoria com as fontes usadas, para explainability (``fase3.logging_utils``).
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Optional

from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from fase3.ehr_tools import get_paciente
from fase3.guardrails import aplicar_guardrails
from fase3.llm_backend import get_llm
from fase3.logging_utils import registrar_interacao
from fase3.prompting import USER_PROMPT_TEMPLATE, formatar_protocolos_prompt
from fase3.retrieval import DEFAULT_TOP_K, BM25Retriever, buscar_protocolos, construir_retriever, formatar_fontes

PROMPT_VERSION = "assistente_medico_v2"

_WORD_RE = re.compile(r"[a-zA-ZÀ-ÿ0-9-]+")
_PROTOCOL_RE = re.compile(r"\bPROT-\d{3}\b", re.IGNORECASE)
_PATIENT_RE = re.compile(r"\bPAC-\d{4}\b", re.IGNORECASE)
_PROTOCOL_LIKE_RE = re.compile(
    r"\[?(?:PROT(?:OCOLOS?)?)[-_\s]*(?:\d{1,3}|[A-Z]+)\]?",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_UNSAFE_SCRIPT_RE = re.compile(r"[\u3400-\u9fff]")
_STOPWORDS = {
    "a", "ao", "aos", "as", "com", "da", "das", "de", "do", "dos",
    "e", "em", "esta", "este", "foi", "o", "os", "para", "por", "que",
    "se", "sem", "ser", "um", "uma", "na", "no", "nas", "nos",
    "fonte", "fontes", "consultada", "consultadas",
}

PROMPT = PromptTemplate.from_template(USER_PROMPT_TEMPLATE)


def construir_chain(llm: LLM) -> Runnable:
    """Monta a chain LCEL prompt -> llm -> texto, reutilizavel com qualquer backend."""
    return PROMPT | llm | StrOutputParser()


def _formatar_protocolos(documentos) -> str:
    return formatar_protocolos_prompt(
        {
            "id": doc.metadata["id"],
            "titulo": doc.metadata["titulo"],
            "conteudo": doc.page_content,
        }
        for doc in documentos
    )


def _formatar_contexto_paciente(paciente_id: Optional[str]) -> str:
    if not paciente_id:
        return "Nenhum paciente informado nesta consulta."
    paciente = get_paciente(paciente_id)
    if paciente is None:
        return f"Codigo de paciente {paciente_id} nao encontrado no prontuario."
    return (
        f"Paciente {paciente['paciente_id']} ({paciente['idade']} anos, {paciente['sexo']}). "
        f"Diagnostico: {paciente['diagnostico']}. Estagio: {paciente['estagio']}. "
        f"Exames pendentes: {', '.join(paciente['exames_pendentes']) or 'nenhum'}. "
        f"Alertas ativos: {', '.join(paciente['alertas_ativos']) or 'nenhum'}."
    )


def _normalizar(texto: str) -> str:
    sem_acentos = "".join(
        char
        for char in unicodedata.normalize("NFKD", texto)
        if not unicodedata.combining(char)
    ).lower()
    return " ".join(_WORD_RE.findall(sem_acentos))


def _extrair_numeros(texto: str) -> set[str]:
    sem_ids = _PATIENT_RE.sub("", _PROTOCOL_RE.sub("", texto))
    return {numero.replace(",", ".") for numero in _NUMBER_RE.findall(sem_ids)}


def _avaliar_grounding(
    resposta: str,
    pergunta: str,
    documentos,
    paciente: Optional[dict],
    paciente_id: Optional[str] = None,
) -> list[str]:
    """Retorna os motivos que tornam a geracao inadequada para exibicao."""
    motivos: list[str] = []
    tokens = _WORD_RE.findall(resposta.lower())
    if len(tokens) < 12:
        motivos.append("resposta_curta")
    if _UNSAFE_SCRIPT_RE.search(resposta):
        motivos.append("caracteres_inesperados")

    ids_recuperados = {doc.metadata["id"] for doc in documentos}
    ids_citados = {item.upper() for item in _PROTOCOL_RE.findall(resposta)}
    fonte_obrigatoria = not (paciente_id and paciente is None)
    if fonte_obrigatoria:
        if not ids_citados:
            motivos.append("fonte_ausente")
        elif not ids_citados <= ids_recuperados:
            motivos.append("fonte_nao_recuperada")

    contexto = " ".join(
        [
            pergunta,
            *(doc.page_content for doc in documentos),
            json.dumps(paciente or {}, ensure_ascii=False),
        ]
    )
    if not _extrair_numeros(resposta) <= _extrair_numeros(contexto):
        motivos.append("valor_numerico_inventado")

    termos_resposta = {
        token for token in _normalizar(resposta).split() if token not in _STOPWORDS and len(token) > 2
    }
    termos_contexto = set(_normalizar(contexto).split())
    cobertura = (
        sum(token in termos_contexto for token in termos_resposta) / len(termos_resposta)
        if termos_resposta
        else 0.0
    )
    if cobertura < 0.25:
        motivos.append("baixa_aderencia_ao_contexto")

    pergunta_normalizada = _normalizar(pergunta)
    tokens_pergunta = set(pergunta_normalizada.split())
    resposta_normalizada = _normalizar(resposta)
    if documentos and any(
        trecho in resposta_normalizada
        for trecho in (
            "nao posso responder",
            "nao possui capacidade",
            "incapaz de responder",
        )
    ):
        motivos.append("resposta_evasiva")

    if paciente_id and paciente is None and not any(
        trecho in resposta_normalizada
        for trecho in ("nao encontrado", "nao foi encontrado", "sem contexto")
    ):
        motivos.append("paciente_nao_encontrado_ignorado")

    pendentes = (paciente or {}).get("exames_pendentes", [])
    consulta_pre_tratamento = "quimioterapia" in pergunta_normalizada or (
        bool(pendentes)
        and any(
            termo in pergunta_normalizada
            for termo in ("ciclo", "tratamento", "liberar", "autorizacao")
        )
    )
    if consulta_pre_tratamento:
        if pendentes and not (
            "pendente" in resposta_normalizada
            and any(
                trecho in resposta_normalizada
                for trecho in (
                    "nao inicie",
                    "nao iniciar",
                    "nenhum ciclo",
                    "antes de iniciar",
                    "nao cumpre",
                    "deve aguardar",
                    "adiar",
                )
            )
        ):
            motivos.append("pendencias_pre_quimioterapia_ignoradas")
        elif "exame" in pergunta_normalizada and not paciente and not (
            "hemograma" in resposta_normalizada
            and "renal" in resposta_normalizada
            and "hepatica" in resposta_normalizada
        ):
            motivos.append("exames_pre_quimioterapia_incompletos")

    if any(termo in pergunta_normalizada for termo in ("bi-rads", "birads", "biopsia")):
        birads_pergunta = re.search(r"bi[-\s]?rads\s*(\d+)", pergunta, re.IGNORECASE)
        birads_resposta = re.search(r"bi[-\s]?rads\s*(\d+)", resposta, re.IGNORECASE)
        if (
            birads_pergunta
            and birads_resposta
            and birads_pergunta.group(1) != birads_resposta.group(1)
        ):
            motivos.append("classificacao_birads_alterada")
        if not (
            any(termo in resposta_normalizada for termo in ("bi-rads", "birads"))
            and "biopsia" in resposta_normalizada
        ):
            motivos.append("conduta_birads_incompleta")

    if any(termo in pergunta_normalizada for termo in ("febre", "taquicardia", "sepse")):
        if not (
            any(termo in resposta_normalizada for termo in ("sepse", "sirs"))
            and any(termo in resposta_normalizada for termo in ("imediat", "acione", "acionamento"))
        ):
            motivos.append("conduta_sepse_incompleta")

    if "dor" in tokens_pergunta:
        escala = re.search(r"(\d+(?:[.,]\d+)?)\s*/\s*10", pergunta)
        escala_preservada = not escala or re.search(
            rf"{re.escape(escala.group(1))}\s*/\s*10", resposta
        )
        if not escala_preservada and not any(
            termo in resposta_normalizada for termo in ("intensa", "grave")
        ):
            motivos.append("intensidade_dor_ignorada")
        if not any(
            termo in resposta_normalizada for termo in ("reavaliacao", "equipe cirurgica")
        ):
            motivos.append("conduta_dor_incompleta")

    bigramas = list(zip(tokens, tokens[1:]))
    if bigramas and 1 - len(set(bigramas)) / len(bigramas) > 0.30:
        motivos.append("repeticao_excessiva")
    return motivos


def _resposta_fallback_segura(
    pergunta: str,
    paciente_id: Optional[str],
    paciente: Optional[dict],
    documentos,
) -> str:
    """Resposta deterministica e fundamentada usada quando a LLM falha no grounding."""
    if paciente_id and paciente is None:
        return (
            f"O codigo {paciente_id} nao foi encontrado no prontuario. Nao e seguro "
            "sugerir conduta sem contexto clinico confirmado; verifique o identificador "
            "com a equipe responsavel."
        )

    contexto_decisorio = " ".join(
        [
            pergunta,
            (paciente or {}).get("observacoes", ""),
            *((paciente or {}).get("exames_pendentes", [])),
            *((paciente or {}).get("alertas_ativos", [])),
        ]
    )
    normalizada = _normalizar(contexto_decisorio)
    ids = {doc.metadata["id"] for doc in documentos}
    if "PROT-011" in ids and any(termo in normalizada for termo in ("febre", "taquicardia", "sepse")):
        alertas = "; ".join((paciente or {}).get("alertas_ativos", []))
        contexto_alerta = f"O prontuario registra: {alertas}. " if alertas else ""
        return (
            f"{contexto_alerta}O protocolo determina acionamento imediato do fluxo de "
            "sepse e comunicacao a equipe medica quando houver criterios de SIRS "
            "associados a suspeita de infeccao. Nao aguarde validacao assincrona. "
            "Fonte: [PROT-011]."
        )

    if "PROT-006" in ids and any(termo in normalizada for termo in ("quimioterapia", "exame")):
        pendentes = (paciente or {}).get("exames_pendentes", [])
        if pendentes:
            return (
                f"Ha exames pendentes no prontuario: {', '.join(pendentes)}. O protocolo "
                "estabelece que nenhum ciclo deve ser iniciado enquanto exames "
                "obrigatorios estiverem pendentes ou fora da validade; confirme-os com "
                "a equipe medica. Fonte: [PROT-006]."
            )
        return (
            "Antes da quimioterapia, confirme hemograma completo, funcao hepatica e "
            "renal, sorologias HBV, HCV e HIV e, quando aplicavel, ecocardiograma ou "
            "MUGA basal. Valide a liberacao com o medico responsavel. Fonte: [PROT-006]."
        )

    if "PROT-004" in ids and "dor" in normalizada:
        contexto_dor = " ".join(
            [pergunta, *((paciente or {}).get("alertas_ativos", []))]
        )
        escala = re.search(r"(\d+(?:[.,]\d+)?)\s*/\s*10", contexto_dor)
        intensidade = (
            f" com intensidade registrada de {escala.group(1)}/10" if escala else ""
        )
        return (
            f"Dor pos-operatoria persistente{intensidade} exige reavaliacao clinica, registro da "
            "intensidade e comunicacao a equipe cirurgica. Siga o protocolo de manejo "
            "da dor sem indicar medicamento ou dose automaticamente. Fonte: [PROT-004]."
        )

    if "PROT-001" in ids and any(termo in normalizada for termo in ("bi-rads", "birads", "biopsia")):
        return (
            "Um achado BI-RADS 4 requer confirmacao histopatologica por biopsia e "
            "revisao da equipe assistente antes da definicao de tratamento. "
            "Fonte: [PROT-001]."
        )

    fonte = documentos[0] if documentos else None
    if fonte:
        primeira_frase = fonte.page_content.split(".", 1)[0].strip()
        return (
            f"O protocolo recuperado informa: {primeira_frase}. Confirme a aplicacao "
            f"ao caso com o medico responsavel. Fonte: [{fonte.metadata['id']}]."
        )
    return (
        "Nao ha informacao suficiente nos protocolos recuperados para responder com "
        "seguranca. Encaminhe a duvida para avaliacao da equipe medica."
    )


def responder_pergunta_clinica(
    pergunta: str,
    paciente_id: Optional[str] = None,
    llm: Optional[LLM] = None,
    retriever: Optional[BM25Retriever] = None,
    top_k: int = DEFAULT_TOP_K,
    incluir_diagnostico: bool = False,
) -> dict:
    """Responde uma pergunta clinica com contexto de protocolos e do paciente.

    Retorna um dicionario com ``resposta`` (ja com guardrails/disclaimer
    aplicados), ``fontes`` (para explainability), ``bloqueado`` e o motivo do
    bloqueio quando aplicavel.
    """
    llm = llm or get_llm()
    retriever = retriever or construir_retriever(k=top_k)

    paciente = get_paciente(paciente_id) if paciente_id else None
    consulta_retrieval = pergunta
    if paciente:
        consulta_retrieval = " ".join(
            [
                pergunta,
                paciente.get("diagnostico", ""),
                paciente.get("observacoes", ""),
                *paciente.get("exames_pendentes", []),
                *paciente.get("alertas_ativos", []),
            ]
        )
    documentos = buscar_protocolos(consulta_retrieval, retriever)
    fontes = formatar_fontes(documentos)

    chain = construir_chain(llm)
    resposta_bruta = chain.invoke(
        {
            "pergunta": pergunta,
            "protocolos": _formatar_protocolos(documentos),
            "contexto_paciente": _formatar_contexto_paciente(paciente_id),
            "plano_factual": _resposta_fallback_segura(
                pergunta,
                paciente_id,
                paciente,
                documentos,
            ),
        }
    )

    # Uma saida insegura deve ser bloqueada antes de qualquer fallback; assim
    # preservamos a evidencia de que o guardrail interceptou a LLM.
    resultado = aplicar_guardrails(resposta_bruta)
    motivos_grounding: list[str] = []
    grounding_fallback = False
    grounding_citation_repair = False
    modo_resposta = "bloqueada" if resultado.bloqueado else "llm"
    if not resultado.bloqueado:
        motivos_grounding = _avaliar_grounding(
            resposta_bruta,
            pergunta,
            documentos,
            paciente,
            paciente_id,
        )
        motivos_sem_citacao = [
            motivo
            for motivo in motivos_grounding
            if motivo not in {"fonte_ausente", "fonte_nao_recuperada"}
        ]
        if not motivos_sem_citacao and documentos and any(
            motivo in {"fonte_ausente", "fonte_nao_recuperada"}
            for motivo in motivos_grounding
        ):
            citacoes = ", ".join(f"[{doc.metadata['id']}]" for doc in documentos)
            resposta_sem_citacoes = _PROTOCOL_LIKE_RE.sub("", resposta_bruta).strip()
            resposta_com_fontes = (
                f"{resposta_sem_citacoes.rstrip()}\n\nFontes consultadas: {citacoes}."
            )
            motivos_apos_reparo = _avaliar_grounding(
                resposta_com_fontes,
                pergunta,
                documentos,
                paciente,
                paciente_id,
            )
            if not motivos_apos_reparo:
                grounding_citation_repair = True
                modo_resposta = "citacao_reparada"
                resultado = aplicar_guardrails(resposta_com_fontes)
        grounding_fallback = bool(motivos_grounding) and not grounding_citation_repair
        if grounding_fallback:
            modo_resposta = "fallback"
            resposta_fallback = _resposta_fallback_segura(
                pergunta,
                paciente_id,
                paciente,
                documentos,
            )
            resultado = aplicar_guardrails(resposta_fallback)

    registrar_interacao(
        "resposta_assistente_medico",
        prompt_version=PROMPT_VERSION,
        paciente_id=paciente_id,
        pergunta=pergunta,
        fontes=fontes,
        grounding_fallback=grounding_fallback,
        grounding_citation_repair=grounding_citation_repair,
        modo_resposta=modo_resposta,
        motivos_grounding=motivos_grounding,
        bloqueado=resultado.bloqueado,
        motivo_bloqueio=resultado.motivo,
    )

    retorno = {
        "resposta": resultado.resposta,
        "fontes": fontes,
        "bloqueado": resultado.bloqueado,
        "motivo_bloqueio": resultado.motivo,
        "paciente_id": paciente_id,
        "grounding_fallback": grounding_fallback,
        "grounding_citation_repair": grounding_citation_repair,
        "motivos_grounding": motivos_grounding,
        "modo_resposta": modo_resposta,
    }
    if incluir_diagnostico:
        retorno["resposta_llm_bruta"] = resposta_bruta
    return retorno
