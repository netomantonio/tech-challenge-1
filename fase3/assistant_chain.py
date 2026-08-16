"""Pipeline LangChain do assistente medico virtual (Fase 3).

Monta uma chain LCEL (``prompt | llm | StrOutputParser``) que integra:

1. Retrieval sobre os protocolos internos (``fase3.retrieval``);
2. Contexto estruturado do paciente vindo do mock de EHR (``fase3.ehr_tools``);
3. O LLM plugavel (``fase3.llm_backend``, "groq" por padrao);
4. Guardrails de seguranca e disclaimer obrigatorio (``fase3.guardrails``);
5. Log de auditoria com as fontes usadas, para explainability (``fase3.logging_utils``).
"""

from __future__ import annotations

from typing import Optional

from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from fase3.ehr_tools import get_paciente
from fase3.guardrails import aplicar_guardrails
from fase3.llm_backend import get_llm
from fase3.logging_utils import registrar_interacao
from fase3.retrieval import DEFAULT_TOP_K, BM25Retriever, buscar_protocolos, construir_retriever, formatar_fontes

PROMPT_VERSION = "assistente_medico_v1"

PROMPT = PromptTemplate.from_template(
    "Contexto do paciente:\n{contexto_paciente}\n\n"
    "Protocolos internos relevantes:\n{protocolos}\n\n"
    "Pergunta do medico: {pergunta}\n\n"
    "Responda em portugues, em 2 a 5 frases objetivas, seguindo obrigatoriamente "
    "estas regras:\n"
    "1. Use somente fatos presentes no contexto do paciente e nos protocolos acima.\n"
    "2. Nao invente diagnosticos, medicamentos, exames, doses ou valores numericos.\n"
    "3. Preserve exatamente os valores numericos recebidos no contexto.\n"
    "4. Cite entre colchetes pelo menos um protocolo usado, por exemplo [PROT-006].\n"
    "5. Condicione qualquer conduta a validacao da equipe medica.\n"
    "6. Se o paciente nao foi encontrado ou faltar informacao, informe a limitacao "
    "e nao sugira conduta."
)


def construir_chain(llm: LLM) -> Runnable:
    """Monta a chain LCEL prompt -> llm -> texto, reutilizavel com qualquer backend."""
    return PROMPT | llm | StrOutputParser()


def _formatar_protocolos(documentos) -> str:
    if not documentos:
        return "Nenhum protocolo relevante encontrado."
    return "\n\n".join(
        f"[{doc.metadata['id']}] {doc.metadata['titulo']}:\n{doc.page_content}"
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


def responder_pergunta_clinica(
    pergunta: str,
    paciente_id: Optional[str] = None,
    llm: Optional[LLM] = None,
    retriever: Optional[BM25Retriever] = None,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """Responde uma pergunta clinica com contexto de protocolos e do paciente.

    Retorna um dicionario com ``resposta`` (ja com guardrails/disclaimer
    aplicados), ``fontes`` (para explainability), ``bloqueado`` e o motivo do
    bloqueio quando aplicavel.
    """
    llm = llm or get_llm()
    retriever = retriever or construir_retriever(k=top_k)

    documentos = buscar_protocolos(pergunta, retriever)
    fontes = formatar_fontes(documentos)

    chain = construir_chain(llm)
    resposta_bruta = chain.invoke(
        {
            "pergunta": pergunta,
            "protocolos": _formatar_protocolos(documentos),
            "contexto_paciente": _formatar_contexto_paciente(paciente_id),
        }
    )

    resultado = aplicar_guardrails(resposta_bruta)

    registrar_interacao(
        "resposta_assistente_medico",
        prompt_version=PROMPT_VERSION,
        paciente_id=paciente_id,
        pergunta=pergunta,
        fontes=fontes,
        bloqueado=resultado.bloqueado,
        motivo_bloqueio=resultado.motivo,
    )

    return {
        "resposta": resultado.resposta,
        "fontes": fontes,
        "bloqueado": resultado.bloqueado,
        "motivo_bloqueio": resultado.motivo,
        "paciente_id": paciente_id,
    }
