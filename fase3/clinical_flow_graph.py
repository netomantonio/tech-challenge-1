"""Fluxo de decisao clinica orquestrado com LangGraph (Fase 3).

Implementa o fluxo pedido no desafio: ao receber informacoes de um
paciente, o sistema verifica exames pendentes, sugere conduta com base nos
protocolos (via ``fase3.assistant_chain``) e emite alertas para a equipe
medica quando necessario — com uma bifurcacao real de decisao (paciente nao
encontrado interrompe o fluxo antes de qualquer sugestao).

    buscar_paciente --(nao encontrado)--> registrar_auditoria --> END
          |
       (encontrado)
          v
    verificar_exames_pendentes --(com pendencias)--> alertar_exames
          |                                      |
          +--(sem pendencias)--------------------+
                                                 v
          sugerir_tratamento -> checar_seguranca -> emitir_alertas
          -> registrar_auditoria -> END
"""

from __future__ import annotations

from typing import Optional, TypedDict

from langchain_core.language_models.llms import LLM
from langgraph.graph import END, StateGraph

from fase3.assistant_chain import responder_pergunta_clinica
from fase3.ehr_tools import get_paciente
from fase3.logging_utils import registrar_interacao
from fase3.retrieval import BM25Retriever, construir_retriever


class EstadoFluxoClinico(TypedDict, total=False):
    paciente_id: str
    pergunta: str
    paciente: Optional[dict]
    paciente_encontrado: bool
    exames_pendentes: list[str]
    tem_exames_pendentes: bool
    rota_exames: str
    sugestao: Optional[dict]
    bloqueado: bool
    alertas: list[str]


def _no_buscar_paciente(state: EstadoFluxoClinico) -> EstadoFluxoClinico:
    paciente = get_paciente(state["paciente_id"])
    return {
        **state,
        "paciente": paciente,
        "paciente_encontrado": paciente is not None,
    }


def _rota_apos_busca(state: EstadoFluxoClinico) -> str:
    return "continuar" if state["paciente_encontrado"] else "paciente_nao_encontrado"


def _no_verificar_exames_pendentes(state: EstadoFluxoClinico) -> EstadoFluxoClinico:
    exames_pendentes = list((state.get("paciente") or {}).get("exames_pendentes", []))
    tem_pendencias = bool(exames_pendentes)
    return {
        **state,
        "exames_pendentes": exames_pendentes,
        "tem_exames_pendentes": tem_pendencias,
        "rota_exames": "com_pendencias" if tem_pendencias else "sem_pendencias",
    }


def _rota_apos_verificar_exames(state: EstadoFluxoClinico) -> str:
    return state["rota_exames"]


def _no_alertar_exames_pendentes(state: EstadoFluxoClinico) -> EstadoFluxoClinico:
    alertas = list(state.get("alertas", []))
    alertas.append(
        f"Exames pendentes para {state['paciente_id']}: "
        f"{', '.join(state['exames_pendentes'])}."
    )
    return {**state, "alertas": alertas}


def _construir_no_sugerir_tratamento(llm: Optional[LLM], retriever: Optional[BM25Retriever]):
    def _no(state: EstadoFluxoClinico) -> EstadoFluxoClinico:
        sugestao = responder_pergunta_clinica(
            pergunta=state["pergunta"],
            paciente_id=state["paciente_id"],
            llm=llm,
            retriever=retriever,
        )
        return {**state, "sugestao": sugestao, "bloqueado": sugestao["bloqueado"]}

    return _no


def _no_checar_seguranca(state: EstadoFluxoClinico) -> EstadoFluxoClinico:
    # O guardrail em si ja roda dentro de `responder_pergunta_clinica`; este
    # no traduz o resultado em uma decisao de fluxo (escalar para revisao
    # humana quando bloqueado).
    alertas = list(state.get("alertas", []))
    if state.get("bloqueado"):
        alertas.append(
            f"Sugestao bloqueada por guardrail ({state['sugestao']['motivo_bloqueio']}) "
            "- requer revisao manual de um medico."
        )
    return {**state, "alertas": alertas}


def _no_emitir_alertas(state: EstadoFluxoClinico) -> EstadoFluxoClinico:
    alertas = list(state.get("alertas", []))
    paciente = state.get("paciente") or {}
    for alerta_ativo in paciente.get("alertas_ativos", []):
        alertas.append(f"Alerta clinico ativo: {alerta_ativo}")
    return {**state, "alertas": alertas}


def _no_registrar_auditoria(state: EstadoFluxoClinico) -> EstadoFluxoClinico:
    registrar_interacao(
        "fluxo_clinico_concluido",
        paciente_id=state["paciente_id"],
        paciente_encontrado=state.get("paciente_encontrado", False),
        exames_pendentes=state.get("exames_pendentes", []),
        tem_exames_pendentes=state.get("tem_exames_pendentes", False),
        rota_exames=state.get("rota_exames"),
        alertas=state.get("alertas", []),
        bloqueado=state.get("bloqueado", False),
        fontes=(state.get("sugestao") or {}).get("fontes", []),
        modo_resposta=(state.get("sugestao") or {}).get("modo_resposta"),
    )
    return state


def construir_grafo(llm: Optional[LLM] = None, retriever: Optional[BM25Retriever] = None):
    retriever = retriever or construir_retriever()
    grafo = StateGraph(EstadoFluxoClinico)

    grafo.add_node("buscar_paciente", _no_buscar_paciente)
    grafo.add_node("verificar_exames_pendentes", _no_verificar_exames_pendentes)
    grafo.add_node("alertar_exames_pendentes", _no_alertar_exames_pendentes)
    grafo.add_node("sugerir_tratamento", _construir_no_sugerir_tratamento(llm, retriever))
    grafo.add_node("checar_seguranca", _no_checar_seguranca)
    grafo.add_node("emitir_alertas", _no_emitir_alertas)
    grafo.add_node("registrar_auditoria", _no_registrar_auditoria)

    grafo.set_entry_point("buscar_paciente")
    grafo.add_conditional_edges(
        "buscar_paciente",
        _rota_apos_busca,
        {
            "continuar": "verificar_exames_pendentes",
            "paciente_nao_encontrado": "registrar_auditoria",
        },
    )
    grafo.add_conditional_edges(
        "verificar_exames_pendentes",
        _rota_apos_verificar_exames,
        {
            "com_pendencias": "alertar_exames_pendentes",
            "sem_pendencias": "sugerir_tratamento",
        },
    )
    grafo.add_edge("alertar_exames_pendentes", "sugerir_tratamento")
    grafo.add_edge("sugerir_tratamento", "checar_seguranca")
    grafo.add_edge("checar_seguranca", "emitir_alertas")
    grafo.add_edge("emitir_alertas", "registrar_auditoria")
    grafo.add_edge("registrar_auditoria", END)

    return grafo.compile()


def executar_fluxo_clinico(
    paciente_id: str,
    pergunta: str,
    llm: Optional[LLM] = None,
    retriever: Optional[BM25Retriever] = None,
) -> EstadoFluxoClinico:
    app = construir_grafo(llm=llm, retriever=retriever)
    estado_inicial: EstadoFluxoClinico = {
        "paciente_id": paciente_id,
        "pergunta": pergunta,
        "alertas": [],
    }
    return app.invoke(estado_inicial)
