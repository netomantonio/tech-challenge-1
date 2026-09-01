"""Guardrails de seguranca e validacao do assistente medico (Fase 3).

Espelha o padrao ja usado em ``src/llm_interpretation.py`` (Fase 2): um
disclaimer fixo, regras explicitas contra prescricao direta e checagem
deterministica de PII — sem depender de um segundo LLM como "juiz".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from fase3.data.build_finetuning_dataset import anonimizar_texto, detectar_pii

DISCLAIMER = (
    "Esta resposta foi gerada por um assistente automatizado com base em "
    "protocolos internos e nao substitui a avaliacao clinica presencial. "
    "Qualquer conduta ou prescricao exige validacao de um medico responsavel "
    "antes de ser aplicada ao paciente."
)

MENSAGEM_PRESCRICAO_BLOQUEADA = (
    "Nao posso indicar diretamente uma prescricao, dose ou tratamento. "
    "Consulte o protocolo relevante e submeta a decisao a um medico "
    "responsavel para validacao antes de qualquer conduta."
)

MENSAGEM_PII_BLOQUEADA = (
    "Esta resposta foi bloqueada porque parece conter informacao "
    "pessoal identificavel do paciente. Reformule a pergunta sem incluir "
    "nome, CPF, telefone ou e-mail, ou solicite revisao de um responsavel "
    "pelo tratamento de dados."
)

MENSAGEM_ENTRADA_PII_BLOQUEADA = (
    "A consulta foi bloqueada antes do processamento porque continha informacao "
    "pessoal identificavel. Remova nome, CPF, telefone ou e-mail e envie a "
    "pergunta novamente com apenas o codigo interno do paciente."
)

# Verbo de acao de prescricao + algo que pareca dose/via/medicamento por perto.
_PRESCRICAO_DIRETA_RE = re.compile(
    r"(?i)\b(tome|toma|administre|aplique|prescrevo|prescreva|receite)\b"
    r"[^.]{0,60}\b(\d+\s?(mg|ml|mcg|g|ui|comprimido|gota)s?|de \w+)\b"
)


@dataclass
class ResultadoGuardrail:
    resposta: str
    bloqueado: bool
    motivo: str | None


@dataclass
class ResultadoGuardrailEntrada:
    texto_redigido: str
    resposta: str | None
    bloqueado: bool
    motivo: str | None


def contem_prescricao_direta(texto: str) -> bool:
    return bool(_PRESCRICAO_DIRETA_RE.search(texto))


def aplicar_guardrails_entrada(texto: str) -> ResultadoGuardrailEntrada:
    """Bloqueia PII antes de retrieval, LLM e auditoria e devolve texto redigido."""
    if detectar_pii(texto):
        return ResultadoGuardrailEntrada(
            texto_redigido=anonimizar_texto(texto),
            resposta=MENSAGEM_ENTRADA_PII_BLOQUEADA,
            bloqueado=True,
            motivo="pii_detectada_na_entrada",
        )
    return ResultadoGuardrailEntrada(
        texto_redigido=texto,
        resposta=None,
        bloqueado=False,
        motivo=None,
    )


def aplicar_guardrails(resposta_llm: str) -> ResultadoGuardrail:
    """Aplica as checagens de seguranca sobre a resposta bruta do LLM.

    Sempre retorna uma resposta segura para exibir ao usuario: quando um
    guardrail e acionado, a resposta original e substituida por uma
    mensagem padronizada em vez de ser exibida (ainda que parcialmente).
    """
    if detectar_pii(resposta_llm):
        return ResultadoGuardrail(
            resposta=MENSAGEM_PII_BLOQUEADA, bloqueado=True, motivo="pii_detectada"
        )
    if contem_prescricao_direta(resposta_llm):
        return ResultadoGuardrail(
            resposta=MENSAGEM_PRESCRICAO_BLOQUEADA,
            bloqueado=True,
            motivo="prescricao_direta_bloqueada",
        )
    resposta_final = f"{resposta_llm.strip()}\n\n{DISCLAIMER}"
    return ResultadoGuardrail(resposta=resposta_final, bloqueado=False, motivo=None)
