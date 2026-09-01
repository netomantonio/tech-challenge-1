"""Contrato de prompt compartilhado pelo treino e pela inferencia da Fase 3."""

from __future__ import annotations

from collections.abc import Iterable, Mapping


SYSTEM_PROMPT_CLINICO = (
    "Voce e um assistente virtual de apoio clinico interno do hospital. "
    "Use somente o contexto e as fontes autorizadas recebidas. Nunca prescreva "
    "medicamentos, doses ou tratamentos diretamente. Toda conduta deve depender "
    "de validacao de um medico responsavel. Quando faltar informacao, declare a "
    "limitacao sem completar dados por conta propria. Preserve exatamente valores "
    "clinicos e identificadores de protocolos."
)

USER_PROMPT_TEMPLATE = (
    "Redija uma resposta curta em portugues obedecendo a estas regras.\n"
    "- Nao acrescente, remova ou substitua fatos e valores clinicos.\n"
    "- Cite somente protocolos presentes nas fontes autorizadas.\n"
    "- Nao indique medicamento ou dose.\n"
    "- Condicione a conduta a validacao da equipe medica.\n"
    "- Retorne apenas a resposta destinada ao profissional de saude.\n\n"
    "<contexto_paciente>\n{contexto_paciente}\n</contexto_paciente>\n\n"
    "<fontes_autorizadas>\n{protocolos}\n</fontes_autorizadas>\n\n"
    "<pergunta_medica>\n{pergunta}\n</pergunta_medica>\n\n"
    "Use fielmente o conteudo abaixo como resposta. Faca somente ajustes "
    "gramaticais e preserve cada valor e identificador.\n"
    "<plano_factual>\n{plano_factual}\n</plano_factual>"
)


def formatar_prompt_usuario(
    *,
    pergunta: str,
    contexto_paciente: str,
    protocolos: str,
    plano_factual: str,
) -> str:
    """Formata exatamente a mensagem de usuario usada pelo treino e pela chain."""
    return USER_PROMPT_TEMPLATE.format(
        pergunta=pergunta.strip(),
        contexto_paciente=contexto_paciente.strip(),
        protocolos=protocolos.strip(),
        plano_factual=plano_factual.strip(),
    )


def formatar_protocolos_prompt(
    protocolos: Iterable[Mapping[str, str]],
    *,
    limite_conteudo: int = 700,
) -> str:
    """Formata fontes com id, titulo e conteudo clinico em tamanho controlado."""
    linhas: list[str] = []
    for protocolo in protocolos:
        conteudo = protocolo.get("conteudo", "").strip()
        if len(conteudo) > limite_conteudo:
            conteudo = conteudo[:limite_conteudo].rsplit(" ", 1)[0].rstrip() + "..."
        linhas.append(
            f"[{protocolo['id']}] {protocolo['titulo']}: {conteudo}"
        )
    return "\n".join(linhas) or "Nenhum protocolo relevante encontrado."
