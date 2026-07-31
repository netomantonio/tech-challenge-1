"""Demo de ponta a ponta do assistente medico da Fase 3, via linha de comando.

Pensado para ser usado na gravacao do video pedido no PDF: roda o fluxo
LangGraph completo (busca do paciente, checagem de exames pendentes,
sugestao com LangChain, guardrails e emissao de alertas) e imprime cada
etapa de forma legivel, incluindo o log de auditoria gerado.

Exemplos:

    python -m fase3.cli_demo --paciente-id PAC-0001 \
        --pergunta "Posso iniciar a quimioterapia hoje?"

    python -m fase3.cli_demo --paciente-id PAC-0005 \
        --pergunta "A paciente esta com febre, qual conduta?" --backend fake
"""

from __future__ import annotations

import argparse
import json

from fase3.clinical_flow_graph import executar_fluxo_clinico
from fase3.llm_backend import get_llm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paciente-id", default=None, help="Codigo do paciente, ex.: PAC-0001")
    parser.add_argument("--pergunta", required=True)
    parser.add_argument("--backend", default="groq", choices=["groq", "local", "fake"])
    args = parser.parse_args()

    llm = get_llm(args.backend) if args.backend != "fake" else get_llm(
        "fake",
        respostas=[
            "Com base no protocolo institucional, recomenda-se seguir a conduta padrao "
            "e reavaliar o paciente conforme os criterios descritos."
        ],
    )

    estado = executar_fluxo_clinico(args.paciente_id, args.pergunta, llm=llm)

    print("=" * 70)
    print(f"Paciente: {args.paciente_id or '(nao informado)'}")
    print(f"Pergunta: {args.pergunta}")
    print("=" * 70)

    if not estado.get("paciente_encontrado", True):
        print("Paciente nao encontrado no prontuario. Fluxo encerrado com seguranca.")
        return

    sugestao = estado.get("sugestao") or {}
    print("\nResposta do assistente:\n")
    print(sugestao.get("resposta", "(sem resposta)"))

    print("\nFontes citadas (explainability):")
    for fonte in sugestao.get("fontes", []):
        print(f"  - [{fonte['id']}] {fonte['titulo']}")

    print(f"\nBloqueado por guardrail: {estado.get('bloqueado', False)}")

    alertas = estado.get("alertas", [])
    print(f"\nAlertas para a equipe medica ({len(alertas)}):")
    for alerta in alertas:
        print(f"  - {alerta}")

    print("\n(evento de auditoria completo registrado em resultados/fase3/auditoria.jsonl)")


if __name__ == "__main__":
    main()
