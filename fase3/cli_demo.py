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

    python -m fase3.cli_demo --paciente-id PAC-0001 \
        --pergunta "Posso iniciar a quimioterapia hoje?" --backend local \
        --base-model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter-path resultados/fase3/finetuning/qwen2.5-1.5b-v4/lora_adapter \
        --lora-scale 0.75
"""

from __future__ import annotations

import argparse

from fase3.clinical_flow_graph import executar_fluxo_clinico
from fase3.llm_backend import DEFAULT_LLM_BACKEND, get_llm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paciente-id", default=None, help="Codigo do paciente, ex.: PAC-0001")
    parser.add_argument("--pergunta", required=True)
    parser.add_argument(
        "--backend",
        default=DEFAULT_LLM_BACKEND,
        choices=["local", "groq", "fake"],
        help="Backend de inferencia (padrao: local).",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Modelo base usado com --backend local (padrao: Qwen2.5-1.5B-Instruct).",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help=(
            "Caminho do adapter LoRA usado com --backend local. Se omitido, "
            "usa resultados/fase3/finetuning/qwen2.5-1.5b-v4/lora_adapter."
        ),
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=None,
        help="Intensidade do adapter local em (0, 1]; padrao calibrado: 0.75.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=160,
        help="Limite de tokens gerados pelo backend local.",
    )
    args = parser.parse_args()

    if args.backend == "fake":
        llm = get_llm(
            "fake",
            respostas=[
                "Com base no protocolo institucional, recomenda-se seguir a conduta padrao "
                "e reavaliar o paciente conforme os criterios descritos."
            ],
        )
    else:
        llm = get_llm(
            args.backend,
            base_model=args.base_model,
            adapter_path=args.adapter_path,
            max_new_tokens=args.max_new_tokens,
            lora_scale=args.lora_scale,
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
    print(f"Modo de resposta: {sugestao.get('modo_resposta', '(nao informado)')}")
    print(f"Rota de exames: {estado.get('rota_exames', '(nao aplicavel)')}")

    alertas = estado.get("alertas", [])
    print(f"\nAlertas para a equipe medica ({len(alertas)}):")
    for alerta in alertas:
        print(f"  - {alerta}")

    print("\n(evento de auditoria completo registrado em resultados/fase3/auditoria.jsonl)")


if __name__ == "__main__":
    main()
