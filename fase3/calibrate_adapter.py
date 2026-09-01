"""Compara modelo-base e escalas LoRA e promove o melhor candidato aprovado."""

from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path

from fase3.evaluation_contract import current_evaluation_contract
from fase3.evaluate_assistant import carregar_casos, executar_avaliacao, salvar_resultados
from fase3.llm_backend import DEFAULT_LOCAL_BASE_MODEL

ROOT = Path(__file__).resolve().parent.parent
RESULTADOS_DIR = ROOT / "resultados" / "fase3"
VALIDATION_PATH = ROOT / "fase3" / "data" / "finetuning_val.jsonl"


def _liberar_memoria() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _carregar_casos_validacao(path: Path = VALIDATION_PATH) -> list[dict]:
    casos: list[dict] = []
    for indice, linha in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        exemplo = json.loads(linha)
        paciente = re.search(r"\bPAC-\d{4}\b", exemplo.get("contexto_paciente", ""))
        casos.append(
            {
                "id": f"VAL-{indice:03d}",
                "tipo": "regular",
                "paciente_id": paciente.group(0) if paciente else None,
                "pergunta": exemplo["instruction"],
                "fonte_obrigatoria": bool(re.search(r"\bPROT-\d{3}\b", exemplo["protocolos"])),
            }
        )
    if len(casos) != 8:
        raise ValueError("A calibracao exige os oito casos do split de validacao.")
    return casos


def calibrar(
    *,
    base_model: str,
    adapter_path: str,
    escalas: list[float],
    max_new_tokens: int,
) -> dict:
    casos_validacao = _carregar_casos_validacao()
    linhas_base = executar_avaliacao(
        "local",
        base_model=base_model,
        max_new_tokens=max_new_tokens,
        use_adapter=False,
        casos=casos_validacao,
    )
    resumo_base = salvar_resultados(
        linhas_base,
        prefix="calibracao_assistente_base",
    )
    _liberar_memoria()

    candidatos: list[tuple[dict, list[dict]]] = []
    for escala in escalas:
        sufixo = str(escala).replace(".", "_")
        linhas = executar_avaliacao(
            "local",
            base_model=base_model,
            adapter_path=adapter_path,
            lora_scale=escala,
            max_new_tokens=max_new_tokens,
            casos=casos_validacao,
        )
        resumo = salvar_resultados(
            linhas,
            prefix=f"calibracao_assistente_escala_{sufixo}",
            baseline=resumo_base,
        )
        candidatos.append((resumo, linhas))
        _liberar_memoria()

    candidatos.sort(
        key=lambda item: (
            bool(item[0]["aprovado"]),
            item[0]["taxa_aceitacao_bruta_regular"],
            item[0]["score_qualidade_bruta_regular"],
            -item[0]["taxa_fallback_regular"],
        ),
        reverse=True,
    )
    melhor_resumo, _ = candidatos[0]
    escala_selecionada = melhor_resumo["lora_scale"]

    casos_finais = carregar_casos()
    casos_regulares = [caso for caso in casos_finais if caso["tipo"] == "regular"]
    linhas_base_finais = executar_avaliacao(
        "local",
        base_model=base_model,
        max_new_tokens=max_new_tokens,
        use_adapter=False,
        casos=casos_regulares,
    )
    resumo_base_final = salvar_resultados(
        linhas_base_finais,
        prefix="avaliacao_assistente_base",
    )
    _liberar_memoria()

    linhas_finais = executar_avaliacao(
        "local",
        base_model=base_model,
        adapter_path=adapter_path,
        lora_scale=escala_selecionada,
        max_new_tokens=max_new_tokens,
        casos=casos_finais,
    )
    resumo_promovido = salvar_resultados(
        linhas_finais,
        prefix="avaliacao_assistente",
        baseline=resumo_base_final,
    )
    calibracao = {
        "base_model": base_model,
        "adapter_path": adapter_path,
        "evaluation_contract": current_evaluation_contract(),
        "calibracao_baseline": resumo_base,
        "calibracao_candidatos": [resumo for resumo, _ in candidatos],
        "avaliacao_final_baseline": resumo_base_final,
        "selecionado": resumo_promovido,
    }
    (RESULTADOS_DIR / "calibracao_adapter.json").write_text(
        json.dumps(calibracao, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return calibracao


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_LOCAL_BASE_MODEL)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument(
        "--scales",
        default="0.25,0.5,0.75,1.0",
        help="Escalas LoRA separadas por virgula.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--enforce-gates", action="store_true")
    args = parser.parse_args()
    escalas = [float(valor) for valor in args.scales.split(",")]
    resultado = calibrar(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        escalas=escalas,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(resultado["selecionado"], ensure_ascii=False, indent=2))
    if args.enforce_gates and not resultado["selecionado"]["aprovado"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
