"""Fine-tuning LoRA/PEFT de um LLM causal com os dados internos do hospital (Fase 3).

Uso tipico (smoke test rapido, sem GPU, modelo pequeno mas com pesos
pretreinados de verdade, so para validar o pipeline de ponta a ponta e
observar a perda caindo):

    python -m fase3.finetuning.train_lora --base-model distilgpt2 \
        --epochs 5 --output-dir resultados/fase3/finetuning/smoke

Para um fine-tuning "de verdade" (recomendado rodar em Colab/GPU, por causa
do tempo e do download do checkpoint), basta trocar o modelo base e ajustar
os modulos de LoRA, por exemplo:

    python -m fase3.finetuning.train_lora \
        --base-model Qwen/Qwen2.5-0.5B-Instruct \
        --lora-target-modules q_proj,v_proj,k_proj,o_proj \
        --epochs 3 --output-dir resultados/fase3/finetuning/qwen2.5-0.5b

As dependencias pesadas (torch/transformers/peft/datasets/accelerate) estao
em ``requirements-fase3.txt`` e sao importadas apenas aqui dentro, para nao
obrigar o restante do repositorio (Fases 1/2, CI) a instala-las.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Evita que o transformers tente importar um TensorFlow eventualmente
# instalado no ambiente (nao usamos TF; so PyTorch). Precisa ser definido
# antes do primeiro `import transformers`.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

FASE3_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = FASE3_ROOT / "data"
DEFAULT_OUTPUT_DIR = FASE3_ROOT.parent / "resultados" / "fase3" / "finetuning"

_MISSING_DEPS_MSG = (
    "As dependencias de fine-tuning nao estao instaladas. Rode:\n"
    "  pip install -r requirements-fase3.txt\n"
    f"(erro original: {{exc}})"
)


def _import_ml_stack():
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:  # pragma: no cover - exercised apenas sem os pacotes
        raise SystemExit(_MISSING_DEPS_MSG.format(exc=exc)) from exc
    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorForLanguageModeling": DataCollatorForLanguageModeling,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


def _default_target_modules(base_model: str) -> list[str]:
    nome = base_model.lower()
    if "gpt2" in nome or "gpt-2" in nome:
        return ["c_attn"]
    if "falcon" in nome:
        return ["query_key_value"]
    # Padrao para a familia Llama/Qwen/Mistral/TinyLlama (arquitetura Llama-like).
    return ["q_proj", "v_proj"]


def _carregar_exemplos(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"Arquivo {path} nao encontrado. Rode antes:\n"
            "  python -m fase3.data.build_finetuning_dataset"
        )
    exemplos = []
    for linha in path.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if linha:
            exemplos.append(json.loads(linha))
    return exemplos


def _formatar_prompt(exemplo: dict) -> str:
    entrada = f"\nContexto: {exemplo['input']}" if exemplo.get("input") else ""
    return (
        f"Instrucao: {exemplo['instruction']}{entrada}\n"
        f"Resposta: {exemplo['output']}"
    )


def treinar(args: argparse.Namespace) -> dict:
    stack = _import_ml_stack()
    torch = stack["torch"]

    train_examples = _carregar_exemplos(args.data_dir / "finetuning_train.jsonl")
    val_examples = _carregar_exemplos(args.data_dir / "finetuning_val.jsonl")

    tokenizer = stack["AutoTokenizer"].from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenizar(exemplo: dict) -> dict:
        texto = _formatar_prompt(exemplo) + tokenizer.eos_token
        saida = tokenizer(texto, truncation=True, max_length=args.max_length)
        return saida

    train_dataset = stack["Dataset"].from_list(train_examples).map(
        tokenizar, remove_columns=list(train_examples[0].keys())
    )
    val_dataset = stack["Dataset"].from_list(val_examples).map(
        tokenizar, remove_columns=list(val_examples[0].keys())
    )

    modelo = stack["AutoModelForCausalLM"].from_pretrained(args.base_model)

    target_modules = args.lora_target_modules or _default_target_modules(args.base_model)
    lora_config = stack["LoraConfig"](
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    modelo = stack["get_peft_model"](modelo, lora_config)
    modelo.print_trainable_parameters()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    training_args = stack["TrainingArguments"](
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="no",
        report_to=[],
        disable_tqdm=True,
    )

    collator = stack["DataCollatorForLanguageModeling"](tokenizer=tokenizer, mlm=False)

    trainer = stack["Trainer"](
        model=modelo,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    inicio = time.time()
    resultado_treino = trainer.train()
    duracao = time.time() - inicio

    modelo.save_pretrained(str(args.output_dir / "lora_adapter"))
    tokenizer.save_pretrained(str(args.output_dir / "lora_adapter"))

    historico_treino = [
        {"step": h["step"], "loss": h["loss"]}
        for h in trainer.state.log_history
        if "loss" in h
    ]
    historico_validacao = [
        {"epoch": h["epoch"], "eval_loss": h["eval_loss"]}
        for h in trainer.state.log_history
        if "eval_loss" in h
    ]
    passos_por_epoca = max(1, len(historico_treino) // max(1, args.epochs))
    perda_media_por_epoca = [
        {
            "epoch": i + 1,
            "loss_medio": sum(
                h["loss"] for h in historico_treino[i * passos_por_epoca : (i + 1) * passos_por_epoca]
            )
            / max(1, len(historico_treino[i * passos_por_epoca : (i + 1) * passos_por_epoca]))
        }
        for i in range(args.epochs)
        if historico_treino[i * passos_por_epoca : (i + 1) * passos_por_epoca]
    ]

    resumo = {
        "base_model": args.base_model,
        "lora_target_modules": target_modules,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "n_exemplos_treino": len(train_examples),
        "n_exemplos_validacao": len(val_examples),
        "duracao_segundos": round(duracao, 2),
        "loss_final_treino": resultado_treino.training_loss,
        "perda_media_por_epoca": perda_media_por_epoca,
        "historico_perda_treino": historico_treino,
        "historico_perda_validacao": historico_validacao,
        "adapter_path": str(args.output_dir / "lora_adapter"),
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return resumo


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default="distilgpt2")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "smoke")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        type=lambda s: [m.strip() for m in s.split(",") if m.strip()],
        default=None,
        help="Lista separada por virgula, ex.: q_proj,v_proj. Se omitido, usa um padrao por arquitetura.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    resumo = treinar(args)
    print(json.dumps(resumo, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
