"""Fine-tuning LoRA/PEFT de um LLM causal com os dados internos do hospital (Fase 3).

Uso recomendado (modelo instrucional e multilingue selecionado):

    python -m fase3.finetuning.train_lora \
        --base-model Qwen/Qwen2.5-1.5B-Instruct \
        --epochs 4 --learning-rate 0.00005 --max-length 512 \
        --output-dir resultados/fase3/finetuning/qwen2.5-1.5b-v5

O treinamento usa loss somente nos tokens da resposta. Os tokens do system
prompt e da instrucao recebem label ``-100`` e nao entram no calculo da loss.
Isso evita que o modelo seja recompensado por copiar a pergunta e aproxima o
treinamento do comportamento esperado no assistente.

O ``distilgpt2`` continua suportado apenas para smoke tests de infraestrutura:

    python -m fase3.finetuning.train_lora \
        --base-model distilgpt2 --lora-target-modules c_attn \
        --epochs 1 --output-dir resultados/fase3/finetuning/smoke-distilgpt2

As dependencias pesadas (torch/transformers/peft/datasets/accelerate) estao
em ``requirements-fase3.txt`` e sao importadas apenas aqui dentro, para nao
obrigar o restante do repositorio (Fases 1/2, CI) a instala-las.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

from fase3.prompting import (
    SYSTEM_PROMPT_CLINICO,
    USER_PROMPT_TEMPLATE,
    formatar_prompt_usuario,
)

# Evita que o transformers tente importar um TensorFlow eventualmente
# instalado no ambiente (nao usamos TF; so PyTorch). Precisa ser definido
# antes do primeiro `import transformers`.
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

FASE3_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = FASE3_ROOT / "data"
DEFAULT_OUTPUT_DIR = FASE3_ROOT.parent / "resultados" / "fase3" / "finetuning"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_QUALITY_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "qwen2.5-1.5b-v5"

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
            DataCollatorForSeq2Seq,
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
        "DataCollatorForSeq2Seq": DataCollatorForSeq2Seq,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


def _default_target_modules(base_model: str) -> list[str]:
    nome = base_model.lower()
    if "gpt2" in nome or "gpt-2" in nome:
        return ["c_attn"]
    if "falcon" in nome:
        return ["query_key_value"]
    # Qwen/Llama-like: adapta atencao e MLP para dar capacidade suficiente ao
    # formato clinico, ainda treinando menos de 2% dos parametros.
    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


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


def _formatar_instrucao(exemplo: dict) -> str:
    return formatar_prompt_usuario(
        pergunta=exemplo["instruction"],
        contexto_paciente=exemplo.get("contexto_paciente")
        or exemplo.get("input")
        or "Nenhum paciente informado.",
        protocolos=exemplo.get("protocolos")
        or "Nenhum protocolo relevante encontrado.",
        plano_factual=exemplo.get("plano_factual") or exemplo["output"],
    )


def _formatar_prefixo(exemplo: dict, tokenizer) -> str:
    """Formata system/user e deixa o cursor exatamente no inicio da resposta."""
    instrucao = _formatar_instrucao(exemplo)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT_CLINICO},
                {"role": "user", "content": instrucao},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    return (
        f"Sistema: {SYSTEM_PROMPT_CLINICO}\n\n"
        f"Instrucao: {instrucao}\n"
        "Resposta:"
    )


def _tokenizar_exemplo_resposta(
    exemplo: dict, tokenizer, max_length: int
) -> dict[str, list[int]]:
    """Tokeniza mantendo labels somente na resposta do assistente.

    Quando o exemplo excede ``max_length``, preserva ao menos 32 tokens de
    resposta e o final do prompt (onde ficam a pergunta e o marcador de
    geracao). O padding fica a cargo do collator e usa ``-100`` nas labels.
    """
    if max_length < 64:
        raise ValueError("max_length deve ser pelo menos 64")

    prefixo = _formatar_prefixo(exemplo, tokenizer)
    eos = tokenizer.eos_token or ""
    resposta = exemplo["output"].strip() + eos

    prompt_ids = tokenizer(prefixo, add_special_tokens=False)["input_ids"]
    resposta_ids = tokenizer(resposta, add_special_tokens=False)["input_ids"]

    if len(resposta_ids) > max_length - 32:
        resposta_ids = resposta_ids[: max_length - 32]
    limite_prompt = max_length - len(resposta_ids)
    if len(prompt_ids) > limite_prompt:
        prompt_ids = prompt_ids[-limite_prompt:]

    input_ids = prompt_ids + resposta_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + list(resposta_ids),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def treinar(args: argparse.Namespace) -> dict:
    stack = _import_ml_stack()
    torch = stack["torch"]

    train_examples = _carregar_exemplos(args.data_dir / "finetuning_train.jsonl")
    val_examples = _carregar_exemplos(args.data_dir / "finetuning_val.jsonl")
    if not args.include_public_data:
        fontes_publicas = {"medquad", "pubmedqa"}
        train_examples = [
            exemplo for exemplo in train_examples if exemplo["source_type"] not in fontes_publicas
        ]
        val_examples = [
            exemplo for exemplo in val_examples if exemplo["source_type"] not in fontes_publicas
        ]

    clinical_examples = [
        exemplo
        for exemplo in train_examples
        if exemplo["source_type"] == "assistente_clinico_sintetico"
    ]
    if args.clinical_repeat > 1:
        train_examples = train_examples + clinical_examples * (args.clinical_repeat - 1)

    if not train_examples or not val_examples:
        raise SystemExit("Os filtros deixaram o split de treino ou validacao vazio.")

    tokenizer = stack["AutoTokenizer"].from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def tokenizar(exemplo: dict) -> dict:
        return _tokenizar_exemplo_resposta(exemplo, tokenizer, args.max_length)

    train_dataset = stack["Dataset"].from_list(train_examples).map(
        tokenizar, remove_columns=list(train_examples[0].keys())
    )
    val_dataset = stack["Dataset"].from_list(val_examples).map(
        tokenizar, remove_columns=list(val_examples[0].keys())
    )

    usar_fp16 = bool(torch.cuda.is_available() and not args.no_fp16)
    model_kwargs = {"torch_dtype": torch.float16} if usar_fp16 else {}
    modelo = stack["AutoModelForCausalLM"].from_pretrained(
        args.base_model, **model_kwargs
    )
    modelo.config.use_cache = False
    if args.gradient_checkpointing:
        modelo.gradient_checkpointing_enable()

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
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="no",
        report_to=[],
        disable_tqdm=True,
        seed=args.seed,
        data_seed=args.seed,
        fp16=usar_fp16,
    )

    collator = stack["DataCollatorForSeq2Seq"](
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

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

    ultimo_eval_loss = (
        historico_validacao[-1]["eval_loss"] if historico_validacao else None
    )
    tokens_supervisionados = sum(
        sum(label != -100 for label in exemplo["labels"])
        for exemplo in train_dataset
    )
    tokens_totais = sum(len(exemplo["labels"]) for exemplo in train_dataset)
    train_path = args.data_dir / "finetuning_train.jsonl"
    val_path = args.data_dir / "finetuning_val.jsonl"

    resumo = {
        "base_model": args.base_model,
        "modelo_instrucional": bool(getattr(tokenizer, "chat_template", None)),
        "loss_apenas_na_resposta": True,
        "lora_target_modules": target_modules,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_length": args.max_length,
        "seed": args.seed,
        "fp16": usar_fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "inclui_dados_publicos": args.include_public_data,
        "repeticao_exemplos_clinicos": args.clinical_repeat,
        "n_exemplos_treino": len(train_examples),
        "n_exemplos_validacao": len(val_examples),
        "tokens_supervisionados_resposta": tokens_supervisionados,
        "fracao_tokens_supervisionados": round(
            tokens_supervisionados / max(1, tokens_totais), 4
        ),
        "dataset_sha256": {
            "treino": _sha256(train_path),
            "validacao": _sha256(val_path),
        },
        "prompt_sha256": hashlib.sha256(
            (SYSTEM_PROMPT_CLINICO + "\n" + USER_PROMPT_TEMPLATE).encode("utf-8")
        ).hexdigest(),
        "device": str(next(modelo.parameters()).device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": stack["torch"].__version__,
        "duracao_segundos": round(duracao, 2),
        "loss_final_treino": resultado_treino.training_loss,
        "loss_final_validacao": ultimo_eval_loss,
        "perplexidade_validacao": (
            round(math.exp(min(ultimo_eval_loss, 20)), 3)
            if ultimo_eval_loss is not None
            else None
        ),
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
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_QUALITY_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-public-data",
        action="store_true",
        help=(
            "Inclui MedQuAD/PubMedQA no treino. Por padrao, o adapter clinico "
            "usa apenas os dados internos em portugues para evitar degradacao de dominio."
        ),
    )
    parser.add_argument(
        "--clinical-repeat",
        type=int,
        default=1,
        help="Fator de oversampling dos exemplos alinhados ao assistente clinico.",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Reduz memoria de ativacoes; recomendado para o modelo de 3B.",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Desativa FP16 mesmo quando uma GPU CUDA esta disponivel.",
    )
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
