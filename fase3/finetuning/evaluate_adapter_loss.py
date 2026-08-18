"""Reavalia a loss supervisionada de um adapter LoRA nos splits da Fase 3."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from fase3.finetuning.train_lora import _carregar_exemplos, _tokenizar_exemplo_resposta
from fase3.llm_backend import DEFAULT_LOCAL_BASE_MODEL

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "fase3" / "data"
DEFAULT_ADAPTER_PATH = (
    ROOT
    / "resultados"
    / "fase3"
    / "finetuning"
    / "qwen2.5-1.5b-v4"
    / "lora_adapter"
)


def avaliar_split(modelo, tokenizer, exemplos: list[dict], max_length: int) -> dict:
    import torch
    from transformers import DataCollatorForSeq2Seq

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    device = next(modelo.parameters()).device
    soma_nll = 0.0
    tokens_supervisionados = 0
    for inicio in range(0, len(exemplos), 2):
        tokenizados = [
            _tokenizar_exemplo_resposta(exemplo, tokenizer, max_length)
            for exemplo in exemplos[inicio : inicio + 2]
        ]
        lote = {chave: valor.to(device) for chave, valor in collator(tokenizados).items()}
        quantidade = int((lote["labels"] != -100).sum().item())
        with torch.inference_mode():
            loss = float(modelo(**lote).loss.item())
        soma_nll += loss * quantidade
        tokens_supervisionados += quantidade

    loss_media = soma_nll / max(1, tokens_supervisionados)
    return {
        "n_exemplos": len(exemplos),
        "tokens_supervisionados": tokens_supervisionados,
        "loss_reavaliada": round(loss_media, 6),
        "perplexidade_reavaliada": round(math.exp(min(loss_media, 20)), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_LOCAL_BASE_MODEL)
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    usar_cuda = torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    modelo = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float16 if usar_cuda else None,
    )
    modelo = PeftModel.from_pretrained(modelo, args.adapter_path)
    if usar_cuda:
        modelo = modelo.to("cuda")
    modelo.eval()

    resultado = {
        "base_model": args.base_model,
        "adapter_path": str(args.adapter_path),
        "device": str(next(modelo.parameters()).device),
        "treino": avaliar_split(
            modelo,
            tokenizer,
            _carregar_exemplos(args.data_dir / "finetuning_train.jsonl"),
            args.max_length,
        ),
        "validacao": avaliar_split(
            modelo,
            tokenizer,
            _carregar_exemplos(args.data_dir / "finetuning_val.jsonl"),
            args.max_length,
        ),
    }
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
