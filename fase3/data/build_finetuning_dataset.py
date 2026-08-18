"""Preprocessamento, anonimizacao e curadoria do dataset alinhado da Fase 3.

A fonte ``assistant_training_cases.json`` descreve oito familias clinicas com
seis variacoes cada. O builder expande essas definicoes em quarenta exemplos
de treino e oito de validacao, usando o mesmo contrato de prompt da inferencia.
MedQuAD e PubMedQA permanecem como referencias no repositorio, mas nao entram
no adapter clinico selecionado.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from fase3.prompting import formatar_protocolos_prompt

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_CASES_PATH = DATA_DIR / "assistant_training_cases.json"
MIN_OUTPUT_CHARS = 20
MAX_OUTPUT_CHARS = 1500

_CPF_RE = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
_PHONE_RE = re.compile(r"\b(?:\+?55\s?)?\(?\d{2}\)?\s?9?\d{4}-?\d{4}\b")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_NAMED_LABEL_RE = re.compile(
    r"(?im)\b(nome|paciente)\s*:\s*(?!\s*\[)[^\n,;]{2,60}"
)


@dataclass
class FinetuningExample:
    # Os cinco primeiros campos preservam a interface historica usada nos testes.
    instruction: str
    input: str
    output: str
    source_type: str
    source_id: str
    contexto_paciente: str = ""
    protocolos: str = ""
    plano_factual: str = ""
    family: str = ""
    split: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def anonimizar_texto(texto: str) -> str:
    """Redige CPF, telefone, e-mail e campos explicitos de nome."""
    texto = _CPF_RE.sub("[CPF_REDIGIDO]", texto)
    texto = _PHONE_RE.sub("[TELEFONE_REDIGIDO]", texto)
    texto = _EMAIL_RE.sub("[EMAIL_REDIGIDO]", texto)
    return _NAMED_LABEL_RE.sub(
        lambda match: f"{match.group(1)}: [NOME_REDIGIDO]", texto
    )


def detectar_pii(texto: str) -> bool:
    return bool(
        _CPF_RE.search(texto)
        or _PHONE_RE.search(texto)
        or _EMAIL_RE.search(texto)
        or _NAMED_LABEL_RE.search(texto)
    )


def curar(exemplos: list[FinetuningExample]) -> list[FinetuningExample]:
    """Remove duplicatas, saidas fora da faixa e qualquer PII residual."""
    vistos: set[str] = set()
    curados: list[FinetuningExample] = []
    for exemplo in exemplos:
        if not MIN_OUTPUT_CHARS <= len(exemplo.output) <= MAX_OUTPUT_CHARS:
            continue
        chave = hashlib.sha256(
            (
                exemplo.instruction.lower().strip()
                + "|"
                + exemplo.output.lower().strip()
            ).encode("utf-8")
        ).hexdigest()
        if chave in vistos:
            continue
        campos_texto = " ".join(
            (
                exemplo.instruction,
                exemplo.input,
                exemplo.output,
                exemplo.contexto_paciente,
                exemplo.protocolos,
                exemplo.plano_factual,
            )
        )
        if detectar_pii(campos_texto):
            continue
        vistos.add(chave)
        curados.append(exemplo)
    return curados


def _carregar_protocolos(path: Path) -> dict[str, dict]:
    return {
        protocolo["id"]: protocolo
        for protocolo in json.loads(path.read_text(encoding="utf-8"))
    }


def _expandir_grupo(
    grupo: dict,
    protocolos_por_id: dict[str, dict],
) -> list[FinetuningExample]:
    variants = grupo.get("variants") or [
        {"question": pergunta} for pergunta in grupo["questions"]
    ]
    if len(variants) != 6:
        raise ValueError(
            f"A familia {grupo['family']} deve possuir exatamente seis variacoes."
        )

    exemplos: list[FinetuningExample] = []
    for indice, variant in enumerate(variants, start=1):
        ids = variant.get("protocol_ids", grupo.get("protocol_ids", []))
        fontes = []
        for protocolo_id in ids:
            if protocolo_id not in protocolos_por_id:
                raise ValueError(f"Protocolo desconhecido no dataset: {protocolo_id}")
            fontes.append(protocolos_por_id[protocolo_id])

        pergunta = anonimizar_texto(variant["question"])
        contexto = anonimizar_texto(
            variant.get("contexto_paciente", grupo["contexto_paciente"])
        )
        plano = anonimizar_texto(
            variant.get("plano_factual", grupo["plano_factual"])
        )
        resposta = anonimizar_texto(
            variant.get("expected_answer", grupo["expected_answer"])
        )
        exemplos.append(
            FinetuningExample(
                instruction=pergunta,
                input=contexto,
                output=resposta,
                source_type="assistente_clinico_sintetico",
                source_id=f"{grupo['family']}-{indice:02d}",
                contexto_paciente=contexto,
                protocolos=formatar_protocolos_prompt(fontes),
                plano_factual=plano,
                family=grupo["family"],
                split="validation" if indice == 6 else "train",
            )
        )
    return exemplos


def construir_dataset(data_dir: Path = DATA_DIR) -> list[FinetuningExample]:
    grupos = json.loads(
        (data_dir / DEFAULT_CASES_PATH.name).read_text(encoding="utf-8")
    )
    protocolos = _carregar_protocolos(data_dir / "protocolos_hospital.json")
    exemplos = [
        exemplo
        for grupo in grupos
        for exemplo in _expandir_grupo(grupo, protocolos)
    ]
    exemplos = curar(exemplos)

    familias = {exemplo.family for exemplo in exemplos}
    if len(exemplos) != 48 or len(familias) != 8:
        raise ValueError(
            "O dataset alinhado deve conter 48 exemplos em oito familias clinicas."
        )
    return exemplos


def dividir_treino_validacao(
    exemplos: list[FinetuningExample], proporcao_validacao: float = 0.15
) -> tuple[list[FinetuningExample], list[FinetuningExample]]:
    """Usa splits explicitos; mantem fallback deterministico para dados legados."""
    if exemplos and all(exemplo.split in {"train", "validation"} for exemplo in exemplos):
        train = [exemplo for exemplo in exemplos if exemplo.split == "train"]
        val = [exemplo for exemplo in exemplos if exemplo.split == "validation"]
        return train, val

    n_val = max(1, round(len(exemplos) * proporcao_validacao))
    passo = max(1, len(exemplos) // n_val)
    val, train = [], []
    for indice, exemplo in enumerate(exemplos):
        if indice % passo == 0 and len(val) < n_val:
            val.append(exemplo)
        else:
            train.append(exemplo)
    return train, val


def _escrever_jsonl(exemplos: list[FinetuningExample], path: Path) -> None:
    with path.open("w", encoding="utf-8") as arquivo:
        for exemplo in exemplos:
            arquivo.write(json.dumps(exemplo.as_dict(), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    exemplos = construir_dataset(args.data_dir)
    train, val = dividir_treino_validacao(exemplos)
    if len(train) != 40 or len(val) != 8:
        raise ValueError(
            "Os splits alinhados devem conter 40 exemplos de treino e oito de validacao."
        )

    _escrever_jsonl(exemplos, args.data_dir / "finetuning_dataset.jsonl")
    _escrever_jsonl(train, args.data_dir / "finetuning_train.jsonl")
    _escrever_jsonl(val, args.data_dir / "finetuning_val.jsonl")
    print(f"Total de exemplos curados: {len(exemplos)}")
    print(f"Treino: {len(train)} | Validacao: {len(val)}")


if __name__ == "__main__":
    main()
