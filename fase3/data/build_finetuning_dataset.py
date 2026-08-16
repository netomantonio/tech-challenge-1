"""Pré-processamento, anonimização e curadoria do dataset de fine-tuning da Fase 3.

Combina três fontes:
  - ``protocolos_hospital.json``: protocolos internos, FAQs e modelos de
    documento (fictícios, em português).
  - ``pacientes_sinteticos.json``: usado apenas para gerar exemplos de
    perguntas frequentes sobre exames pendentes/alertas, nunca para expor
    dados de paciente "reais" no dataset de treino.
  - ``sample_medquad.jsonl``: amostra real do MedQuAD/PubMedQA (ver campos
    ``source``/``source_url`` de cada linha para atribuição).

Gera ``finetuning_dataset.jsonl`` (todos os exemplos) e os splits
``finetuning_train.jsonl`` / ``finetuning_val.jsonl`` usados por
``fase3/finetuning/train_lora.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

MIN_OUTPUT_CHARS = 20
MAX_OUTPUT_CHARS = 1500

_CPF_RE = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
_PHONE_RE = re.compile(r"\b(?:\+?55\s?)?\(?\d{2}\)?\s?9?\d{4}-?\d{4}\b")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_NAMED_LABEL_RE = re.compile(
    # `(?!\s*\[)` evita re-detectar o proprio placeholder ja redigido (ex.:
    # "Paciente: [NOME_REDIGIDO]") como se fosse PII em uma segunda checagem.
    # Precisa cobrir o espaco opcional tambem dentro do lookahead, senao o
    # `\s*` anterior pode dar backtrack e "pular" a checagem.
    r"(?im)\b(nome|paciente)\s*:\s*(?!\s*\[)[^\n,;]{2,60}"
)


@dataclass
class FinetuningExample:
    instruction: str
    input: str
    output: str
    source_type: str
    source_id: str

    def as_dict(self) -> dict:
        return asdict(self)


def anonimizar_texto(texto: str) -> str:
    """Redige padroes de PII (CPF, telefone, e-mail, rotulos de nome) em ``texto``.

    Aplicado defensivamente a toda fonte antes de entrar no dataset de
    treino, mesmo quando a fonte já é sintética — é a mesma checagem que
    deveria rodar sobre dados reais do hospital em um cenário de produção.
    """
    texto = _CPF_RE.sub("[CPF_REDIGIDO]", texto)
    texto = _PHONE_RE.sub("[TELEFONE_REDIGIDO]", texto)
    texto = _EMAIL_RE.sub("[EMAIL_REDIGIDO]", texto)
    texto = _NAMED_LABEL_RE.sub(lambda m: f"{m.group(1)}: [NOME_REDIGIDO]", texto)
    return texto


def detectar_pii(texto: str) -> bool:
    """Indica se ``texto`` contem algum padrao de PII (CPF, telefone, e-mail ou rotulo de nome)."""
    return bool(
        _CPF_RE.search(texto)
        or _PHONE_RE.search(texto)
        or _EMAIL_RE.search(texto)
        or _NAMED_LABEL_RE.search(texto)
    )


def _split_faq(conteudo: str) -> list[tuple[str, str]]:
    """Quebra um bloco de FAQ no formato 'P: ... R: ...' em pares (pergunta, resposta)."""
    pares = []
    partes = re.split(r"P:\s*", conteudo)
    for parte in partes:
        parte = parte.strip()
        if not parte or "R:" not in parte:
            continue
        pergunta, resposta = parte.split("R:", 1)
        pergunta, resposta = pergunta.strip(), resposta.strip()
        if pergunta and resposta:
            pares.append((pergunta, resposta))
    return pares


def carregar_protocolos(path: Path) -> list[FinetuningExample]:
    protocolos = json.loads(path.read_text(encoding="utf-8"))
    exemplos: list[FinetuningExample] = []
    for protocolo in protocolos:
        conteudo = anonimizar_texto(protocolo["conteudo"])
        if protocolo["categoria"] == "faq":
            for pergunta, resposta in _split_faq(conteudo):
                exemplos.append(
                    FinetuningExample(
                        instruction=pergunta,
                        input="",
                        output=resposta,
                        source_type="faq_interna",
                        source_id=protocolo["id"],
                    )
                )
        else:
            exemplos.append(
                FinetuningExample(
                    instruction=f"Explique o protocolo interno: {protocolo['titulo']}",
                    input="",
                    output=conteudo,
                    source_type=f"protocolo_{protocolo['categoria']}",
                    source_id=protocolo["id"],
                )
            )
    return exemplos


def carregar_exemplos_pacientes(path: Path) -> list[FinetuningExample]:
    """Gera perguntas de exemplo sobre pendências a partir dos prontuários fictícios.

    Nunca inclui identificação real — os pacientes já são sintéticos e o
    identificador usado é um código interno (``PAC-000x``), nunca nome.
    """
    pacientes = json.loads(path.read_text(encoding="utf-8"))
    exemplos: list[FinetuningExample] = []
    for paciente in pacientes:
        if not paciente["exames_pendentes"]:
            continue
        pendentes = ", ".join(paciente["exames_pendentes"])
        pergunta = (
            f"Quais exames estao pendentes para o paciente {paciente['paciente_id']} "
            "antes de prosseguir com o tratamento?"
        )
        resposta = (
            f"O paciente {paciente['paciente_id']} possui os seguintes exames "
            f"pendentes: {pendentes}. Consulte o protocolo de exames "
            "pre-tratamento (PROT-006) antes de iniciar ou dar continuidade a "
            "qualquer conduta terapeutica."
        )
        exemplos.append(
            FinetuningExample(
                instruction=pergunta,
                input="",
                output=anonimizar_texto(resposta),
                source_type="exemplo_prontuario",
                source_id=paciente["paciente_id"],
            )
        )
    return exemplos


def carregar_exemplos_assistente(path: Path) -> list[FinetuningExample]:
    """Cria exemplos sinteticos alinhados ao uso real do assistente clinico.

    Diferente dos exemplos de memorizacao de protocolo, estes pares ensinam o
    formato esperado na inferencia: resposta objetiva, citacao explicita,
    preservacao de valores do prontuario e validacao medica obrigatoria.
    """
    pacientes = {
        paciente["paciente_id"]: paciente
        for paciente in json.loads(path.read_text(encoding="utf-8"))
    }
    p1 = pacientes["PAC-0001"]
    p3 = pacientes["PAC-0003"]
    p5 = pacientes["PAC-0005"]
    p6 = pacientes["PAC-0006"]

    exemplos = [
        FinetuningExample(
            instruction="Posso iniciar a quimioterapia neoadjuvante hoje?",
            input=(
                f"Codigo {p1['paciente_id']}; exames pendentes: "
                f"{', '.join(p1['exames_pendentes'])}."
            ),
            output=(
                "Nao inicie o ciclo enquanto houver exames obrigatorios pendentes. "
                "Confirme ecocardiograma basal e sorologias HBV, HCV e HIV; a decisao "
                "final exige validacao do medico responsavel. Fonte: [PROT-006]."
            ),
            source_type="assistente_clinico_sintetico",
            source_id="QA-CLIN-001",
        ),
        FinetuningExample(
            instruction="O que fazer com achado BI-RADS 4 pendente de biopsia?",
            input=f"Codigo {p3['paciente_id']}; diagnostico: {p3['diagnostico']}.",
            output=(
                "O achado BI-RADS 4 requer confirmacao histopatologica por biopsia. "
                "Encaminhe o resultado para revisao da equipe assistente antes de "
                "definir qualquer tratamento. Fonte: [PROT-001]."
            ),
            source_type="assistente_clinico_sintetico",
            source_id="QA-CLIN-002",
        ),
        FinetuningExample(
            instruction="A paciente esta com febre e taquicardia. Qual conduta seguir?",
            input=(
                f"Codigo {p5['paciente_id']}; alertas ativos: "
                f"{', '.join(p5['alertas_ativos'])}."
            ),
            output=(
                "Febre de 38.6 C associada a frequencia cardiaca de 110 bpm atende a "
                "dois criterios de SIRS. Acione imediatamente o protocolo de sepse, "
                "comunique a equipe medica e siga as coletas previstas no protocolo; "
                "nao aguarde validacao assincrona. Fonte: [PROT-011]."
            ),
            source_type="assistente_clinico_sintetico",
            source_id="QA-CLIN-003",
        ),
        FinetuningExample(
            instruction="A dor pos-operatoria persiste em 7/10. O que fazer?",
            input=f"Codigo {p6['paciente_id']}; observacao: {p6['observacoes']}.",
            output=(
                "Dor persistente em 7/10 exige reavaliacao clinica e comunicacao a "
                "equipe cirurgica. Siga o protocolo institucional de manejo da dor, "
                "sem indicar medicamento ou dose automaticamente. Fonte: [PROT-004]."
            ),
            source_type="assistente_clinico_sintetico",
            source_id="QA-CLIN-004",
        ),
        FinetuningExample(
            instruction="Quais exames sao obrigatorios antes da quimioterapia sistemica?",
            input="Nenhum paciente informado.",
            output=(
                "Verifique hemograma completo, funcao hepatica e renal, sorologias "
                "HBV, HCV e HIV e, quando o esquema exigir, ecocardiograma ou MUGA "
                "basal. Confirme a validade dos exames antes do ciclo. Fonte: [PROT-006]."
            ),
            source_type="assistente_clinico_sintetico",
            source_id="QA-CLIN-005",
        ),
        FinetuningExample(
            instruction="Qual protocolo usar para um codigo de paciente inexistente?",
            input="Codigo PAC-9999 nao encontrado no prontuario.",
            output=(
                "O codigo do paciente nao foi encontrado. Nao sugira conduta sem "
                "contexto clinico confirmado; solicite a verificacao do identificador "
                "e encerre o fluxo com seguranca."
            ),
            source_type="assistente_clinico_sintetico",
            source_id="QA-CLIN-006",
        ),
    ]
    return [
        FinetuningExample(
            instruction=anonimizar_texto(exemplo.instruction),
            input=anonimizar_texto(exemplo.input),
            output=anonimizar_texto(exemplo.output),
            source_type=exemplo.source_type,
            source_id=exemplo.source_id,
        )
        for exemplo in exemplos
    ]


def carregar_medquad(path: Path) -> list[FinetuningExample]:
    exemplos: list[FinetuningExample] = []
    for linha in path.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha:
            continue
        registro = json.loads(linha)
        exemplos.append(
            FinetuningExample(
                instruction=registro["question"],
                input="",
                output=anonimizar_texto(registro["answer"]),
                source_type=registro["dataset"].lower(),
                source_id=registro["id"],
            )
        )
    return exemplos


def curar(exemplos: list[FinetuningExample]) -> list[FinetuningExample]:
    """Remove duplicatas e exemplos fora da faixa de tamanho aceitavel."""
    vistos: set[str] = set()
    curados: list[FinetuningExample] = []
    for exemplo in exemplos:
        tamanho = len(exemplo.output)
        if tamanho < MIN_OUTPUT_CHARS or tamanho > MAX_OUTPUT_CHARS:
            continue
        chave = hashlib.sha256(
            (exemplo.instruction.lower().strip() + "|" + exemplo.output.lower().strip()).encode("utf-8")
        ).hexdigest()
        if chave in vistos:
            continue
        vistos.add(chave)
        curados.append(exemplo)
    return curados


def dividir_treino_validacao(
    exemplos: list[FinetuningExample], proporcao_validacao: float = 0.15
) -> tuple[list[FinetuningExample], list[FinetuningExample]]:
    n_val = max(1, round(len(exemplos) * proporcao_validacao))
    # Split deterministico (sem random) para reprodutibilidade: pega 1 a cada N para validacao.
    passo = max(1, len(exemplos) // n_val)
    val, train = [], []
    for i, exemplo in enumerate(exemplos):
        if i % passo == 0 and len(val) < n_val:
            val.append(exemplo)
        else:
            train.append(exemplo)
    return train, val


def construir_dataset(data_dir: Path = DATA_DIR) -> list[FinetuningExample]:
    exemplos = (
        carregar_protocolos(data_dir / "protocolos_hospital.json")
        + carregar_exemplos_pacientes(data_dir / "pacientes_sinteticos.json")
        + carregar_exemplos_assistente(data_dir / "pacientes_sinteticos.json")
        + carregar_medquad(data_dir / "sample_medquad.jsonl")
    )
    return curar(exemplos)


def _escrever_jsonl(exemplos: list[FinetuningExample], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for exemplo in exemplos:
            f.write(json.dumps(exemplo.as_dict(), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    args = parser.parse_args()

    exemplos = construir_dataset(args.data_dir)
    train, val = dividir_treino_validacao(exemplos, args.val_fraction)

    _escrever_jsonl(exemplos, args.data_dir / "finetuning_dataset.jsonl")
    _escrever_jsonl(train, args.data_dir / "finetuning_train.jsonl")
    _escrever_jsonl(val, args.data_dir / "finetuning_val.jsonl")

    print(f"Total de exemplos curados: {len(exemplos)}")
    print(f"Treino: {len(train)} | Validacao: {len(val)}")


if __name__ == "__main__":
    main()
