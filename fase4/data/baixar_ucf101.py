"""Recorte reproduzivel do UCF101 para avaliar percepcao em videos reais.

Baixa tres clipes de um espelho com revisao fixa e converte para a interface.
Nao ha treinamento nem rotulos clinicos: as classes sao acoes humanas.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from fase4.config import DIR_DADOS

DIR_UCF101 = DIR_DADOS / "ucf101"
URL = "https://storage.googleapis.com/thumos14_files/UCF101_videos.zip"
ARTIGO = "https://arxiv.org/abs/1212.0402"
CLASSES = ("JumpingJack", "BodyWeightSquats", "Lunges")
MANIFESTO = Path(__file__).with_name("ucf101_manifesto.json")


def id_do_clipe(nome: str) -> str:
    if not re.fullmatch(r"v_(JumpingJack|BodyWeightSquats|Lunges)_g\d{2}_c\d{2}\.avi", nome):
        raise ValueError("Clipe fora do recorte UCF101 definido para esta avaliacao")
    return "UCF-" + Path(nome).stem.removeprefix("v_")


def converter_video(origem: Path, destino: Path) -> dict:
    import cv2
    captura = cv2.VideoCapture(str(origem))
    fps = captura.get(cv2.CAP_PROP_FPS)
    largura = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not captura.isOpened() or fps <= 0 or not largura or not altura:
        captura.release()
        raise ValueError("Video UCF101 ilegivel")
    escritor = cv2.VideoWriter(str(destino), cv2.VideoWriter_fourcc(*"mp4v"), fps, (largura, altura))
    if not escritor.isOpened():
        captura.release()
        raise RuntimeError("Nao foi possivel abrir o gravador MP4")
    quadros = 0
    try:
        while True:
            ok, imagem = captura.read()
            if not ok:
                break
            escritor.write(imagem)
            quadros += 1
    finally:
        captura.release()
        escritor.release()
    if not quadros:
        raise ValueError("Video sem quadros")
    return {"fps": fps, "largura": largura, "altura": altura, "quadros": quadros, "duracao_s": quadros / fps}


def _baixar_clipes(destino: Path):
    """Espelho com revisao registrada; preserva os nomes originais UCF101."""
    import requests
    manifesto = json.loads(MANIFESTO.read_text(encoding="utf-8"))
    for fonte in manifesto:
        nome, url = fonte["clipe"], fonte["url"]
        original = destino / id_do_clipe(nome) / nome
        if original.exists() and hashlib.sha256(original.read_bytes()).hexdigest() == fonte["sha256_original"]:
            conteudo = original.read_bytes()
        else:
            resposta = requests.get(url, timeout=60)
            resposta.raise_for_status()
            conteudo = resposta.content
        if hashlib.sha256(conteudo).hexdigest() != fonte["sha256_original"]:
            raise ValueError(f"Hash inesperado para {nome}; download nao utilizado")
        yield nome, conteudo, url, fonte["revisao_espelho"]


def preparar_recorte(destino: Path = DIR_UCF101) -> list[dict]:
    fontes = []
    for nome, conteudo, url, revisao in _baixar_clipes(destino):
        pasta = destino / id_do_clipe(nome)
        pasta.mkdir(parents=True, exist_ok=True)
        original = pasta / nome
        original.write_bytes(conteudo)
        info = converter_video(original, pasta / "fisioterapia.mp4")
        fonte = {
            "dataset": "UCF101", "clipe": nome, "classe": nome.split("_")[1],
            "grupo": nome.split("_")[2], "url": url, "revisao_espelho": revisao, "artigo": ARTIGO,
            "espelho": "sumith2425/UCF101", "url_distribuicao_alternativa": URL,
            "catalogo": "https://github.com/xiaobai1217/Awesome-Video-Datasets",
            "sha256_original": hashlib.sha256(original.read_bytes()).hexdigest(),
            "sha256_mp4": hashlib.sha256((pasta / "fisioterapia.mp4").read_bytes()).hexdigest(),
            "conversao": "AVI para MP4, todos os quadros, sem audio, FPS e resolucao preservados",
            "uso": "avaliacao exploratoria de percepcao; nao e video clinico nem rotulo de anomalia",
            **info,
        }
        (pasta / "fonte.json").write_text(json.dumps(fonte, ensure_ascii=False, indent=2), encoding="utf-8")
        fontes.append(fonte)
        print(f"Preparado {pasta.name}: {info['quadros']} quadros", flush=True)
    return fontes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destino", type=Path, default=DIR_UCF101)
    args = parser.parse_args()
    preparar_recorte(args.destino)


if __name__ == "__main__":
    main()
