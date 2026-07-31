"""Retrieval sobre os protocolos internos do hospital (Fase 3).

Usa BM25 (``rank_bm25`` via ``langchain_community``) em vez de embeddings,
de proposito: e determinístico, roda 100% offline/CPU e nao exige baixar
nenhum modelo de embedding, o que mantem o pipeline reproduzivel em CI e no
notebook sem depender de rede.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_PROTOCOLOS_PATH = DATA_DIR / "protocolos_hospital.json"
DEFAULT_TOP_K = 3


def carregar_documentos(path: Path = DEFAULT_PROTOCOLOS_PATH) -> list[Document]:
    protocolos = json.loads(path.read_text(encoding="utf-8"))
    return [
        Document(
            page_content=protocolo["conteudo"],
            metadata={
                "id": protocolo["id"],
                "titulo": protocolo["titulo"],
                "categoria": protocolo["categoria"],
            },
        )
        for protocolo in protocolos
    ]


def construir_retriever(
    path: Path = DEFAULT_PROTOCOLOS_PATH, k: int = DEFAULT_TOP_K
) -> BM25Retriever:
    documentos = carregar_documentos(path)
    retriever = BM25Retriever.from_documents(documentos)
    retriever.k = k
    return retriever


def buscar_protocolos(pergunta: str, retriever: Optional[BM25Retriever] = None) -> list[Document]:
    retriever = retriever or construir_retriever()
    return retriever.invoke(pergunta)


def formatar_fontes(documentos: list[Document]) -> list[dict]:
    """Extrai id/titulo dos documentos recuperados para citacao (explainability)."""
    return [
        {"id": doc.metadata["id"], "titulo": doc.metadata["titulo"]}
        for doc in documentos
    ]
