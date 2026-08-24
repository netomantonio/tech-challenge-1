"""Aplicacao web local para demonstrar o assistente clinico da Fase 3."""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import webbrowser
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from fase3.clinical_flow_graph import executar_fluxo_clinico
from fase3.ehr_tools import DEFAULT_PACIENTES_JSON
from fase3.llm_backend import (
    DEFAULT_LLM_BACKEND,
    DEFAULT_LOCAL_ADAPTER_PATH,
    DEFAULT_LOCAL_BASE_MODEL,
    DEFAULT_LOCAL_LORA_SCALE,
    LLMUnavailableError,
    get_llm,
    resolver_backend,
)

STATIC_DIR = Path(__file__).resolve().parent / "web"
logger = logging.getLogger("fase3.web")


class ConsultaRequest(BaseModel):
    paciente_id: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9-]+$")
    pergunta: str = Field(min_length=3, max_length=1200)


class AssistantRuntime:
    """Mantem uma unica instancia do modelo e serializa o uso da GPU."""

    def __init__(self, backend: str | None = None) -> None:
        self.backend = resolver_backend(backend)
        self._llm: Any = None
        self._lock = threading.Lock()

    @property
    def carregado(self) -> bool:
        return self._llm is not None

    def configure(self, backend: str) -> None:
        if self._llm is not None and backend != self.backend:
            raise RuntimeError("O backend nao pode ser alterado depois de carregar o modelo.")
        self.backend = backend

    def _carregar_llm(self):
        if self._llm is None:
            if self.backend == "fake":
                self._llm = get_llm(
                    "fake",
                    respostas=[
                        "A conduta deve seguir o plano factual recuperado e ser "
                        "confirmada pela equipe medica. Fonte: [PROT-006]."
                    ],
                )
            else:
                self._llm = get_llm(self.backend)
        return self._llm

    def consultar(self, paciente_id: str, pergunta: str) -> dict[str, Any]:
        with self._lock:
            llm = self._carregar_llm()
            return executar_fluxo_clinico(paciente_id, pergunta, llm=llm)


def _carregar_pacientes() -> list[dict[str, Any]]:
    return json.loads(DEFAULT_PACIENTES_JSON.read_text(encoding="utf-8"))


def _serializar_estado(estado: dict[str, Any]) -> dict[str, Any]:
    sugestao = estado.get("sugestao") or {}
    return {
        "paciente_encontrado": estado.get("paciente_encontrado", False),
        "paciente": estado.get("paciente"),
        "resposta": sugestao.get("resposta"),
        "fontes": sugestao.get("fontes", []),
        "modo_resposta": sugestao.get("modo_resposta"),
        "bloqueado": estado.get("bloqueado", False),
        "motivo_bloqueio": sugestao.get("motivo_bloqueio"),
        "exames_pendentes": estado.get("exames_pendentes", []),
        "tem_exames_pendentes": estado.get("tem_exames_pendentes", False),
        "rota_exames": estado.get("rota_exames"),
        "alertas": estado.get("alertas", []),
    }


def create_app(runtime: AssistantRuntime | None = None) -> FastAPI:
    assistant_runtime = runtime or AssistantRuntime()
    app = FastAPI(
        title="Assistente Clinico - Fase 3",
        description="Interface local do fluxo clinico academico com LangGraph e LoRA.",
        version="1.0.0",
    )

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        return {
            "status": "ready",
            "backend": assistant_runtime.backend,
            "modelo_carregado": assistant_runtime.carregado,
            "modelo_base": DEFAULT_LOCAL_BASE_MODEL if assistant_runtime.backend == "local" else None,
            "adapter": str(DEFAULT_LOCAL_ADAPTER_PATH) if assistant_runtime.backend == "local" else None,
            "escala_lora": DEFAULT_LOCAL_LORA_SCALE if assistant_runtime.backend == "local" else None,
            "cache_hf": os.getenv("HF_HOME") if assistant_runtime.backend == "local" else None,
            "offline": os.getenv("HF_HUB_OFFLINE") == "1" if assistant_runtime.backend == "local" else None,
        }

    @app.get("/api/pacientes")
    async def pacientes() -> list[dict[str, Any]]:
        return _carregar_pacientes()

    @app.post("/api/consultas")
    async def consultar(request: ConsultaRequest) -> dict[str, Any]:
        try:
            estado = await run_in_threadpool(
                assistant_runtime.consultar,
                request.paciente_id,
                request.pergunta.strip(),
            )
            return _serializar_estado(estado)
        except LLMUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (OSError, RuntimeError, ValueError) as exc:
            logger.exception("Falha ao executar consulta clinica")
            raise HTTPException(
                status_code=500,
                detail="Nao foi possivel executar o assistente local. Consulte o terminal do servico.",
            ) from exc

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> FileResponse:
        return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")

    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")
    return app


runtime = AssistantRuntime()
app = create_app(runtime)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=["local", "groq", "fake"],
        default=DEFAULT_LLM_BACKEND,
        help="Backend de inferencia (padrao: local).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--no-open-browser", action="store_true")
    args = parser.parse_args()

    runtime.configure(args.backend)
    url = f"http://{args.host}:{args.port}"
    if not args.no_open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
