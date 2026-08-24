"""Aplicacao web local para demonstrar o assistente clinico da Fase 3."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import threading
import webbrowser
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from fase3.clinical_flow_graph import executar_fluxo_clinico
from fase3.ehr_tools import DEFAULT_PACIENTES_JSON
from fase3.llm_backend import (
    DEFAULT_LLM_BACKEND,
    DEFAULT_LOCAL_BASE_MODEL,
    LLMUnavailableError,
    get_promoted_local_config,
    get_llm,
    resolver_backend,
)
from fase3.training_service import TrainingJobManager

STATIC_DIR = Path(__file__).resolve().parent / "web"
logger = logging.getLogger("fase3.web")


class ConsultaRequest(BaseModel):
    paciente_id: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9-]+$")
    pergunta: str = Field(min_length=3, max_length=1200)


class TreinoRequest(BaseModel):
    version: str = Field(pattern=r"^qwen2\.5-1\.5b-v[1-9][0-9]*$")
    epochs: int = Field(default=6, ge=1, le=12)
    batch_size: int = Field(default=2, ge=1, le=8)
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=32)
    learning_rate: float = Field(default=2e-5, gt=0, le=1e-3)
    max_length: int = Field(default=512, ge=128, le=2048)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    lora_r: int = Field(default=16, ge=1, le=256)
    lora_alpha: int = Field(default=32, ge=1, le=512)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)


class AdapterRequest(BaseModel):
    version: str = Field(pattern=r"^qwen2\.5-1\.5b-v[1-9][0-9]*$")


class CalibracaoRequest(AdapterRequest):
    scales: list[float] = Field(
        default_factory=lambda: [0.25, 0.5, 0.75, 1.0],
        min_length=1,
        max_length=8,
    )


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

    def descarregar(self) -> None:
        """Libera o modelo de inferencia antes de um job que use a GPU."""
        with self._lock:
            self._llm = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass


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


def create_app(
    runtime: AssistantRuntime | None = None,
    training_manager: TrainingJobManager | None = None,
) -> FastAPI:
    assistant_runtime = runtime or AssistantRuntime()
    jobs = training_manager or TrainingJobManager()
    app = FastAPI(
        title="Assistente Clinico - Fase 3",
        description="Interface local do fluxo clinico academico com LangGraph e LoRA.",
        version="1.0.0",
    )

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        promoted = get_promoted_local_config()
        return {
            "status": "ready",
            "backend": assistant_runtime.backend,
            "modelo_carregado": assistant_runtime.carregado,
            "modelo_base": DEFAULT_LOCAL_BASE_MODEL if assistant_runtime.backend == "local" else None,
            "adapter": promoted["adapter_path"] if assistant_runtime.backend == "local" else None,
            "escala_lora": promoted["lora_scale"] if assistant_runtime.backend == "local" else None,
            "cache_hf": os.getenv("HF_HOME") if assistant_runtime.backend == "local" else None,
            "offline": os.getenv("HF_HUB_OFFLINE") == "1" if assistant_runtime.backend == "local" else None,
        }

    @app.get("/api/pacientes")
    async def pacientes() -> list[dict[str, Any]]:
        return _carregar_pacientes()

    @app.post("/api/consultas")
    async def consultar(request: ConsultaRequest) -> dict[str, Any]:
        if jobs.ativo:
            raise HTTPException(
                status_code=409,
                detail="Uma operacao de modelo esta em andamento. Aguarde sua conclusao antes de consultar.",
            )
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

    def validar_acesso_local(http_request: Request) -> None:
        host = http_request.client.host if http_request.client else None
        if host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
            raise HTTPException(status_code=403, detail="Operacoes de modelo aceitam apenas acesso local.")
        if assistant_runtime.backend != "local":
            raise HTTPException(status_code=409, detail="Operacoes exigem o backend local.")

    async def preparar_job(http_request: Request) -> None:
        validar_acesso_local(http_request)
        if jobs.ativo:
            raise HTTPException(status_code=409, detail="Ja existe uma operacao em andamento.")
        await run_in_threadpool(assistant_runtime.descarregar)

    @app.get("/api/treinamento")
    async def treinamento_overview(http_request: Request) -> dict[str, Any]:
        validar_acesso_local(http_request)
        return jobs.overview()

    @app.get("/api/treinamento/job")
    async def treinamento_job(http_request: Request) -> dict[str, Any]:
        validar_acesso_local(http_request)
        return {"job": jobs.job()}

    @app.post("/api/treinamento/dataset")
    async def treinamento_dataset(http_request: Request) -> dict[str, Any]:
        await preparar_job(http_request)
        return jobs.iniciar_dataset()

    @app.post("/api/treinamento/iniciar")
    async def treinamento_iniciar(request: TreinoRequest, http_request: Request) -> dict[str, Any]:
        await preparar_job(http_request)
        try:
            return jobs.iniciar_treino(request.model_dump())
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/treinamento/loss")
    async def treinamento_loss(request: AdapterRequest, http_request: Request) -> dict[str, Any]:
        await preparar_job(http_request)
        try:
            return jobs.iniciar_loss(request.version)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/treinamento/calibrar")
    async def treinamento_calibrar(
        request: CalibracaoRequest, http_request: Request
    ) -> dict[str, Any]:
        if any(not 0.0 < scale <= 1.0 for scale in request.scales):
            raise HTTPException(status_code=422, detail="Escalas LoRA devem estar em (0, 1].")
        await preparar_job(http_request)
        try:
            return jobs.iniciar_calibracao(request.version, request.scales)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/treinamento/cancelar")
    async def treinamento_cancelar(http_request: Request) -> dict[str, Any]:
        validar_acesso_local(http_request)
        try:
            return jobs.cancelar()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/treinamento/promover")
    async def treinamento_promover(request: AdapterRequest, http_request: Request) -> dict[str, Any]:
        validar_acesso_local(http_request)
        try:
            config = jobs.promover(request.version)
            await run_in_threadpool(assistant_runtime.descarregar)
            return config
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> FileResponse:
        return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")

    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")
    return app


runtime = AssistantRuntime()
training_manager = TrainingJobManager()
app = create_app(runtime, training_manager)


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
