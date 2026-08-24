"""Aplicacao web local para demonstrar o assistente clinico da Fase 3."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import socket
import threading
import webbrowser
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
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
from fase3.model_registry import (
    allowed_model_roots,
    get_model,
    hardware_capabilities,
    model_capabilities,
    register_model,
    unregister_model,
)
from fase3.training_service import TrainingJobManager

STATIC_DIR = Path(__file__).resolve().parent / "web"
logger = logging.getLogger("fase3.web")
API_VERSION = "2.0.0"


class ConsultaRequest(BaseModel):
    paciente_id: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9-]+$")
    pergunta: str = Field(min_length=3, max_length=1200)


class TreinoRequest(BaseModel):
    version: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,63}-v[1-9][0-9]*$")
    model_alias: str = Field(default="qwen2.5-1.5b", pattern=r"^[a-z0-9][a-z0-9._-]{1,63}$")
    precision: str = Field(default="auto", pattern=r"^(auto|fp32|fp16|bf16|nf4)$")
    gradient_checkpointing: bool = False
    trust_remote_code: bool = False
    target_modules: list[str] = Field(default_factory=list, max_length=64)
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
    version: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,63}-v[1-9][0-9]*$")


class ModelRequest(BaseModel):
    alias: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,63}$")
    label: str = Field(min_length=2, max_length=100)
    source_type: str = Field(pattern=r"^(huggingface|local)$")
    source: str = Field(min_length=1, max_length=500)
    revision: str | None = Field(default=None, max_length=160)
    target_modules: list[str] = Field(default_factory=list, max_length=64)
    trust_remote_code: bool = False


class ModelInstallRequest(BaseModel):
    alias: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,63}$")


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
        "etapas_executadas": estado.get("etapas_executadas", []),
    }


def _allowed_origins() -> list[str]:
    configured = os.getenv("FASE3_ALLOWED_ORIGINS", "")
    return [item.strip().rstrip("/") for item in configured.split(",") if item.strip()]


def _remote_training_enabled() -> bool:
    return os.getenv("FASE3_ALLOW_REMOTE_TRAINING", "0").strip().lower() in {"1", "true", "yes"}


def _private_addresses() -> list[str]:
    addresses: set[str] = set()
    try:
        for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = item[4][0]
            if address != "127.0.0.1":
                addresses.add(address)
    except OSError:
        pass
    return sorted(addresses)


def create_app(
    runtime: AssistantRuntime | None = None,
    training_manager: TrainingJobManager | None = None,
) -> FastAPI:
    assistant_runtime = runtime or AssistantRuntime()
    jobs = training_manager or TrainingJobManager()
    app = FastAPI(
        title="Assistente Clinico - Fase 3",
        description="Interface local do fluxo clinico academico com LangGraph e LoRA.",
        version=API_VERSION,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Accept", "X-Fase3-Client"],
        expose_headers=["X-Fase3-Api-Version"],
        max_age=600,
        allow_private_network=True,
    )

    @app.middleware("http")
    async def private_network_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Fase3-Api-Version"] = API_VERSION
        if request.headers.get("access-control-request-private-network", "").lower() == "true":
            response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response

    @app.get("/api/capabilities")
    async def capabilities() -> dict[str, Any]:
        promoted = get_promoted_local_config()
        return {
            "api_version": API_VERSION,
            "instance_name": os.getenv("FASE3_INSTANCE_NAME", socket.gethostname()),
            "backend": assistant_runtime.backend,
            "hardware": hardware_capabilities(),
            "models": model_capabilities(),
            "promoted": promoted,
            "features": {
                "clinical_chat": True,
                "training": assistant_runtime.backend == "local",
                "remote_training": _remote_training_enabled(),
                "model_registration": assistant_runtime.backend == "local",
                "model_installation": assistant_runtime.backend == "local",
                "private_network_access": True,
                "api_major": 2,
            },
            "allowed_origins": _allowed_origins(),
            "permissions": {
                "model_roots": allowed_model_roots(),
                "trust_remote_code_requires_revision": True,
            },
        }

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        promoted = get_promoted_local_config()
        return {
            "status": "ready",
            "backend": assistant_runtime.backend,
            "modelo_carregado": assistant_runtime.carregado,
            "modelo_base": promoted.get("base_model", DEFAULT_LOCAL_BASE_MODEL) if assistant_runtime.backend == "local" else None,
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

    def validar_operacao_modelo(http_request: Request) -> None:
        host = http_request.client.host if http_request.client else None
        local = host in {"127.0.0.1", "::1", "localhost", "testclient"}
        if not local and not _remote_training_enabled():
            raise HTTPException(
                status_code=403,
                detail="Operacoes remotas estao desativadas. O proprietario deve definir FASE3_ALLOW_REMOTE_TRAINING=1.",
            )
        if assistant_runtime.backend != "local":
            raise HTTPException(status_code=409, detail="Operacoes exigem o backend local.")

    async def preparar_job(http_request: Request) -> None:
        validar_operacao_modelo(http_request)
        if jobs.ativo:
            raise HTTPException(status_code=409, detail="Ja existe uma operacao em andamento.")
        await run_in_threadpool(assistant_runtime.descarregar)

    @app.get("/api/treinamento")
    async def treinamento_overview(http_request: Request) -> dict[str, Any]:
        validar_operacao_modelo(http_request)
        return jobs.overview()

    @app.get("/api/treinamento/job")
    async def treinamento_job(http_request: Request) -> dict[str, Any]:
        validar_operacao_modelo(http_request)
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
        validar_operacao_modelo(http_request)
        try:
            return jobs.cancelar()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/treinamento/promover")
    async def treinamento_promover(request: AdapterRequest, http_request: Request) -> dict[str, Any]:
        validar_operacao_modelo(http_request)
        try:
            config = jobs.promover(request.version)
            await run_in_threadpool(assistant_runtime.descarregar)
            return config
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/modelos")
    async def modelos() -> list[dict[str, Any]]:
        return model_capabilities()

    @app.post("/api/modelos")
    async def cadastrar_modelo(request: ModelRequest, http_request: Request) -> dict[str, Any]:
        validar_operacao_modelo(http_request)
        try:
            return register_model(request.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete("/api/modelos/{alias}")
    async def remover_modelo(alias: str, http_request: Request) -> dict[str, bool]:
        validar_operacao_modelo(http_request)
        try:
            unregister_model(alias)
            return {"removed": True}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/modelos/instalar")
    async def instalar_modelo(request: ModelInstallRequest, http_request: Request) -> dict[str, Any]:
        await preparar_job(http_request)
        try:
            get_model(request.alias)
            return jobs.iniciar_instalacao(request.alias)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

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
    parser.add_argument("--allowed-origin", action="append", default=[])
    parser.add_argument("--allow-remote-training", action="store_true")
    parser.add_argument("--instance-name")
    parser.add_argument("--no-open-browser", action="store_true")
    args = parser.parse_args()

    if args.allowed_origin:
        current = _allowed_origins()
        os.environ["FASE3_ALLOWED_ORIGINS"] = ",".join(dict.fromkeys(current + args.allowed_origin))
    if args.allow_remote_training:
        os.environ["FASE3_ALLOW_REMOTE_TRAINING"] = "1"
    if args.instance_name:
        os.environ["FASE3_INSTANCE_NAME"] = args.instance_name

    runtime.configure(args.backend)
    url = f"http://{args.host}:{args.port}"
    if not args.no_open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    import uvicorn

    served_app = create_app(runtime, training_manager)
    print(f"Backend: http://127.0.0.1:{args.port}")
    if args.host == "0.0.0.0":
        for address in _private_addresses():
            print(f"Rede local: http://{address}:{args.port}")
    print(f"Origens autorizadas: {', '.join(_allowed_origins()) or 'somente mesma origem'}")
    print(f"Treinamento remoto: {'ATIVADO' if _remote_training_enabled() else 'desativado'}")
    if _remote_training_enabled():
        print("AVISO: operacoes remotas de modelo estao habilitadas sem autenticacao.")
    uvicorn.run(served_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
