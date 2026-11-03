"""Interface web local do monitoramento multimodal (Fase 4).

Mesmo padrao da Fase 3: FastAPI servindo uma pagina em HTML/JS puro de
`fase4/web/`, com a logica toda no backend. A pagina permite processar um
paciente, ler os relatorios, ouvir o audio e ver o video anotado, acompanhar o
monitor de leito em tempo real (Server-Sent Events), receber os alertas num
painel da equipe (inclusive por webhook), enviar gravacoes proprias e marcar a
area critica sobre um quadro do video.

    python -m fase4.web_app                 # http://127.0.0.1:8020
    python -m fase4.web_app --port 8021 --no-open-browser

Para a CLI entregar alertas nesta interface:
    $env:FASE4_WEBHOOK_URL = "http://127.0.0.1:8020/api/webhook/alertas"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import threading
import webbrowser
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from fase4.azure_services import ServicoAzureIndisponivelError, status_servicos
from fase4.markdown_simples import markdown_para_html
from fase4.video_analysis import BackendIndisponivelError
from fase4.web_jobs import JobOcupadoError
from fase4.web_service import (
    ARQUIVO_RESULTADO_WEB,
    ARQUIVOS_UPLOAD,
    RELATORIOS,
    SSE_HEADERS,
    ConteudoInvalido,
    Deps,
    IdInvalido,
    Paciente,
    PacienteNaoEncontrado,
    RepositorioPacientes,
    backends_disponiveis,
    caminho_video,
    descrever_video,
    emitir_sse,
    executar_processamento,
    faixas_serializadas,
    fluxo_tempo_real,
    formatar_sse,
    precisa_de_job,
    quadro_jpeg,
    remover_paciente_enviado,
    resolver_midia,
    tipo_do_upload,
    transcodificar_h264,
    validar_area_critica,
    validar_conteudo,
    validar_id,
)

STATIC_DIR = Path(__file__).resolve().parent / "web"
logger = logging.getLogger("fase4.web")
API_VERSION = "1.0.0"
LOOPBACK = {"127.0.0.1", "::1", "localhost", "testclient", None}
CSP = (
    "default-src 'self'; img-src 'self' data: blob:; media-src 'self'; connect-src 'self'; "
    "script-src 'self'; style-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
)
TAMANHO_MAXIMO_WEBHOOK = 256 * 1024


# ---------------------------------------------------------------------------
# Modelos de requisicao
# ---------------------------------------------------------------------------


class ProcessarRequest(BaseModel):
    provedor_azure: Literal["auto", "azure", "offline", "local"] = "auto"
    backend_pose: Literal["anotado", "mediapipe"] = "anotado"
    backend_detector: Literal["anotado", "yolov8"] = "anotado"
    area_critica: list[float] | None = Field(default=None, min_length=4, max_length=4)
    arquivo_audio: str = Field(default="consulta.wav", pattern=r"^consulta[a-z0-9_-]*\.wav$")
    gravar_video_anotado: bool = True
    modo: Literal["auto", "sincrono", "job"] = "auto"


class NovoPacienteRequest(BaseModel):
    paciente_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{1,39}$")


class PhysionetRequest(BaseModel):
    registro: str = Field(default="p04/p044083/p044083-2112-05-04-19-50n", pattern=r"^[A-Za-z0-9][A-Za-z0-9_/-]{2,80}$")
    minutos: int = Field(default=720, ge=0, le=3000)


# ---------------------------------------------------------------------------
# Aplicacao
# ---------------------------------------------------------------------------


def _remoto_permitido() -> bool:
    return os.getenv("FASE4_WEB_REMOTO", "0").strip().lower() in {"1", "true", "yes"}


def _exigir_loopback(request: Request) -> None:
    host = request.client.host if request.client else None
    if host not in LOOPBACK and not _remoto_permitido():
        raise HTTPException(status_code=403, detail="Operacao permitida apenas a partir desta maquina.")


def create_app(deps: Deps | None = None) -> FastAPI:
    deps = deps or Deps()
    repositorio = RepositorioPacientes(deps)
    app = FastAPI(
        title="Monitoramento multimodal - Fase 4",
        description="Interface local do pipeline de video, audio, sinais vitais, fusao e alertas.",
        version=API_VERSION,
    )
    app.state.deps = deps

    @app.middleware("http")
    async def cabecalhos_de_seguranca(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Fase4-Api-Version"] = API_VERSION
        response.headers["Content-Security-Policy"] = CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    # --- helpers -------------------------------------------------------------

    def resolver(paciente_id: str) -> Paciente:
        try:
            return repositorio.resolver(paciente_id)
        except IdInvalido as erro:
            raise HTTPException(status_code=422, detail=str(erro)) from erro
        except PacienteNaoEncontrado as erro:
            raise HTTPException(status_code=404, detail=str(erro)) from erro

    def job_ou_409() -> None:
        if deps.jobs.ativo:
            raise HTTPException(
                status_code=409,
                detail="Ha uma execucao em segundo plano em andamento. Aguarde ou cancele antes de processar.",
            )

    # --- estado --------------------------------------------------------------

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        return {
            "status": "ready",
            "versao": API_VERSION,
            "servicos": status_servicos(),
            "backends": backends_disponiveis(deps),
            "faixas": faixas_serializadas(),
            "job": deps.jobs.snapshot(),
            "webhook_url_sugerida": "/api/webhook/alertas",
            "diretorios": {
                "amostras": str(deps.dir_amostras),
                "enviados": str(deps.dir_enviados),
                "physionet": str(deps.dir_physionet),
                "resultados": str(deps.dir_resultados),
            },
            "uploads": {"arquivos": list(ARQUIVOS_UPLOAD), "limites_bytes": deps.limites_upload},
        }

    # --- pacientes -----------------------------------------------------------

    @app.get("/api/pacientes")
    async def listar_pacientes() -> list[dict[str, Any]]:
        return await run_in_threadpool(repositorio.listar)

    @app.get("/api/pacientes/{paciente_id}")
    async def detalhar_paciente(paciente_id: str) -> dict[str, Any]:
        paciente = resolver(paciente_id)
        return await run_in_threadpool(repositorio.descrever, paciente, True)

    @app.post("/api/pacientes/{paciente_id}/processar")
    async def processar(paciente_id: str, pedido: ProcessarRequest, response: Response) -> dict[str, Any]:
        paciente = resolver(paciente_id)
        opcoes = pedido.model_dump()
        video = caminho_video(paciente, "original")
        try:
            info = await run_in_threadpool(descrever_video, video, deps) if video else None
            opcoes["area_critica"] = validar_area_critica(pedido.area_critica, info)
        except ValueError as erro:
            raise HTTPException(status_code=422, detail=str(erro)) from erro
        except BackendIndisponivelError as erro:
            raise HTTPException(status_code=503, detail=str(erro)) from erro
        if opcoes["area_critica"] is not None:
            opcoes["area_critica"] = list(opcoes["area_critica"])

        em_job = pedido.modo == "job" or (pedido.modo == "auto" and precisa_de_job(paciente, opcoes))
        job_ou_409()

        if em_job:
            backends = backends_disponiveis(deps)
            faltando = [
                nome for nome, chave in (("mediapipe", "mediapipe"), ("yolov8", "yolov8"))
                if opcoes.get("backend_pose" if nome == "mediapipe" else "backend_detector") != "anotado"
                and not backends[chave]
            ]
            if opcoes["provedor_azure"] == "local" and not backends["whisper"]:
                faltando.append("faster-whisper")
            if faltando:
                raise HTTPException(
                    status_code=503,
                    detail="Backend nao instalado: " + ", ".join(faltando) + " (pip install -r requirements-fase4.txt).",
                )

            def tarefa(informar, cancelado):
                return executar_processamento(deps, paciente, opcoes, informar, cancelado)

            try:
                job = deps.jobs.iniciar(
                    "processamento", f"Processar {paciente.paciente_id}", tarefa,
                    metadados={"paciente_id": paciente.paciente_id, **opcoes},
                )
            except JobOcupadoError as erro:
                raise HTTPException(status_code=409, detail=str(erro)) from erro
            response.status_code = 202
            return {"modo": "job", "job": job}

        if not deps.trava_sincrona.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Outro processamento esta em andamento nesta interface.")
        try:
            resultado = await run_in_threadpool(executar_processamento, deps, paciente, opcoes)
        except (BackendIndisponivelError, ServicoAzureIndisponivelError) as erro:
            raise HTTPException(status_code=503, detail=str(erro)) from erro
        except FileNotFoundError as erro:
            raise HTTPException(status_code=404, detail=str(erro)) from erro
        finally:
            deps.trava_sincrona.release()
        return {"modo": "sincrono", "resultado": resultado}

    @app.get("/api/pacientes/{paciente_id}/resultado")
    async def resultado(paciente_id: str) -> Any:
        paciente = resolver(paciente_id)
        caminho = paciente.pasta_resultados / ARQUIVO_RESULTADO_WEB
        if not caminho.exists():
            raise HTTPException(status_code=404, detail="Paciente ainda nao processado pela interface.")
        return JSONResponse(content=__import__("json").loads(caminho.read_text(encoding="utf-8")))

    @app.get("/api/pacientes/{paciente_id}/relatorios/{nome}")
    async def relatorio(paciente_id: str, nome: str) -> dict[str, Any]:
        paciente = resolver(paciente_id)
        if nome not in RELATORIOS:
            raise HTTPException(status_code=404, detail="Relatorio desconhecido.")
        caminho = paciente.pasta_resultados / f"{nome}.md"
        if not caminho.exists():
            raise HTTPException(status_code=404, detail="Relatorio ainda nao gerado para este paciente.")
        markdown = caminho.read_text(encoding="utf-8")
        return {
            "nome": nome,
            "html": markdown_para_html(markdown),
            "markdown": markdown,
            "modificado_em": __import__("datetime").datetime.fromtimestamp(
                caminho.stat().st_mtime, tz=__import__("datetime").timezone.utc
            ).isoformat(),
        }

    @app.get("/api/pacientes/{paciente_id}/midia/{nome}")
    async def midia(paciente_id: str, nome: str, h264: int = 0) -> FileResponse:
        paciente = resolver(paciente_id)
        encontrado = resolver_midia(paciente, nome, h264=bool(h264))
        if encontrado is None:
            raise HTTPException(status_code=404, detail="Midia nao encontrada.")
        caminho, media_type = encontrado
        return FileResponse(caminho, media_type=media_type, filename=caminho.name)

    @app.get("/api/pacientes/{paciente_id}/video/info")
    async def video_info(paciente_id: str, fonte: Literal["original", "anotado"] = "original", transcodificar: int = 0) -> dict[str, Any]:
        paciente = resolver(paciente_id)
        caminho = caminho_video(paciente, fonte)
        if caminho is None:
            raise HTTPException(status_code=404, detail="Video nao encontrado.")
        try:
            if transcodificar:
                await run_in_threadpool(transcodificar_h264, deps, caminho)
            info = await run_in_threadpool(descrever_video, caminho, deps)
        except BackendIndisponivelError as erro:
            raise HTTPException(status_code=503, detail=str(erro)) from erro
        info["fonte"] = fonte
        info["url"] = f"/api/pacientes/{paciente_id}/midia/" + (
            "fisioterapia_anotado.mp4" if fonte == "anotado" else "fisioterapia.mp4"
        )
        return info

    @app.get("/api/pacientes/{paciente_id}/quadro.jpg")
    async def quadro(paciente_id: str, t: float = 0.0, fonte: Literal["original", "anotado"] = "original") -> Response:
        paciente = resolver(paciente_id)
        try:
            dados, largura, altura = await run_in_threadpool(quadro_jpeg, paciente, fonte, t)
        except PacienteNaoEncontrado as erro:
            raise HTTPException(status_code=404, detail=str(erro)) from erro
        except ValueError as erro:
            raise HTTPException(status_code=422, detail=str(erro)) from erro
        except BackendIndisponivelError as erro:
            raise HTTPException(status_code=503, detail=str(erro)) from erro
        return Response(
            content=dados, media_type="image/jpeg",
            headers={"X-Quadro-Largura": str(largura), "X-Quadro-Altura": str(altura), "Cache-Control": "no-store"},
        )

    # --- tempo real ----------------------------------------------------------

    @app.get("/api/pacientes/{paciente_id}/tempo-real")
    async def tempo_real(
        request: Request,
        paciente_id: str,
        ms_por_minuto: int = Query(default=50, ge=0, le=2000),
        multivariado: int = Query(default=1, ge=0, le=1),
        limite_minutos: int | None = Query(default=None, ge=1, le=100000),
    ) -> StreamingResponse:
        paciente = resolver(paciente_id)
        if not (paciente.pasta_amostras / "sinais_vitais.csv").exists():
            raise HTTPException(status_code=404, detail="Paciente sem serie de sinais vitais.")
        loop = asyncio.get_running_loop()
        fila: asyncio.Queue = asyncio.Queue()
        parar = threading.Event()
        threading.Thread(
            target=fluxo_tempo_real,
            args=(deps, paciente, ms_por_minuto, bool(multivariado), limite_minutos, loop, fila, parar),
            name=f"tempo-real-{paciente_id}",
            daemon=True,
        ).start()
        return StreamingResponse(
            emitir_sse(request, fila, deps.sse_keepalive_s, encerrar_em=("fim", "erro"), ao_terminar=parar.set),
            media_type="text/event-stream; charset=utf-8",
            headers=SSE_HEADERS,
        )

    # --- alertas / equipe ----------------------------------------------------

    @app.get("/api/alertas")
    async def alertas(desde: int = Query(default=0, ge=0), limite: int = Query(default=200, ge=1, le=500)) -> dict[str, Any]:
        return {"alertas": deps.barramento.historico(desde, limite), "seq_atual": deps.barramento.seq_atual}

    @app.get("/api/alertas/stream")
    async def alertas_stream(
        request: Request,
        desde: int | None = Query(default=None, ge=0),
        maximo: int | None = Query(default=None, ge=1, le=10000),
    ) -> StreamingResponse:
        if desde is None:
            cabecalho = request.headers.get("last-event-id")
            desde = int(cabecalho) if cabecalho and cabecalho.isdigit() else 0
        loop = asyncio.get_running_loop()
        identificador, fila, pendentes = deps.barramento.inscrever(loop, desde)
        iniciais = [formatar_sse("alerta", item, identificador=item["seq"]) for item in pendentes]
        return StreamingResponse(
            emitir_sse(
                request, fila, deps.sse_keepalive_s,
                ao_terminar=lambda: deps.barramento.cancelar_inscricao(identificador),
                iniciais=iniciais,
                maximo_alertas=maximo,
            ),
            media_type="text/event-stream; charset=utf-8",
            headers=SSE_HEADERS,
        )

    @app.post("/api/webhook/alertas", status_code=202)
    async def receber_webhook(request: Request) -> dict[str, Any]:
        _exigir_loopback(request)
        corpo = await request.body()
        if len(corpo) > TAMANHO_MAXIMO_WEBHOOK:
            raise HTTPException(status_code=413, detail="Alerta grande demais.")
        try:
            dados = __import__("json").loads(corpo.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as erro:
            raise HTTPException(status_code=422, detail="JSON invalido.") from erro
        if not isinstance(dados, dict):
            raise HTTPException(status_code=422, detail="O corpo deve ser um objeto JSON.")
        faltando = [c for c in ("alerta_id", "paciente_id", "prioridade", "titulo") if not dados.get(c)]
        if faltando:
            raise HTTPException(status_code=422, detail="Campos obrigatorios ausentes: " + ", ".join(faltando))
        registro = deps.barramento.publicar(dados, origem="webhook")
        return {"recebido": True, "alerta_id": dados["alerta_id"], "seq": registro["seq"]}

    # --- jobs ----------------------------------------------------------------

    @app.get("/api/jobs/atual")
    async def job_atual() -> dict[str, Any]:
        return {"job": deps.jobs.snapshot()}

    @app.post("/api/jobs/cancelar")
    async def cancelar_job() -> dict[str, Any]:
        try:
            return {"job": deps.jobs.cancelar()}
        except JobOcupadoError as erro:
            raise HTTPException(status_code=409, detail=str(erro)) from erro

    # --- enviados ------------------------------------------------------------

    @app.post("/api/pacientes/enviados", status_code=201)
    async def criar_enviado(request: Request, pedido: NovoPacienteRequest) -> dict[str, Any]:
        _exigir_loopback(request)
        if repositorio.existe_em_alguma_origem(pedido.paciente_id):
            raise HTTPException(status_code=409, detail="Ja existe um paciente com esse id.")
        pasta = deps.dir_enviados / pedido.paciente_id
        pasta.mkdir(parents=True, exist_ok=False)
        paciente = repositorio.resolver(pedido.paciente_id)
        return repositorio.descrever(paciente)

    @app.put("/api/pacientes/enviados/{paciente_id}/arquivos/{nome}", status_code=201)
    async def enviar_arquivo(request: Request, paciente_id: str, nome: str) -> dict[str, Any]:
        _exigir_loopback(request)
        paciente = resolver(paciente_id)
        if paciente.origem != "enviados":
            raise HTTPException(status_code=403, detail="So pacientes enviados pela interface recebem arquivos.")
        tipo = tipo_do_upload(nome)
        if tipo is None:
            raise HTTPException(status_code=404, detail="Nome de arquivo nao permitido: " + ", ".join(ARQUIVOS_UPLOAD))
        tamanho_declarado = request.headers.get("content-length")
        if tamanho_declarado is None or not tamanho_declarado.isdigit():
            raise HTTPException(status_code=411, detail="Content-Length obrigatorio.")
        limite = deps.limites_upload.get(tipo, 0)
        if int(tamanho_declarado) > limite:
            raise HTTPException(status_code=413, detail=f"Arquivo acima do limite de {limite} bytes para .{tipo}.")

        destino = paciente.pasta_amostras / nome
        parcial = destino.with_suffix(destino.suffix + ".parte")
        recebidos = 0
        try:
            with parcial.open("wb") as f:
                async for bloco in request.stream():
                    recebidos += len(bloco)
                    if recebidos > limite:
                        raise HTTPException(status_code=413, detail=f"Arquivo acima do limite de {limite} bytes para .{tipo}.")
                    f.write(bloco)
            validacao = await run_in_threadpool(validar_conteudo, nome, parcial)
            os.replace(parcial, destino)
        except ConteudoInvalido as erro:
            parcial.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail=str(erro)) from erro
        except HTTPException:
            parcial.unlink(missing_ok=True)
            raise
        return {"nome": nome, "bytes": recebidos, "validacao": validacao, "paciente": repositorio.descrever(paciente)}

    @app.delete("/api/pacientes/enviados/{paciente_id}")
    async def remover_enviado(request: Request, paciente_id: str) -> dict[str, Any]:
        _exigir_loopback(request)
        paciente = resolver(paciente_id)
        try:
            await run_in_threadpool(remover_paciente_enviado, repositorio, paciente)
        except PermissionError as erro:
            raise HTTPException(status_code=403, detail=str(erro)) from erro
        return {"removido": True, "paciente_id": paciente_id}

    # --- physionet -----------------------------------------------------------

    @app.post("/api/physionet/baixar", status_code=202)
    async def baixar_physionet(request: Request, pedido: PhysionetRequest) -> dict[str, Any]:
        _exigir_loopback(request)
        job_ou_409()
        nome = pedido.registro.strip("/").rsplit("/", 1)[-1]
        try:
            validar_id(nome)
        except IdInvalido as erro:
            raise HTTPException(status_code=422, detail=str(erro)) from erro

        def tarefa(informar, cancelado):
            informar(progresso=0.1, etapa=f"Baixando e convertendo {pedido.registro} ({pedido.minutos or 'todos'} min)")
            resumo = deps.baixar_physionet(pedido.registro, deps.dir_physionet, pedido.minutos or None)
            if cancelado():
                return {"registro": nome, "download": resumo, "cancelado": True}
            informar(progresso=0.6, etapa="Rodando os detectores no registro")
            paciente = repositorio.resolver(nome)
            resultado = executar_processamento(
                deps, paciente, {"provedor_azure": "offline", "backend_pose": "anotado", "backend_detector": "anotado"},
                informar, cancelado,
            )
            return {"registro": nome, "download": resumo, "resultado": resultado}

        try:
            job = deps.jobs.iniciar("physionet", f"PhysioNet {nome}", tarefa, metadados=pedido.model_dump())
        except JobOcupadoError as erro:
            raise HTTPException(status_code=409, detail=str(erro)) from erro
        return {"modo": "job", "job": job}

    # --- estaticos -----------------------------------------------------------

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> FileResponse:
        return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")

    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")
    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.getenv("FASE4_WEB_PORT", "8020")))
    parser.add_argument("--no-open-browser", action="store_true")
    parser.add_argument(
        "--permitir-remoto", action="store_true",
        help="Aceita uploads, exclusoes e webhooks de outras maquinas (sem autenticacao; use so em rede confiavel).",
    )
    args = parser.parse_args()
    if args.permitir_remoto:
        os.environ["FASE4_WEB_REMOTO"] = "1"

    url = f"http://{args.host}:{args.port}"
    print(f"Interface da Fase 4: {url}")
    print(f'Para a CLI entregar alertas aqui: $env:FASE4_WEBHOOK_URL = "{url}/api/webhook/alertas"')
    if args.host != "127.0.0.1":
        print("AVISO: a interface nao tem autenticacao; evite expor fora da maquina local.")
    if not args.no_open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
