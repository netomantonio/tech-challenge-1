"""Servicos da interface web da Fase 4: pacientes, processamento, alertas e streams.

Tudo que a API precisa e que nao e roteamento HTTP mora aqui, com as
dependencias injetadas por `Deps` para os testes apontarem cada pasta para um
diretorio temporario e trocarem `processar_paciente` por uma funcao lenta
falsa. A camada web nunca reimplementa regra clinica: ela chama o pipeline,
serializa o que ele devolve e entrega os alertas pelos mesmos canais da CLI
mais um canal em memoria (`BarramentoAlertas`) que alimenta o navegador.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import struct
import subprocess
import threading
import wave
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from fase4 import anomaly_detection
from fase4.alertas import Alerta, CanalEntrega, MotorDeAlertas, canais_padrao
from fase4.config import (
    DIR_ENVIADOS,
    DIR_PHYSIONET,
    DIR_RESULTADOS,
    LIMIARES_ANOMALIA,
)
from fase4.data.baixar_physionet import preparar_registro
from fase4.data.gerar_dados_sinteticos import DIR_AMOSTRAS
from fase4.pipeline import ARQUIVO_AUDIO_PADRAO, ResultadoPaciente, localizar_amostras, processar_paciente
from fase4.validacao_physionet import validar_registro
from fase4.video_analysis import extrair_quadro_jpeg, info_video
from fase4.web_jobs import JobManager

PADRAO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,39}$")
PADRAO_AUDIO = re.compile(r"^consulta[a-z0-9_-]*\.wav$")
ORIGENS = ("sinteticos", "enviados", "physionet")
RELATORIOS = ("relatorio_monitoramento", "relatorio_audio", "relatorio_video", "relatorio_physionet")
MIDIAS_RESULTADO = {"fisioterapia_anotado.mp4", "fisioterapia_anotado.h264.mp4"}
ARQUIVO_RESULTADO_WEB = "resultado_web.json"

# Arquivos aceitos no upload (nome fixo -> tipo), com os nomes que
# `localizar_amostras` procura. Limites em bytes por tipo.
ARQUIVOS_UPLOAD: dict[str, str] = {
    "sinais_vitais.csv": "csv",
    "prescricoes.json": "json",
    "movimentacao.csv": "csv",
    "consulta.wav": "wav",
    "consulta_fala.wav": "wav",
    "consulta.transcricao.json": "json",
    "consulta_fala.transcricao.json": "json",
    "fisioterapia.mp4": "mp4",
    "fisioterapia.keypoints.json": "json",
    "fisioterapia.objetos.json": "json",
}
LIMITES_UPLOAD: dict[str, int] = {
    "wav": 50 * 1024 * 1024,
    "mp4": 500 * 1024 * 1024,
    "csv": 20 * 1024 * 1024,
    "json": 20 * 1024 * 1024,
}
FOURCC_NAVEGADOR = {"avc1", "h264", "H264", "AVC1", "x264", "X264"}


class PacienteNaoEncontrado(LookupError):
    """Nao existe paciente com esse id em nenhuma das origens."""


class IdInvalido(ValueError):
    """O id nao respeita `PADRAO_ID`."""


class ConteudoInvalido(ValueError):
    """O arquivo enviado nao tem o formato esperado."""


def agora_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Barramento de alertas (canal em memoria + pub/sub para SSE)
# ---------------------------------------------------------------------------


class BarramentoAlertas:
    """`CanalEntrega` em memoria que tambem publica para assinantes assincronos.

    Entra por ultimo na lista de canais do `MotorDeAlertas`: como o motor
    marca o canal antes de entregar, o registro guardado aqui ja traz
    `canais_entrega` completo (fila JSONL, webhook e `painel_web`).
    """

    nome = "painel_web"

    def __init__(self, maximo: int = 500, maximo_fila: int = 200) -> None:
        self._lock = threading.Lock()
        self._historico: deque[dict[str, Any]] = deque(maxlen=maximo)
        self._seq = 0
        self._inscritos: dict[int, tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = {}
        self._proximo_inscrito = 0
        self._maximo_fila = maximo_fila

    @property
    def seq_atual(self) -> int:
        with self._lock:
            return self._seq

    def entregar(self, alerta: Alerta) -> None:
        self.publicar(alerta.to_dict(), origem="canal_memoria")

    def publicar(self, dados: dict[str, Any], origem: str) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
            registro = {"seq": self._seq, "recebido_em": agora_iso(), "origem": origem, **dados}
            self._historico.append(registro)
            inscritos = list(self._inscritos.values())
        for loop, fila in inscritos:
            try:
                loop.call_soon_threadsafe(_enfileirar, fila, registro, self._maximo_fila)
            except RuntimeError:  # loop fechado: o assinante sumiu
                continue
        return registro

    def inscrever(
        self, loop: asyncio.AbstractEventLoop, desde_seq: int = 0
    ) -> tuple[int, asyncio.Queue, list[dict[str, Any]]]:
        """Registra um assinante e devolve o historico pendente sob o mesmo lock."""
        fila: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._proximo_inscrito += 1
            identificador = self._proximo_inscrito
            self._inscritos[identificador] = (loop, fila)
            pendentes = [r for r in self._historico if r["seq"] > desde_seq]
        return identificador, fila, pendentes

    def cancelar_inscricao(self, identificador: int) -> None:
        with self._lock:
            self._inscritos.pop(identificador, None)

    def historico(self, desde_seq: int = 0, limite: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            itens = [r for r in self._historico if r["seq"] > desde_seq]
        return itens[-limite:]


def _enfileirar(fila: asyncio.Queue, item: Any, maximo: int) -> None:
    while fila.qsize() >= maximo:
        try:
            fila.get_nowait()
        except asyncio.QueueEmpty:
            break
    fila.put_nowait(item)


# ---------------------------------------------------------------------------
# Dependencias e repositorio
# ---------------------------------------------------------------------------


@dataclass
class Deps:
    dir_amostras: Path = DIR_AMOSTRAS
    dir_enviados: Path = DIR_ENVIADOS
    dir_physionet: Path = DIR_PHYSIONET
    dir_resultados: Path = DIR_RESULTADOS
    jobs: JobManager = field(default_factory=JobManager)
    barramento: BarramentoAlertas = field(default_factory=BarramentoAlertas)
    limites_upload: dict[str, int] = field(default_factory=lambda: dict(LIMITES_UPLOAD))
    sse_keepalive_s: float = 15.0
    processar: Callable[..., ResultadoPaciente] = processar_paciente
    validar_physionet: Callable[..., dict] = validar_registro
    baixar_physionet: Callable[..., dict] = preparar_registro
    ffmpeg: str | None = field(default_factory=lambda: shutil.which("ffmpeg"))
    trava_sincrona: threading.Lock = field(default_factory=threading.Lock)


@dataclass(frozen=True)
class Paciente:
    paciente_id: str
    origem: str
    raiz_amostras: Path
    raiz_resultados: Path

    @property
    def pasta_amostras(self) -> Path:
        return self.raiz_amostras / self.paciente_id

    @property
    def pasta_resultados(self) -> Path:
        return self.raiz_resultados / self.paciente_id


def validar_id(paciente_id: str) -> str:
    if not isinstance(paciente_id, str) or not PADRAO_ID.match(paciente_id):
        raise IdInvalido("id de paciente invalido: use letras, numeros, '-' e '_' (2 a 40 caracteres)")
    return paciente_id


def _dentro_de(pasta: Path, raiz: Path) -> bool:
    try:
        pasta.resolve().relative_to(raiz.resolve())
    except (ValueError, OSError):
        return False
    return True


class RepositorioPacientes:
    """Localiza pacientes nas tres origens e descreve o que cada um tem."""

    def __init__(self, deps: Deps) -> None:
        self.deps = deps

    def raizes(self) -> dict[str, tuple[Path, Path]]:
        d = self.deps
        return {
            "sinteticos": (d.dir_amostras, d.dir_resultados),
            "enviados": (d.dir_enviados, d.dir_resultados / "enviados"),
            "physionet": (d.dir_physionet, d.dir_resultados / "physionet"),
        }

    def _existe(self, origem: str, pasta: Path) -> bool:
        if not pasta.is_dir():
            return False
        if origem == "physionet":
            return (pasta / "sinais_vitais.csv").exists()
        return True

    def resolver(self, paciente_id: str) -> Paciente:
        validar_id(paciente_id)
        for origem, (raiz_amostras, raiz_resultados) in self.raizes().items():
            pasta = raiz_amostras / paciente_id
            if self._existe(origem, pasta) and _dentro_de(pasta, raiz_amostras):
                return Paciente(paciente_id, origem, raiz_amostras, raiz_resultados)
        raise PacienteNaoEncontrado(f"paciente '{paciente_id}' nao encontrado")

    def existe_em_alguma_origem(self, paciente_id: str) -> bool:
        try:
            self.resolver(paciente_id)
        except PacienteNaoEncontrado:
            return False
        return True

    def listar(self) -> list[dict[str, Any]]:
        itens: list[dict[str, Any]] = []
        for origem, (raiz_amostras, raiz_resultados) in self.raizes().items():
            if not raiz_amostras.is_dir():
                continue
            for pasta in sorted(p for p in raiz_amostras.iterdir() if p.is_dir()):
                if not PADRAO_ID.match(pasta.name) or not self._existe(origem, pasta):
                    continue
                itens.append(self.descrever(Paciente(pasta.name, origem, raiz_amostras, raiz_resultados)))
        return itens

    def descrever(self, paciente: Paciente, detalhado: bool = False) -> dict[str, Any]:
        amostras = localizar_amostras(paciente.paciente_id, paciente.raiz_amostras)
        audios = sorted(
            p.name for p in paciente.pasta_amostras.glob("consulta*.wav") if PADRAO_AUDIO.match(p.name)
        )
        resultado_web = paciente.pasta_resultados / ARQUIVO_RESULTADO_WEB
        processado_em = None
        if resultado_web.exists():
            try:
                processado_em = json.loads(resultado_web.read_text(encoding="utf-8")).get("executado_em")
            except (OSError, json.JSONDecodeError):
                processado_em = None
        descricao: dict[str, Any] = {
            "paciente_id": paciente.paciente_id,
            "origem": paciente.origem,
            "arquivos": {nome: caminho.name for nome, caminho in amostras.items() if nome != "audio"},
            "audios": audios,
            "modalidades": sorted(
                {"video" if n in ("video", "keypoints") else n for n in amostras}
                - {"keypoints", "objetos"}
            ),
            "resultado_disponivel": resultado_web.exists(),
            "relatorios": [n for n in RELATORIOS if (paciente.pasta_resultados / f"{n}.md").exists()],
            "processado_em": processado_em,
        }
        if detalhado:
            plantados = paciente.pasta_amostras / "eventos_plantados.json"
            fonte = paciente.pasta_amostras / "fonte.json"
            descricao["eventos_plantados"] = _ler_json(plantados)
            descricao["fonte"] = _ler_json(fonte)
            video = amostras.get("video")
            descricao["video"] = descrever_video(video, self.deps) if video else None
        return descricao


def _ler_json(caminho: Path) -> Any:
    if not caminho.exists():
        return None
    try:
        return json.loads(caminho.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Processamento e serializacao
# ---------------------------------------------------------------------------


def faixas_serializadas() -> list[dict[str, Any]]:
    return [
        {
            "nome": f.nome,
            "unidade": f.unidade,
            "minimo": f.minimo,
            "maximo": f.maximo,
            "critico_baixo": f.critico_baixo,
            "critico_alto": f.critico_alto,
        }
        for f in LIMIARES_ANOMALIA.faixas
    ]


def url_midia(paciente_id: str, nome: str) -> str:
    return f"/api/pacientes/{paciente_id}/midia/{nome}"


def serializar_resultado(
    resultado: ResultadoPaciente, paciente: Paciente, opcoes: dict[str, Any], deps: Deps
) -> dict[str, Any]:
    dados = resultado.to_dict()
    video = resultado.resultado_video
    audio = resultado.resultado_audio
    dados["video"] = (
        {k: v for k, v in video.to_dict().items() if k != "achados"} if video is not None else None
    )
    dados["audio"] = (
        {
            "caminho_audio": Path(audio.caminho_audio).name,
            "atributos_acusticos": audio.atributos.to_dict(),
            "transcricao": audio.transcricao.to_dict(),
            "analise_texto": audio.analise_texto.to_dict(),
            "wer_referencia": audio.wer_referencia,
        }
        if audio is not None
        else None
    )
    dados["origem"] = paciente.origem
    dados["opcoes"] = opcoes
    dados["executado_em"] = agora_iso()
    dados["relatorios"] = [n for n in RELATORIOS if (paciente.pasta_resultados / f"{n}.md").exists()]
    midias: dict[str, str | None] = {"audio": None, "video_original": None, "video_anotado": None}
    arquivo_audio = opcoes.get("arquivo_audio", ARQUIVO_AUDIO_PADRAO)
    if (paciente.pasta_amostras / arquivo_audio).exists():
        midias["audio"] = url_midia(paciente.paciente_id, arquivo_audio)
    if (paciente.pasta_amostras / "fisioterapia.mp4").exists():
        midias["video_original"] = url_midia(paciente.paciente_id, "fisioterapia.mp4")
    if (paciente.pasta_resultados / "fisioterapia_anotado.mp4").exists():
        midias["video_anotado"] = url_midia(paciente.paciente_id, "fisioterapia_anotado.mp4")
    dados["midias"] = midias
    return dados


def executar_processamento(
    deps: Deps,
    paciente: Paciente,
    opcoes: dict[str, Any],
    informar: Callable[..., None] | None = None,
    cancelado: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Roda o pipeline para o paciente e grava `resultado_web.json`."""
    def _informar(progresso: float, etapa: str) -> None:
        if informar is not None:
            informar(progresso=progresso, etapa=etapa)

    _informar(0.05, f"Localizando amostras de {paciente.paciente_id}")
    paciente.raiz_resultados.mkdir(parents=True, exist_ok=True)
    canais: list[CanalEntrega] = canais_padrao(paciente.raiz_resultados) + [deps.barramento]
    area = opcoes.get("area_critica")
    _informar(0.15, "Executando as modalidades (video, audio, series, prescricoes, movimentacao)")
    resultado = deps.processar(
        paciente.paciente_id,
        raiz_amostras=paciente.raiz_amostras,
        dir_saida=paciente.raiz_resultados,
        provedor_azure=opcoes.get("provedor_azure", "auto"),
        backend_pose=opcoes.get("backend_pose", "anotado"),
        backend_detector=opcoes.get("backend_detector", "anotado"),
        area_critica=tuple(area) if area else None,
        arquivo_audio=opcoes.get("arquivo_audio", ARQUIVO_AUDIO_PADRAO),
        gravar_video_anotado=bool(opcoes.get("gravar_video_anotado", True)),
        canais=canais,
    )
    _informar(0.85, "Fusao e alertas concluidos; gravando resultados")
    if paciente.origem == "physionet" and not (cancelado and cancelado()):
        _informar(0.9, "Gerando relatorio de validacao do registro PhysioNet")
        try:
            deps.validar_physionet(
                paciente.paciente_id, dir_dados=paciente.raiz_amostras, dir_saida=paciente.raiz_resultados
            )
        except Exception as erro:  # noqa: BLE001 - o relatorio extra nao invalida o processamento
            if informar is not None:
                informar(log=f"relatorio PhysioNet nao gerado: {erro}")
    dados = serializar_resultado(resultado, paciente, opcoes, deps)
    paciente.pasta_resultados.mkdir(parents=True, exist_ok=True)
    (paciente.pasta_resultados / ARQUIVO_RESULTADO_WEB).write_text(
        json.dumps(dados, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    _informar(1.0, "Concluido")
    return dados


def precisa_de_job(paciente: Paciente, opcoes: dict[str, Any]) -> bool:
    """Backends reais, whisper local e registros PhysioNet levam minutos."""
    return (
        opcoes.get("backend_pose", "anotado") != "anotado"
        or opcoes.get("backend_detector", "anotado") != "anotado"
        or opcoes.get("provedor_azure", "auto") == "local"
        or paciente.origem == "physionet"
    )


def backends_disponiveis(deps: Deps) -> dict[str, bool]:
    import importlib.util

    def tem(modulo: str) -> bool:
        try:
            return importlib.util.find_spec(modulo) is not None
        except (ValueError, ModuleNotFoundError):
            return False

    return {
        "cv2": tem("cv2"),
        "mediapipe": tem("mediapipe"),
        "yolov8": tem("ultralytics"),
        "whisper": tem("faster_whisper"),
        "wfdb": tem("wfdb"),
        "ffmpeg": bool(deps.ffmpeg),
    }


# ---------------------------------------------------------------------------
# Midia, quadros e video
# ---------------------------------------------------------------------------


def resolver_midia(paciente: Paciente, nome: str, h264: bool = False) -> tuple[Path, str] | None:
    """Caminho e media type de uma midia permitida, ou None."""
    if PADRAO_AUDIO.match(nome):
        caminho = paciente.pasta_amostras / nome
        return (caminho, "audio/wav") if caminho.exists() else None
    if nome == "fisioterapia.mp4":
        caminho = paciente.pasta_amostras / nome
        return (caminho, "video/mp4") if caminho.exists() else None
    if nome in MIDIAS_RESULTADO:
        caminho = paciente.pasta_resultados / nome
        if h264 and nome == "fisioterapia_anotado.mp4":
            transcodificado = paciente.pasta_resultados / "fisioterapia_anotado.h264.mp4"
            if transcodificado.exists():
                caminho = transcodificado
        return (caminho, "video/mp4") if caminho.exists() else None
    return None


def caminho_video(paciente: Paciente, fonte: str) -> Path | None:
    if fonte == "anotado":
        caminho = paciente.pasta_resultados / "fisioterapia_anotado.mp4"
    else:
        caminho = paciente.pasta_amostras / "fisioterapia.mp4"
    return caminho if caminho.exists() else None


def descrever_video(caminho: Path, deps: Deps) -> dict[str, Any]:
    info = info_video(caminho)
    h264 = caminho.with_name(caminho.stem + ".h264.mp4")
    info["reproduzivel_no_navegador"] = info["fourcc"] in FOURCC_NAVEGADOR
    info["h264_disponivel"] = h264.exists()
    info["ffmpeg_disponivel"] = bool(deps.ffmpeg)
    return info


def transcodificar_h264(deps: Deps, origem: Path, timeout_s: int = 300) -> Path | None:
    """Gera `<nome>.h264.mp4` com ffmpeg quando ele existe; None se nao for possivel."""
    if not deps.ffmpeg or not origem.exists():
        return None
    destino = origem.with_name(origem.stem + ".h264.mp4")
    if destino.exists() and destino.stat().st_mtime >= origem.stat().st_mtime:
        return destino
    comando = [
        deps.ffmpeg, "-y", "-loglevel", "error", "-i", str(origem),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(destino),
    ]
    try:
        subprocess.run(comando, check=True, capture_output=True, timeout=timeout_s)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        destino.unlink(missing_ok=True)
        return None
    return destino


def quadro_jpeg(paciente: Paciente, fonte: str, instante_s: float) -> tuple[bytes, int, int]:
    caminho = caminho_video(paciente, fonte)
    if caminho is None:
        raise PacienteNaoEncontrado("video nao encontrado para este paciente")
    info = info_video(caminho)
    if instante_s < 0 or (info["duracao_s"] and instante_s > info["duracao_s"]):
        raise ValueError(f"instante fora da duracao do video (0 a {info['duracao_s']:.2f} s)")
    return extrair_quadro_jpeg(caminho, instante_s)


def validar_area_critica(area: Iterable[float] | None, video: dict[str, Any] | None) -> tuple[float, ...] | None:
    if area is None:
        return None
    valores = [float(v) for v in area]
    if len(valores) != 4:
        raise ValueError("area critica exige quatro numeros: x1, y1, x2, y2")
    x1, y1, x2, y2 = valores
    if x2 <= x1 or y2 <= y1:
        raise ValueError("area critica exige x2 > x1 e y2 > y1")
    if min(valores) < 0:
        raise ValueError("area critica nao pode ter coordenadas negativas")
    if video and video.get("largura") and video.get("altura"):
        if x2 > video["largura"] or y2 > video["altura"]:
            raise ValueError(
                f"area critica fora do quadro ({video['largura']}x{video['altura']} px)"
            )
    return tuple(valores)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def tipo_do_upload(nome: str) -> str | None:
    return ARQUIVOS_UPLOAD.get(nome)


def validar_conteudo(nome: str, caminho: Path) -> dict[str, Any]:
    """Confere o formato do arquivo enviado; ConteudoInvalido se nao servir."""
    tipo = tipo_do_upload(nome)
    tamanho = caminho.stat().st_size
    if tamanho == 0:
        raise ConteudoInvalido("arquivo vazio")

    if tipo == "wav":
        try:
            with wave.open(str(caminho), "rb") as wav:
                canais, largura, taxa, quadros = (
                    wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
                )
        except (wave.Error, EOFError, struct.error) as erro:
            raise ConteudoInvalido(f"WAV invalido: {erro}") from erro
        return {"tipo": "wav", "canais": canais, "bits": largura * 8, "taxa_hz": taxa, "duracao_s": round(quadros / taxa, 2) if taxa else 0}

    if tipo == "mp4":
        with caminho.open("rb") as f:
            cabecalho = f.read(12)
        if len(cabecalho) < 12 or cabecalho[4:8] != b"ftyp":
            raise ConteudoInvalido("MP4 invalido: cabecalho 'ftyp' ausente")
        try:
            info = info_video(caminho)
        except Exception as erro:  # noqa: BLE001
            raise ConteudoInvalido(f"video nao pode ser aberto: {erro}") from erro
        if not info["quadros"]:
            raise ConteudoInvalido("video sem quadros legiveis")
        return {"tipo": "mp4", **info}

    if tipo == "csv":
        with caminho.open("r", encoding="utf-8", errors="replace") as f:
            cabecalho = f.readline().strip().split(",")
        colunas = [c.strip() for c in cabecalho]
        if "timestamp" not in colunas:
            raise ConteudoInvalido("CSV precisa da coluna 'timestamp'")
        if nome == "sinais_vitais.csv":
            conhecidos = [c for c in colunas if c in LIMIARES_ANOMALIA.canais]
            if not conhecidos:
                raise ConteudoInvalido(
                    "sinais_vitais.csv precisa de ao menos um canal: " + ", ".join(LIMIARES_ANOMALIA.canais)
                )
            return {"tipo": "csv", "colunas": colunas, "canais": conhecidos}
        if nome == "movimentacao.csv" and "indice_atividade" not in colunas:
            raise ConteudoInvalido("movimentacao.csv precisa da coluna 'indice_atividade'")
        return {"tipo": "csv", "colunas": colunas}

    if tipo == "json":
        try:
            dados = json.loads(caminho.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as erro:
            raise ConteudoInvalido(f"JSON invalido: {erro}") from erro
        if nome == "prescricoes.json" and not (isinstance(dados, list) and all(isinstance(i, dict) for i in dados)):
            raise ConteudoInvalido("prescricoes.json deve ser uma lista de eventos")
        if nome.startswith("fisioterapia.") and not (isinstance(dados, dict) and "quadros" in dados):
            raise ConteudoInvalido(f"{nome} deve ter a chave 'quadros'")
        if nome.endswith("transcricao.json") and not (isinstance(dados, dict) and "segmentos" in dados):
            raise ConteudoInvalido(f"{nome} deve ter a chave 'segmentos'")
        return {"tipo": "json", "itens": len(dados) if isinstance(dados, (list, dict)) else None}

    raise ConteudoInvalido("tipo de arquivo nao permitido")


def remover_paciente_enviado(repositorio: RepositorioPacientes, paciente: Paciente) -> None:
    if paciente.origem != "enviados":
        raise PermissionError("so pacientes enviados pela interface podem ser removidos")
    for pasta in (paciente.pasta_amostras, paciente.pasta_resultados):
        if pasta.exists() and _dentro_de(pasta, pasta.parent):
            shutil.rmtree(pasta, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tempo real (produtor em thread) e SSE
# ---------------------------------------------------------------------------


def _enviar_evento(loop: asyncio.AbstractEventLoop, fila: asyncio.Queue, evento: str, dados: Any) -> None:
    try:
        loop.call_soon_threadsafe(_enfileirar, fila, {"evento": evento, "dados": dados}, 5000)
    except RuntimeError:
        pass


def fluxo_tempo_real(
    deps: Deps,
    paciente: Paciente,
    ms_por_minuto: int,
    incluir_multivariado: bool,
    limite_minutos: int | None,
    loop: asyncio.AbstractEventLoop,
    fila: asyncio.Queue,
    parar: threading.Event,
) -> None:
    """Percorre a serie minuto a minuto e publica cada leitura (roda numa thread)."""
    try:
        amostras = localizar_amostras(paciente.paciente_id, paciente.raiz_amostras)
        if "sinais_vitais" not in amostras:
            _enviar_evento(loop, fila, "erro", {"mensagem": "paciente sem serie de sinais vitais"})
            return
        df = anomaly_detection.carregar_sinais_vitais(amostras["sinais_vitais"])
        canais_serie = [c for c in LIMIARES_ANOMALIA.canais if c in df.columns]
        _enviar_evento(
            loop, fila, "inicio",
            {
                "paciente_id": paciente.paciente_id,
                "total_minutos": int(len(df)),
                "inicio": df["timestamp"].iloc[0].isoformat() if len(df) else None,
                "fim": df["timestamp"].iloc[-1].isoformat() if len(df) else None,
                "canais": canais_serie,
                "faixas": [f for f in faixas_serializadas() if f["nome"] in canais_serie],
                "multivariado": incluir_multivariado,
            },
        )
        paciente.raiz_resultados.mkdir(parents=True, exist_ok=True)
        motor = MotorDeAlertas(canais=canais_padrao(paciente.raiz_resultados) + [deps.barramento])
        total_alertas = 0
        minutos = 0
        for instante, valores, achados in anomaly_detection.monitorar_leituras(
            df, incluir_multivariado=incluir_multivariado
        ):
            if parar.is_set():
                break
            alertas = motor.gerar(paciente.paciente_id, achados, agora=instante) if achados else []
            total_alertas += len(alertas)
            _enviar_evento(
                loop, fila, "leitura",
                {
                    "minuto": minutos,
                    "instante": instante.isoformat(),
                    "valores": valores,
                    "achados": [a.to_dict() for a in achados],
                    "alertas": [a.to_dict() for a in alertas],
                },
            )
            minutos += 1
            if limite_minutos is not None and minutos >= limite_minutos:
                break
            if ms_por_minuto > 0 and parar.wait(ms_por_minuto / 1000.0):
                break
        _enviar_evento(
            loop, fila, "fim",
            {"minutos": minutos, "total_alertas": total_alertas, "entregas_falhas": motor.entregas_falhas,
             "interrompido": parar.is_set()},
        )
    except Exception as erro:  # noqa: BLE001 - vira evento para o navegador
        _enviar_evento(loop, fila, "erro", {"mensagem": f"{type(erro).__name__}: {erro}"})


def formatar_sse(evento: str, dados: Any, identificador: int | None = None) -> str:
    corpo = json.dumps(dados, ensure_ascii=False, default=str)
    linhas = [f"event: {evento}"]
    if identificador is not None:
        linhas.append(f"id: {identificador}")
    linhas.append(f"data: {corpo}")
    return "\n".join(linhas) + "\n\n"


async def emitir_sse(
    request: Any,
    fila: asyncio.Queue,
    keepalive_s: float,
    encerrar_em: Iterable[str] = (),
    ao_terminar: Callable[[], None] | None = None,
    iniciais: Iterable[str] = (),
    maximo_alertas: int | None = None,
):
    """Gerador assincrono que transforma itens da fila em frames SSE.

    `maximo_alertas` encerra o stream depois de N alertas (uteis em testes e
    em clientes que so querem uma amostra); sem ele, o stream dura ate o
    cliente desconectar.
    """
    terminais = set(encerrar_em)
    enviados = 0
    try:
        for frame in iniciais:
            yield frame
            enviados += 1
            if maximo_alertas is not None and enviados >= maximo_alertas:
                return
        while True:
            if await request.is_disconnected():
                break
            try:
                item = await asyncio.wait_for(fila.get(), timeout=keepalive_s)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            if "evento" in item and "dados" in item and "seq" not in item:
                yield formatar_sse(item["evento"], item["dados"])
                if item["evento"] in terminais:
                    break
            else:
                yield formatar_sse("alerta", item, identificador=item.get("seq"))
                enviados += 1
                if maximo_alertas is not None and enviados >= maximo_alertas:
                    break
    finally:
        if ao_terminar is not None:
            ao_terminar()


SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}
