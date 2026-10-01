"""Analise de video clinico: pose e deteccao de objetos (Fase 4).

O enunciado pede para processar videos de fisioterapia ou cirurgia, detectar
movimentos e eventos fora do padrao com modelos como OpenPose (postura) e
YOLOv8 (objetos e areas criticas) e gerar relatorios automaticos.

A organizacao do modulo separa duas coisas que costumam vir misturadas:

1. de onde vem a percepcao - um `estimador de pose` que devolve keypoints por
   quadro e um `detector` que devolve caixas por quadro;
2. o que se faz com ela - os angulos articulares, as regras de desvio e o
   relatorio, que e a parte propria deste trabalho.

Por isso existem backends. O backend `anotado` le os keypoints e as caixas
de arquivos JSON (usado com as amostras sinteticas e nos testes). O backend
`mediapipe` roda o MediaPipe Pose quadro a quadro e o `yolov8` roda o
Ultralytics YOLOv8; os dois sao opcionais e estao em `requirements-fase4.txt`.
Usamos MediaPipe Pose no lugar do OpenPose porque ele entrega o mesmo tipo de
saida (keypoints 2D do corpo) com instalacao via pip, enquanto o OpenPose
exige compilar Caffe/CUDA; a analise a jusante e identica, ja que depende
apenas das coordenadas das articulacoes.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from fase4.alertas import Achado
from fase4.config import LIMIARES_VIDEO, LimiaresVideo

BACKENDS_POSE = ("anotado", "mediapipe")
BACKENDS_DETECTOR = ("anotado", "yolov8")

# Correspondencia entre os pontos do MediaPipe Pose e os nomes usados aqui.
_INDICES_MEDIAPIPE = {
    "nariz": 0,
    "ombro_esquerdo": 11,
    "ombro_direito": 12,
    "cotovelo_esquerdo": 13,
    "cotovelo_direito": 14,
    "punho_esquerdo": 15,
    "punho_direito": 16,
    "quadril_esquerdo": 23,
    "quadril_direito": 24,
    "joelho_esquerdo": 25,
    "joelho_direito": 26,
    "tornozelo_esquerdo": 27,
    "tornozelo_direito": 28,
}


class BackendIndisponivelError(RuntimeError):
    """O backend pedido nao esta instalado ou nao tem como rodar aqui."""


@dataclass
class Articulacao:
    x: float
    y: float
    confianca: float


@dataclass
class QuadroPose:
    quadro: int
    instante_s: float
    articulacoes: dict[str, Articulacao]


@dataclass
class Objeto:
    classe: str
    confianca: float
    caixa: tuple[float, float, float, float]
    area_critica: bool = False


@dataclass
class QuadroObjetos:
    quadro: int
    instante_s: float
    objetos: list[Objeto]


@dataclass
class MedidasQuadro:
    """Angulos calculados num quadro, usados pelas regras de desvio."""

    quadro: int
    instante_s: float
    abducao_ombro_esquerdo: float | None
    abducao_ombro_direito: float | None
    inclinacao_tronco: float | None
    assimetria_ombros: float | None

    def to_dict(self) -> dict:
        return {
            "quadro": self.quadro,
            "instante_s": round(self.instante_s, 3),
            "abducao_ombro_esquerdo": _arredondar(self.abducao_ombro_esquerdo),
            "abducao_ombro_direito": _arredondar(self.abducao_ombro_direito),
            "inclinacao_tronco": _arredondar(self.inclinacao_tronco),
            "assimetria_ombros": _arredondar(self.assimetria_ombros),
        }


@dataclass
class ResultadoVideo:
    paciente_id: str
    caminho_video: str
    backend_pose: str
    backend_detector: str
    fps: float
    quadros_analisados: int
    duracao_s: float
    medidas: list[MedidasQuadro]
    achados: list[Achado]
    resumo_angulos: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "paciente_id": self.paciente_id,
            "caminho_video": self.caminho_video,
            "backend_pose": self.backend_pose,
            "backend_detector": self.backend_detector,
            "fps": self.fps,
            "quadros_analisados": self.quadros_analisados,
            "duracao_s": round(self.duracao_s, 3),
            "resumo_angulos": {k: round(v, 2) for k, v in self.resumo_angulos.items()},
            "achados": [a.to_dict() for a in self.achados],
        }


def _arredondar(valor: float | None) -> float | None:
    return None if valor is None else round(valor, 2)


# ---------------------------------------------------------------------------
# Backends de percepcao
# ---------------------------------------------------------------------------


def carregar_pose_anotada(caminho_json: Path | str) -> tuple[list[QuadroPose], float]:
    with Path(caminho_json).open(encoding="utf-8") as f:
        dados = json.load(f)

    quadros = [
        QuadroPose(
            quadro=int(item["quadro"]),
            instante_s=float(item["instante_s"]),
            articulacoes={
                nome: Articulacao(
                    x=float(ponto["x"]), y=float(ponto["y"]), confianca=float(ponto["confianca"])
                )
                for nome, ponto in item["articulacoes"].items()
            },
        )
        for item in dados["quadros"]
    ]
    return quadros, float(dados.get("fps", 30.0))


def carregar_objetos_anotados(caminho_json: Path | str) -> list[QuadroObjetos]:
    with Path(caminho_json).open(encoding="utf-8") as f:
        dados = json.load(f)

    return [
        QuadroObjetos(
            quadro=int(item["quadro"]),
            instante_s=float(item["instante_s"]),
            objetos=[
                Objeto(
                    classe=obj["classe"],
                    confianca=float(obj["confianca"]),
                    caixa=tuple(float(v) for v in obj["caixa"]),  # type: ignore[arg-type]
                    area_critica=bool(obj.get("area_critica", False)),
                )
                for obj in item["objetos"]
            ],
        )
        for item in dados["quadros"]
    ]


def _fps_do_video(caminho_video: Path | str, padrao: float = 30.0) -> float:
    """Le o fps do arquivo. A taxa importa porque as regras de evento medem tempo."""
    try:
        import cv2
    except ImportError:  # pragma: no cover - depende de lib opcional
        return padrao

    captura = cv2.VideoCapture(str(caminho_video))
    fps = captura.get(cv2.CAP_PROP_FPS) if captura.isOpened() else 0.0
    captura.release()
    return float(fps) if fps and fps > 0 else padrao


def extrair_pose_mediapipe(caminho_video: Path | str) -> tuple[list[QuadroPose], float]:
    """Roda o MediaPipe Pose quadro a quadro num video real."""
    try:
        import cv2
        import mediapipe as mp
    except ImportError as erro:  # pragma: no cover - depende de lib opcional
        raise BackendIndisponivelError(
            "o backend 'mediapipe' exige mediapipe e opencv-python instalados "
            "(pip install -r requirements-fase4.txt)"
        ) from erro

    captura = cv2.VideoCapture(str(caminho_video))
    if not captura.isOpened():  # pragma: no cover
        raise BackendIndisponivelError(f"nao foi possivel abrir o video {caminho_video}")

    fps = captura.get(cv2.CAP_PROP_FPS) or 30.0
    quadros: list[QuadroPose] = []

    with mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=1, min_detection_confidence=0.5
    ) as pose:
        indice = 0
        while True:
            ok, imagem = captura.read()
            if not ok:
                break
            altura, largura = imagem.shape[:2]
            resultado = pose.process(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
            articulacoes: dict[str, Articulacao] = {}
            if resultado.pose_landmarks:
                pontos = resultado.pose_landmarks.landmark
                for nome, idx in _INDICES_MEDIAPIPE.items():
                    ponto = pontos[idx]
                    articulacoes[nome] = Articulacao(
                        x=ponto.x * largura,
                        y=ponto.y * altura,
                        confianca=float(getattr(ponto, "visibility", 0.0)),
                    )
            quadros.append(
                QuadroPose(quadro=indice, instante_s=indice / fps, articulacoes=articulacoes)
            )
            indice += 1

    captura.release()
    return quadros, float(fps)


def detectar_objetos_yolov8(
    caminho_video: Path | str,
    modelo: str = "yolov8n.pt",
    classes_criticas: Iterable[str] = ("dreno",),
) -> list[QuadroObjetos]:
    """Roda o YOLOv8 (Ultralytics) quadro a quadro num video real."""
    try:
        from ultralytics import YOLO
    except ImportError as erro:  # pragma: no cover - depende de lib opcional
        raise BackendIndisponivelError(
            "o backend 'yolov8' exige ultralytics instalado "
            "(pip install -r requirements-fase4.txt)"
        ) from erro

    rede = YOLO(modelo)
    criticas = set(classes_criticas)
    quadros: list[QuadroObjetos] = []
    fps = _fps_do_video(caminho_video)

    for indice, predicao in enumerate(
        rede.predict(source=str(caminho_video), stream=True, verbose=False)
    ):
        objetos: list[Objeto] = []
        for caixa in predicao.boxes:
            nome = rede.names[int(caixa.cls)]
            x1, y1, x2, y2 = (float(v) for v in caixa.xyxy[0])
            objetos.append(
                Objeto(
                    classe=nome,
                    confianca=float(caixa.conf),
                    caixa=(x1, y1, x2, y2),
                    area_critica=nome in criticas,
                )
            )
        quadros.append(
            QuadroObjetos(quadro=indice, instante_s=indice / fps, objetos=objetos)
        )
    return quadros


# ---------------------------------------------------------------------------
# Angulos articulares
# ---------------------------------------------------------------------------


def _valido(articulacoes: dict[str, Articulacao], nomes: Iterable[str], minimo: float) -> bool:
    return all(
        nome in articulacoes and articulacoes[nome].confianca >= minimo for nome in nomes
    )


def angulo_abducao_ombro(
    ombro: Articulacao, cotovelo: Articulacao, lado: str
) -> float:
    """Angulo entre o braco e o eixo vertical do corpo, em graus.

    Zero grau e o braco ao longo do tronco (apontando para baixo) e 180 graus
    e o braco acima da cabeca, que e a convencao clinica de abducao de ombro.
    Em coordenadas de imagem o eixo y cresce para baixo, por isso o vetor de
    referencia e (0, 1).
    """
    dx = cotovelo.x - ombro.x
    dy = cotovelo.y - ombro.y
    norma = math.hypot(dx, dy)
    if norma == 0:
        return 0.0
    cosseno = max(-1.0, min(1.0, dy / norma))
    angulo = math.degrees(math.acos(cosseno))
    # O sinal de dx diz se o braco abriu para fora ou cruzou o corpo. Um
    # cruzamento conta como abducao negativa, que nao interessa aqui.
    esperado = 1.0 if lado == "esquerdo" else -1.0
    if dx * esperado < 0:
        return -angulo
    return angulo


def angulo_inclinacao_tronco(
    ombro_esq: Articulacao,
    ombro_dir: Articulacao,
    quadril_esq: Articulacao,
    quadril_dir: Articulacao,
) -> float:
    """Inclinacao lateral do tronco em graus, medida do eixo vertical."""
    centro_ombros_x = (ombro_esq.x + ombro_dir.x) / 2
    centro_ombros_y = (ombro_esq.y + ombro_dir.y) / 2
    centro_quadris_x = (quadril_esq.x + quadril_dir.x) / 2
    centro_quadris_y = (quadril_esq.y + quadril_dir.y) / 2

    dx = centro_ombros_x - centro_quadris_x
    dy = centro_quadris_y - centro_ombros_y  # positivo quando os ombros estao acima
    if dy == 0:
        return 90.0
    return abs(math.degrees(math.atan2(dx, dy)))


def medir_quadros(
    quadros: list[QuadroPose], limiares: LimiaresVideo = LIMIARES_VIDEO
) -> list[MedidasQuadro]:
    """Calcula, quadro a quadro, os angulos usados nas regras posturais."""
    medidas: list[MedidasQuadro] = []
    minimo = limiares.confianca_minima_keypoint

    for quadro in quadros:
        articulacoes = quadro.articulacoes

        esquerdo = direito = tronco = assimetria = None

        if _valido(articulacoes, ("ombro_esquerdo", "cotovelo_esquerdo"), minimo):
            esquerdo = angulo_abducao_ombro(
                articulacoes["ombro_esquerdo"], articulacoes["cotovelo_esquerdo"], "esquerdo"
            )
        if _valido(articulacoes, ("ombro_direito", "cotovelo_direito"), minimo):
            direito = angulo_abducao_ombro(
                articulacoes["ombro_direito"], articulacoes["cotovelo_direito"], "direito"
            )
        if esquerdo is not None and direito is not None:
            assimetria = abs(esquerdo - direito)

        if _valido(
            articulacoes,
            ("ombro_esquerdo", "ombro_direito", "quadril_esquerdo", "quadril_direito"),
            minimo,
        ):
            tronco = angulo_inclinacao_tronco(
                articulacoes["ombro_esquerdo"],
                articulacoes["ombro_direito"],
                articulacoes["quadril_esquerdo"],
                articulacoes["quadril_direito"],
            )

        medidas.append(
            MedidasQuadro(
                quadro=quadro.quadro,
                instante_s=quadro.instante_s,
                abducao_ombro_esquerdo=esquerdo,
                abducao_ombro_direito=direito,
                inclinacao_tronco=tronco,
                assimetria_ombros=assimetria,
            )
        )
    return medidas


# ---------------------------------------------------------------------------
# Regras de desvio
# ---------------------------------------------------------------------------


def _janelas_acima(
    valores: list[tuple[int, float, float | None]], limite: float, minimo_quadros: int
) -> list[dict]:
    """Agrupa sequencias de quadros em que a medida passou do limite.

    Exigir um numero minimo de quadros seguidos e o que separa um desvio real
    de um pico de ruido do estimador de pose.
    """
    janelas: list[dict] = []
    atual: list[tuple[int, float, float]] = []

    for quadro, instante, valor in valores:
        if valor is not None and valor > limite:
            atual.append((quadro, instante, valor))
            continue
        if len(atual) >= minimo_quadros:
            janelas.append(_fechar_janela(atual))
        atual = []

    if len(atual) >= minimo_quadros:
        janelas.append(_fechar_janela(atual))
    return janelas


def _fechar_janela(amostras: list[tuple[int, float, float]]) -> dict:
    picos = [v for _, _, v in amostras]
    indice_pico = picos.index(max(picos))
    return {
        "quadro_inicial": amostras[0][0],
        "quadro_final": amostras[-1][0],
        "instante_inicial_s": amostras[0][1],
        "instante_final_s": amostras[-1][1],
        "valor_maximo": max(picos),
        "instante_pico_s": amostras[indice_pico][1],
        "quadros": len(amostras),
    }


def _formatar_instante(segundos: float) -> str:
    minutos = int(segundos // 60)
    resto = segundos - minutos * 60
    return f"{minutos:02d}:{resto:05.2f}"


def avaliar_postura(
    medidas: list[MedidasQuadro], limiares: LimiaresVideo = LIMIARES_VIDEO
) -> list[Achado]:
    """Regras posturais: inclinacao de tronco, amplitude e assimetria."""
    achados: list[Achado] = []

    regras = (
        (
            "inclinacao_tronco",
            [(m.quadro, m.instante_s, m.inclinacao_tronco) for m in medidas],
            limiares.inclinacao_tronco_max_graus,
            "desvio_postural_tronco",
            "Compensacao de tronco acima do tolerado durante o exercicio",
            "atencao",
        ),
        (
            "abducao_ombro_esquerdo",
            [(m.quadro, m.instante_s, m.abducao_ombro_esquerdo) for m in medidas],
            limiares.abducao_ombro_max_graus,
            "amplitude_ombro_esquerdo_excedida",
            "Amplitude de abducao do ombro esquerdo acima da liberada no pos-operatorio",
            "critico",
        ),
        (
            "abducao_ombro_direito",
            [(m.quadro, m.instante_s, m.abducao_ombro_direito) for m in medidas],
            limiares.abducao_ombro_max_graus,
            "amplitude_ombro_direito_excedida",
            "Amplitude de abducao do ombro direito acima da liberada no pos-operatorio",
            "critico",
        ),
        (
            "assimetria_ombros",
            [(m.quadro, m.instante_s, m.assimetria_ombros) for m in medidas],
            limiares.assimetria_ombros_max_graus,
            "assimetria_entre_ombros",
            "Assimetria entre os ombros na mesma repeticao",
            "atencao",
        ),
    )

    for medida, valores, limite, tipo, descricao, severidade in regras:
        for janela in _janelas_acima(valores, limite, limiares.quadros_minimos_desvio):
            excesso = janela["valor_maximo"] - limite
            achados.append(
                Achado(
                    modalidade="video",
                    tipo=tipo,
                    descricao=f"{descricao} ({janela['valor_maximo']:.1f} graus)",
                    severidade=severidade,
                    score=min(0.97, 0.55 + excesso / 60.0),
                    instante=_formatar_instante(janela["instante_pico_s"]),
                    evidencias={
                        "medida": medida,
                        "limite_graus": limite,
                        "valor_maximo_graus": round(janela["valor_maximo"], 1),
                        "janela_s": (
                            f"{janela['instante_inicial_s']:.2f}-{janela['instante_final_s']:.2f}"
                        ),
                        "quadros": janela["quadros"],
                    },
                )
            )
    return achados


def _sobreposicao(caixa_a: tuple[float, ...], caixa_b: tuple[float, ...]) -> float:
    """Fracao da caixa A que esta dentro da caixa B."""
    x1 = max(caixa_a[0], caixa_b[0])
    y1 = max(caixa_a[1], caixa_b[1])
    x2 = min(caixa_a[2], caixa_b[2])
    y2 = min(caixa_a[3], caixa_b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersecao = (x2 - x1) * (y2 - y1)
    area_a = (caixa_a[2] - caixa_a[0]) * (caixa_a[3] - caixa_a[1])
    return intersecao / area_a if area_a > 0 else 0.0


def avaliar_eventos(
    quadros: list[QuadroObjetos],
    fps: float,
    limiares: LimiaresVideo = LIMIARES_VIDEO,
    classe_paciente: str = "mao_paciente",
    classe_profissional: str = "profissional",
) -> list[Achado]:
    """Regras sobre objetos: invasao de area critica e ausencia do profissional."""
    achados: list[Achado] = []
    if not quadros:
        return achados

    # Invasao da area critica (regiao do dreno cirurgico).
    invasoes: list[tuple[int, float, float]] = []
    for quadro in quadros:
        criticas = [o for o in quadro.objetos if o.area_critica]
        mao = next((o for o in quadro.objetos if o.classe == classe_paciente), None)
        maior = 0.0
        if mao is not None:
            for area in criticas:
                maior = max(maior, _sobreposicao(mao.caixa, area.caixa))
        invasoes.append((quadro.quadro, quadro.instante_s, maior))

    for janela in _janelas_acima(
        invasoes, limiares.sobreposicao_area_critica, limiares.quadros_minimos_desvio
    ):
        achados.append(
            Achado(
                modalidade="video",
                tipo="invasao_area_critica",
                descricao=(
                    "Mao do paciente dentro da area critica do dreno durante a sessao"
                ),
                severidade="critico",
                score=min(0.95, 0.70 + janela["valor_maximo"] / 4.0),
                instante=_formatar_instante(janela["instante_pico_s"]),
                evidencias={
                    "sobreposicao_maxima": round(janela["valor_maximo"], 3),
                    "duracao_s": round(janela["quadros"] / fps, 2),
                    "janela_s": (
                        f"{janela['instante_inicial_s']:.2f}-{janela['instante_final_s']:.2f}"
                    ),
                },
            )
        )

    # Ausencia do profissional em cena.
    ausencias: list[tuple[int, float, float]] = [
        (
            quadro.quadro,
            quadro.instante_s,
            0.0 if any(o.classe == classe_profissional for o in quadro.objetos) else 1.0,
        )
        for quadro in quadros
    ]
    quadros_minimos = max(1, int(limiares.segundos_max_sem_profissional * fps))
    for janela in _janelas_acima(ausencias, 0.5, quadros_minimos):
        duracao = janela["quadros"] / fps
        achados.append(
            Achado(
                modalidade="video",
                tipo="profissional_ausente",
                descricao=(
                    f"Paciente executou o exercicio sem profissional em cena por "
                    f"{duracao:.1f} s"
                ),
                severidade="atencao",
                score=min(0.9, 0.55 + duracao / 20.0),
                instante=_formatar_instante(janela["instante_inicial_s"]),
                evidencias={
                    "duracao_s": round(duracao, 2),
                    "limite_s": limiares.segundos_max_sem_profissional,
                    "janela_s": (
                        f"{janela['instante_inicial_s']:.2f}-{janela['instante_final_s']:.2f}"
                    ),
                },
            )
        )

    return achados


# ---------------------------------------------------------------------------
# Pipeline da modalidade
# ---------------------------------------------------------------------------


def analisar_sessao(
    paciente_id: str,
    caminho_video: Path | str | None = None,
    caminho_keypoints: Path | str | None = None,
    caminho_objetos: Path | str | None = None,
    backend_pose: str = "anotado",
    backend_detector: str = "anotado",
    limiares: LimiaresVideo = LIMIARES_VIDEO,
) -> ResultadoVideo:
    """Analisa uma sessao clinica gravada e devolve medidas e achados."""
    if backend_pose not in BACKENDS_POSE:
        raise ValueError(f"backend de pose invalido: {backend_pose}")
    if backend_detector not in BACKENDS_DETECTOR:
        raise ValueError(f"backend de detector invalido: {backend_detector}")

    if backend_pose == "anotado":
        if caminho_keypoints is None:
            raise ValueError("o backend 'anotado' exige caminho_keypoints")
        quadros_pose, fps = carregar_pose_anotada(caminho_keypoints)
    else:
        if caminho_video is None:
            raise ValueError("o backend 'mediapipe' exige caminho_video")
        quadros_pose, fps = extrair_pose_mediapipe(caminho_video)

    if backend_detector == "anotado":
        quadros_objetos = (
            carregar_objetos_anotados(caminho_objetos) if caminho_objetos else []
        )
    else:
        if caminho_video is None:
            raise ValueError("o backend 'yolov8' exige caminho_video")
        quadros_objetos = detectar_objetos_yolov8(caminho_video)

    medidas = medir_quadros(quadros_pose, limiares)
    achados = avaliar_postura(medidas, limiares) + avaliar_eventos(
        quadros_objetos, fps, limiares
    )

    duracao = (len(quadros_pose) / fps) if fps else 0.0
    return ResultadoVideo(
        paciente_id=paciente_id,
        caminho_video=str(caminho_video or caminho_keypoints or ""),
        backend_pose=backend_pose,
        backend_detector=backend_detector,
        fps=fps,
        quadros_analisados=len(quadros_pose),
        duracao_s=duracao,
        medidas=medidas,
        achados=achados,
        resumo_angulos=_resumo_angulos(medidas),
    )


def _resumo_angulos(medidas: list[MedidasQuadro]) -> dict[str, float]:
    def maximo(valores: list[float | None]) -> float:
        validos = [v for v in valores if v is not None]
        return max(validos) if validos else 0.0

    return {
        "abducao_maxima_ombro_esquerdo": maximo([m.abducao_ombro_esquerdo for m in medidas]),
        "abducao_maxima_ombro_direito": maximo([m.abducao_ombro_direito for m in medidas]),
        "inclinacao_maxima_tronco": maximo([m.inclinacao_tronco for m in medidas]),
        "assimetria_maxima_ombros": maximo([m.assimetria_ombros for m in medidas]),
    }


def relatorio_markdown(resultado: ResultadoVideo) -> str:
    """Relatorio automatico da sessao, conforme pedido no enunciado."""
    linhas = [
        f"# Relatório de análise de vídeo - {resultado.paciente_id}",
        "",
        f"Arquivo: `{Path(resultado.caminho_video).name}`  ",
        f"Duração: {resultado.duracao_s:.1f} s ({resultado.quadros_analisados} quadros "
        f"a {resultado.fps:.0f} fps)  ",
        f"Estimador de pose: `{resultado.backend_pose}`  ",
        f"Detector de objetos: `{resultado.backend_detector}`",
        "",
        "## Amplitudes observadas",
        "",
        "| Medida | Máximo | Limite |",
        "| --- | ---: | ---: |",
        f"| Abdução do ombro esquerdo | "
        f"{resultado.resumo_angulos['abducao_maxima_ombro_esquerdo']:.1f}° | "
        f"{LIMIARES_VIDEO.abducao_ombro_max_graus:.0f}° |",
        f"| Abdução do ombro direito | "
        f"{resultado.resumo_angulos['abducao_maxima_ombro_direito']:.1f}° | "
        f"{LIMIARES_VIDEO.abducao_ombro_max_graus:.0f}° |",
        f"| Inclinação lateral do tronco | "
        f"{resultado.resumo_angulos['inclinacao_maxima_tronco']:.1f}° | "
        f"{LIMIARES_VIDEO.inclinacao_tronco_max_graus:.0f}° |",
        f"| Assimetria entre ombros | "
        f"{resultado.resumo_angulos['assimetria_maxima_ombros']:.1f}° | "
        f"{LIMIARES_VIDEO.assimetria_ombros_max_graus:.0f}° |",
        "",
        "## Desvios e eventos detectados",
        "",
    ]

    if resultado.achados:
        linhas += ["| Instante | Severidade | Evento | Evidência |", "| --- | --- | --- | --- |"]
        for achado in resultado.achados:
            evidencias = "; ".join(f"{k}={v}" for k, v in achado.evidencias.items())
            linhas.append(
                f"| `{achado.instante or '-'}` | {achado.severidade} | "
                f"{achado.descricao} | {evidencias} |"
            )
    else:
        linhas.append("Nenhum desvio acima dos limites do protocolo foi detectado.")

    linhas += [
        "",
        "> Relatório gerado automaticamente a partir de vídeo sintético. Não "
        "substitui a avaliação do fisioterapeuta responsável.",
        "",
    ]
    return "\n".join(linhas)


def renderizar_video_anotado(
    caminho_video: Path | str,
    resultado: ResultadoVideo,
    destino: Path | str,
    quadros_pose: list[QuadroPose] | None = None,
) -> bool:
    """Grava uma copia do video com os angulos e os desvios sobrepostos.

    Serve para a demonstracao em video: a equipe ve o instante exato em que a
    regra disparou, em vez de so ler o numero no relatorio.
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover - depende de lib opcional
        return False

    captura = cv2.VideoCapture(str(caminho_video))
    if not captura.isOpened():
        return False

    largura = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = captura.get(cv2.CAP_PROP_FPS) or resultado.fps or 30.0

    escritor = cv2.VideoWriter(
        str(destino), cv2.VideoWriter_fourcc(*"mp4v"), fps, (largura, altura)
    )
    if not escritor.isOpened():
        captura.release()
        return False

    por_quadro = {m.quadro: m for m in resultado.medidas}
    # Marca os intervalos em que cada achado estava ativo.
    intervalos: list[tuple[int, int, str]] = []
    for achado in resultado.achados:
        janela = achado.evidencias.get("janela_s")
        if not isinstance(janela, str) or "-" not in janela:
            continue
        inicio_s, fim_s = (float(v) for v in janela.split("-"))
        intervalos.append((int(inicio_s * fps), int(fim_s * fps) + 1, achado.tipo))

    indice = 0
    while True:
        ok, imagem = captura.read()
        if not ok:
            break

        medida = por_quadro.get(indice)
        if medida is not None:
            texto = (
                f"ombro E {medida.abducao_ombro_esquerdo or 0:5.1f}  "
                f"ombro D {medida.abducao_ombro_direito or 0:5.1f}  "
                f"tronco {medida.inclinacao_tronco or 0:4.1f}"
            )
            cv2.putText(
                imagem, texto, (8, altura - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (220, 220, 220), 1,
            )

        ativos = [tipo for inicio, fim, tipo in intervalos if inicio <= indice < fim]
        for posicao, tipo in enumerate(ativos):
            cv2.putText(
                imagem, f"! {tipo}", (8, 40 + posicao * 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (60, 80, 240), 1,
            )
        if ativos:
            cv2.rectangle(imagem, (1, 1), (largura - 2, altura - 2), (60, 80, 240), 2)

        escritor.write(imagem)
        indice += 1

    escritor.release()
    captura.release()
    return True
