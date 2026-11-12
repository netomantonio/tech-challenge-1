"""Gera as amostras multimodais sinteticas usadas na Fase 4.

Nenhum dado real de paciente entra neste projeto. O PhysioNet e o AudioSet
sao citados no enunciado como sugestao de dataset, mas os dois exigem
cadastro/licenca e o PhysioNet nao permite redistribuir os sinais no
repositorio. Para manter a entrega reproduzivel em qualquer maquina, geramos
aqui amostras sinteticas com a mesma estrutura desses datasets:

* `sinais_vitais.csv` - serie temporal multivariada de 12 horas, 1 amostra
  por minuto, no formato tabular do MIMIC/PhysioNet (timestamp + canais);
* `consulta.wav` - audio mono 16 kHz com fala sintetizada, acompanhado de
  `consulta.transcricao.json` com as falas e seus instantes (usado como
  referencia quando o Azure Speech to Text nao esta configurado);
* `fisioterapia.mp4` - video de uma sessao de fisioterapia representada por
  um esqueleto articulado, com `keypoints.json` (saida equivalente a de um
  estimador de pose) e `objetos.json` (saida equivalente a de um detector);
* `prescricoes.json` - evolucao das prescricoes do periodo;
* `movimentacao.csv` - indice horario de atividade do paciente no leito.

Os dois pacientes sao os mesmos da Fase 3: PAC-0006 (pos-operatorio de
mastectomia, caso com deterioracao) e PAC-0002 (caso estavel, usado como
controle para checar se o pipeline nao alerta sem motivo).

Uso:

    python -m fase4.data.gerar_dados_sinteticos
    python -m fase4.data.gerar_dados_sinteticos --paciente PAC-0006 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import wave
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

DIR_AMOSTRAS = Path(__file__).resolve().parent / "amostras"

INICIO_MONITORAMENTO = datetime(2025, 6, 10, 8, 0, 0)
MINUTOS_MONITORAMENTO = 12 * 60
TAXA_AUDIO = 16_000
FPS_VIDEO = 30
LARGURA_VIDEO = 480
ALTURA_VIDEO = 360

CANAIS_VITAIS = (
    "frequencia_cardiaca",
    "pressao_sistolica",
    "pressao_diastolica",
    "saturacao_oxigenio",
    "temperatura",
    "frequencia_respiratoria",
)

# Articulacoes exportadas pelo estimador de pose. E o subconjunto do formato
# COCO que basta para os angulos usados na analise postural.
ARTICULACOES = (
    "nariz",
    "ombro_esquerdo",
    "ombro_direito",
    "cotovelo_esquerdo",
    "cotovelo_direito",
    "punho_esquerdo",
    "punho_direito",
    "quadril_esquerdo",
    "quadril_direito",
    "joelho_esquerdo",
    "joelho_direito",
    "tornozelo_esquerdo",
    "tornozelo_direito",
)

LIGACOES_ESQUELETO = (
    ("ombro_esquerdo", "ombro_direito"),
    ("ombro_esquerdo", "cotovelo_esquerdo"),
    ("cotovelo_esquerdo", "punho_esquerdo"),
    ("ombro_direito", "cotovelo_direito"),
    ("cotovelo_direito", "punho_direito"),
    ("ombro_esquerdo", "quadril_esquerdo"),
    ("ombro_direito", "quadril_direito"),
    ("quadril_esquerdo", "quadril_direito"),
    ("quadril_esquerdo", "joelho_esquerdo"),
    ("joelho_esquerdo", "tornozelo_esquerdo"),
    ("quadril_direito", "joelho_direito"),
    ("joelho_direito", "tornozelo_direito"),
)

# Area critica da sessao: regiao do dreno cirurgico, que o paciente nao deve
# tocar durante o exercicio (PROT-004 do hospital ficticio).
AREA_CRITICA_DRENO = (250, 178, 296, 220)  # x1, y1, x2, y2
CENTRO_DRENO = (
    (AREA_CRITICA_DRENO[0] + AREA_CRITICA_DRENO[2]) / 2,
    (AREA_CRITICA_DRENO[1] + AREA_CRITICA_DRENO[3]) / 2,
)


@dataclass(frozen=True)
class RepeticaoExercicio:
    """Uma repeticao do exercicio de abducao de ombro no video sintetico."""

    pico_ombro_esquerdo: float
    pico_ombro_direito: float
    pico_inclinacao_tronco: float


# Sessao do PAC-0006: tres repeticoes corretas, duas com compensacao de
# tronco e amplitude acima do liberado, e uma ultima repeticao corrigida.
SESSAO_PAC_0006 = (
    RepeticaoExercicio(95.0, 93.0, 3.0),
    RepeticaoExercicio(97.0, 95.0, 4.0),
    RepeticaoExercicio(96.0, 94.0, 2.5),
    RepeticaoExercicio(138.0, 92.0, 19.0),
    RepeticaoExercicio(132.0, 95.0, 17.0),
    RepeticaoExercicio(95.0, 94.0, 3.0),
)

# Janela em que o profissional sai de cena e janela em que o paciente leva a
# mao a area do dreno.
JANELA_SEM_PROFISSIONAL = (300, 420)
JANELA_MAO_NO_DRENO = (382, 432)


# ---------------------------------------------------------------------------
# Sinais vitais
# ---------------------------------------------------------------------------


def gerar_sinais_vitais(paciente_id: str, seed: int) -> tuple[list[dict], dict]:
    """Serie de 12 horas por minuto, com ou sem deterioracao progressiva.

    O caso PAC-0006 recebe uma piora gradual a partir do minuto 480 (taquicardia,
    hipotensao, queda de saturacao, febre e taquipneia) mais um pico isolado de
    frequencia cardiaca no minuto 150, que representa artefato de sensor. O
    artefato existe de proposito: ele testa se o detector exige pontos
    consecutivos antes de confirmar o desvio.
    """
    rng = np.random.default_rng(seed)
    minutos = np.arange(MINUTOS_MONITORAMENTO)

    base = {
        "frequencia_cardiaca": 82.0,
        "pressao_sistolica": 124.0,
        "pressao_diastolica": 78.0,
        "saturacao_oxigenio": 97.0,
        "temperatura": 36.8,
        "frequencia_respiratoria": 16.0,
    }
    ruido = {
        "frequencia_cardiaca": 2.2,
        "pressao_sistolica": 3.0,
        "pressao_diastolica": 2.2,
        "saturacao_oxigenio": 0.5,
        "temperatura": 0.08,
        "frequencia_respiratoria": 0.7,
    }
    # Amplitude da variacao circadiana de cada canal, em unidade propria. Sao
    # valores pequenos de proposito: a variacao fisiologica normal de um dia
    # nao pode ser maior que a deterioracao que queremos detectar, senao o
    # detector de tendencia alerta o paciente estavel.
    amplitude_circadiana = {
        "frequencia_cardiaca": 3.0,
        "pressao_sistolica": 4.0,
        "pressao_diastolica": 3.0,
        "saturacao_oxigenio": 0.3,
        "temperatura": 0.15,
        "frequencia_respiratoria": 0.6,
    }

    series = {}
    for canal, valor in base.items():
        # Ciclo de 24 horas: nas 12 horas monitoradas aparece meia volta.
        oscilacao = amplitude_circadiana[canal] * np.sin(2 * math.pi * minutos / 1440.0)
        series[canal] = valor + oscilacao + rng.normal(0, ruido[canal], minutos.size)

    eventos: dict = {"deterioracao": None, "artefato": None}

    if paciente_id == "PAC-0006":
        inicio, duracao = 480, 180
        rampa = np.zeros(minutos.size)
        fim = min(inicio + duracao, minutos.size)
        rampa[inicio:fim] = np.linspace(0.0, 1.0, fim - inicio)
        rampa[fim:] = 1.0

        series["frequencia_cardiaca"] += rampa * 46.0
        series["pressao_sistolica"] -= rampa * 28.0
        series["pressao_diastolica"] -= rampa * 20.0
        series["saturacao_oxigenio"] -= rampa * 6.0
        series["temperatura"] += rampa * 1.6
        series["frequencia_respiratoria"] += rampa * 10.0

        series["frequencia_cardiaca"][150] = 185.0
        eventos["deterioracao"] = {
            "minuto_inicio": inicio,
            "minuto_fim": fim,
            "descricao": "piora progressiva compativel com resposta inflamatoria sistemica",
        }
        eventos["artefato"] = {
            "minuto": 150,
            "descricao": "pico isolado de frequencia cardiaca (artefato de sensor)",
        }

    series["saturacao_oxigenio"] = np.clip(series["saturacao_oxigenio"], 80.0, 100.0)

    registros = []
    for i, minuto in enumerate(minutos):
        instante = INICIO_MONITORAMENTO + timedelta(minutes=int(minuto))
        registro = {"timestamp": instante.isoformat()}
        for canal in CANAIS_VITAIS:
            casas = 2 if canal == "temperatura" else 1
            registro[canal] = round(float(series[canal][i]), casas)
        registros.append(registro)

    return registros, eventos


def salvar_sinais_vitais(registros: list[dict], destino: Path) -> None:
    cabecalho = ["timestamp", *CANAIS_VITAIS]
    linhas = [",".join(cabecalho)]
    for registro in registros:
        linhas.append(",".join(str(registro[coluna]) for coluna in cabecalho))
    destino.write_text("\n".join(linhas) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Prescricoes
# ---------------------------------------------------------------------------


def gerar_prescricoes(paciente_id: str) -> list[dict]:
    """Evolucao de prescricoes do periodo monitorado.

    Para o PAC-0006 foram plantadas tres alteracoes inesperadas: um salto de
    dose de opioide, a suspensao do antimicrobiano antes do fim do esquema e
    a prescricao de dois anti-inflamatorios da mesma classe ao mesmo tempo.
    """

    def evento(hora: str, **campos) -> dict:
        instante = datetime.fromisoformat(f"2025-06-10T{hora}:00")
        return {"timestamp": instante.isoformat(), **campos}

    if paciente_id == "PAC-0006":
        return [
            evento(
                "08:10",
                acao="inicio",
                medicamento="dipirona",
                classe="analgesico_nao_opioide",
                dose=1000.0,
                unidade="mg",
                frequencia_horas=6,
                via="IV",
                prescritor="MED-204",
            ),
            evento(
                "09:00",
                acao="inicio",
                medicamento="ibuprofeno",
                classe="anti_inflamatorio_nao_esteroidal",
                dose=600.0,
                unidade="mg",
                frequencia_horas=8,
                via="oral",
                prescritor="MED-204",
            ),
            evento(
                "09:30",
                acao="inicio",
                medicamento="ceftriaxona",
                classe="antimicrobiano",
                dose=1000.0,
                unidade="mg",
                frequencia_horas=12,
                via="IV",
                prescritor="MED-204",
                duracao_prevista_dias=7,
                dia_tratamento=1,
            ),
            evento(
                "10:30",
                acao="inicio",
                medicamento="morfina",
                classe="opioide",
                dose=2.0,
                unidade="mg",
                frequencia_horas=4,
                via="SC",
                prescritor="MED-204",
            ),
            evento(
                "14:05",
                acao="ajuste",
                medicamento="morfina",
                classe="opioide",
                dose=10.0,
                unidade="mg",
                frequencia_horas=4,
                via="SC",
                prescritor="MED-311",
            ),
            evento(
                "15:40",
                acao="suspensao",
                medicamento="ceftriaxona",
                classe="antimicrobiano",
                dose=1000.0,
                unidade="mg",
                frequencia_horas=12,
                via="IV",
                prescritor="MED-311",
                duracao_prevista_dias=7,
                dia_tratamento=2,
            ),
            evento(
                "16:20",
                acao="inicio",
                medicamento="cetoprofeno",
                classe="anti_inflamatorio_nao_esteroidal",
                dose=100.0,
                unidade="mg",
                frequencia_horas=12,
                via="IV",
                prescritor="MED-311",
            ),
        ]

    return [
        evento(
            "08:15",
            acao="inicio",
            medicamento="dipirona",
            classe="analgesico_nao_opioide",
            dose=1000.0,
            unidade="mg",
            frequencia_horas=6,
            via="oral",
            prescritor="MED-118",
        ),
        evento(
            "08:20",
            acao="inicio",
            medicamento="tamoxifeno",
            classe="hormonioterapia",
            dose=20.0,
            unidade="mg",
            frequencia_horas=24,
            via="oral",
            prescritor="MED-118",
        ),
        evento(
            "14:00",
            acao="ajuste",
            medicamento="dipirona",
            classe="analgesico_nao_opioide",
            dose=1000.0,
            unidade="mg",
            frequencia_horas=8,
            via="oral",
            prescritor="MED-118",
        ),
    ]


# ---------------------------------------------------------------------------
# Movimentacao no leito
# ---------------------------------------------------------------------------


def gerar_movimentacao(paciente_id: str, seed: int) -> list[dict]:
    """Indice horario de atividade (0-100) derivado do sensor do leito.

    No PAC-0006 ha agitacao entre 02h e 04h e imobilidade prolongada entre
    09h e 15h; no PAC-0002 o padrao segue o esperado para a internacao.
    """
    rng = np.random.default_rng(seed + 7)
    inicio = datetime(2025, 6, 10, 0, 0, 0)

    if paciente_id == "PAC-0006":
        perfil = [
            6, 4, 62, 70, 58, 9, 12, 20,  # 00h-07h
            26, 4, 3, 2, 4, 3, 3, 22,     # 08h-15h
            34, 30, 28, 24, 18, 14, 10, 8,  # 16h-23h
        ]
    else:
        perfil = [
            8, 6, 5, 7, 6, 10, 18, 32,
            44, 52, 48, 40, 46, 50, 44, 38,
            42, 46, 40, 34, 28, 20, 14, 10,
        ]

    registros = []
    for hora, valor in enumerate(perfil):
        indice = float(np.clip(valor + rng.normal(0, 1.5), 0.0, 100.0))
        registros.append(
            {
                "timestamp": (inicio + timedelta(hours=hora)).isoformat(),
                "indice_atividade": round(indice, 1),
                "fonte": "sensor_leito",
            }
        )
    return registros


def salvar_movimentacao(registros: list[dict], destino: Path) -> None:
    linhas = ["timestamp,indice_atividade,fonte"]
    for registro in registros:
        linhas.append(
            f"{registro['timestamp']},{registro['indice_atividade']},{registro['fonte']}"
        )
    destino.write_text("\n".join(linhas) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Audio da consulta
# ---------------------------------------------------------------------------

FALAS_PAC_0006 = [
    "Doutora, desde ontem eu estou muito cansada.",
    "Quando eu levanto da cama eu fico sem ar.",
    "Hoje de manha senti um aperto no peito.",
    "A ferida esta mais quente e tive febre de noite.",
    "A dor aumentou e eu nao aguento fazer os exercicios.",
    "Preciso parar um pouco para respirar.",
]

FALAS_PAC_0002 = [
    "Bom dia doutora, eu estou bem melhor esta semana.",
    "Consegui fazer a caminhada todos os dias sem dor.",
    "Tomei o remedio nos horarios e nao tive enjoo.",
    "Estou mais animada e dormindo tranquila a noite.",
]


@dataclass(frozen=True)
class PerfilVocal:
    """Parametros de sintese que diferenciam uma voz cansada de uma estavel."""

    f0_inicial: float
    f0_final: float
    jitter: float
    taxa_silabas: float
    amplitude_inicial: float
    amplitude_final: float
    duracao_fala: tuple[float, float]
    duracao_pausa: tuple[float, float]
    ruido_respiratorio: float


PERFIL_FADIGA = PerfilVocal(
    f0_inicial=196.0,
    f0_final=152.0,
    jitter=0.045,
    taxa_silabas=1.9,
    amplitude_inicial=0.30,
    amplitude_final=0.15,
    duracao_fala=(1.0, 1.5),
    duracao_pausa=(1.3, 2.0),
    ruido_respiratorio=0.030,
)

PERFIL_ESTAVEL = PerfilVocal(
    f0_inicial=205.0,
    f0_final=201.0,
    jitter=0.006,
    taxa_silabas=4.3,
    amplitude_inicial=0.34,
    amplitude_final=0.32,
    duracao_fala=(2.6, 3.6),
    duracao_pausa=(0.35, 0.60),
    ruido_respiratorio=0.004,
)


def _media_movel(sinal: np.ndarray, janela: int) -> np.ndarray:
    if janela <= 1:
        return sinal
    nucleo = np.ones(janela) / janela
    return np.convolve(sinal, nucleo, mode="same")


def _sintetizar_fala(
    duracao_s: float,
    f0_inicial: float,
    f0_final: float,
    jitter: float,
    taxa_silabas: float,
    amplitude: float,
    ruido_respiratorio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sintetiza um trecho de fala como soma de harmonicos modulada por silabas.

    Nao e um sintetizador de voz: e um sinal vozeado controlado, com F0,
    jitter, taxa silabica e ruido de respiracao ajustaveis. Isso permite
    gerar os dois casos (fala cansada e fala estavel) de forma reproduzivel e
    verificar se o extrator de atributos acusticos reage ao que deveria.
    """
    n = max(1, int(duracao_s * TAXA_AUDIO))
    t = np.arange(n) / TAXA_AUDIO

    f0 = np.linspace(f0_inicial, f0_final, n)
    perturbacao = _media_movel(rng.normal(0.0, jitter, n), int(TAXA_AUDIO * 0.004))
    frequencia_instantanea = f0 * (1.0 + perturbacao)

    fase = 2 * math.pi * np.cumsum(frequencia_instantanea) / TAXA_AUDIO
    sinal = np.zeros(n)
    for harmonico, peso in enumerate((1.0, 0.52, 0.34, 0.22, 0.14), start=1):
        sinal += peso * np.sin(harmonico * fase)
    sinal /= 2.22

    # Envelope silabico: abre e fecha a amplitude na taxa de silabas.
    envelope_silabico = 0.25 + 0.75 * np.abs(np.sin(math.pi * taxa_silabas * t))
    sinal *= envelope_silabico

    # Ruido de respiracao, mais audivel no caso com esforco respiratorio.
    sinal += _media_movel(rng.normal(0.0, ruido_respiratorio, n), 9)

    # Fade de 25 ms nas pontas para nao gerar estalo na juncao dos trechos.
    fade = min(int(0.025 * TAXA_AUDIO), n // 2)
    if fade > 0:
        rampa = np.linspace(0.0, 1.0, fade)
        sinal[:fade] *= rampa
        sinal[-fade:] *= rampa[::-1]

    pico = float(np.max(np.abs(sinal))) or 1.0
    return sinal / pico * amplitude


def gerar_audio_consulta(
    paciente_id: str, seed: int
) -> tuple[np.ndarray, dict]:
    """Monta o audio da consulta e a transcricao de referencia correspondente."""
    rng = np.random.default_rng(seed + 13)
    perfil = PERFIL_FADIGA if paciente_id == "PAC-0006" else PERFIL_ESTAVEL
    falas = FALAS_PAC_0006 if paciente_id == "PAC-0006" else FALAS_PAC_0002

    blocos: list[np.ndarray] = []
    segmentos: list[dict] = []
    posicao_s = 0.6
    blocos.append(np.zeros(int(0.6 * TAXA_AUDIO)))

    total = len(falas)
    for indice, fala in enumerate(falas):
        progresso = indice / max(1, total - 1)
        duracao = float(rng.uniform(*perfil.duracao_fala))
        amplitude = perfil.amplitude_inicial + progresso * (
            perfil.amplitude_final - perfil.amplitude_inicial
        )
        f0_bloco_inicial = perfil.f0_inicial + progresso * (
            perfil.f0_final - perfil.f0_inicial
        )
        f0_bloco_final = f0_bloco_inicial - 6.0 * progresso

        trecho = _sintetizar_fala(
            duracao_s=duracao,
            f0_inicial=f0_bloco_inicial,
            f0_final=f0_bloco_final,
            jitter=perfil.jitter,
            taxa_silabas=perfil.taxa_silabas,
            amplitude=amplitude,
            ruido_respiratorio=perfil.ruido_respiratorio,
            rng=rng,
        )
        blocos.append(trecho)
        segmentos.append(
            {
                "inicio_s": round(posicao_s, 3),
                "fim_s": round(posicao_s + duracao, 3),
                "texto": fala,
                "confianca": round(float(rng.uniform(0.82, 0.94)), 3),
            }
        )
        posicao_s += duracao

        if indice < total - 1:
            pausa = float(rng.uniform(*perfil.duracao_pausa))
            blocos.append(np.zeros(int(pausa * TAXA_AUDIO)))
            posicao_s += pausa

    blocos.append(np.zeros(int(0.8 * TAXA_AUDIO)))
    sinal = np.concatenate(blocos)
    # Piso de ruido ambiente do consultorio.
    sinal = sinal + rng.normal(0.0, 0.0015, sinal.size)

    transcricao = {
        "paciente_id": paciente_id,
        "idioma": "pt-BR",
        "duracao_s": round(sinal.size / TAXA_AUDIO, 3),
        "origem": "transcricao de referencia do audio sintetico",
        "segmentos": segmentos,
    }
    return sinal, transcricao


def salvar_wav(sinal: np.ndarray, destino: Path) -> None:
    amostras = np.clip(sinal, -1.0, 1.0)
    inteiros = (amostras * 32767).astype("<i2")
    with wave.open(str(destino), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(TAXA_AUDIO)
        wav.writeframes(inteiros.tobytes())


# ---------------------------------------------------------------------------
# Video da sessao de fisioterapia
# ---------------------------------------------------------------------------


def _angulos_do_quadro(quadro: int, total_quadros: int) -> tuple[float, float, float]:
    """Angulos de abducao dos dois ombros e inclinacao do tronco no quadro."""
    quadros_por_repeticao = total_quadros / len(SESSAO_PAC_0006)
    indice = min(int(quadro / quadros_por_repeticao), len(SESSAO_PAC_0006) - 1)
    repeticao = SESSAO_PAC_0006[indice]

    fase = (quadro - indice * quadros_por_repeticao) / quadros_por_repeticao
    # Meia senoide: sobe ate o pico no meio da repeticao e volta ao repouso.
    forma = math.sin(math.pi * min(max(fase, 0.0), 1.0))

    repouso = 10.0
    esquerdo = repouso + (repeticao.pico_ombro_esquerdo - repouso) * forma
    direito = repouso + (repeticao.pico_ombro_direito - repouso) * forma
    tronco = repeticao.pico_inclinacao_tronco * forma
    return esquerdo, direito, tronco


def _keypoints_do_quadro(quadro: int, total_quadros: int) -> dict[str, tuple[float, float, float]]:
    """Posicao em pixels de cada articulacao, no formato de um estimador de pose."""
    abducao_esq, abducao_dir, inclinacao = _angulos_do_quadro(quadro, total_quadros)

    quadril_centro = np.array([LARGURA_VIDEO / 2, 250.0])
    altura_tronco = 100.0
    rad = math.radians(inclinacao)
    # O tronco inclina para o lado esquerdo do paciente (x crescente na imagem).
    ombro_centro = quadril_centro + np.array(
        [altura_tronco * math.sin(rad), -altura_tronco * math.cos(rad)]
    )

    meia_largura_ombros = 30.0
    meia_largura_quadris = 22.0
    braco, antebraco, coxa, perna = 52.0, 48.0, 58.0, 55.0

    pontos: dict[str, tuple[float, float, float]] = {}

    def registrar(nome: str, ponto: np.ndarray, confianca: float = 0.95) -> None:
        pontos[nome] = (float(ponto[0]), float(ponto[1]), confianca)

    ombro_esq = ombro_centro + np.array([meia_largura_ombros, 0.0])
    ombro_dir = ombro_centro + np.array([-meia_largura_ombros, 0.0])
    registrar("ombro_esquerdo", ombro_esq)
    registrar("ombro_direito", ombro_dir)
    registrar("nariz", ombro_centro + np.array([0.0, -42.0]))

    for lado, sinal, ombro, abducao in (
        ("esquerdo", 1.0, ombro_esq, abducao_esq),
        ("direito", -1.0, ombro_dir, abducao_dir),
    ):
        # Abducao medida a partir do braco ao longo do corpo (0 grau aponta
        # para baixo), abrindo para fora do tronco.
        angulo = math.radians(abducao)
        direcao = np.array([sinal * math.sin(angulo), math.cos(angulo)])
        cotovelo = ombro + braco * direcao
        punho = cotovelo + antebraco * direcao
        registrar(f"cotovelo_{lado}", cotovelo)
        registrar(f"punho_{lado}", punho)

    inicio, fim = JANELA_MAO_NO_DRENO
    if inicio <= quadro < fim:
        # O paciente flexiona o cotovelo e leva a mao a regiao do dreno.
        avanco = (quadro - inicio) / (fim - inicio)
        peso = math.sin(math.pi * avanco)
        punho_atual = np.array(pontos["punho_esquerdo"][:2])
        destino = np.array(CENTRO_DRENO)
        punho_novo = punho_atual + (destino - punho_atual) * peso
        registrar("punho_esquerdo", punho_novo)
        cotovelo_atual = np.array(pontos["cotovelo_esquerdo"][:2])
        meio = (np.array(pontos["ombro_esquerdo"][:2]) + punho_novo) / 2 + np.array([26.0, 6.0])
        registrar("cotovelo_esquerdo", cotovelo_atual + (meio - cotovelo_atual) * peso)

    quadril_esq = quadril_centro + np.array([meia_largura_quadris, 0.0])
    quadril_dir = quadril_centro + np.array([-meia_largura_quadris, 0.0])
    registrar("quadril_esquerdo", quadril_esq)
    registrar("quadril_direito", quadril_dir)
    registrar("joelho_esquerdo", quadril_esq + np.array([2.0, coxa]))
    registrar("joelho_direito", quadril_dir + np.array([-2.0, coxa]))
    registrar("tornozelo_esquerdo", quadril_esq + np.array([3.0, coxa + perna]))
    registrar("tornozelo_direito", quadril_dir + np.array([-3.0, coxa + perna]))

    return pontos


def gerar_anotacoes_video(
    duracao_s: float, seed: int
) -> tuple[dict, dict]:
    """Produz os keypoints por quadro e as caixas de objetos por quadro.

    As duas estruturas tem o mesmo formato que `fase4.video_analysis` espera
    receber de um estimador de pose e de um detector de objetos. Assim a
    analise e identica, venha ela de um modelo real ou destas anotacoes.
    """
    rng = np.random.default_rng(seed + 29)
    total_quadros = int(duracao_s * FPS_VIDEO)

    quadros_pose = []
    quadros_objetos = []

    for quadro in range(total_quadros):
        pontos = _keypoints_do_quadro(quadro, total_quadros)
        # Ruido de estimacao de pose, para a analise nao receber angulos exatos.
        articulacoes = {}
        for nome, (x, y, confianca) in pontos.items():
            articulacoes[nome] = {
                "x": round(float(x + rng.normal(0, 0.8)), 2),
                "y": round(float(y + rng.normal(0, 0.8)), 2),
                "confianca": round(float(min(0.99, confianca + rng.normal(0, 0.02))), 3),
            }
        quadros_pose.append(
            {"quadro": quadro, "instante_s": round(quadro / FPS_VIDEO, 3), "articulacoes": articulacoes}
        )

        xs = [a["x"] for a in articulacoes.values()]
        ys = [a["y"] for a in articulacoes.values()]
        objetos = [
            {
                "classe": "paciente",
                "confianca": 0.93,
                "caixa": [round(min(xs) - 8, 1), round(min(ys) - 8, 1), round(max(xs) + 8, 1), round(max(ys) + 8, 1)],
            },
            {"classe": "maca", "confianca": 0.88, "caixa": [40.0, 268.0, 440.0, 320.0]},
            {
                "classe": "dreno",
                "confianca": 0.81,
                "caixa": [float(v) for v in AREA_CRITICA_DRENO],
                "area_critica": True,
            },
            {
                "classe": "mao_paciente",
                "confianca": 0.86,
                "caixa": [
                    round(articulacoes["punho_esquerdo"]["x"] - 11, 1),
                    round(articulacoes["punho_esquerdo"]["y"] - 11, 1),
                    round(articulacoes["punho_esquerdo"]["x"] + 11, 1),
                    round(articulacoes["punho_esquerdo"]["y"] + 11, 1),
                ],
            },
        ]
        inicio, fim = JANELA_SEM_PROFISSIONAL
        if not (inicio <= quadro < fim):
            objetos.append(
                {"classe": "profissional", "confianca": 0.90, "caixa": [26.0, 120.0, 96.0, 300.0]}
            )
        quadros_objetos.append(
            {"quadro": quadro, "instante_s": round(quadro / FPS_VIDEO, 3), "objetos": objetos}
        )

    pose = {
        "fonte": "anotacao sintetica da sessao de fisioterapia",
        "fps": FPS_VIDEO,
        "largura": LARGURA_VIDEO,
        "altura": ALTURA_VIDEO,
        "articulacoes": list(ARTICULACOES),
        "quadros": quadros_pose,
    }
    deteccoes = {
        "fonte": "anotacao sintetica equivalente a saida de um detector de objetos",
        "fps": FPS_VIDEO,
        "classes": ["paciente", "profissional", "maca", "dreno", "mao_paciente"],
        "quadros": quadros_objetos,
    }
    return pose, deteccoes


def renderizar_video(pose: dict, deteccoes: dict, destino: Path) -> bool:
    """Desenha o esqueleto e as caixas num MP4. Requer OpenCV instalado."""
    try:
        import cv2
    except ImportError:
        return False

    escritor = cv2.VideoWriter(
        str(destino),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS_VIDEO,
        (LARGURA_VIDEO, ALTURA_VIDEO),
    )
    if not escritor.isOpened():
        return False

    for quadro_pose, quadro_obj in zip(pose["quadros"], deteccoes["quadros"]):
        imagem = np.full((ALTURA_VIDEO, LARGURA_VIDEO, 3), 32, dtype=np.uint8)
        cv2.rectangle(imagem, (40, 268), (440, 320), (70, 70, 70), -1)

        x1, y1, x2, y2 = (int(v) for v in AREA_CRITICA_DRENO)
        cv2.rectangle(imagem, (x1, y1), (x2, y2), (40, 120, 220), 1)
        cv2.putText(
            imagem, "dreno", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (40, 120, 220), 1
        )

        articulacoes = quadro_pose["articulacoes"]
        for origem, fim_ligacao in LIGACOES_ESQUELETO:
            p1 = articulacoes[origem]
            p2 = articulacoes[fim_ligacao]
            cv2.line(
                imagem,
                (int(p1["x"]), int(p1["y"])),
                (int(p2["x"]), int(p2["y"])),
                (210, 210, 210),
                2,
            )
        for ponto in articulacoes.values():
            cv2.circle(imagem, (int(ponto["x"]), int(ponto["y"])), 3, (80, 200, 120), -1)

        tem_profissional = any(o["classe"] == "profissional" for o in quadro_obj["objetos"])
        if tem_profissional:
            cv2.rectangle(imagem, (26, 120), (96, 300), (150, 150, 90), 1)
            cv2.putText(
                imagem, "profissional", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 90), 1
            )

        cv2.putText(
            imagem,
            f"t={quadro_pose['instante_s']:5.2f}s",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
        )
        escritor.write(imagem)

    escritor.release()
    return True


# ---------------------------------------------------------------------------
# Orquestracao
# ---------------------------------------------------------------------------

PACIENTES_PADRAO = ("PAC-0006", "PAC-0002")
PACIENTES_COM_VIDEO = ("PAC-0006",)
DURACAO_VIDEO_S = 20.0


def gerar_amostras_paciente(
    paciente_id: str, seed: int = 42, destino_raiz: Path | None = None
) -> dict:
    """Gera todas as amostras de um paciente e devolve os caminhos criados."""
    raiz = (destino_raiz or DIR_AMOSTRAS) / paciente_id
    raiz.mkdir(parents=True, exist_ok=True)
    criados: dict[str, str] = {}

    registros, eventos = gerar_sinais_vitais(paciente_id, seed)
    caminho_vitais = raiz / "sinais_vitais.csv"
    salvar_sinais_vitais(registros, caminho_vitais)
    criados["sinais_vitais"] = str(caminho_vitais)

    caminho_prescricoes = raiz / "prescricoes.json"
    caminho_prescricoes.write_text(
        json.dumps(gerar_prescricoes(paciente_id), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    criados["prescricoes"] = str(caminho_prescricoes)

    caminho_movimentacao = raiz / "movimentacao.csv"
    salvar_movimentacao(gerar_movimentacao(paciente_id, seed), caminho_movimentacao)
    criados["movimentacao"] = str(caminho_movimentacao)

    sinal, transcricao = gerar_audio_consulta(paciente_id, seed)
    caminho_wav = raiz / "consulta.wav"
    salvar_wav(sinal, caminho_wav)
    (raiz / "consulta.transcricao.json").write_text(
        json.dumps(transcricao, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    criados["audio"] = str(caminho_wav)

    if paciente_id in PACIENTES_COM_VIDEO:
        pose, deteccoes = gerar_anotacoes_video(DURACAO_VIDEO_S, seed)
        (raiz / "fisioterapia.keypoints.json").write_text(
            json.dumps(pose, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (raiz / "fisioterapia.objetos.json").write_text(
            json.dumps(deteccoes, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        caminho_video = raiz / "fisioterapia.mp4"
        if renderizar_video(pose, deteccoes, caminho_video):
            criados["video"] = str(caminho_video)
        criados["keypoints"] = str(raiz / "fisioterapia.keypoints.json")
        criados["objetos"] = str(raiz / "fisioterapia.objetos.json")

    (raiz / "eventos_plantados.json").write_text(
        json.dumps(
            {
                "paciente_id": paciente_id,
                "seed": seed,
                "sinais_vitais": eventos,
                "observacao": (
                    "Lista do que foi injetado de proposito nas amostras. Serve de "
                    "gabarito para conferir o que o pipeline detectou."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return criados


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--paciente",
        action="append",
        default=None,
        help="Paciente a gerar (pode repetir). Padrao: PAC-0006 e PAC-0002.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--destino",
        default=None,
        help="Diretorio raiz das amostras (padrao: fase4/data/amostras).",
    )
    args = parser.parse_args()

    pacientes = args.paciente or list(PACIENTES_PADRAO)
    raiz = Path(args.destino) if args.destino else None

    for paciente in pacientes:
        criados = gerar_amostras_paciente(paciente, seed=args.seed, destino_raiz=raiz)
        print(f"{paciente}:")
        for chave, caminho in criados.items():
            print(f"  {chave:14s} {caminho}")


if __name__ == "__main__":
    main()
