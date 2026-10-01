"""Analise de audio das consultas medicas (Fase 4).

O enunciado pede tres coisas desta modalidade: detectar alteracoes vocais
indicativas de condicao medica (cansaco, dificuldade respiratoria),
transcrever com Azure Speech to Text e identificar termos criticos e
sentimento com Azure Text Analytics.

A parte vocal nao depende da nuvem: os atributos acusticos sao calculados
aqui, direto da forma de onda, com `wave` e numpy. A escolha foi proposital.
O Speech to Text devolve texto, nao fisiologia da voz; quem indica fadiga ou
esforco respiratorio e a energia da fala, o tamanho das pausas, a taxa de
fala e a estabilidade da frequencia fundamental. A nuvem entra na camada de
linguagem (transcricao, sentimento e frases-chave), via `fase4.azure_services`.

Limitacao assumida: nossos atributos sao aproximacoes de medidas usadas em
acustica clinica. O `jitter` aqui e a variacao relativa da F0 entre quadros
consecutivos de 10 ms, nao o jitter periodo a periodo medido por softwares
especializados. Serve para comparar gravacoes dentro deste projeto, nao para
diagnostico.
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fase4.alertas import Achado
from fase4.azure_services import (
    ResultadoAnaliseTexto,
    ResultadoTranscricao,
    analisar_texto,
    normalizar_texto,
    transcrever_audio,
)
from fase4.config import LIMIARES_AUDIO, LimiaresAudio

TAMANHO_QUADRO_S = 0.025
PASSO_QUADRO_S = 0.010
F0_MINIMA = 70.0
F0_MAXIMA = 350.0
DURACAO_MINIMA_SEGMENTO_S = 0.20


@dataclass
class SegmentoVoz:
    inicio_s: float
    fim_s: float

    @property
    def duracao_s(self) -> float:
        return self.fim_s - self.inicio_s


@dataclass
class AtributosAcusticos:
    """Atributos extraidos da forma de onda da consulta."""

    duracao_total_s: float
    duracao_fala_s: float
    proporcao_pausa: float
    quantidade_segmentos: int
    duracao_media_frase_s: float
    pausa_media_s: float
    pausa_maxima_s: float
    f0_media_hz: float
    f0_desvio_hz: float
    jitter_f0: float
    taxa_fala_silabas_s: float
    energia_media: float
    queda_energia_relativa: float
    taxa_cruzamento_zero: float
    centroide_espectral_hz: float
    segmentos: list[SegmentoVoz] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "duracao_total_s": round(self.duracao_total_s, 3),
            "duracao_fala_s": round(self.duracao_fala_s, 3),
            "proporcao_pausa": round(self.proporcao_pausa, 4),
            "quantidade_segmentos": self.quantidade_segmentos,
            "duracao_media_frase_s": round(self.duracao_media_frase_s, 3),
            "pausa_media_s": round(self.pausa_media_s, 3),
            "pausa_maxima_s": round(self.pausa_maxima_s, 3),
            "f0_media_hz": round(self.f0_media_hz, 2),
            "f0_desvio_hz": round(self.f0_desvio_hz, 2),
            "jitter_f0": round(self.jitter_f0, 5),
            "taxa_fala_silabas_s": round(self.taxa_fala_silabas_s, 3),
            "energia_media": round(self.energia_media, 5),
            "queda_energia_relativa": round(self.queda_energia_relativa, 4),
            "taxa_cruzamento_zero": round(self.taxa_cruzamento_zero, 4),
            "centroide_espectral_hz": round(self.centroide_espectral_hz, 1),
        }


@dataclass
class ResultadoAudio:
    paciente_id: str
    caminho_audio: str
    atributos: AtributosAcusticos
    transcricao: ResultadoTranscricao
    analise_texto: ResultadoAnaliseTexto
    achados: list[Achado]

    def to_dict(self) -> dict:
        return {
            "paciente_id": self.paciente_id,
            "caminho_audio": self.caminho_audio,
            "atributos_acusticos": self.atributos.to_dict(),
            "transcricao": self.transcricao.to_dict(),
            "analise_texto": self.analise_texto.to_dict(),
            "achados": [a.to_dict() for a in self.achados],
        }


# ---------------------------------------------------------------------------
# Leitura e atributos
# ---------------------------------------------------------------------------


def carregar_wav(caminho: Path | str) -> tuple[np.ndarray, int]:
    """Le um WAV PCM mono (ou converte estereo para mono) normalizado em [-1, 1]."""
    with wave.open(str(caminho), "rb") as wav:
        canais = wav.getnchannels()
        largura = wav.getsampwidth()
        taxa = wav.getframerate()
        quadros = wav.readframes(wav.getnframes())

    if largura != 2:
        raise ValueError("esperado WAV PCM de 16 bits")

    amostras = np.frombuffer(quadros, dtype="<i2").astype(np.float64) / 32768.0
    if canais > 1:
        amostras = amostras.reshape(-1, canais).mean(axis=1)
    return amostras, taxa


def _enquadrar(sinal: np.ndarray, taxa: int) -> np.ndarray:
    tamanho = int(TAMANHO_QUADRO_S * taxa)
    passo = int(PASSO_QUADRO_S * taxa)
    if sinal.size < tamanho:
        return sinal.reshape(1, -1) if sinal.size else np.zeros((1, 1))
    quantidade = 1 + (sinal.size - tamanho) // passo
    indices = np.arange(tamanho)[None, :] + passo * np.arange(quantidade)[:, None]
    return sinal[indices]


def _detectar_voz(rms: np.ndarray) -> np.ndarray:
    """VAD simples por energia, com limiar adaptado ao piso de ruido.

    O limiar combina o piso de ruido (percentil 10) com uma fracao do pico.
    Isso evita tanto marcar o ruido ambiente como fala quanto perder o fim de
    uma frase que termina fraca, que e justamente o caso do paciente cansado.
    """
    if rms.size == 0:
        return np.zeros(0, dtype=bool)
    piso = float(np.percentile(rms, 10))
    pico = float(np.max(rms))
    limiar = max(piso * 3.0, pico * 0.10)
    return rms > limiar


def _agrupar_segmentos(voz: np.ndarray, passo_s: float) -> list[SegmentoVoz]:
    segmentos: list[SegmentoVoz] = []
    inicio: int | None = None
    for i, ativo in enumerate(voz):
        if ativo and inicio is None:
            inicio = i
        elif not ativo and inicio is not None:
            segmentos.append(SegmentoVoz(inicio * passo_s, i * passo_s))
            inicio = None
    if inicio is not None:
        segmentos.append(SegmentoVoz(inicio * passo_s, voz.size * passo_s))
    return [s for s in segmentos if s.duracao_s >= DURACAO_MINIMA_SEGMENTO_S]


def _f0_por_autocorrelacao(quadro: np.ndarray, taxa: int) -> float:
    """Estima a F0 de um quadro pelo primeiro pico da autocorrelacao.

    Metodo classico e barato: remove a media, calcula a autocorrelacao e
    procura o maior pico dentro do intervalo de periodos plausiveis para voz
    humana. Devolve 0.0 quando o pico e fraco (quadro nao vozeado).
    """
    janela = quadro - quadro.mean()
    if np.allclose(janela, 0.0):
        return 0.0
    correlacao = np.correlate(janela, janela, mode="full")[janela.size - 1 :]
    if correlacao[0] <= 0:
        return 0.0

    atraso_min = int(taxa / F0_MAXIMA)
    atraso_max = min(int(taxa / F0_MINIMA), correlacao.size - 1)
    if atraso_max <= atraso_min:
        return 0.0

    faixa = correlacao[atraso_min:atraso_max]
    melhor = int(np.argmax(faixa)) + atraso_min
    if correlacao[melhor] / correlacao[0] < 0.30:
        return 0.0
    return float(taxa) / melhor


def _contar_silabas(rms: np.ndarray, passo_s: float) -> int:
    """Conta nucleos silabicos como picos do envelope de energia suavizado."""
    if rms.size < 3:
        return 0
    janela = max(1, int(0.05 / passo_s))
    suave = np.convolve(rms, np.ones(janela) / janela, mode="same")
    limiar = float(np.percentile(suave, 55))
    picos = 0
    for i in range(1, suave.size - 1):
        if suave[i] > limiar and suave[i] >= suave[i - 1] and suave[i] > suave[i + 1]:
            picos += 1
    return picos


def _centroide_espectral(quadros: np.ndarray, taxa: int) -> float:
    if quadros.size == 0:
        return 0.0
    espectro = np.abs(np.fft.rfft(quadros, axis=1))
    frequencias = np.fft.rfftfreq(quadros.shape[1], d=1.0 / taxa)
    energia = espectro.sum(axis=1)
    validos = energia > 0
    if not np.any(validos):
        return 0.0
    centroides = (espectro[validos] * frequencias).sum(axis=1) / energia[validos]
    return float(centroides.mean())


def extrair_atributos(caminho_audio: Path | str) -> AtributosAcusticos:
    """Calcula os atributos acusticos usados nas regras de alteracao vocal."""
    sinal, taxa = carregar_wav(caminho_audio)
    duracao_total = sinal.size / taxa

    quadros = _enquadrar(sinal, taxa)
    rms = np.sqrt((quadros**2).mean(axis=1))
    voz = _detectar_voz(rms)
    segmentos = _agrupar_segmentos(voz, PASSO_QUADRO_S)

    duracao_fala = sum(s.duracao_s for s in segmentos)
    proporcao_pausa = 1.0 - (duracao_fala / duracao_total if duracao_total else 0.0)

    pausas = []
    for anterior, atual in zip(segmentos, segmentos[1:]):
        pausas.append(atual.inicio_s - anterior.fim_s)

    quadros_vozeados = quadros[voz] if np.any(voz) else np.zeros((0, quadros.shape[1]))
    f0s = np.array(
        [_f0_por_autocorrelacao(quadro, taxa) for quadro in quadros_vozeados]
    )
    f0s_validas = f0s[f0s > 0]

    # Jitter: variacao relativa da F0 entre quadros vozeados consecutivos.
    if f0s_validas.size > 1:
        diferencas = np.abs(np.diff(f0s_validas))
        jitter = float(diferencas.mean() / f0s_validas.mean())
    else:
        jitter = 0.0

    silabas = _contar_silabas(rms[voz] if np.any(voz) else rms, PASSO_QUADRO_S)
    taxa_fala = silabas / duracao_total if duracao_total else 0.0

    energia_vozeada = rms[voz] if np.any(voz) else rms
    if energia_vozeada.size >= 6:
        terco = energia_vozeada.size // 3
        inicio = float(energia_vozeada[:terco].mean())
        fim = float(energia_vozeada[-terco:].mean())
        queda = (inicio - fim) / inicio if inicio > 0 else 0.0
    else:
        queda = 0.0

    if quadros_vozeados.size:
        cruzamentos = float(
            np.mean(np.abs(np.diff(np.sign(quadros_vozeados), axis=1)) > 0)
        )
    else:
        cruzamentos = 0.0

    return AtributosAcusticos(
        duracao_total_s=duracao_total,
        duracao_fala_s=duracao_fala,
        proporcao_pausa=max(0.0, proporcao_pausa),
        quantidade_segmentos=len(segmentos),
        duracao_media_frase_s=(duracao_fala / len(segmentos)) if segmentos else 0.0,
        pausa_media_s=float(np.mean(pausas)) if pausas else 0.0,
        pausa_maxima_s=float(np.max(pausas)) if pausas else 0.0,
        f0_media_hz=float(f0s_validas.mean()) if f0s_validas.size else 0.0,
        f0_desvio_hz=float(f0s_validas.std()) if f0s_validas.size else 0.0,
        jitter_f0=jitter,
        taxa_fala_silabas_s=taxa_fala,
        energia_media=float(energia_vozeada.mean()) if energia_vozeada.size else 0.0,
        queda_energia_relativa=queda,
        taxa_cruzamento_zero=cruzamentos,
        centroide_espectral_hz=_centroide_espectral(quadros_vozeados, taxa),
        segmentos=segmentos,
    )


# ---------------------------------------------------------------------------
# Regras de alteracao vocal
# ---------------------------------------------------------------------------


def _formatar_instante(segundos: float) -> str:
    minutos = int(segundos // 60)
    resto = segundos - minutos * 60
    return f"{minutos:02d}:{resto:05.2f}"


def avaliar_alteracoes_vocais(
    atributos: AtributosAcusticos, limiares: LimiaresAudio = LIMIARES_AUDIO
) -> list[Achado]:
    """Aplica as regras acusticas de fadiga e esforco respiratorio."""
    achados: list[Achado] = []

    fala_entrecortada = (
        atributos.proporcao_pausa >= limiares.proporcao_pausa_alta
        and atributos.duracao_media_frase_s <= limiares.duracao_frase_curta_s
    )
    if fala_entrecortada:
        # Frases curtas separadas por pausas longas e o padrao classico de
        # quem precisa interromper a fala para respirar.
        excesso = atributos.proporcao_pausa - limiares.proporcao_pausa_alta
        achados.append(
            Achado(
                modalidade="audio",
                tipo="fala_entrecortada",
                descricao=(
                    "Fala entrecortada com frases curtas e pausas longas, compativel "
                    "com esforco respiratorio"
                ),
                severidade="atencao",
                score=min(0.95, 0.60 + excesso * 1.5),
                instante=_formatar_instante(
                    atributos.segmentos[0].inicio_s if atributos.segmentos else 0.0
                ),
                evidencias={
                    "proporcao_pausa": round(atributos.proporcao_pausa, 3),
                    "duracao_media_frase_s": round(atributos.duracao_media_frase_s, 2),
                    "pausa_maxima_s": round(atributos.pausa_maxima_s, 2),
                },
            )
        )

    fadiga = (
        atributos.queda_energia_relativa >= limiares.queda_energia_relativa
        or atributos.taxa_fala_silabas_s <= limiares.taxa_fala_lenta
    )
    if fadiga:
        criterios = []
        if atributos.queda_energia_relativa >= limiares.queda_energia_relativa:
            criterios.append("queda de energia ao longo da consulta")
        if atributos.taxa_fala_silabas_s <= limiares.taxa_fala_lenta:
            criterios.append("taxa de fala reduzida")
        achados.append(
            Achado(
                modalidade="audio",
                tipo="fadiga_vocal",
                descricao="Sinais vocais de cansaco: " + " e ".join(criterios),
                severidade="atencao",
                score=0.55 + 0.2 * len(criterios),
                instante=_formatar_instante(atributos.duracao_total_s),
                evidencias={
                    "queda_energia_relativa": round(atributos.queda_energia_relativa, 3),
                    "taxa_fala_silabas_s": round(atributos.taxa_fala_silabas_s, 2),
                    "energia_media": round(atributos.energia_media, 4),
                },
            )
        )

    if atributos.jitter_f0 >= limiares.jitter_alto:
        achados.append(
            Achado(
                modalidade="audio",
                tipo="instabilidade_vocal",
                descricao=(
                    "Frequencia fundamental instavel, compativel com voz tremula ou "
                    "articulacao prejudicada"
                ),
                severidade="atencao",
                score=min(0.9, 0.5 + (atributos.jitter_f0 - limiares.jitter_alto) * 100),
                instante=None,
                evidencias={
                    "jitter_f0": round(atributos.jitter_f0, 4),
                    "f0_media_hz": round(atributos.f0_media_hz, 1),
                    "f0_desvio_hz": round(atributos.f0_desvio_hz, 1),
                },
            )
        )

    return achados


def avaliar_conteudo(
    analise: ResultadoAnaliseTexto, transcricao: ResultadoTranscricao
) -> list[Achado]:
    """Transforma termos criticos e sentimento da transcricao em achados."""
    achados: list[Achado] = []

    # Um termo critico pode aparecer varias vezes; agrupamos por categoria
    # para nao emitir cinco achados do mesmo problema.
    por_categoria: dict[str, list] = {}
    for termo in analise.termos_criticos:
        por_categoria.setdefault(termo.categoria, []).append(termo)

    for categoria, termos in por_categoria.items():
        peso = max(t.peso for t in termos)
        severidade = "critico" if peso >= 0.8 else "atencao" if peso >= 0.55 else "informativo"
        instante = _instante_do_termo(termos[0].termo, transcricao)
        achados.append(
            Achado(
                modalidade="audio",
                tipo=f"termo_critico_{categoria}",
                descricao=(
                    f"Queixa de categoria '{categoria}' relatada pelo paciente: "
                    + ", ".join(sorted({t.termo for t in termos}))
                ),
                severidade=severidade,
                score=peso,
                instante=instante,
                evidencias={
                    "trecho": termos[0].trecho,
                    "confianca_transcricao": round(transcricao.confianca, 2),
                    "provedor_transcricao": transcricao.provedor,
                },
            )
        )

    if analise.sentimento == "negativo" and analise.scores_sentimento.get("negativo", 0) >= 0.6:
        achados.append(
            Achado(
                modalidade="audio",
                tipo="sentimento_negativo",
                descricao="Sentimento predominante negativo na fala do paciente",
                severidade="informativo",
                score=float(analise.scores_sentimento["negativo"]),
                instante=None,
                evidencias={
                    "scores": {k: round(v, 3) for k, v in analise.scores_sentimento.items()},
                    "provedor": analise.provedor,
                },
            )
        )

    return achados


def _instante_do_termo(termo: str, transcricao: ResultadoTranscricao) -> str | None:
    """Localiza em qual segmento da transcricao o termo critico apareceu."""
    alvo = normalizar_texto(termo)
    for segmento in transcricao.segmentos:
        if alvo in normalizar_texto(segmento.texto):
            return _formatar_instante(segmento.inicio_s)
    return None


def analisar_consulta(
    caminho_audio: Path | str,
    paciente_id: str,
    provedor_azure: str = "auto",
    limiares: LimiaresAudio = LIMIARES_AUDIO,
) -> ResultadoAudio:
    """Pipeline completo da modalidade audio de uma consulta.

    Ordem: atributos acusticos da forma de onda -> transcricao (Azure Speech
    to Text ou substituto offline) -> sentimento e termos criticos (Azure Text
    Analytics ou substituto offline) -> achados.
    """
    caminho = Path(caminho_audio)
    atributos = extrair_atributos(caminho)
    transcricao = transcrever_audio(caminho, provedor=provedor_azure)

    if transcricao.texto and transcricao.confianca >= limiares.confianca_minima_transcricao:
        analise = analisar_texto(transcricao.texto, provedor=provedor_azure)
        achados_conteudo = avaliar_conteudo(analise, transcricao)
    else:
        # Sem transcricao confiavel nao aplicamos as regras de texto: seria
        # alertar a equipe com base em palavra que o reconhecedor pode ter
        # errado. Os atributos acusticos continuam valendo.
        analise = ResultadoAnaliseTexto(
            sentimento="indisponivel",
            scores_sentimento={},
            frases_chave=[],
            termos_criticos=[],
            provedor=transcricao.provedor,
        )
        achados_conteudo = []

    achados = avaliar_alteracoes_vocais(atributos, limiares) + achados_conteudo

    return ResultadoAudio(
        paciente_id=paciente_id,
        caminho_audio=str(caminho),
        atributos=atributos,
        transcricao=transcricao,
        analise_texto=analise,
        achados=achados,
    )


def relatorio_markdown(resultado: ResultadoAudio) -> str:
    """Relatorio legivel da consulta, para anexar ao prontuario."""
    atributos = resultado.atributos
    linhas = [
        f"# Relatório de análise de áudio - {resultado.paciente_id}",
        "",
        f"Arquivo: `{Path(resultado.caminho_audio).name}`  ",
        f"Duração: {atributos.duracao_total_s:.1f} s  ",
        f"Transcrição: provedor `{resultado.transcricao.provedor}`, "
        f"confiança média {resultado.transcricao.confianca:.2f}  ",
        f"Análise de texto: provedor `{resultado.analise_texto.provedor}`",
        "",
        "## Atributos acústicos",
        "",
        "| Atributo | Valor |",
        "| --- | ---: |",
        f"| Tempo efetivo de fala | {atributos.duracao_fala_s:.1f} s |",
        f"| Proporção de pausa | {atributos.proporcao_pausa:.2f} |",
        f"| Frases detectadas | {atributos.quantidade_segmentos} |",
        f"| Duração média da frase | {atributos.duracao_media_frase_s:.2f} s |",
        f"| Pausa máxima | {atributos.pausa_maxima_s:.2f} s |",
        f"| F0 média | {atributos.f0_media_hz:.1f} Hz |",
        f"| Jitter da F0 | {atributos.jitter_f0:.4f} |",
        f"| Taxa de fala | {atributos.taxa_fala_silabas_s:.2f} sílabas/s |",
        f"| Queda de energia | {atributos.queda_energia_relativa:.2f} |",
        "",
        "## Transcrição",
        "",
    ]

    for segmento in resultado.transcricao.segmentos:
        linhas.append(
            f"- `{_formatar_instante(segmento.inicio_s)}` {segmento.texto}"
        )

    linhas += [
        "",
        "## Sentimento e termos críticos",
        "",
        f"Sentimento: **{resultado.analise_texto.sentimento}**",
        "Frases-chave: " + (", ".join(resultado.analise_texto.frases_chave) or "nenhuma"),
        "",
    ]
    if resultado.analise_texto.termos_criticos:
        linhas += ["| Termo | Categoria | Peso |", "| --- | --- | ---: |"]
        for termo in resultado.analise_texto.termos_criticos:
            linhas.append(f"| {termo.termo} | {termo.categoria} | {termo.peso:.2f} |")
    else:
        linhas.append("Nenhum termo crítico do léxico institucional foi encontrado.")

    linhas += ["", "## Achados", ""]
    if resultado.achados:
        for achado in resultado.achados:
            instante = f" (`{achado.instante}`)" if achado.instante else ""
            linhas.append(
                f"- **{achado.severidade}**{instante} {achado.descricao} "
                f"(confiança {achado.score:.2f})"
            )
    else:
        linhas.append("Nenhuma alteração vocal ou queixa crítica detectada.")

    linhas += [
        "",
        "> Relatório gerado automaticamente por um protótipo acadêmico. Não "
        "substitui avaliação clínica.",
        "",
    ]
    return "\n".join(linhas)
