"""Integracao com os servicos gerenciados da Azure usados na Fase 4.

O enunciado pede Azure Speech to Text para transcrever as consultas e Azure
Text Analytics para sentimento e termos criticos. Como o repositorio precisa
continuar rodando no CI e nas maquinas da equipe sem chave paga, cada
servico tem dois provedores:

* `azure`: chama o servico real; exige as variaveis de ambiente
  `AZURE_SPEECH_KEY`/`AZURE_SPEECH_REGION` e
  `AZURE_LANGUAGE_KEY`/`AZURE_LANGUAGE_ENDPOINT` e os SDKs opcionais de
  `requirements-fase4.txt`;
* `offline`: substituto deterministico, no mesmo espirito do `FakeLLM` da
  Fase 3. A transcricao vem do arquivo `.txt` de referencia gravado junto do
  audio sintetico e o sentimento sai de um lexico em
  `fase4/data/termos_criticos_audio.json`.

O provedor `auto` (padrao) usa a Azure quando ha credencial e SDK, e cai
para o modo offline caso contrario. O resto do pipeline nao muda: os dois
provedores devolvem exatamente as mesmas estruturas.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
import wave
from threading import Event
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fase4.config import (
    CAMINHO_TERMOS_CRITICOS,
    ENV_LANGUAGE_ENDPOINT,
    ENV_LANGUAGE_KEY,
    ENV_SPEECH_KEY,
    ENV_SPEECH_REGION,
    IDIOMA_PADRAO,
    azure_language_configurada,
    azure_speech_configurada,
)

PROVEDORES = ("auto", "azure", "offline")


class ServicoAzureIndisponivelError(RuntimeError):
    """Pedimos explicitamente o provedor `azure`, mas ele nao pode ser usado."""


@dataclass
class SegmentoTranscricao:
    inicio_s: float
    fim_s: float
    texto: str
    confianca: float


@dataclass
class ResultadoTranscricao:
    texto: str
    confianca: float
    duracao_s: float
    idioma: str
    provedor: str
    segmentos: list[SegmentoTranscricao] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "texto": self.texto,
            "confianca": round(self.confianca, 4),
            "duracao_s": round(self.duracao_s, 3),
            "idioma": self.idioma,
            "provedor": self.provedor,
            "segmentos": [
                {
                    "inicio_s": round(s.inicio_s, 3),
                    "fim_s": round(s.fim_s, 3),
                    "texto": s.texto,
                    "confianca": round(s.confianca, 4),
                }
                for s in self.segmentos
            ],
        }


@dataclass
class TermoCritico:
    termo: str
    categoria: str
    peso: float
    trecho: str


@dataclass
class ResultadoAnaliseTexto:
    sentimento: str
    scores_sentimento: dict[str, float]
    frases_chave: list[str]
    termos_criticos: list[TermoCritico]
    provedor: str

    def to_dict(self) -> dict:
        return {
            "sentimento": self.sentimento,
            "scores_sentimento": {k: round(v, 4) for k, v in self.scores_sentimento.items()},
            "frases_chave": self.frases_chave,
            "termos_criticos": [
                {
                    "termo": t.termo,
                    "categoria": t.categoria,
                    "peso": t.peso,
                    "trecho": t.trecho,
                }
                for t in self.termos_criticos
            ],
            "provedor": self.provedor,
        }

    @property
    def peso_maximo(self) -> float:
        return max((t.peso for t in self.termos_criticos), default=0.0)

    def categorias(self) -> list[str]:
        vistas: list[str] = []
        for termo in self.termos_criticos:
            if termo.categoria not in vistas:
                vistas.append(termo.categoria)
        return vistas


def normalizar_texto(texto: str) -> str:
    """Remove acentos e baixa a caixa, para casar o lexico com a transcricao."""
    sem_acento = unicodedata.normalize("NFKD", texto)
    sem_acento = "".join(c for c in sem_acento if not unicodedata.combining(c))
    return sem_acento.lower()


def carregar_lexico(caminho: Path | None = None) -> dict[str, Any]:
    caminho = caminho or CAMINHO_TERMOS_CRITICOS
    with Path(caminho).open(encoding="utf-8") as f:
        return json.load(f)


def duracao_wav(caminho: Path | str) -> float:
    with wave.open(str(caminho), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate())


# ---------------------------------------------------------------------------
# Speech to Text
# ---------------------------------------------------------------------------


def _sdk_speech_disponivel() -> bool:
    try:
        import azure.cognitiveservices.speech  # noqa: F401
    except Exception:
        return False
    return True


def _transcrever_azure(caminho_wav: Path, idioma: str) -> ResultadoTranscricao:
    """Reconhecimento continuo do arquivo inteiro via Azure Speech to Text."""
    import azure.cognitiveservices.speech as speechsdk

    config = speechsdk.SpeechConfig(
        subscription=os.environ[ENV_SPEECH_KEY], region=os.environ[ENV_SPEECH_REGION]
    )
    config.speech_recognition_language = idioma
    config.output_format = speechsdk.OutputFormat.Detailed
    audio_config = speechsdk.audio.AudioConfig(filename=str(caminho_wav))
    reconhecedor = speechsdk.SpeechRecognizer(speech_config=config, audio_config=audio_config)

    segmentos: list[SegmentoTranscricao] = []
    terminou = Event()
    falhas: list[str] = []

    def _ao_reconhecer(evento: Any) -> None:
        resultado = evento.result
        if resultado.reason != speechsdk.ResultReason.RecognizedSpeech:
            return
        inicio = resultado.offset / 1e7
        fim = inicio + resultado.duration / 1e7
        segmentos.append(
            SegmentoTranscricao(
                inicio_s=inicio,
                fim_s=fim,
                texto=resultado.text,
                confianca=_confianca_detalhada(resultado),
            )
        )

    reconhecedor.recognized.connect(_ao_reconhecer)
    def _ao_cancelar(evento: Any) -> None:
        if evento.cancellation_details.reason == speechsdk.CancellationReason.Error:
            falhas.append("Azure Speech cancelou o reconhecimento por erro. Verifique o recurso e as credenciais.")
        terminou.set()

    reconhecedor.session_stopped.connect(lambda _: terminou.set())
    reconhecedor.canceled.connect(_ao_cancelar)

    reconhecedor.start_continuous_recognition()
    try:
        if not terminou.wait(timeout=max(60.0, duracao_wav(caminho_wav) * 2 + 30)):
            raise ServicoAzureIndisponivelError("Tempo limite excedido no Azure Speech.")
    finally:
        reconhecedor.stop_continuous_recognition()
    if falhas:
        raise ServicoAzureIndisponivelError(falhas[0])

    texto = " ".join(s.texto for s in segmentos).strip()
    if not texto:
        raise ServicoAzureIndisponivelError("Azure Speech nao reconheceu fala no arquivo informado.")
    confianca = (
        sum(s.confianca for s in segmentos) / len(segmentos) if segmentos else 0.0
    )
    return ResultadoTranscricao(
        texto=texto,
        confianca=confianca,
        duracao_s=duracao_wav(caminho_wav),
        idioma=idioma,
        provedor="azure",
        segmentos=segmentos,
    )


def _confianca_detalhada(resultado: Any) -> float:
    """Le a confianca do melhor n-best do JSON detalhado do Speech."""
    try:
        detalhes = json.loads(resultado.json)
        melhores = detalhes.get("NBest") or []
        if melhores:
            return float(melhores[0].get("Confidence", 0.0))
    except Exception:
        pass
    return 0.0


def _transcrever_offline(caminho_wav: Path, idioma: str) -> ResultadoTranscricao:
    """Le a transcricao de referencia gravada junto do audio sintetico.

    O gerador de dados salva, ao lado de `consulta.wav`, um
    `consulta.transcricao.json` com as falas e seus instantes. Assim o modo
    offline reproduz o formato por segmentos que a Azure devolveria, sem
    inventar um texto que o audio nao contem.
    """
    referencia = caminho_wav.with_suffix(".transcricao.json")
    duracao = duracao_wav(caminho_wav) if caminho_wav.exists() else 0.0

    if not referencia.exists():
        return ResultadoTranscricao(
            texto="",
            confianca=0.0,
            duracao_s=duracao,
            idioma=idioma,
            provedor="offline",
            segmentos=[],
        )

    with referencia.open(encoding="utf-8") as f:
        dados = json.load(f)

    segmentos = [
        SegmentoTranscricao(
            inicio_s=float(item["inicio_s"]),
            fim_s=float(item["fim_s"]),
            texto=item["texto"],
            confianca=float(item.get("confianca", 0.8)),
        )
        for item in dados.get("segmentos", [])
    ]
    texto = " ".join(s.texto for s in segmentos).strip()
    confianca = sum(s.confianca for s in segmentos) / len(segmentos) if segmentos else 0.0
    return ResultadoTranscricao(
        texto=texto,
        confianca=confianca,
        duracao_s=duracao or float(dados.get("duracao_s", 0.0)),
        idioma=dados.get("idioma", idioma),
        provedor="offline",
        segmentos=segmentos,
    )


def transcrever_audio(
    caminho_wav: Path | str,
    provedor: str = "auto",
    idioma: str = IDIOMA_PADRAO,
) -> ResultadoTranscricao:
    """Transcreve um WAV com Azure Speech to Text ou com o substituto offline."""
    if provedor not in PROVEDORES:
        raise ValueError(f"provedor invalido: {provedor}")

    caminho = Path(caminho_wav)
    pode_azure = azure_speech_configurada() and _sdk_speech_disponivel()

    if provedor == "azure":
        if not pode_azure:
            raise ServicoAzureIndisponivelError(
                "Azure Speech to Text exige azure-cognitiveservices-speech instalado e "
                f"as variaveis {ENV_SPEECH_KEY} e {ENV_SPEECH_REGION} definidas."
            )
        return _transcrever_azure(caminho, idioma)

    if provedor == "auto" and pode_azure:
        return _transcrever_azure(caminho, idioma)

    return _transcrever_offline(caminho, idioma)


# ---------------------------------------------------------------------------
# Text Analytics
# ---------------------------------------------------------------------------


def _sdk_language_disponivel() -> bool:
    try:
        import azure.ai.textanalytics  # noqa: F401
    except Exception:
        return False
    return True


def _analisar_texto_azure(texto: str, idioma: str) -> ResultadoAnaliseTexto:
    """Sentimento e frases-chave pelo Azure Text Analytics (Azure AI Language)."""
    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential

    cliente = TextAnalyticsClient(
        endpoint=os.environ[ENV_LANGUAGE_ENDPOINT],
        credential=AzureKeyCredential(os.environ[ENV_LANGUAGE_KEY]),
    )
    codigo_idioma = idioma.split("-")[0]

    sentimento = cliente.analyze_sentiment([texto], language=codigo_idioma)[0]
    frases = cliente.extract_key_phrases([texto], language=codigo_idioma)[0]

    if sentimento.is_error or frases.is_error:
        raise ServicoAzureIndisponivelError("Azure Text Analytics nao concluiu a analise do texto.")

    scores = {
        "positivo": float(sentimento.confidence_scores.positive),
        "neutro": float(sentimento.confidence_scores.neutral),
        "negativo": float(sentimento.confidence_scores.negative),
    }
    traducao = {"positive": "positivo", "neutral": "neutro", "negative": "negativo", "mixed": "misto"}

    # As frases-chave da Azure nao dizem o que e clinicamente critico, so o
    # que e saliente no texto. Por isso o lexico institucional continua
    # valendo nos dois provedores: ele define o que vira alerta.
    frases_chave = list(frases.key_phrases) if not frases.is_error else []
    termos = _buscar_termos_criticos(texto)

    return ResultadoAnaliseTexto(
        sentimento=traducao.get(sentimento.sentiment, sentimento.sentiment),
        scores_sentimento=scores,
        frases_chave=frases_chave,
        termos_criticos=termos,
        provedor="azure",
    )


def _buscar_termos_criticos(texto: str) -> list[TermoCritico]:
    """Casa o lexico clinico com a transcricao e devolve os trechos achados."""
    lexico = carregar_lexico()
    normalizado = normalizar_texto(texto)
    encontrados: list[TermoCritico] = []

    for categoria, dados in lexico.get("categorias", {}).items():
        peso = float(dados.get("peso", 0.5))
        for termo in dados.get("termos", []):
            alvo = normalizar_texto(termo)
            posicao = normalizado.find(alvo)
            if posicao < 0:
                continue
            inicio = max(0, posicao - 35)
            fim = min(len(texto), posicao + len(alvo) + 35)
            encontrados.append(
                TermoCritico(
                    termo=termo,
                    categoria=categoria,
                    peso=peso,
                    trecho=texto[inicio:fim].strip(),
                )
            )
    encontrados.sort(key=lambda t: t.peso, reverse=True)
    return encontrados


def _analisar_texto_offline(texto: str) -> ResultadoAnaliseTexto:
    """Sentimento por lexico, usado quando o Text Analytics nao esta disponivel.

    A pontuacao satura os pesos dos termos negativos e positivos encontrados,
    soma uma massa neutra fixa e normaliza. A massa neutra existe para o
    rotulo nao ficar em 100% negativo so porque o lexico casou com uma
    palavra: isso daria uma falsa impressao de precisao. E um substituto
    simples, mas suficiente para o pipeline, porque quem decide o alerta sao
    os termos criticos, nao o rotulo de sentimento.
    """
    lexico = carregar_lexico()
    normalizado = normalizar_texto(texto)

    termos = _buscar_termos_criticos(texto)
    peso_negativo = min(1.0, sum(t.peso for t in termos) / 2.0)
    peso_positivo = min(
        1.0,
        sum(
            0.4
            for palavra in lexico.get("termos_positivos", [])
            if normalizar_texto(palavra) in normalizado
        )
        / 1.2,
    )
    massa_neutra = 0.5

    soma = peso_negativo + peso_positivo + massa_neutra
    scores = {
        "positivo": peso_positivo / soma,
        "neutro": massa_neutra / soma,
        "negativo": peso_negativo / soma,
    }
    sentimento = max(scores, key=scores.get)

    return ResultadoAnaliseTexto(
        sentimento=sentimento,
        scores_sentimento=scores,
        frases_chave=_frases_chave_offline(texto),
        termos_criticos=termos,
        provedor="offline",
    )


def _frases_chave_offline(texto: str, maximo: int = 8) -> list[str]:
    """Aproximacao de frases-chave: substantivos longos sem palavras vazias."""
    vazias = {
        "que", "para", "como", "pelo", "pela", "uma", "uns", "umas", "dos", "das",
        "mas", "com", "sem", "mais", "menos", "muito", "pouco", "tenho", "estou",
        "esta", "este", "isso", "aqui", "quando", "porque", "entao", "voce", "doutora",
        "doutor", "senhora", "senhor", "agora", "ontem", "hoje", "fazer", "ficar",
    }
    palavras = re.findall(r"[A-Za-zÀ-ÿ]{4,}", texto)
    vistas: list[str] = []
    for palavra in palavras:
        if normalizar_texto(palavra) in vazias:
            continue
        if palavra.lower() not in [v.lower() for v in vistas]:
            vistas.append(palavra.lower())
        if len(vistas) >= maximo:
            break
    return vistas


def analisar_texto(
    texto: str,
    provedor: str = "auto",
    idioma: str = IDIOMA_PADRAO,
) -> ResultadoAnaliseTexto:
    """Extrai sentimento, frases-chave e termos criticos de uma transcricao."""
    if provedor not in PROVEDORES:
        raise ValueError(f"provedor invalido: {provedor}")

    pode_azure = azure_language_configurada() and _sdk_language_disponivel()

    if provedor == "azure":
        if not pode_azure:
            raise ServicoAzureIndisponivelError(
                "Azure Text Analytics exige azure-ai-textanalytics instalado e as "
                f"variaveis {ENV_LANGUAGE_KEY} e {ENV_LANGUAGE_ENDPOINT} definidas."
            )
        return _analisar_texto_azure(texto, idioma)

    if provedor == "auto" and pode_azure:
        return _analisar_texto_azure(texto, idioma)

    return _analisar_texto_offline(texto)


def status_servicos() -> dict[str, Any]:
    """Resumo usado pela CLI e pelo relatorio para mostrar o que rodou na nuvem."""
    return {
        "speech_to_text": {
            "credencial": azure_speech_configurada(),
            "sdk": _sdk_speech_disponivel(),
            "provedor_efetivo": "azure"
            if azure_speech_configurada() and _sdk_speech_disponivel()
            else "offline",
        },
        "text_analytics": {
            "credencial": azure_language_configurada(),
            "sdk": _sdk_language_disponivel(),
            "provedor_efetivo": "azure"
            if azure_language_configurada() and _sdk_language_disponivel()
            else "offline",
        },
    }
