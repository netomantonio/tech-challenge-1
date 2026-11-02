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

Ha ainda um terceiro provedor, `local`, para a transcricao: roda o
faster-whisper na propria maquina. Ele existe para demonstrar Speech to Text
em fala real quando nao ha credencial da Azure, e precisa ser pedido
explicitamente (`auto` nunca o escolhe). A analise de texto no modo `local`
usa o lexico, como no `offline`.
"""

from __future__ import annotations

import json
import math
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
    DIR_MODELOS,
    ENV_LANGUAGE_ENDPOINT,
    ENV_LANGUAGE_KEY,
    ENV_SPEECH_KEY,
    ENV_SPEECH_REGION,
    ENV_WHISPER_MODELO,
    IDIOMA_PADRAO,
    MODELO_WHISPER_PADRAO,
    azure_language_configurada,
    azure_speech_configurada,
)

PROVEDORES = ("auto", "azure", "offline", "local")


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
    # Entidades clinicas reconhecidas pela Azure (Text Analytics for health ou
    # NER generico), cada uma como {"texto", "categoria", "confianca"}. Vazio
    # no provedor offline.
    entidades: list[dict[str, Any]] = field(default_factory=list)
    # Em quantos documentos o texto foi dividido para respeitar o limite por
    # documento do servico (1 no offline).
    documentos_enviados: int = 1

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
            "entidades": self.entidades,
            "documentos_enviados": self.documentos_enviados,
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


def _sdk_whisper_disponivel() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except Exception:
        return False
    return True


def nome_modelo_whisper() -> str:
    return os.getenv(ENV_WHISPER_MODELO, MODELO_WHISPER_PADRAO)


TAXA_WHISPER = 16_000


def _carregar_wav_para_whisper(caminho_wav: Path) -> "np.ndarray":
    """Le o WAV PCM como float32 mono a 16 kHz, formato que o whisper consome.

    Passar o vetor pronto evita o decodificador de midia do faster-whisper
    (PyAV), cuja versao instalada pode ser incompativel; para WAV nao ha
    motivo para depender dele.
    """
    import numpy as np

    with wave.open(str(caminho_wav), "rb") as wav:
        canais, largura, taxa, quadros = (
            wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
        )
        bruto = wav.readframes(quadros)
    if largura == 2:
        sinal = np.frombuffer(bruto, dtype="<i2").astype(np.float32) / 32768.0
    elif largura == 4:
        sinal = np.frombuffer(bruto, dtype="<i4").astype(np.float32) / 2147483648.0
    elif largura == 1:
        sinal = (np.frombuffer(bruto, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ServicoAzureIndisponivelError(f"WAV com {largura * 8} bits nao suportado pelo provedor local.")
    if canais > 1:
        sinal = sinal.reshape(-1, canais).mean(axis=1)
    if taxa != TAXA_WHISPER and sinal.size:
        duracao = sinal.size / taxa
        novos = int(round(duracao * TAXA_WHISPER))
        origem = np.linspace(0.0, duracao, num=sinal.size, endpoint=False)
        destino = np.linspace(0.0, duracao, num=novos, endpoint=False)
        sinal = np.interp(destino, origem, sinal).astype(np.float32)
    return sinal.astype(np.float32)


def _transcrever_local(caminho_wav: Path, idioma: str) -> ResultadoTranscricao:
    """Speech to Text local com faster-whisper (CPU, int8).

    A confianca de cada segmento e `exp(avg_logprob)`, a probabilidade media
    por token que o modelo atribuiu ao proprio texto. Nao e comparavel ao
    `Confidence` da Azure, mas serve ao mesmo filtro de confianca minima.
    """
    from faster_whisper import WhisperModel

    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    modelo = WhisperModel(
        nome_modelo_whisper(), device="cpu", compute_type="int8", download_root=str(DIR_MODELOS)
    )
    segmentos_brutos, _info = modelo.transcribe(
        _carregar_wav_para_whisper(caminho_wav),
        language=idioma.split("-")[0], beam_size=5, vad_filter=True,
    )
    segmentos = [
        SegmentoTranscricao(
            inicio_s=float(s.start),
            fim_s=float(s.end),
            texto=s.text.strip(),
            confianca=max(0.0, min(1.0, math.exp(float(s.avg_logprob)))),
        )
        for s in segmentos_brutos
        if s.text.strip()
    ]
    texto = " ".join(s.texto for s in segmentos).strip()
    if not texto:
        raise ServicoAzureIndisponivelError(
            "O provedor local (faster-whisper) nao reconheceu fala no arquivo informado."
        )
    duracoes = [max(s.fim_s - s.inicio_s, 1e-3) for s in segmentos]
    confianca = sum(s.confianca * d for s, d in zip(segmentos, duracoes)) / sum(duracoes)
    return ResultadoTranscricao(
        texto=texto,
        confianca=confianca,
        duracao_s=duracao_wav(caminho_wav),
        idioma=idioma,
        provedor="local",
        segmentos=segmentos,
    )


def transcrever_audio(
    caminho_wav: Path | str,
    provedor: str = "auto",
    idioma: str = IDIOMA_PADRAO,
) -> ResultadoTranscricao:
    """Transcreve um WAV com Azure Speech to Text, com o whisper local ou offline."""
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

    if provedor == "local":
        if not _sdk_whisper_disponivel():
            raise ServicoAzureIndisponivelError(
                "O provedor local exige faster-whisper instalado "
                "(pip install -r requirements-fase4.txt)."
            )
        return _transcrever_local(caminho, idioma)

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


# Limite de caracteres por documento aceito pelo Azure AI Language (5.120) e
# de documentos por requisicao nas operacoes de sentimento e frases-chave.
LIMITE_CARACTERES_DOCUMENTO = 5_000
LIMITE_DOCUMENTOS_POR_CHAMADA = 10

# Categorias do Text Analytics for health que interessam ao monitoramento.
CATEGORIAS_SAUDE_RELEVANTES = {
    "SymptomOrSign": "sintoma",
    "Diagnosis": "diagnostico",
    "MedicationName": "medicamento",
    "BodyStructure": "estrutura_corporal",
    "ExaminationName": "exame",
    "TreatmentName": "tratamento",
}
# Confianca minima para uma entidade fora do lexico virar termo informativo.
CONFIANCA_MINIMA_ENTIDADE = 0.8
CATEGORIA_SINTOMA_AZURE = "sintoma_relatado_azure"
PESO_SINTOMA_AZURE = 0.45


def dividir_em_documentos(texto: str, limite: int = LIMITE_CARACTERES_DOCUMENTO) -> list[str]:
    """Divide o texto em documentos de ate `limite` caracteres, em fim de frase.

    Uma consulta de 20 minutos transcrita passa com folga dos 5.120
    caracteres que o servico aceita por documento; sem dividir, a chamada
    inteira falha. A quebra procura o fim de frase mais proximo do limite e,
    se nao houver, quebra em espaco.
    """
    texto = texto.strip()
    if not texto:
        return []
    documentos: list[str] = []
    while len(texto) > limite:
        corte = -1
        for pontuacao in (". ", "! ", "? ", "; ", ", "):
            corte = max(corte, texto.rfind(pontuacao, 0, limite))
        if corte > 0:
            corte += 1  # mantem a pontuacao no documento anterior
        else:
            corte = texto.rfind(" ", 0, limite)
        if corte <= 0:
            corte = limite
        documentos.append(texto[:corte].strip())
        texto = texto[corte:].strip()
    if texto:
        documentos.append(texto)
    return documentos


def _lotes(itens: list[str], tamanho: int) -> list[list[str]]:
    return [itens[i:i + tamanho] for i in range(0, len(itens), tamanho)]


def _entidades_azure(cliente: Any, documentos: list[str], codigo_idioma: str) -> list[dict[str, Any]]:
    """Entidades clinicas: Text Analytics for health, com NER generico de reserva.

    O recurso de saude e opcional no SDK/recurso; quando indisponivel caimos
    para `recognize_entities`, que ainda marca categorias uteis (Quantity,
    DateTime, Person) mas nao sintomas. Falha aqui nao derruba a analise: as
    entidades enriquecem o relatorio, quem decide alerta e o lexico.
    """
    entidades: list[dict[str, Any]] = []
    try:
        poller = cliente.begin_analyze_healthcare_entities(documentos, language=codigo_idioma)
        for resultado in poller.result():
            if getattr(resultado, "is_error", False):
                continue
            for entidade in resultado.entities:
                entidades.append(
                    {
                        "texto": str(entidade.text),
                        "categoria": str(entidade.category),
                        "confianca": float(getattr(entidade, "confidence_score", 0.0) or 0.0),
                        "origem": "text_analytics_for_health",
                    }
                )
        return entidades
    except Exception:  # noqa: BLE001 - recurso opcional; tentamos o NER generico
        pass

    try:
        for lote in _lotes(documentos, LIMITE_DOCUMENTOS_POR_CHAMADA):
            for resultado in cliente.recognize_entities(lote, language=codigo_idioma):
                if getattr(resultado, "is_error", False):
                    continue
                for entidade in resultado.entities:
                    entidades.append(
                        {
                            "texto": str(entidade.text),
                            "categoria": str(entidade.category),
                            "confianca": float(getattr(entidade, "confidence_score", 0.0) or 0.0),
                            "origem": "ner_generico",
                        }
                    )
    except Exception:  # noqa: BLE001
        return []
    return entidades


def _termos_a_partir_de_entidades(
    entidades: list[dict[str, Any]], texto: str, ja_encontrados: list[TermoCritico]
) -> list[TermoCritico]:
    """Cruza as entidades da Azure com o lexico institucional.

    Entidade que casa com um termo do lexico herda a categoria e o peso dele
    (e so e acrescentada se o lexico ainda nao a tinha achado). Sintoma ou
    diagnostico reconhecido pela Azure com confianca alta, mas fora do
    lexico, entra como termo informativo: a Azure identificou, o hospital
    ainda nao decidiu a prioridade.
    """
    lexico = carregar_lexico()
    normalizado = normalizar_texto(texto)
    vistos = {(normalizar_texto(t.termo), t.categoria) for t in ja_encontrados}
    novos: list[TermoCritico] = []

    for entidade in entidades:
        alvo = normalizar_texto(entidade["texto"]).strip()
        if not alvo:
            continue
        casou = False
        for categoria, dados in lexico.get("categorias", {}).items():
            for termo in dados.get("termos", []):
                termo_norm = normalizar_texto(termo)
                if alvo == termo_norm or alvo in termo_norm or termo_norm in alvo:
                    casou = True
                    chave = (termo_norm, categoria)
                    if chave in vistos:
                        continue
                    vistos.add(chave)
                    novos.append(
                        TermoCritico(
                            termo=termo,
                            categoria=categoria,
                            peso=float(dados.get("peso", 0.5)),
                            trecho=_trecho(texto, normalizado, termo_norm) or entidade["texto"],
                        )
                    )
        if casou:
            continue
        categoria_azure = entidade.get("categoria", "")
        if (
            categoria_azure in ("SymptomOrSign", "Diagnosis")
            and entidade.get("confianca", 0.0) >= CONFIANCA_MINIMA_ENTIDADE
            and (alvo, CATEGORIA_SINTOMA_AZURE) not in vistos
        ):
            vistos.add((alvo, CATEGORIA_SINTOMA_AZURE))
            novos.append(
                TermoCritico(
                    termo=entidade["texto"],
                    categoria=CATEGORIA_SINTOMA_AZURE,
                    peso=PESO_SINTOMA_AZURE,
                    trecho=_trecho(texto, normalizado, alvo) or entidade["texto"],
                )
            )
    return novos


def _trecho(texto: str, normalizado: str, alvo: str) -> str | None:
    posicao = normalizado.find(alvo)
    if posicao < 0:
        return None
    inicio = max(0, posicao - 35)
    fim = min(len(texto), posicao + len(alvo) + 35)
    return texto[inicio:fim].strip()


def _analisar_texto_azure(texto: str, idioma: str) -> ResultadoAnaliseTexto:
    """Sentimento, frases-chave e entidades pelo Azure Text Analytics (Azure AI Language).

    O texto e dividido em documentos de ate 5.000 caracteres e enviado em
    lotes de 10. O sentimento final e a media dos scores ponderada pelo
    tamanho de cada documento; as frases-chave sao unidas sem repeticao.
    """
    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential

    cliente = TextAnalyticsClient(
        endpoint=os.environ[ENV_LANGUAGE_ENDPOINT],
        credential=AzureKeyCredential(os.environ[ENV_LANGUAGE_KEY]),
    )
    codigo_idioma = idioma.split("-")[0]
    documentos = dividir_em_documentos(texto) or [texto]

    sentimentos: list[Any] = []
    frases_por_doc: list[Any] = []
    for lote in _lotes(documentos, LIMITE_DOCUMENTOS_POR_CHAMADA):
        sentimentos.extend(cliente.analyze_sentiment(lote, language=codigo_idioma))
        frases_por_doc.extend(cliente.extract_key_phrases(lote, language=codigo_idioma))

    if any(getattr(s, "is_error", False) for s in sentimentos) or any(
        getattr(f, "is_error", False) for f in frases_por_doc
    ):
        raise ServicoAzureIndisponivelError("Azure Text Analytics nao concluiu a analise do texto.")

    pesos = [max(len(d), 1) for d in documentos]
    total = float(sum(pesos))
    scores = {"positivo": 0.0, "neutro": 0.0, "negativo": 0.0}
    for sentimento, peso in zip(sentimentos, pesos):
        scores["positivo"] += float(sentimento.confidence_scores.positive) * peso / total
        scores["neutro"] += float(sentimento.confidence_scores.neutral) * peso / total
        scores["negativo"] += float(sentimento.confidence_scores.negative) * peso / total

    traducao = {"positive": "positivo", "neutral": "neutro", "negative": "negativo", "mixed": "misto"}
    if len(sentimentos) == 1:
        rotulo = traducao.get(sentimentos[0].sentiment, sentimentos[0].sentiment)
    else:
        rotulo = max(scores, key=scores.get)

    frases_chave: list[str] = []
    for resultado in frases_por_doc:
        for frase in resultado.key_phrases:
            if frase not in frases_chave:
                frases_chave.append(frase)

    # As frases-chave da Azure nao dizem o que e clinicamente critico, so o
    # que e saliente no texto. Por isso o lexico institucional continua
    # valendo nos dois provedores: ele define o que vira alerta. As entidades
    # clinicas da Azure ampliam a busca: o que ela reconhece e cruzado com o
    # lexico, e sintomas fora dele entram como informativos.
    termos = _buscar_termos_criticos(texto)
    entidades = _entidades_azure(cliente, documentos, codigo_idioma)
    termos += _termos_a_partir_de_entidades(entidades, texto, termos)
    termos.sort(key=lambda t: t.peso, reverse=True)

    return ResultadoAnaliseTexto(
        sentimento=rotulo,
        scores_sentimento=scores,
        frases_chave=frases_chave,
        termos_criticos=termos,
        provedor="azure",
        entidades=entidades,
        documentos_enviados=len(documentos),
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
    """Extrai sentimento, frases-chave e termos criticos de uma transcricao.

    O provedor `local` so existe para a transcricao; aqui ele equivale ao
    `offline` (lexico institucional), porque nao ha modelo de linguagem local
    no projeto.
    """
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
        "speech_to_text_local": {
            "credencial": True,
            "sdk": _sdk_whisper_disponivel(),
            "provedor_efetivo": "local" if _sdk_whisper_disponivel() else "indisponivel",
            "modelo": nome_modelo_whisper(),
        },
    }
