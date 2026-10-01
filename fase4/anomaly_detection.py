"""Deteccao de anomalias em sinais vitais, prescricoes e movimentacao (Fase 4).

O enunciado pede deteccao de anomalias em tres frentes: series temporais de
sinais vitais, evolucao das prescricoes e padroes de movimentacao durante a
internacao. Cada frente tem natureza diferente, entao usamos tecnicas
diferentes em vez de forcar um unico modelo.

Para os sinais vitais ha tres detectores complementares, porque cada um pega
um tipo de anomalia que os outros deixam passar:

1. **z-score robusto** (mediana e MAD moveis) pega mudanca abrupta. Nao pega
   deterioracao lenta: quando a piora entra na propria janela de referencia,
   a mediana acompanha o paciente e o desvio desaparece;
2. **desvio de tendencia** compara a mediana recente com a mediana das
   primeiras horas de internacao, que servem de referencia. E o detector que
   percebe a piora gradual, antes de qualquer limite ser cruzado;
3. **limite clinico** usa as faixas de `config.py`. E o mais simples e o
   ultimo a disparar, mas e o que a equipe reconhece de imediato.

Os tres exigem pontos consecutivos antes de confirmar. Essa regra e o que
separa deterioracao de artefato de sensor: um pico isolado de frequencia
cardiaca gera apenas um registro informativo, nao um alerta.

Alem disso, um **Isolation Forest** treinado nas primeiras horas de
monitoramento avalia os seis canais em conjunto. Ele captura combinacoes
anomalas (taquicardia junto com hipotensao e queda de saturacao) que cada
canal isolado ainda nao denunciaria.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from fase4.alertas import Achado
from fase4.config import LIMIARES_ANOMALIA, LimiaresAnomalia

# Deslocamento minimo, por canal, para a mediana recente ser considerada
# diferente da referencia. Abaixo disso o desvio e variacao fisiologica
# normal (inclusive o ritmo circadiano) e nao vale alertar.
DESLOCAMENTO_MINIMO = {
    "frequencia_cardiaca": 10.0,
    "pressao_sistolica": 12.0,
    "pressao_diastolica": 8.0,
    "saturacao_oxigenio": 2.0,
    "temperatura": 0.4,
    "frequencia_respiratoria": 3.0,
}

# Classes em que duas medicacoes simultaneas caracterizam duplicidade.
CLASSES_SEM_DUPLICIDADE = {
    "anti_inflamatorio_nao_esteroidal",
    "opioide",
    "anticoagulante",
}

MINUTOS_REFERENCIA = 120
MINUTOS_TENDENCIA = 30


# ---------------------------------------------------------------------------
# Carga dos dados
# ---------------------------------------------------------------------------


def carregar_sinais_vitais(caminho_csv: Path | str) -> pd.DataFrame:
    df = pd.read_csv(caminho_csv, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def carregar_movimentacao(caminho_csv: Path | str) -> pd.DataFrame:
    df = pd.read_csv(caminho_csv, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Detector online de sinais vitais
# ---------------------------------------------------------------------------


@dataclass
class EstadoRegra:
    """Contadores de uma regra num canal: confirmacao e histerese.

    `consecutivos` conta quantas leituras seguidas satisfizeram a condicao;
    `disparado` evita repetir o mesmo achado a cada minuto enquanto o problema
    persiste; `recuperacao` conta quantas leituras seguidas voltaram ao normal
    e e o que libera a regra para disparar de novo.
    """

    consecutivos: int = 0
    disparado: bool = False
    recuperacao: int = 0
    candidato: dict | None = None


@dataclass
class EstadoCanal:
    """Memoria do detector para um canal de sinal vital."""

    historico: deque = field(default_factory=deque)
    referencia_mediana: float | None = None
    referencia_mad: float | None = None
    amostras_referencia: list[float] = field(default_factory=list)
    regras: dict[str, EstadoRegra] = field(default_factory=dict)

    def regra(self, nome: str) -> EstadoRegra:
        return self.regras.setdefault(nome, EstadoRegra())


CONFIRMADO = "confirmado"
NAO_CONFIRMADO = "nao_confirmado"
SEM_ACAO = "sem_acao"


def _avancar_regra(
    estado: EstadoRegra, condicao: bool, exigidos: int, recuperacao: int
) -> str:
    """Maquina de estados compartilhada pelas tres regras de sinais vitais.

    Devolve `confirmado` no exato momento em que a condicao completa as
    leituras consecutivas exigidas, `nao_confirmado` quando uma sequencia
    comecou mas morreu antes de confirmar (o caso do artefato de sensor) e
    `sem_acao` no resto.
    """
    if condicao:
        estado.recuperacao = 0
        estado.consecutivos += 1
        if estado.disparado:
            return SEM_ACAO
        if estado.consecutivos >= exigidos:
            estado.disparado = True
            return CONFIRMADO
        return SEM_ACAO

    pendentes = estado.consecutivos
    estado.consecutivos = 0
    estado.recuperacao += 1
    if estado.disparado and estado.recuperacao >= recuperacao:
        estado.disparado = False
        return SEM_ACAO
    if estado.disparado:
        return SEM_ACAO
    if 0 < pendentes < exigidos:
        return NAO_CONFIRMADO
    return SEM_ACAO


class DetectorSinaisVitais:
    """Detector incremental: recebe uma amostra por vez e devolve achados.

    Trabalhar amostra a amostra e o que permite usar o mesmo codigo no modo
    de monitoramento em tempo real e no processamento em lote de uma serie
    historica (`avaliar_serie`), sem duplicar as regras.
    """

    def __init__(self, limiares: LimiaresAnomalia = LIMIARES_ANOMALIA) -> None:
        self.limiares = limiares
        self._estados: dict[str, EstadoCanal] = {
            canal: EstadoCanal(historico=deque(maxlen=limiares.janela_minutos))
            for canal in limiares.canais
        }
        self._amostras_vistas = 0

    @property
    def referencia_pronta(self) -> bool:
        return self._amostras_vistas >= MINUTOS_REFERENCIA

    def atualizar(self, instante: datetime, valores: dict[str, float]) -> list[Achado]:
        """Processa uma amostra (um minuto) e devolve os achados confirmados."""
        achados: list[Achado] = []
        self._amostras_vistas += 1

        for canal, valor in valores.items():
            estado = self._estados.get(canal)
            if estado is None:
                continue
            achados.extend(self._avaliar_canal(canal, estado, instante, float(valor)))

        return achados

    def _avaliar_canal(
        self, canal: str, estado: EstadoCanal, instante: datetime, valor: float
    ) -> list[Achado]:
        achados: list[Achado] = []

        # A referencia e congelada apos as primeiras horas de monitoramento,
        # que representam o estado basal do paciente naquele leito.
        if not self.referencia_pronta:
            estado.amostras_referencia.append(valor)
        elif estado.referencia_mediana is None and estado.amostras_referencia:
            base = np.asarray(estado.amostras_referencia, dtype=float)
            estado.referencia_mediana = float(np.median(base))
            estado.referencia_mad = float(np.median(np.abs(base - np.median(base))))

        achados.extend(self._regra_z_robusto(canal, estado, instante, valor))
        achados.extend(self._regra_tendencia(canal, estado, instante, valor))
        achados.extend(self._regra_limite_clinico(canal, estado, instante, valor))

        estado.historico.append(valor)
        return achados

    def _regra_z_robusto(
        self, canal: str, estado: EstadoCanal, instante: datetime, valor: float
    ) -> list[Achado]:
        limiares = self.limiares
        regra = estado.regra("z_robusto")

        # Antes da janela encher nao ha referencia confiavel de mediana e MAD,
        # e julgar com poucos pontos foi o que mais gerou falso positivo nos
        # primeiros testes.
        if len(estado.historico) < limiares.janela_minutos:
            return []

        janela = np.asarray(estado.historico, dtype=float)
        mediana = float(np.median(janela))
        mad = float(np.median(np.abs(janela - mediana)))
        if mad <= 0:
            return []

        # z modificado de Iglewicz e Hoaglin: 0,6745 * (x - mediana) / MAD.
        z = 0.6745 * (valor - mediana) / mad
        condicao = abs(z) >= limiares.z_robusto
        if condicao and regra.consecutivos == 0:
            regra.candidato = {
                "valor": valor,
                "mediana_janela": round(mediana, 2),
                "z_robusto": round(z, 2),
                "instante": instante.isoformat(),
            }

        situacao = _avancar_regra(
            regra, condicao, limiares.pontos_consecutivos, limiares.minutos_recuperacao
        )

        if situacao == NAO_CONFIRMADO:
            candidato = regra.candidato or {}
            regra.candidato = None
            # A sequencia morreu antes de confirmar: tratamos como artefato de
            # sensor e deixamos registrado, sem incomodar a equipe.
            return [
                Achado(
                    modalidade="sinais_vitais",
                    tipo=f"desvio_pontual_nao_confirmado_{canal}",
                    descricao=(
                        f"Leitura isolada de {canal.replace('_', ' ')} fora do padrao "
                        f"recente ({candidato.get('valor', valor):g}); nao se repetiu nas "
                        "leituras seguintes e foi tratada como artefato de sensor"
                    ),
                    severidade="informativo",
                    score=min(0.6, abs(float(candidato.get("z_robusto", z))) / 20.0),
                    instante=candidato.get("instante", instante.isoformat()),
                    evidencias=candidato,
                )
            ]

        if situacao != CONFIRMADO:
            return []

        regra.candidato = None
        return [
            Achado(
                modalidade="sinais_vitais",
                tipo=f"desvio_abrupto_{canal}",
                descricao=(
                    f"Mudanca abrupta de {canal.replace('_', ' ')} confirmada em "
                    f"{limiares.pontos_consecutivos} leituras seguidas ({valor:g})"
                ),
                severidade="atencao",
                score=min(0.95, 0.6 + abs(z) / 40.0),
                instante=instante.isoformat(),
                evidencias={
                    "valor": valor,
                    "mediana_janela": round(mediana, 2),
                    "z_robusto": round(z, 2),
                    "janela_minutos": limiares.janela_minutos,
                },
            )
        ]

    def _regra_tendencia(
        self, canal: str, estado: EstadoCanal, instante: datetime, valor: float
    ) -> list[Achado]:
        limiares = self.limiares
        regra = estado.regra("tendencia")
        if estado.referencia_mediana is None or len(estado.historico) < MINUTOS_TENDENCIA:
            return []

        recente = np.asarray(list(estado.historico)[-MINUTOS_TENDENCIA:], dtype=float)
        mediana_recente = float(np.median(recente))
        deslocamento = mediana_recente - estado.referencia_mediana

        tolerancia = max(
            DESLOCAMENTO_MINIMO.get(canal, 0.0),
            3.0 * (estado.referencia_mad or 0.0),
        )
        situacao = _avancar_regra(
            regra,
            abs(deslocamento) >= tolerancia,
            limiares.pontos_consecutivos,
            limiares.minutos_recuperacao,
        )
        if situacao != CONFIRMADO:
            return []

        sentido = "alta" if deslocamento > 0 else "queda"
        return [
            Achado(
                modalidade="sinais_vitais",
                tipo=f"tendencia_{sentido}_{canal}",
                descricao=(
                    f"Tendencia de {sentido} sustentada de {canal.replace('_', ' ')}: "
                    f"{mediana_recente:.1f} contra {estado.referencia_mediana:.1f} do basal"
                ),
                severidade="atencao",
                score=min(0.95, 0.55 + abs(deslocamento) / (tolerancia * 6)),
                instante=instante.isoformat(),
                evidencias={
                    "mediana_recente": round(mediana_recente, 2),
                    "mediana_basal": round(estado.referencia_mediana, 2),
                    "deslocamento": round(deslocamento, 2),
                    "tolerancia": round(tolerancia, 2),
                    "minutos_comparados": MINUTOS_TENDENCIA,
                },
            )
        ]

    def _regra_limite_clinico(
        self, canal: str, estado: EstadoCanal, instante: datetime, valor: float
    ) -> list[Achado]:
        limiares = self.limiares
        regra = estado.regra("limite_clinico")
        faixa = limiares.faixa(canal)
        if faixa is None:
            return []

        estourou_critico = (
            faixa.critico_alto is not None and valor > faixa.critico_alto
        ) or (faixa.critico_baixo is not None and valor < faixa.critico_baixo)

        # O mesmo cuidado do z-score: um unico ponto fora do limite critico
        # pode ser sensor deslocado, e nao o paciente.
        situacao = _avancar_regra(
            regra,
            estourou_critico,
            limiares.pontos_consecutivos,
            limiares.minutos_recuperacao,
        )
        if situacao != CONFIRMADO:
            return []

        lado = "acima" if valor > (faixa.critico_alto or math.inf) else "abaixo"
        limite = faixa.critico_alto if lado == "acima" else faixa.critico_baixo
        return [
            Achado(
                modalidade="sinais_vitais",
                tipo=f"limite_critico_{canal}",
                descricao=(
                    f"{canal.replace('_', ' ').capitalize()} {lado} do limite critico: "
                    f"{valor:g} {faixa.unidade} (limite {limite:g})"
                ),
                severidade="critico",
                score=0.9,
                instante=instante.isoformat(),
                evidencias={
                    "valor": valor,
                    "limite": limite,
                    "faixa_normal": f"{faixa.minimo:g}-{faixa.maximo:g} {faixa.unidade}",
                },
            )
        ]


def avaliar_serie(
    df: pd.DataFrame, limiares: LimiaresAnomalia = LIMIARES_ANOMALIA
) -> list[Achado]:
    """Passa uma serie historica inteira pelo detector incremental."""
    detector = DetectorSinaisVitais(limiares)
    achados: list[Achado] = []
    canais = [c for c in limiares.canais if c in df.columns]

    for _, linha in df.iterrows():
        valores = {canal: float(linha[canal]) for canal in canais}
        achados.extend(detector.atualizar(linha["timestamp"].to_pydatetime(), valores))
    return achados


# ---------------------------------------------------------------------------
# Detector multivariado
# ---------------------------------------------------------------------------


def detectar_isolation_forest(
    df: pd.DataFrame,
    limiares: LimiaresAnomalia = LIMIARES_ANOMALIA,
    minutos_referencia: int = MINUTOS_REFERENCIA,
) -> tuple[list[Achado], pd.DataFrame]:
    """Isolation Forest treinado no basal e aplicado ao resto da serie.

    Treinar somente nas primeiras horas (e nao na serie toda) foi uma escolha
    consciente: no monitoramento real, o que existe quando o paciente e
    admitido e o proprio basal dele. Treinar com a serie completa usaria
    informacao do futuro para decidir o presente e inflaria o resultado.

    O corte de decisao e o percentil 1 dos scores do proprio basal: um ponto
    e anomalo quando fica mais isolado que 99% do que ja se viu daquele
    paciente.
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    canais = [c for c in limiares.canais if c in df.columns]
    matriz = df[canais].to_numpy(dtype=float)
    treino = matriz[:minutos_referencia]
    if treino.shape[0] < 30:
        return [], df.assign(score_isolation_forest=np.nan, anomalia_multivariada=False)

    escalador = StandardScaler().fit(treino)
    modelo = IsolationForest(
        n_estimators=200,
        contamination=limiares.contaminacao_isolation_forest,
        random_state=limiares.seed,
    ).fit(escalador.transform(treino))

    scores = modelo.decision_function(escalador.transform(matriz))
    corte = float(np.percentile(scores[:minutos_referencia], 1))
    anomalo = scores < corte

    resultado = df.assign(score_isolation_forest=scores, anomalia_multivariada=anomalo)

    # Pontos isolados abaixo do corte sao esperados mesmo no basal (o corte e
    # um percentil dele). O que interessa e o episodio: minutos vizinhos sao
    # unidos e so viram achado se o conjunto durar o minimo configurado.
    blocos = _unir_blocos(
        _blocos_verdadeiros(anomalo, minimo=1), limiares.minutos_uniao_episodio
    )

    achados: list[Achado] = []
    for inicio, fim in blocos:
        if fim <= minutos_referencia or (fim - inicio) < limiares.minutos_minimos_episodio:
            continue
        trecho = resultado.iloc[inicio:fim]
        pior = trecho["score_isolation_forest"].idxmin()
        canais_alterados = _canais_mais_alterados(df, canais, inicio, fim, minutos_referencia)
        achados.append(
            Achado(
                modalidade="sinais_vitais",
                tipo="anomalia_multivariada",
                descricao=(
                    "Combinacao anomala de sinais vitais em relacao ao basal do paciente "
                    f"({', '.join(canais_alterados)})"
                ),
                severidade="critico",
                score=0.85,
                instante=df.loc[pior, "timestamp"].isoformat(),
                evidencias={
                    "inicio": df.loc[inicio, "timestamp"].isoformat(),
                    "fim": df.loc[fim - 1, "timestamp"].isoformat(),
                    "minutos": int(fim - inicio),
                    "score_minimo": round(float(trecho["score_isolation_forest"].min()), 4),
                    "corte_basal": round(corte, 4),
                },
            )
        )

    return achados, resultado


def _blocos_verdadeiros(mascara: np.ndarray, minimo: int) -> list[tuple[int, int]]:
    """Intervalos [inicio, fim) de True com pelo menos `minimo` elementos."""
    blocos: list[tuple[int, int]] = []
    inicio: int | None = None
    for i, valor in enumerate(mascara):
        if valor and inicio is None:
            inicio = i
        elif not valor and inicio is not None:
            if i - inicio >= minimo:
                blocos.append((inicio, i))
            inicio = None
    if inicio is not None and mascara.size - inicio >= minimo:
        blocos.append((inicio, int(mascara.size)))
    return blocos


def _unir_blocos(blocos: list[tuple[int, int]], intervalo: int) -> list[tuple[int, int]]:
    """Junta blocos separados por menos de `intervalo` posicoes."""
    if not blocos:
        return []
    unidos = [blocos[0]]
    for inicio, fim in blocos[1:]:
        ultimo_inicio, ultimo_fim = unidos[-1]
        if inicio - ultimo_fim <= intervalo:
            unidos[-1] = (ultimo_inicio, fim)
        else:
            unidos.append((inicio, fim))
    return unidos


def _canais_mais_alterados(
    df: pd.DataFrame, canais: list[str], inicio: int, fim: int, minutos_referencia: int
) -> list[str]:
    """Ordena os canais pelo quanto se afastaram do basal no trecho anomalo."""
    desvios = []
    for canal in canais:
        base = df[canal].iloc[:minutos_referencia]
        mediana = float(base.median())
        escala = float((base - mediana).abs().median()) or 1.0
        trecho = df[canal].iloc[inicio:fim]
        desvios.append((abs(float(trecho.median()) - mediana) / escala, canal))
    desvios.sort(reverse=True)
    return [canal for _, canal in desvios[:3]]


# ---------------------------------------------------------------------------
# Prescricoes
# ---------------------------------------------------------------------------


def detectar_anomalias_prescricoes(
    eventos: list[dict], limiares: LimiaresAnomalia = LIMIARES_ANOMALIA
) -> list[Achado]:
    """Procura alteracoes inesperadas na evolucao das prescricoes.

    Tres regras, escolhidas por serem as que mais aparecem em revisao de
    prescricao no hospital ficticio do projeto: salto de dose, interrupcao
    precoce de antimicrobiano e duplicidade terapeutica na mesma classe.
    """
    achados: list[Achado] = []
    dose_vigente: dict[str, float] = {}
    ativos_por_classe: dict[str, set[str]] = {}

    for evento in sorted(eventos, key=lambda e: e["timestamp"]):
        medicamento = evento["medicamento"]
        classe = evento.get("classe", "nao_classificado")
        acao = evento.get("acao", "inicio")
        dose = float(evento.get("dose", 0.0))
        instante = evento["timestamp"]

        if acao == "suspensao":
            ativos_por_classe.get(classe, set()).discard(medicamento)
            dose_vigente.pop(medicamento, None)
            achados.extend(_regra_suspensao_precoce(evento, limiares))
            continue

        anterior = dose_vigente.get(medicamento)
        if anterior is not None and anterior > 0:
            variacao = (dose - anterior) / anterior * 100.0
            if variacao >= limiares.aumento_dose_percentual:
                severidade = "critico" if classe == "opioide" else "atencao"
                achados.append(
                    Achado(
                        modalidade="prescricao",
                        tipo="salto_de_dose",
                        descricao=(
                            f"Aumento de {variacao:.0f}% na dose de {medicamento} "
                            f"({anterior:g} para {dose:g} {evento.get('unidade', '')})".strip()
                        ),
                        severidade=severidade,
                        score=min(0.95, 0.6 + variacao / 500.0),
                        instante=instante,
                        evidencias={
                            "medicamento": medicamento,
                            "classe": classe,
                            "dose_anterior": anterior,
                            "dose_nova": dose,
                            "variacao_percentual": round(variacao, 1),
                            "prescritor": evento.get("prescritor"),
                        },
                    )
                )

        dose_vigente[medicamento] = dose
        ativos = ativos_por_classe.setdefault(classe, set())
        ativos.add(medicamento)

        if classe in CLASSES_SEM_DUPLICIDADE and len(ativos) > 1:
            achados.append(
                Achado(
                    modalidade="prescricao",
                    tipo="duplicidade_terapeutica",
                    descricao=(
                        f"Dois medicamentos da classe {classe} ativos ao mesmo tempo: "
                        + ", ".join(sorted(ativos))
                    ),
                    severidade="critico",
                    score=0.88,
                    instante=instante,
                    evidencias={
                        "classe": classe,
                        "medicamentos": sorted(ativos),
                        "prescritor": evento.get("prescritor"),
                    },
                )
            )

    return achados


def _regra_suspensao_precoce(
    evento: dict, limiares: LimiaresAnomalia
) -> list[Achado]:
    if evento.get("classe") != "antimicrobiano":
        return []
    previsto = evento.get("duracao_prevista_dias")
    atual = evento.get("dia_tratamento")
    if not previsto or not atual:
        return []

    fracao = float(atual) / float(previsto)
    if fracao >= limiares.fracao_minima_antimicrobiano:
        return []

    return [
        Achado(
            modalidade="prescricao",
            tipo="suspensao_precoce_antimicrobiano",
            descricao=(
                f"{evento['medicamento']} suspenso no dia {atual} de um esquema "
                f"previsto para {previsto} dias"
            ),
            severidade="critico",
            score=min(0.95, 0.7 + (1 - fracao) * 0.3),
            instante=evento["timestamp"],
            evidencias={
                "medicamento": evento["medicamento"],
                "dia_tratamento": atual,
                "duracao_prevista_dias": previsto,
                "fracao_cumprida": round(fracao, 2),
                "prescritor": evento.get("prescritor"),
            },
        )
    ]


# ---------------------------------------------------------------------------
# Movimentacao no leito
# ---------------------------------------------------------------------------

HORAS_NOITE = set(range(22, 24)) | set(range(0, 6))


def detectar_anomalias_movimentacao(
    df: pd.DataFrame, limiares: LimiaresAnomalia = LIMIARES_ANOMALIA
) -> list[Achado]:
    """Imobilidade diurna prolongada e agitacao noturna no indice de atividade."""
    achados: list[Achado] = []
    if df.empty:
        return achados

    registros = [
        (linha["timestamp"].to_pydatetime(), float(linha["indice_atividade"]))
        for _, linha in df.iterrows()
    ]

    # Imobilidade: horas diurnas seguidas com atividade muito baixa. A lista
    # recebe um marcador final (None) para a ultima sequencia ser fechada pelo
    # mesmo caminho das anteriores.
    sequencia: list[tuple[datetime, float]] = []
    fim_da_lista: list[tuple[datetime, float] | None] = [*registros, None]
    for registro in fim_da_lista:
        if registro is not None:
            instante, indice = registro
            diurno = instante.hour not in HORAS_NOITE
            if diurno and indice < limiares.atividade_imobilidade:
                sequencia.append((instante, indice))
                continue
        if len(sequencia) >= limiares.horas_imobilidade:
            achados.append(
                Achado(
                    modalidade="movimentacao",
                    tipo="imobilidade_prolongada",
                    descricao=(
                        f"Paciente praticamente imovel por {len(sequencia)} horas seguidas "
                        "durante o dia"
                    ),
                    severidade="atencao",
                    score=min(0.9, 0.6 + len(sequencia) * 0.04),
                    instante=sequencia[0][0].isoformat(),
                    evidencias={
                        "horas": len(sequencia),
                        "inicio": sequencia[0][0].isoformat(),
                        "fim": sequencia[-1][0].isoformat(),
                        "indice_medio": round(
                            sum(v for _, v in sequencia) / len(sequencia), 1
                        ),
                        "limite_indice": limiares.atividade_imobilidade,
                    },
                )
            )
        sequencia = []

    # Agitacao noturna: atividade alta em horario de sono.
    noturnos = [
        (instante, indice)
        for instante, indice in registros
        if instante.hour in HORAS_NOITE and indice > limiares.atividade_agitacao_noturna
    ]
    if len(noturnos) >= 2:
        achados.append(
            Achado(
                modalidade="movimentacao",
                tipo="agitacao_noturna",
                descricao=(
                    f"Atividade elevada em {len(noturnos)} horas da madrugada, padrao "
                    "compativel com agitacao ou delirium"
                ),
                severidade="atencao",
                score=min(0.9, 0.6 + len(noturnos) * 0.07),
                instante=noturnos[0][0].isoformat(),
                evidencias={
                    "horas_afetadas": [i.strftime("%H:%M") for i, _ in noturnos],
                    "indice_maximo": max(v for _, v in noturnos),
                    "limite_indice": limiares.atividade_agitacao_noturna,
                },
            )
        )

    return achados


# ---------------------------------------------------------------------------
# Monitoramento em tempo real
# ---------------------------------------------------------------------------


def monitorar_em_tempo_real(
    df: pd.DataFrame,
    limiares: LimiaresAnomalia = LIMIARES_ANOMALIA,
) -> Iterator[tuple[datetime, list[Achado]]]:
    """Percorre a serie como se as amostras estivessem chegando do monitor.

    O gerador existe para deixar claro que as regras dos sinais vitais nao
    dependem de ver a serie inteira: a cada minuto o detector decide com o
    que ja passou. O Isolation Forest, por depender de um modelo treinado,
    roda separado em `detectar_isolation_forest`.
    """
    detector = DetectorSinaisVitais(limiares)
    canais = [c for c in limiares.canais if c in df.columns]

    for _, linha in df.iterrows():
        instante = linha["timestamp"].to_pydatetime()
        valores = {canal: float(linha[canal]) for canal in canais}
        achados = detector.atualizar(instante, valores)
        if achados:
            yield instante, achados
