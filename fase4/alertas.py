"""Achados das modalidades e alertas automaticos para a equipe medica.

Cada analisador (video, audio, sinais vitais, prescricao e movimentacao)
devolve uma lista de `Achado`. O `MotorDeAlertas` traduz esses achados em
`Alerta`, define prioridade, escolhe o destinatario e evita repetir o mesmo
aviso a cada ciclo de monitoramento. Deixar essa traducao num unico lugar
foi o que permitiu usar o mesmo motor tanto no modo batch quanto no modo de
monitoramento em tempo real.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

from fase4.config import DIR_RESULTADOS, ENV_FILA_ALERTAS, ENV_WEBHOOK_URL, NOME_FILA_ALERTAS
from fase4.logging_utils import registrar_evento

SEVERIDADES = ("informativo", "atencao", "critico")

# Prioridade exibida para a equipe, no padrao de triagem por cor.
PRIORIDADE_POR_SEVERIDADE = {
    "critico": "vermelho",
    "atencao": "laranja",
    "informativo": "amarelo",
}

# Quem recebe o alerta. O roteamento e por modalidade porque, no hospital
# ficticio do enunciado, quem acompanha a sessao de fisioterapia nao e quem
# responde por uma alteracao de prescricao.
DESTINO_POR_MODALIDADE = {
    "video": "Equipe de reabilitacao e cirurgia responsavel",
    "audio": "Medico assistente da consulta",
    "sinais_vitais": "Equipe de plantao do leito",
    "prescricao": "Farmacia clinica e medico prescritor",
    "movimentacao": "Equipe de enfermagem do leito",
    "multimodal": "Medico de plantao e enfermagem do leito",
}

# Tempo minimo entre dois alertas do mesmo tipo para o mesmo paciente.
JANELA_SUPRESSAO = timedelta(minutes=15)

# Familias de achado. Achados da mesma familia descrevem o mesmo problema
# visto de angulos diferentes e por isso viram um alerta so. Sem esse
# agrupamento, um turno com deterioracao gera dezenas de notificacoes para a
# mesma equipe - o cenario de fadiga de alarme, em que a equipe passa a
# ignorar o painel.
GRUPOS_ACHADO = (
    ("tendencia_", "tendencia_sinais_vitais"),
    ("limite_critico_", "limite_critico_sinais_vitais"),
    ("desvio_abrupto_", "desvio_abrupto_sinais_vitais"),
    ("desvio_pontual_nao_confirmado_", "leitura_nao_confirmada"),
    ("termo_critico_", "queixas_criticas"),
    ("fala_entrecortada", "alteracoes_vocais"),
    ("fadiga_vocal", "alteracoes_vocais"),
    ("instabilidade_vocal", "alteracoes_vocais"),
    ("padrao_disartrico", "alteracoes_vocais"),
    ("desvio_postural_", "execucao_exercicio"),
    ("amplitude_ombro_", "execucao_exercicio"),
    ("assimetria_entre_ombros", "execucao_exercicio"),
    ("salto_de_dose", "alteracao_prescricao"),
    ("duplicidade_terapeutica", "alteracao_prescricao"),
    ("suspensao_precoce_", "alteracao_prescricao"),
    ("imobilidade_prolongada", "padrao_movimentacao"),
    ("agitacao_noturna", "padrao_movimentacao"),
)

ROTULOS_GRUPO = {
    "tendencia_sinais_vitais": "tendencia sustentada em sinais vitais",
    "limite_critico_sinais_vitais": "sinal vital fora do limite critico",
    "desvio_abrupto_sinais_vitais": "mudanca abrupta de sinal vital",
    "queixas_criticas": "queixa critica relatada na consulta",
    "alteracoes_vocais": "alteracao vocal na consulta",
    "execucao_exercicio": "execucao do exercicio fora do protocolo",
    "alteracao_prescricao": "alteracao inesperada na prescricao",
    "padrao_movimentacao": "padrao anormal de movimentacao no leito",
}


def grupo_do_achado(tipo: str) -> str:
    """Familia a que o tipo de achado pertence (o proprio tipo, se nao houver)."""
    for prefixo, grupo in GRUPOS_ACHADO:
        if tipo.startswith(prefixo):
            return grupo
    return tipo


@dataclass
class Achado:
    """Evento suspeito detectado por uma das modalidades."""

    modalidade: str
    tipo: str
    descricao: str
    severidade: str
    score: float
    instante: str | None = None
    evidencias: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severidade not in SEVERIDADES:
            raise ValueError(f"severidade invalida: {self.severidade}")
        self.score = float(min(max(self.score, 0.0), 1.0))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Alerta:
    """Notificacao pronta para ser entregue a equipe medica."""

    alerta_id: str
    paciente_id: str
    prioridade: str
    titulo: str
    mensagem: str
    destino: str
    acao_recomendada: str
    modalidades: list[str]
    score: float
    criado_em: str
    achados: list[dict] = field(default_factory=list)
    # Nomes dos canais que receberam este alerta (preenchido pelo motor).
    canais_entrega: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def resumo(self) -> str:
        return f"[{self.prioridade.upper()}] {self.titulo} -> {self.destino}"


# ---------------------------------------------------------------------------
# Canais de entrega
# ---------------------------------------------------------------------------


class CanalEntrega(Protocol):
    """Destino concreto de um alerta: fila, webhook, e-mail, pager..."""

    nome: str

    def entregar(self, alerta: Alerta) -> None: ...


class FilaJsonl:
    """Fila append-only em JSON Lines, um alerta por linha.

    E o contrato mais simples que um sistema hospitalar consegue consumir: o
    integrador le o arquivo (ou o diretorio) e processa as linhas novas. Cada
    linha e o `Alerta.to_dict()` completo, com os achados que o sustentam.
    """

    nome = "fila_jsonl"

    def __init__(self, caminho: Path | str) -> None:
        self.caminho = Path(caminho)

    def entregar(self, alerta: Alerta) -> None:
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        registro = {"entregue_em": datetime.now(timezone.utc).isoformat(), **alerta.to_dict()}
        with self.caminho.open("a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False, default=str) + "\n")


class WebhookHttp:
    """POST do alerta em JSON para uma URL (painel, Teams, pager, HIS).

    Usa apenas a biblioteca padrao. Qualquer falha de rede e levantada para
    o motor registrar `entrega_falhou`; o pipeline nunca para por causa do
    canal.
    """

    nome = "webhook"

    def __init__(self, url: str, timeout_s: float = 5.0) -> None:
        self.url = url
        self.timeout_s = timeout_s

    def entregar(self, alerta: Alerta) -> None:
        corpo = json.dumps(alerta.to_dict(), ensure_ascii=False, default=str).encode("utf-8")
        requisicao = urllib.request.Request(
            self.url,
            data=corpo,
            method="POST",
            headers={"Content-Type": "application/json; charset=utf-8", "User-Agent": "fase4-alertas/1.0"},
        )
        with urllib.request.urlopen(requisicao, timeout=self.timeout_s) as resposta:
            if resposta.status >= 300:
                raise RuntimeError(f"webhook respondeu HTTP {resposta.status}")


def canais_padrao(dir_saida: Path | str | None = None) -> list[CanalEntrega]:
    """Canais ativos nesta execucao.

    A fila JSONL e sempre gravada (em `FASE4_FILA_ALERTAS` ou em
    `<dir_saida>/fila_alertas.jsonl`); o webhook so quando `FASE4_WEBHOOK_URL`
    esta definida.
    """
    caminho_fila = os.getenv(ENV_FILA_ALERTAS)
    base = Path(dir_saida) if dir_saida is not None else DIR_RESULTADOS
    canais: list[CanalEntrega] = [FilaJsonl(caminho_fila or base / NOME_FILA_ALERTAS)]
    url = os.getenv(ENV_WEBHOOK_URL, "").strip()
    if url:
        canais.append(WebhookHttp(url))
    return canais


def descrever_canais(canais: Iterable[CanalEntrega]) -> list[str]:
    descricoes = []
    for canal in canais:
        if isinstance(canal, FilaJsonl):
            descricoes.append(f"fila JSONL em {canal.caminho}")
        elif isinstance(canal, WebhookHttp):
            descricoes.append(f"webhook POST {canal.url}")
        else:
            descricoes.append(getattr(canal, "nome", type(canal).__name__))
    return descricoes


ACOES_RECOMENDADAS = {
    "video": (
        "Revisar a gravacao no instante indicado e corrigir a execucao do "
        "exercicio com o paciente antes da proxima serie."
    ),
    "audio": (
        "Reavaliar queixa respiratoria e nivel de fadiga do paciente na "
        "propria consulta e registrar no prontuario."
    ),
    "sinais_vitais": (
        "Conferir o sinal vital a beira do leito, descartar artefato de "
        "sensor e reavaliar o escore de deterioracao."
    ),
    "prescricao": (
        "Validar a alteracao com o medico prescritor antes da proxima "
        "administracao."
    ),
    "movimentacao": (
        "Checar mobilizacao no leito e necessidade de reavaliacao de risco "
        "(queda, lesao por pressao, delirium)."
    ),
    "multimodal": (
        "Avaliar o paciente presencialmente: duas ou mais modalidades "
        "apontaram o mesmo quadro na mesma janela de tempo."
    ),
}


class MotorDeAlertas:
    """Converte achados em alertas, com consolidacao, prioridade e supressao.

    Sao tres mecanismos para o painel da equipe continuar legivel:

    * **consolidacao**: achados da mesma familia viram um unico alerta que
      lista todos eles;
    * **absorcao**: quando a fusao multimodal identifica um quadro clinico, as
      familias que sustentam esse quadro nao geram alerta proprio - quem avisa
      e o alerta do quadro, que e mais informativo e tem prioridade maior;
    * **supressao temporal**: o mesmo grupo nao e reenviado dentro da janela
      configurada, para o monitoramento em tempo real nao repetir o aviso a
      cada minuto enquanto a deterioracao continua.
    """

    def __init__(
        self,
        janela_supressao: timedelta = JANELA_SUPRESSAO,
        canais: Iterable[CanalEntrega] = (),
    ) -> None:
        self.janela_supressao = janela_supressao
        self.canais: list[CanalEntrega] = list(canais)
        self._ultimo_envio: dict[tuple[str, str], datetime] = {}
        self._contador = 0
        self.entregas_falhas = 0

    def _entregar(self, alertas: list[Alerta]) -> None:
        """Envia cada alerta a cada canal; falha de canal vira auditoria, nao excecao."""
        for alerta in alertas:
            for canal in self.canais:
                nome = getattr(canal, "nome", type(canal).__name__)
                # O canal e marcado antes do envio para o proprio registro
                # entregue (linha da fila, corpo do POST) dizer por onde passou.
                alerta.canais_entrega.append(nome)
                try:
                    canal.entregar(alerta)
                except Exception as erro:  # noqa: BLE001 - o canal nao pode derrubar o monitoramento
                    alerta.canais_entrega.remove(nome)
                    self.entregas_falhas += 1
                    registrar_evento(
                        "entrega_falhou",
                        paciente_id=alerta.paciente_id,
                        alerta_id=alerta.alerta_id,
                        canal=nome,
                        erro=str(erro)[:200],
                    )
                    continue
                registrar_evento(
                    "alerta_entregue",
                    paciente_id=alerta.paciente_id,
                    alerta_id=alerta.alerta_id,
                    canal=getattr(canal, "nome", type(canal).__name__),
                    prioridade=alerta.prioridade,
                )

    def _proximo_id(self, paciente_id: str) -> str:
        self._contador += 1
        return f"ALERTA-{paciente_id}-{self._contador:03d}"

    def _suprimido(self, paciente_id: str, grupo: str, agora: datetime) -> bool:
        chave = (paciente_id, grupo)
        anterior = self._ultimo_envio.get(chave)
        if anterior is not None and agora - anterior < self.janela_supressao:
            return True
        self._ultimo_envio[chave] = agora
        return False

    def gerar(
        self,
        paciente_id: str,
        achados: Iterable[Achado],
        grupos_absorvidos: Iterable[str] = (),
        agora: datetime | None = None,
    ) -> list[Alerta]:
        """Emite os alertas de um ciclo de monitoramento, do mais grave ao menos."""
        agora = agora or datetime.now(timezone.utc)
        absorvidos = set(grupos_absorvidos)

        # Achado informativo entra no relatorio, mas nao interrompe a equipe.
        relevantes = [a for a in achados if a.severidade != "informativo"]

        grupos: dict[str, list[Achado]] = {}
        for achado in relevantes:
            grupo = grupo_do_achado(achado.tipo)
            if grupo in absorvidos:
                continue
            grupos.setdefault(grupo, []).append(achado)

        alertas: list[Alerta] = []
        for grupo, membros in grupos.items():
            if self._suprimido(paciente_id, grupo, agora):
                continue
            alertas.append(self._montar(paciente_id, grupo, membros, agora))

        alertas.sort(
            key=lambda a: (list(PRIORIDADE_POR_SEVERIDADE.values()).index(a.prioridade), -a.score)
        )
        self._entregar(alertas)
        return alertas

    def _montar(
        self, paciente_id: str, grupo: str, membros: list[Achado], agora: datetime
    ) -> Alerta:
        membros = sorted(
            membros, key=lambda a: (SEVERIDADES.index(a.severidade), a.score), reverse=True
        )
        principal = membros[0]

        modalidades: list[str] = []
        for achado in membros:
            if achado.modalidade not in modalidades:
                modalidades.append(achado.modalidade)

        if len(membros) == 1:
            titulo = principal.descricao
        else:
            rotulo = ROTULOS_GRUPO.get(grupo, grupo.replace("_", " "))
            titulo = f"{len(membros)} achados de {rotulo}"

        return Alerta(
            alerta_id=self._proximo_id(paciente_id),
            paciente_id=paciente_id,
            prioridade=PRIORIDADE_POR_SEVERIDADE[principal.severidade],
            titulo=titulo,
            mensagem=self._mensagem(paciente_id, membros),
            destino=self._destino(principal, modalidades),
            acao_recomendada=self._acao(principal),
            modalidades=modalidades,
            score=principal.score,
            criado_em=agora.isoformat(),
            achados=[a.to_dict() for a in membros],
        )

    @staticmethod
    def _destino(principal: Achado, modalidades: list[str]) -> str:
        """Destinatarios do alerta.

        Para um quadro multimodal, avisamos as equipes de todas as modalidades
        envolvidas: foi justamente a soma delas que levantou o quadro, e cada
        uma tem uma parte da conduta.
        """
        if principal.modalidade == "multimodal":
            envolvidas = str(principal.evidencias.get("modalidades", "")).split(", ")
            destinos: list[str] = [DESTINO_POR_MODALIDADE["multimodal"]]
            for modalidade in envolvidas:
                destino = DESTINO_POR_MODALIDADE.get(modalidade.strip())
                if destino and destino not in destinos:
                    destinos.append(destino)
            return "; ".join(destinos)

        destinos = []
        for modalidade in modalidades:
            destino = DESTINO_POR_MODALIDADE.get(modalidade, "Equipe medica")
            if destino not in destinos:
                destinos.append(destino)
        return "; ".join(destinos)

    @staticmethod
    def _acao(principal: Achado) -> str:
        acao = principal.evidencias.get("acao")
        if isinstance(acao, str) and acao:
            return acao
        return ACOES_RECOMENDADAS.get(principal.modalidade, "Avaliar o paciente.")

    @staticmethod
    def _mensagem(paciente_id: str, membros: list[Achado]) -> str:
        principal = membros[0]
        instante = f" em {principal.instante}" if principal.instante else ""
        texto = (
            f"Paciente {paciente_id}: {principal.descricao}{instante}. "
            f"Modalidade: {principal.modalidade}. Confianca: {principal.score:.2f}."
        )
        evidencias = ", ".join(f"{k}={v}" for k, v in principal.evidencias.items())
        if evidencias:
            texto += f" Evidencias: {evidencias}."
        if len(membros) > 1:
            detalhes = "; ".join(
                f"{a.descricao} ({a.instante})" if a.instante else a.descricao
                for a in membros[1:]
            )
            texto += f" Tambem no mesmo grupo: {detalhes}."
        return texto
