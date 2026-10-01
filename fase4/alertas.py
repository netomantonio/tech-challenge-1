"""Achados das modalidades e alertas automaticos para a equipe medica.

Cada analisador (video, audio, sinais vitais, prescricao e movimentacao)
devolve uma lista de `Achado`. O `MotorDeAlertas` traduz esses achados em
`Alerta`, define prioridade, escolhe o destinatario e evita repetir o mesmo
aviso a cada ciclo de monitoramento. Deixar essa traducao num unico lugar
foi o que permitiu usar o mesmo motor tanto no modo batch quanto no modo de
monitoramento em tempo real.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

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

    def to_dict(self) -> dict:
        return asdict(self)

    def resumo(self) -> str:
        return f"[{self.prioridade.upper()}] {self.titulo} -> {self.destino}"


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

    def __init__(self, janela_supressao: timedelta = JANELA_SUPRESSAO) -> None:
        self.janela_supressao = janela_supressao
        self._ultimo_envio: dict[tuple[str, str], datetime] = {}
        self._contador = 0

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
