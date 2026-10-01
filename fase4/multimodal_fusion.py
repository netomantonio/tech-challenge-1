"""Fusao dos achados das tres modalidades (Fase 4).

Esta e a parte que justifica chamar a solucao de multimodal. Sem fusao,
teriamos tres sistemas independentes avisando a equipe em paralelo; o valor
de olhar video, audio e sinais vitais juntos esta em perceber que achados
diferentes descrevem o mesmo quadro clinico.

A fusao e feita em dois niveis:

1. **sindromes**: combinacoes conhecidas de achados de modalidades
   diferentes. Quando o paciente relata falta de ar na consulta (audio) e a
   saturacao cai junto com a frequencia respiratoria subindo (sinais vitais),
   isso nao sao dois avisos: e um unico quadro de deterioracao respiratoria,
   e a prioridade dele e maior que a de cada parte isolada;
2. **risco do paciente**: um escore 0-100 que agrega todos os achados com
   peso por severidade e por quantidade de modalidades envolvidas, para
   ordenar pacientes num painel de plantao.

A regra de combinacao e deliberadamente explicita, em tabela, e nao um
modelo treinado. Nao temos dados rotulados de desfecho para treinar uma
fusao supervisionada, e um escore aprendido de dados sinteticos daria uma
falsa impressao de validade. Regras escritas podem ser conferidas por um
medico, que e o que faz sentido num projeto academico.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from fase4.alertas import Achado, SEVERIDADES, grupo_do_achado

PESO_SEVERIDADE = {"informativo": 1.0, "atencao": 5.0, "critico": 12.0}

# Janela em que achados de modalidades diferentes sao considerados parte do
# mesmo quadro. Doze horas cobre um turno de plantao, que e a unidade de
# tempo com que a equipe trabalha.
JANELA_CORRELACAO = timedelta(hours=12)


@dataclass(frozen=True)
class RegraSindrome:
    """Combinacao de achados que, juntos, descrevem um quadro clinico."""

    nome: str
    descricao: str
    acao: str
    # Cada entrada e um conjunto de prefixos de tipo; a sindrome exige pelo
    # menos um achado de cada entrada, e as entradas vem de modalidades
    # diferentes de proposito.
    requisitos: tuple[tuple[str, ...], ...]
    severidade: str = "critico"


REGRAS_SINDROME = (
    RegraSindrome(
        nome="deterioracao_respiratoria",
        descricao=(
            "Queixa respiratoria na consulta junto com piora objetiva de "
            "oxigenacao ou ventilacao"
        ),
        acao=(
            "Avaliar o paciente presencialmente, repetir oximetria e considerar "
            "suporte de oxigenio conforme protocolo"
        ),
        requisitos=(
            ("termo_critico_respiratorio", "fala_entrecortada"),
            (
                "tendencia_queda_saturacao_oxigenio",
                "limite_critico_saturacao_oxigenio",
                "tendencia_alta_frequencia_respiratoria",
                "limite_critico_frequencia_respiratoria",
            ),
        ),
    ),
    RegraSindrome(
        nome="suspeita_infeccao",
        descricao=(
            "Relato de febre ou alteracao da ferida junto com febre medida e "
            "repercussao hemodinamica"
        ),
        acao=(
            "Coletar culturas, reavaliar a ferida operatoria e checar o esquema "
            "antimicrobiano antes da proxima dose"
        ),
        requisitos=(
            ("termo_critico_infeccao",),
            ("limite_critico_temperatura", "tendencia_alta_temperatura"),
            (
                "tendencia_alta_frequencia_cardiaca",
                "limite_critico_frequencia_cardiaca",
                "tendencia_queda_pressao_sistolica",
                "anomalia_multivariada",
            ),
        ),
    ),
    RegraSindrome(
        nome="risco_medicamentoso",
        descricao=(
            "Alteracao inesperada de prescricao em paciente que ja apresenta "
            "piora clinica ou queixa de dor"
        ),
        acao=(
            "Revisar a prescricao com a farmacia clinica antes da proxima "
            "administracao e registrar a justificativa no prontuario"
        ),
        requisitos=(
            ("salto_de_dose", "duplicidade_terapeutica", "suspensao_precoce_antimicrobiano"),
            (
                "termo_critico_dor",
                "tendencia_",
                "limite_critico_",
                "anomalia_multivariada",
            ),
        ),
        severidade="critico",
    ),
    RegraSindrome(
        nome="reabilitacao_insegura",
        descricao=(
            "Execucao do exercicio fora do protocolo em paciente que relata dor "
            "ou cansaco na mesma janela"
        ),
        acao=(
            "Rever o plano de reabilitacao com o fisioterapeuta antes da proxima "
            "sessao e reavaliar a liberacao de amplitude"
        ),
        requisitos=(
            (
                "amplitude_ombro",
                "desvio_postural_tronco",
                "invasao_area_critica",
            ),
            ("termo_critico_dor", "termo_critico_fadiga", "fadiga_vocal"),
        ),
        severidade="atencao",
    ),
    RegraSindrome(
        nome="risco_de_imobilidade",
        descricao=(
            "Imobilidade prolongada no leito em paciente com piora de sinais "
            "vitais ou agitacao noturna"
        ),
        acao=(
            "Reavaliar risco de lesao por pressao e de trombose e reforcar "
            "mobilizacao assistida"
        ),
        requisitos=(
            ("imobilidade_prolongada",),
            ("agitacao_noturna", "tendencia_", "limite_critico_", "anomalia_multivariada"),
        ),
        severidade="atencao",
    ),
)


@dataclass
class Sindrome:
    """Quadro clinico identificado pela combinacao de modalidades."""

    nome: str
    descricao: str
    acao: str
    severidade: str
    modalidades: list[str]
    achados: list[Achado] = field(default_factory=list)

    @property
    def grupos(self) -> list[str]:
        """Familias de achado que esta sindrome explica.

        O motor de alertas usa essa lista para nao notificar duas vezes o mesmo
        problema: se o quadro multimodal ja cobre a familia, ela nao gera um
        alerta separado.
        """
        vistos: list[str] = []
        for achado in self.achados:
            grupo = grupo_do_achado(achado.tipo)
            if grupo not in vistos:
                vistos.append(grupo)
        return vistos

    def to_dict(self) -> dict:
        return {
            "nome": self.nome,
            "descricao": self.descricao,
            "acao": self.acao,
            "severidade": self.severidade,
            "modalidades": self.modalidades,
            "grupos": self.grupos,
            "achados": [a.tipo for a in self.achados],
            "instantes": [a.instante for a in self.achados],
        }

    def como_achado(self) -> Achado:
        """Representa a sindrome como um achado multimodal, para o motor de alertas."""
        tipos = sorted({a.tipo for a in self.achados})
        return Achado(
            modalidade="multimodal",
            tipo=f"sindrome_{self.nome}",
            descricao=self.descricao,
            severidade=self.severidade,
            score=min(0.99, max(a.score for a in self.achados) * 1.05),
            instante=self._instante_representativo(),
            evidencias={
                "modalidades": ", ".join(self.modalidades),
                "quantidade_achados": len(self.achados),
                "tipos_de_achado": ", ".join(tipos),
                "acao": self.acao,
            },
        )

    def _instante_representativo(self) -> str | None:
        """Primeiro instante absoluto do quadro, quando existir.

        Video e audio marcam posicao dentro da gravacao (`mm:ss`), que nao e
        comparavel com o timestamp do monitor. Preferimos sempre o timestamp
        absoluto para o alerta nao exibir um horario que a equipe leria errado.
        """
        absolutos = [i for i in (_instante(a) for a in self.achados) if i is not None]
        if absolutos:
            return min(absolutos).isoformat()
        return next((a.instante for a in self.achados if a.instante), None)


@dataclass
class ResultadoFusao:
    paciente_id: str
    risco: float
    nivel_risco: str
    sindromes: list[Sindrome]
    contagem_por_modalidade: dict[str, int]
    contagem_por_severidade: dict[str, int]
    achados_multimodais: list[Achado]

    @property
    def grupos_absorvidos(self) -> list[str]:
        """Familias de achado cobertas por algum quadro multimodal."""
        grupos: list[str] = []
        for sindrome in self.sindromes:
            for grupo in sindrome.grupos:
                if grupo not in grupos:
                    grupos.append(grupo)
        return grupos

    def to_dict(self) -> dict:
        return {
            "paciente_id": self.paciente_id,
            "risco": round(self.risco, 1),
            "nivel_risco": self.nivel_risco,
            "sindromes": [s.to_dict() for s in self.sindromes],
            "grupos_absorvidos": self.grupos_absorvidos,
            "contagem_por_modalidade": self.contagem_por_modalidade,
            "contagem_por_severidade": self.contagem_por_severidade,
        }


def _casa_requisito(achado: Achado, prefixos: tuple[str, ...]) -> bool:
    return any(achado.tipo.startswith(prefixo) for prefixo in prefixos)


def _instante(achado: Achado) -> datetime | None:
    """Le o instante do achado quando ele for um timestamp absoluto.

    Achados de video e audio usam a posicao dentro da gravacao (`mm:ss`), que
    nao e comparavel com o timestamp do monitor. Nesses casos devolvemos None
    e a correlacao temporal nao e exigida: a gravacao pertence ao mesmo
    periodo de monitoramento por construcao do pipeline.
    """
    if not achado.instante:
        return None
    try:
        return datetime.fromisoformat(achado.instante)
    except ValueError:
        return None


def _dentro_da_janela(grupo: list[Achado]) -> bool:
    instantes = [i for i in (_instante(a) for a in grupo) if i is not None]
    if len(instantes) < 2:
        return True
    return (max(instantes) - min(instantes)) <= JANELA_CORRELACAO


def identificar_sindromes(achados: list[Achado]) -> list[Sindrome]:
    """Procura, entre os achados, as combinacoes previstas em REGRAS_SINDROME."""
    sindromes: list[Sindrome] = []

    for regra in REGRAS_SINDROME:
        selecionados: list[Achado] = []
        familias: set[str] = set()
        completa = True
        for prefixos in regra.requisitos:
            candidatos = [a for a in achados if _casa_requisito(a, prefixos)]
            if not candidatos:
                completa = False
                break
            # Entre os candidatos de um requisito, o mais grave representa o
            # requisito na checagem temporal. Mas todas as familias que
            # casaram contam como sustentando o quadro, nao so a do
            # representante: senao a absorcao dependeria de qual achado teve
            # score maior, que e um critério arbitrário.
            candidatos.sort(key=lambda a: (SEVERIDADES.index(a.severidade), a.score), reverse=True)
            selecionados.append(candidatos[0])
            familias.update(grupo_do_achado(a.tipo) for a in candidatos)

        if not completa or not _dentro_da_janela(selecionados):
            continue

        modalidades = []
        for achado in selecionados:
            if achado.modalidade not in modalidades:
                modalidades.append(achado.modalidade)
        if len(modalidades) < 2:
            # Sem pelo menos duas modalidades nao e fusao, e so repetir o que
            # uma modalidade sozinha ja disse.
            continue

        # A sindrome passa a carregar todos os achados das familias que ela
        # representa, nao so o representante de cada requisito. Assim nada se
        # perde quando o alerta do quadro substitui os alertas das familias.
        completos = [a for a in achados if grupo_do_achado(a.tipo) in familias]
        completos.sort(key=lambda a: (SEVERIDADES.index(a.severidade), a.score), reverse=True)

        modalidades = []
        for achado in completos:
            if achado.modalidade not in modalidades:
                modalidades.append(achado.modalidade)

        # O alerta do quadro substitui os alertas das familias absorvidas, por
        # isso ele nunca pode ser menos grave que o pior achado que absorveu.
        severidade = max(
            [regra.severidade, *(a.severidade for a in completos)],
            key=SEVERIDADES.index,
        )

        sindromes.append(
            Sindrome(
                nome=regra.nome,
                descricao=regra.descricao,
                acao=regra.acao,
                severidade=severidade,
                modalidades=modalidades,
                achados=completos,
            )
        )

    return sindromes


def calcular_risco(achados: list[Achado], sindromes: list[Sindrome]) -> float:
    """Escore 0-100 de risco do paciente, para ordenar o painel de plantao.

    A soma e saturada por uma funcao de rendimento decrescente: vinte achados
    de atencao nao devem valer mais que um achado critico confirmado por duas
    modalidades. Cada sindrome adiciona um bonus, porque concordancia entre
    modalidades e justamente o que aumenta a confianca no quadro.
    """
    if not achados:
        return 0.0

    bruto = sum(PESO_SEVERIDADE[a.severidade] * (0.5 + a.score / 2) for a in achados)
    bruto += sum(15.0 for _ in sindromes)

    modalidades = {a.modalidade for a in achados}
    bruto *= 1.0 + 0.10 * (len(modalidades) - 1)

    # Saturacao: 60 pontos brutos chegam a cerca de 74 na escala final.
    return float(100.0 * (1.0 - math.exp(-bruto / 45.0)))


def classificar_risco(risco: float) -> str:
    if risco >= 70.0:
        return "alto"
    if risco >= 40.0:
        return "moderado"
    if risco > 0.0:
        return "baixo"
    return "sem_achados"


def fundir(paciente_id: str, achados: list[Achado]) -> ResultadoFusao:
    """Combina os achados de todas as modalidades de um paciente."""
    sindromes = identificar_sindromes(achados)
    achados_multimodais = [s.como_achado() for s in sindromes]

    contagem_modalidade: dict[str, int] = {}
    for achado in achados:
        contagem_modalidade[achado.modalidade] = (
            contagem_modalidade.get(achado.modalidade, 0) + 1
        )

    contagem_severidade = {sev: 0 for sev in SEVERIDADES}
    for achado in achados:
        contagem_severidade[achado.severidade] += 1

    risco = calcular_risco(achados, sindromes)

    return ResultadoFusao(
        paciente_id=paciente_id,
        risco=risco,
        nivel_risco=classificar_risco(risco),
        sindromes=sindromes,
        contagem_por_modalidade=contagem_modalidade,
        contagem_por_severidade=contagem_severidade,
        achados_multimodais=achados_multimodais,
    )
