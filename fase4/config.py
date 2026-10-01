"""Caminhos, variaveis de ambiente e limiares clinicos da Fase 4.

Centralizar isso num unico modulo evita repetir numeros magicos nos tres
analisadores (video, audio e sinais vitais) e deixa explicito no relatorio
de onde veio cada limiar. Os valores sao academicos: foram escolhidos a
partir dos protocolos ficticios usados desde a Fase 3 e de faixas de
referencia comuns em escores de deterioracao clinica (tipo NEWS), nao de
uma validacao com dados reais de pacientes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
DIR_DADOS = Path(__file__).resolve().parent / "data"
DIR_RESULTADOS = RAIZ_PROJETO / "resultados" / "fase4"

CAMINHO_TERMOS_CRITICOS = DIR_DADOS / "termos_criticos_audio.json"

# Nomes das variaveis de ambiente dos servicos gerenciados da Azure.
ENV_SPEECH_KEY = "AZURE_SPEECH_KEY"
ENV_SPEECH_REGION = "AZURE_SPEECH_REGION"
ENV_LANGUAGE_KEY = "AZURE_LANGUAGE_KEY"
ENV_LANGUAGE_ENDPOINT = "AZURE_LANGUAGE_ENDPOINT"

IDIOMA_PADRAO = "pt-BR"


def azure_speech_configurada() -> bool:
    return bool(os.getenv(ENV_SPEECH_KEY) and os.getenv(ENV_SPEECH_REGION))


def azure_language_configurada() -> bool:
    return bool(os.getenv(ENV_LANGUAGE_KEY) and os.getenv(ENV_LANGUAGE_ENDPOINT))


@dataclass(frozen=True)
class LimiaresVideo:
    """Limiares da analise postural e de eventos em video clinico."""

    # Inclinacao lateral do tronco aceitavel durante exercicio supervisionado.
    inclinacao_tronco_max_graus: float = 15.0
    # Amplitude de elevacao do ombro liberada no pos-operatorio recente
    # (PROT-004 restringe exercicios amplos nas primeiras semanas).
    abducao_ombro_max_graus: float = 120.0
    # Assimetria tolerada entre os dois ombros na mesma repeticao.
    assimetria_ombros_max_graus: float = 20.0
    # Um desvio so vira evento depois de persistir alguns quadros seguidos,
    # para nao reportar ruido de estimacao de pose.
    quadros_minimos_desvio: int = 5
    # Confianca minima do keypoint para entrar no calculo de angulo.
    confianca_minima_keypoint: float = 0.35
    # Tempo maximo sem o profissional em cena durante a sessao.
    segundos_max_sem_profissional: float = 2.0
    # Fracao do corpo do paciente dentro da area critica (dreno) que
    # caracteriza invasao de area.
    sobreposicao_area_critica: float = 0.0


@dataclass(frozen=True)
class LimiaresAudio:
    """Limiares acusticos para suspeita de fadiga ou esforco respiratorio."""

    # Proporcao de silencio na gravacao acima da qual a fala e considerada
    # entrecortada.
    proporcao_pausa_alta: float = 0.42
    # Duracao media de frase (segundos) abaixo da qual ha suspeita de
    # dificuldade em sustentar a fala.
    duracao_frase_curta_s: float = 1.6
    # Variacao relativa da F0 entre quadros vozeados consecutivos acima da
    # qual a voz e considerada instavel. Nao e o jitter periodo a periodo da
    # literatura (cujos valores de corte ficam na casa de 1% a 2%): a escala
    # desta medida e menor porque cada quadro de 25 ms ja media varios ciclos.
    # O corte foi calibrado nas amostras sinteticas (0,0006 na voz estavel
    # contra 0,0038 na voz cansada).
    jitter_alto: float = 0.0025
    # Queda relativa de energia entre o inicio e o fim da gravacao.
    queda_energia_relativa: float = 0.30
    # Taxa de fala (silabas estimadas por segundo) considerada lenta.
    taxa_fala_lenta: float = 2.2
    # Confianca minima da transcricao para usar o texto nas regras.
    confianca_minima_transcricao: float = 0.50


@dataclass(frozen=True)
class FaixaVital:
    """Faixa de normalidade de um sinal vital, com limites de alerta."""

    nome: str
    unidade: str
    minimo: float
    maximo: float
    critico_baixo: float | None = None
    critico_alto: float | None = None


@dataclass(frozen=True)
class LimiaresAnomalia:
    """Parametros dos detectores de anomalia em series e prescricoes."""

    # Janela da mediana/MAD moveis usadas no z-score robusto online. O
    # detector so comeca a julgar quando essa janela esta cheia.
    janela_minutos: int = 60
    # |z| robusto a partir do qual o ponto e marcado como anomalo. Como o z
    # modificado de Iglewicz e Hoaglin tem escala proxima do z comum, 4,0
    # corresponde a cerca de 4 desvios: com 720 leituras por canal, isso
    # mantem o numero esperado de falsos positivos abaixo de 1 por paciente.
    z_robusto: float = 4.0
    # Pontos anomalos consecutivos necessarios para confirmar o desvio
    # (filtra artefato isolado de sensor).
    pontos_consecutivos: int = 3
    # Minutos em que a condicao precisa ficar ausente para a mesma regra
    # poder disparar de novo no mesmo canal (histerese).
    minutos_recuperacao: int = 30
    # Contaminacao informada ao Isolation Forest multivariado.
    contaminacao_isolation_forest: float = 0.03
    # Duracao minima e intervalo de uniao dos episodios multivariados. Com 15
    # e 5 minutos, o paciente estavel nao gera nenhum episodio e o paciente em
    # deterioracao gera um unico, continuo; valores menores fragmentavam o
    # mesmo episodio em varios achados.
    minutos_minimos_episodio: int = 15
    minutos_uniao_episodio: int = 5
    seed: int = 42
    # Aumento percentual de dose que exige revisao.
    aumento_dose_percentual: float = 50.0
    # Fracao minima do tratamento antimicrobiano que deve ser cumprida.
    fracao_minima_antimicrobiano: float = 0.8
    # Indice de atividade (0-100) abaixo do qual o paciente e considerado imovel.
    atividade_imobilidade: float = 8.0
    # Horas seguidas de imobilidade diurna que geram alerta.
    horas_imobilidade: int = 5
    # Indice de atividade noturna compativel com agitacao.
    atividade_agitacao_noturna: float = 55.0
    faixas: tuple[FaixaVital, ...] = field(
        default_factory=lambda: (
            FaixaVital("frequencia_cardiaca", "bpm", 60.0, 100.0, 45.0, 130.0),
            FaixaVital("pressao_sistolica", "mmHg", 100.0, 140.0, 90.0, 180.0),
            FaixaVital("pressao_diastolica", "mmHg", 60.0, 90.0, 50.0, 110.0),
            FaixaVital("saturacao_oxigenio", "%", 94.0, 100.0, 90.0, None),
            FaixaVital("temperatura", "C", 35.5, 37.5, 35.0, 38.0),
            FaixaVital("frequencia_respiratoria", "irpm", 12.0, 20.0, 9.0, 24.0),
        )
    )

    def faixa(self, nome: str) -> FaixaVital | None:
        for faixa in self.faixas:
            if faixa.nome == nome:
                return faixa
        return None

    @property
    def canais(self) -> tuple[str, ...]:
        return tuple(faixa.nome for faixa in self.faixas)


LIMIARES_VIDEO = LimiaresVideo()
LIMIARES_AUDIO = LimiaresAudio()
LIMIARES_ANOMALIA = LimiaresAnomalia()
