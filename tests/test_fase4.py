"""Testes da Fase 4 (monitoramento multimodal de pacientes).

Mesmo estilo de `tests/test_fase3.py`: `unittest.TestCase` puro, sem rede e
sem depender de chave paga. Os servicos da Azure sao sempre chamados com
`provedor="offline"`, que e o substituto deterministico de
`fase4.azure_services` - equivalente ao `FakeLLM` da Fase 3.

As amostras sinteticas sao geradas num diretorio temporario em `setUpClass`,
para o teste nao depender do que ja existe em `fase4/data/amostras`.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from fase4 import anomaly_detection, audio_analysis, multimodal_fusion, video_analysis
from fase4.alertas import Achado, MotorDeAlertas, grupo_do_achado
from fase4.azure_services import (
    ServicoAzureIndisponivelError,
    analisar_texto,
    carregar_lexico,
    transcrever_audio,
)
from fase4.config import LIMIARES_ANOMALIA, LIMIARES_AUDIO, LIMIARES_VIDEO
from fase4.data import gerar_dados_sinteticos as gerador
from fase4.pipeline import localizar_amostras, processar_paciente, relatorio_markdown
from fase4.video_analysis import Articulacao


class AmostrasTemporarias(unittest.TestCase):
    """Base que gera as amostras sinteticas uma vez para a classe inteira."""

    diretorio: tempfile.TemporaryDirectory
    raiz: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.diretorio = tempfile.TemporaryDirectory()
        cls.raiz = Path(cls.diretorio.name)
        for paciente in gerador.PACIENTES_PADRAO:
            gerador.gerar_amostras_paciente(paciente, seed=42, destino_raiz=cls.raiz)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.diretorio.cleanup()

    def caminho(self, paciente: str, arquivo: str) -> Path:
        return self.raiz / paciente / arquivo


# ---------------------------------------------------------------------------
# Geracao de dados
# ---------------------------------------------------------------------------


class GeracaoDeAmostrasTests(AmostrasTemporarias):
    def test_gera_todas_as_modalidades_do_caso_completo(self) -> None:
        for arquivo in (
            "sinais_vitais.csv",
            "prescricoes.json",
            "movimentacao.csv",
            "consulta.wav",
            "consulta.transcricao.json",
            "fisioterapia.keypoints.json",
            "fisioterapia.objetos.json",
        ):
            self.assertTrue(
                self.caminho("PAC-0006", arquivo).exists(), f"faltou {arquivo}"
            )

    def test_paciente_controle_nao_tem_video(self) -> None:
        self.assertFalse(self.caminho("PAC-0002", "fisioterapia.keypoints.json").exists())
        self.assertTrue(self.caminho("PAC-0002", "consulta.wav").exists())

    def test_serie_vital_tem_doze_horas_por_minuto(self) -> None:
        df = anomaly_detection.carregar_sinais_vitais(
            self.caminho("PAC-0006", "sinais_vitais.csv")
        )
        self.assertEqual(len(df), 12 * 60)
        intervalos = df["timestamp"].diff().dropna().unique()
        self.assertEqual(list(intervalos), [pd.Timedelta(minutes=1)])

    def test_geracao_e_reproduzivel_com_a_mesma_seed(self) -> None:
        registros_a, _ = gerador.gerar_sinais_vitais("PAC-0006", seed=7)
        registros_b, _ = gerador.gerar_sinais_vitais("PAC-0006", seed=7)
        registros_c, _ = gerador.gerar_sinais_vitais("PAC-0006", seed=8)
        self.assertEqual(registros_a, registros_b)
        self.assertNotEqual(registros_a, registros_c)


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------


class AngulosArticularesTests(unittest.TestCase):
    def test_braco_ao_longo_do_corpo_e_zero_grau(self) -> None:
        ombro = Articulacao(100.0, 100.0, 0.9)
        cotovelo = Articulacao(100.0, 150.0, 0.9)  # direto para baixo
        angulo = video_analysis.angulo_abducao_ombro(ombro, cotovelo, "esquerdo")
        self.assertAlmostEqual(angulo, 0.0, places=5)

    def test_braco_na_horizontal_e_noventa_graus(self) -> None:
        ombro = Articulacao(100.0, 100.0, 0.9)
        cotovelo = Articulacao(150.0, 100.0, 0.9)
        angulo = video_analysis.angulo_abducao_ombro(ombro, cotovelo, "esquerdo")
        self.assertAlmostEqual(angulo, 90.0, places=5)

    def test_braco_acima_da_cabeca_passa_de_cento_e_vinte(self) -> None:
        ombro = Articulacao(100.0, 100.0, 0.9)
        # 140 graus de abducao para o lado esquerdo do paciente.
        rad = math.radians(140.0)
        cotovelo = Articulacao(
            100.0 + 50 * math.sin(rad), 100.0 + 50 * math.cos(rad), 0.9
        )
        angulo = video_analysis.angulo_abducao_ombro(ombro, cotovelo, "esquerdo")
        self.assertAlmostEqual(angulo, 140.0, places=4)
        self.assertGreater(angulo, LIMIARES_VIDEO.abducao_ombro_max_graus)

    def test_braco_cruzando_o_corpo_devolve_angulo_negativo(self) -> None:
        ombro = Articulacao(100.0, 100.0, 0.9)
        cotovelo = Articulacao(60.0, 90.0, 0.9)  # para o lado oposto
        angulo = video_analysis.angulo_abducao_ombro(ombro, cotovelo, "esquerdo")
        self.assertLess(angulo, 0.0)

    def test_tronco_vertical_nao_tem_inclinacao(self) -> None:
        inclinacao = video_analysis.angulo_inclinacao_tronco(
            Articulacao(130.0, 100.0, 0.9),
            Articulacao(70.0, 100.0, 0.9),
            Articulacao(125.0, 200.0, 0.9),
            Articulacao(75.0, 200.0, 0.9),
        )
        self.assertAlmostEqual(inclinacao, 0.0, places=5)

    def test_tronco_inclinado_quarenta_e_cinco_graus(self) -> None:
        inclinacao = video_analysis.angulo_inclinacao_tronco(
            Articulacao(230.0, 100.0, 0.9),
            Articulacao(170.0, 100.0, 0.9),
            Articulacao(125.0, 200.0, 0.9),
            Articulacao(75.0, 200.0, 0.9),
        )
        self.assertAlmostEqual(inclinacao, 45.0, places=5)


class FiltroDeRuidoDePoseTests(unittest.TestCase):
    def test_pico_isolado_nao_vira_desvio(self) -> None:
        # Um unico quadro acima do limite: menos que quadros_minimos_desvio.
        valores = [(i, i / 30.0, 10.0) for i in range(30)]
        valores[15] = (15, 0.5, 40.0)
        janelas = video_analysis._janelas_acima(
            valores, limite=15.0, minimo_quadros=LIMIARES_VIDEO.quadros_minimos_desvio
        )
        self.assertEqual(janelas, [])

    def test_desvio_sustentado_vira_uma_janela(self) -> None:
        valores = [(i, i / 30.0, 10.0) for i in range(30)]
        for i in range(10, 20):
            valores[i] = (i, i / 30.0, 25.0)
        janelas = video_analysis._janelas_acima(
            valores, limite=15.0, minimo_quadros=LIMIARES_VIDEO.quadros_minimos_desvio
        )
        self.assertEqual(len(janelas), 1)
        self.assertEqual(janelas[0]["quadros"], 10)
        self.assertEqual(janelas[0]["valor_maximo"], 25.0)

    def test_keypoint_com_confianca_baixa_nao_entra_na_medida(self) -> None:
        quadro = video_analysis.QuadroPose(
            quadro=0,
            instante_s=0.0,
            articulacoes={
                "ombro_esquerdo": Articulacao(100.0, 100.0, 0.05),
                "cotovelo_esquerdo": Articulacao(150.0, 100.0, 0.05),
            },
        )
        medidas = video_analysis.medir_quadros([quadro])
        self.assertIsNone(medidas[0].abducao_ombro_esquerdo)


class AnaliseDeVideoTests(AmostrasTemporarias):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.resultado = video_analysis.analisar_sessao(
            paciente_id="PAC-0006",
            caminho_keypoints=cls.raiz / "PAC-0006" / "fisioterapia.keypoints.json",
            caminho_objetos=cls.raiz / "PAC-0006" / "fisioterapia.objetos.json",
        )

    def test_processa_todos_os_quadros_da_sessao(self) -> None:
        esperado = int(gerador.DURACAO_VIDEO_S * gerador.FPS_VIDEO)
        self.assertEqual(self.resultado.quadros_analisados, esperado)
        self.assertEqual(len(self.resultado.medidas), esperado)

    def test_detecta_os_cinco_tipos_de_evento_plantados(self) -> None:
        tipos = {a.tipo for a in self.resultado.achados}
        for esperado in (
            "desvio_postural_tronco",
            "amplitude_ombro_esquerdo_excedida",
            "assimetria_entre_ombros",
            "invasao_area_critica",
            "profissional_ausente",
        ):
            self.assertIn(esperado, tipos)

    def test_ombro_sem_desvio_nao_gera_achado(self) -> None:
        # O ombro direito fica sempre abaixo do limite nas repeticoes geradas.
        tipos = {a.tipo for a in self.resultado.achados}
        self.assertNotIn("amplitude_ombro_direito_excedida", tipos)
        self.assertLess(
            self.resultado.resumo_angulos["abducao_maxima_ombro_direito"],
            LIMIARES_VIDEO.abducao_ombro_max_graus,
        )

    def test_ausencia_do_profissional_respeita_o_limite_de_tempo(self) -> None:
        achado = next(
            a for a in self.resultado.achados if a.tipo == "profissional_ausente"
        )
        self.assertGreater(
            achado.evidencias["duracao_s"], LIMIARES_VIDEO.segundos_max_sem_profissional
        )

    def test_relatorio_tem_as_secoes_pedidas_no_enunciado(self) -> None:
        relatorio = video_analysis.relatorio_markdown(self.resultado)
        self.assertIn("Amplitudes observadas", relatorio)
        self.assertIn("Desvios e eventos detectados", relatorio)
        self.assertIn("PAC-0006", relatorio)

    def test_backend_invalido_e_rejeitado(self) -> None:
        with self.assertRaises(ValueError):
            video_analysis.analisar_sessao("PAC-0006", backend_pose="openpose")


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


class AtributosAcusticosTests(AmostrasTemporarias):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.fadiga = audio_analysis.extrair_atributos(
            cls.raiz / "PAC-0006" / "consulta.wav"
        )
        cls.estavel = audio_analysis.extrair_atributos(
            cls.raiz / "PAC-0002" / "consulta.wav"
        )

    def test_separa_fala_de_pausa(self) -> None:
        self.assertGreater(self.fadiga.quantidade_segmentos, 0)
        self.assertGreater(self.estavel.quantidade_segmentos, 0)
        self.assertLess(self.fadiga.duracao_fala_s, self.fadiga.duracao_total_s)

    def test_voz_cansada_tem_mais_pausa_e_frases_mais_curtas(self) -> None:
        self.assertGreater(self.fadiga.proporcao_pausa, self.estavel.proporcao_pausa)
        self.assertLess(
            self.fadiga.duracao_media_frase_s, self.estavel.duracao_media_frase_s
        )

    def test_voz_cansada_fala_mais_devagar_e_perde_energia(self) -> None:
        self.assertLess(self.fadiga.taxa_fala_silabas_s, self.estavel.taxa_fala_silabas_s)
        self.assertGreater(
            self.fadiga.queda_energia_relativa, self.estavel.queda_energia_relativa
        )

    def test_voz_cansada_tem_f0_mais_instavel(self) -> None:
        self.assertGreater(self.fadiga.jitter_f0, self.estavel.jitter_f0)

    def test_f0_estimada_fica_na_faixa_da_voz_humana(self) -> None:
        for atributos in (self.fadiga, self.estavel):
            self.assertGreater(atributos.f0_media_hz, audio_analysis.F0_MINIMA)
            self.assertLess(atributos.f0_media_hz, audio_analysis.F0_MAXIMA)

    def test_regras_vocais_disparam_so_no_caso_com_fadiga(self) -> None:
        achados_fadiga = {
            a.tipo for a in audio_analysis.avaliar_alteracoes_vocais(self.fadiga)
        }
        achados_estavel = {
            a.tipo for a in audio_analysis.avaliar_alteracoes_vocais(self.estavel)
        }
        self.assertIn("fala_entrecortada", achados_fadiga)
        self.assertIn("fadiga_vocal", achados_fadiga)
        self.assertIn("instabilidade_vocal", achados_fadiga)
        self.assertEqual(achados_estavel, set())


class ServicosAzureOfflineTests(AmostrasTemporarias):
    def test_transcricao_offline_usa_a_referencia_do_audio(self) -> None:
        resultado = transcrever_audio(
            self.caminho("PAC-0006", "consulta.wav"), provedor="offline"
        )
        self.assertEqual(resultado.provedor, "offline")
        self.assertTrue(resultado.segmentos)
        self.assertGreater(resultado.confianca, 0.5)
        self.assertIn("cansada", resultado.texto)
        # Os segmentos precisam estar em ordem e dentro da duracao do audio.
        for anterior, atual in zip(resultado.segmentos, resultado.segmentos[1:]):
            self.assertLessEqual(anterior.fim_s, atual.inicio_s)
        self.assertLessEqual(resultado.segmentos[-1].fim_s, resultado.duracao_s)

    def test_termos_criticos_sao_encontrados_sem_acento(self) -> None:
        analise = analisar_texto(
            "A paciente relata falta de ar e febre há dois dias.", provedor="offline"
        )
        categorias = analise.categorias()
        self.assertIn("respiratorio", categorias)
        self.assertIn("infeccao", categorias)
        self.assertEqual(analise.sentimento, "negativo")

    def test_fala_positiva_nao_gera_termo_critico(self) -> None:
        analise = analisar_texto(
            "Estou bem melhor, sem dor e consegui caminhar.", provedor="offline"
        )
        self.assertEqual(analise.termos_criticos, [])
        self.assertEqual(analise.sentimento, "positivo")

    def test_provedor_azure_sem_credencial_falha_explicitamente(self) -> None:
        with self.assertRaises(ServicoAzureIndisponivelError):
            analisar_texto("texto qualquer", provedor="azure")
        with self.assertRaises(ServicoAzureIndisponivelError):
            transcrever_audio(
                self.caminho("PAC-0006", "consulta.wav"), provedor="azure"
            )

    def test_provedor_invalido_e_rejeitado(self) -> None:
        with self.assertRaises(ValueError):
            analisar_texto("texto", provedor="watson")

    def test_lexico_tem_peso_e_termos_em_todas_as_categorias(self) -> None:
        lexico = carregar_lexico()
        self.assertTrue(lexico["categorias"])
        for categoria, dados in lexico["categorias"].items():
            self.assertGreater(dados["peso"], 0.0, categoria)
            self.assertTrue(dados["termos"], categoria)


class AnaliseDeConsultaTests(AmostrasTemporarias):
    def test_consulta_com_queixas_gera_achados_criticos(self) -> None:
        resultado = audio_analysis.analisar_consulta(
            self.caminho("PAC-0006", "consulta.wav"), "PAC-0006", provedor_azure="offline"
        )
        tipos = {a.tipo for a in resultado.achados}
        self.assertIn("termo_critico_respiratorio", tipos)
        self.assertIn("fala_entrecortada", tipos)
        criticos = [a for a in resultado.achados if a.severidade == "critico"]
        self.assertTrue(criticos)

    def test_consulta_estavel_nao_gera_alerta(self) -> None:
        resultado = audio_analysis.analisar_consulta(
            self.caminho("PAC-0002", "consulta.wav"), "PAC-0002", provedor_azure="offline"
        )
        relevantes = [a for a in resultado.achados if a.severidade != "informativo"]
        self.assertEqual(relevantes, [])

    def test_transcricao_pouco_confiavel_bloqueia_as_regras_de_texto(self) -> None:
        # Sem o arquivo de referencia a transcricao volta vazia; nesse caso as
        # regras de texto nao podem rodar, para nao alertar com base em
        # palavra que o reconhecedor nao ouviu.
        with tempfile.TemporaryDirectory() as temporario:
            destino = Path(temporario) / "consulta.wav"
            destino.write_bytes(
                self.caminho("PAC-0006", "consulta.wav").read_bytes()
            )
            resultado = audio_analysis.analisar_consulta(
                destino, "PAC-0006", provedor_azure="offline"
            )
        self.assertEqual(resultado.analise_texto.sentimento, "indisponivel")
        self.assertTrue(
            all(not a.tipo.startswith("termo_critico") for a in resultado.achados)
        )
        # Os atributos acusticos continuam valendo.
        self.assertIn("fala_entrecortada", {a.tipo for a in resultado.achados})

    def test_instante_do_termo_aponta_para_o_segmento_certo(self) -> None:
        resultado = audio_analysis.analisar_consulta(
            self.caminho("PAC-0006", "consulta.wav"), "PAC-0006", provedor_azure="offline"
        )
        achado = next(
            a for a in resultado.achados if a.tipo == "termo_critico_respiratorio"
        )
        segmento = next(
            s for s in resultado.transcricao.segmentos if "sem ar" in s.texto
        )
        minutos, segundos = achado.instante.split(":")
        self.assertAlmostEqual(
            int(minutos) * 60 + float(segundos), segmento.inicio_s, places=2
        )


# ---------------------------------------------------------------------------
# Anomalias
# ---------------------------------------------------------------------------


class MaquinaDeEstadosDeRegraTests(unittest.TestCase):
    def test_confirma_apenas_na_contagem_exigida(self) -> None:
        estado = anomaly_detection.EstadoRegra()
        self.assertEqual(
            anomaly_detection._avancar_regra(estado, True, 3, 30),
            anomaly_detection.SEM_ACAO,
        )
        self.assertEqual(
            anomaly_detection._avancar_regra(estado, True, 3, 30),
            anomaly_detection.SEM_ACAO,
        )
        self.assertEqual(
            anomaly_detection._avancar_regra(estado, True, 3, 30),
            anomaly_detection.CONFIRMADO,
        )

    def test_nao_repete_enquanto_a_condicao_persiste(self) -> None:
        estado = anomaly_detection.EstadoRegra()
        for _ in range(3):
            anomaly_detection._avancar_regra(estado, True, 3, 30)
        for _ in range(20):
            self.assertEqual(
                anomaly_detection._avancar_regra(estado, True, 3, 30),
                anomaly_detection.SEM_ACAO,
            )

    def test_sequencia_curta_e_tratada_como_artefato(self) -> None:
        estado = anomaly_detection.EstadoRegra()
        anomaly_detection._avancar_regra(estado, True, 3, 30)
        self.assertEqual(
            anomaly_detection._avancar_regra(estado, False, 3, 30),
            anomaly_detection.NAO_CONFIRMADO,
        )

    def test_libera_novo_disparo_so_apos_a_recuperacao(self) -> None:
        estado = anomaly_detection.EstadoRegra()
        for _ in range(3):
            anomaly_detection._avancar_regra(estado, True, 3, 30)
        for _ in range(5):
            anomaly_detection._avancar_regra(estado, False, 3, 30)
        # Ainda nao recuperou: nao pode confirmar de novo.
        for _ in range(3):
            resultado = anomaly_detection._avancar_regra(estado, True, 3, 30)
        self.assertEqual(resultado, anomaly_detection.SEM_ACAO)

        for _ in range(30):
            anomaly_detection._avancar_regra(estado, False, 3, 30)
        for _ in range(2):
            anomaly_detection._avancar_regra(estado, True, 3, 30)
        self.assertEqual(
            anomaly_detection._avancar_regra(estado, True, 3, 30),
            anomaly_detection.CONFIRMADO,
        )


class DetectorDeSinaisVitaisTests(AmostrasTemporarias):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.df_piora = anomaly_detection.carregar_sinais_vitais(
            cls.raiz / "PAC-0006" / "sinais_vitais.csv"
        )
        cls.df_estavel = anomaly_detection.carregar_sinais_vitais(
            cls.raiz / "PAC-0002" / "sinais_vitais.csv"
        )
        cls.achados_piora = anomaly_detection.avaliar_serie(cls.df_piora)
        cls.achados_estavel = anomaly_detection.avaliar_serie(cls.df_estavel)

    def test_paciente_estavel_nao_gera_alerta(self) -> None:
        relevantes = [a for a in self.achados_estavel if a.severidade != "informativo"]
        self.assertEqual(relevantes, [])

    def test_deterioracao_e_detectada_pela_tendencia_e_pelo_limite(self) -> None:
        tipos = {a.tipo for a in self.achados_piora}
        self.assertTrue(any(t.startswith("tendencia_") for t in tipos))
        self.assertTrue(any(t.startswith("limite_critico_") for t in tipos))

    def test_artefato_isolado_nao_vira_alerta(self) -> None:
        artefatos = [
            a
            for a in self.achados_piora
            if a.tipo == "desvio_pontual_nao_confirmado_frequencia_cardiaca"
            and a.evidencias.get("valor") == 185.0
        ]
        self.assertEqual(len(artefatos), 1)
        self.assertEqual(artefatos[0].severidade, "informativo")
        # O pico de 185 bpm passa do limite critico, mas nao pode gerar
        # alerta por ser uma leitura unica.
        criticos_fc = [
            a
            for a in self.achados_piora
            if a.tipo == "limite_critico_frequencia_cardiaca"
            and a.instante.endswith("10:30:00")
        ]
        self.assertEqual(criticos_fc, [])

    def test_tendencia_antecipa_o_limite_critico(self) -> None:
        tendencias = [a for a in self.achados_piora if a.tipo.startswith("tendencia_")]
        limites = [a for a in self.achados_piora if a.tipo.startswith("limite_critico_")]
        self.assertTrue(tendencias and limites)
        primeira_tendencia = min(datetime.fromisoformat(a.instante) for a in tendencias)
        primeiro_limite = min(datetime.fromisoformat(a.instante) for a in limites)
        self.assertLess(primeira_tendencia, primeiro_limite)

    def test_cada_regra_dispara_uma_vez_por_canal(self) -> None:
        contagem: dict[str, int] = {}
        for achado in self.achados_piora:
            if achado.severidade == "informativo":
                continue
            contagem[achado.tipo] = contagem.get(achado.tipo, 0) + 1
        for tipo, quantidade in contagem.items():
            self.assertEqual(quantidade, 1, f"{tipo} disparou {quantidade} vezes")

    def test_monitoramento_em_tempo_real_devolve_os_mesmos_achados(self) -> None:
        em_tempo_real = [
            achado
            for _, achados in anomaly_detection.monitorar_em_tempo_real(self.df_piora)
            for achado in achados
        ]
        self.assertEqual(
            [a.tipo for a in em_tempo_real], [a.tipo for a in self.achados_piora]
        )

    def test_detector_nao_julga_antes_da_janela_encher(self) -> None:
        detector = anomaly_detection.DetectorSinaisVitais()
        instante = datetime(2025, 6, 10, 8, 0)
        achados = []
        for i in range(20):
            achados.extend(
                detector.atualizar(
                    instante + timedelta(minutes=i),
                    {"frequencia_cardiaca": 80.0 if i < 19 else 200.0},
                )
            )
        self.assertEqual(achados, [])


class IsolationForestTests(AmostrasTemporarias):
    def test_detecta_um_episodio_no_caso_com_piora(self) -> None:
        df = anomaly_detection.carregar_sinais_vitais(
            self.caminho("PAC-0006", "sinais_vitais.csv")
        )
        achados, enriquecido = anomaly_detection.detectar_isolation_forest(df)
        self.assertEqual(len(achados), 1)
        self.assertEqual(achados[0].tipo, "anomalia_multivariada")
        self.assertGreaterEqual(
            achados[0].evidencias["minutos"], LIMIARES_ANOMALIA.minutos_minimos_episodio
        )
        self.assertIn("score_isolation_forest", enriquecido.columns)

    def test_nao_acusa_o_paciente_estavel(self) -> None:
        df = anomaly_detection.carregar_sinais_vitais(
            self.caminho("PAC-0002", "sinais_vitais.csv")
        )
        achados, _ = anomaly_detection.detectar_isolation_forest(df)
        self.assertEqual(achados, [])

    def test_serie_curta_nao_quebra_o_detector(self) -> None:
        df = anomaly_detection.carregar_sinais_vitais(
            self.caminho("PAC-0002", "sinais_vitais.csv")
        ).head(10)
        achados, enriquecido = anomaly_detection.detectar_isolation_forest(df)
        self.assertEqual(achados, [])
        self.assertEqual(len(enriquecido), 10)


class AnomaliasDePrescricaoTests(AmostrasTemporarias):
    def _eventos(self, paciente: str) -> list[dict]:
        with self.caminho(paciente, "prescricoes.json").open(encoding="utf-8") as f:
            return json.load(f)

    def test_detecta_as_tres_alteracoes_plantadas(self) -> None:
        achados = anomaly_detection.detectar_anomalias_prescricoes(
            self._eventos("PAC-0006")
        )
        tipos = {a.tipo for a in achados}
        self.assertEqual(
            tipos,
            {"salto_de_dose", "suspensao_precoce_antimicrobiano", "duplicidade_terapeutica"},
        )
        self.assertTrue(all(a.severidade == "critico" for a in achados))

    def test_salto_de_dose_registra_a_variacao(self) -> None:
        achado = next(
            a
            for a in anomaly_detection.detectar_anomalias_prescricoes(
                self._eventos("PAC-0006")
            )
            if a.tipo == "salto_de_dose"
        )
        self.assertEqual(achado.evidencias["medicamento"], "morfina")
        self.assertEqual(achado.evidencias["variacao_percentual"], 400.0)

    def test_prescricao_coerente_nao_gera_achado(self) -> None:
        self.assertEqual(
            anomaly_detection.detectar_anomalias_prescricoes(self._eventos("PAC-0002")), []
        )

    def test_ajuste_pequeno_de_dose_nao_alerta(self) -> None:
        eventos = [
            {
                "timestamp": "2025-06-10T08:00:00",
                "acao": "inicio",
                "medicamento": "dipirona",
                "classe": "analgesico_nao_opioide",
                "dose": 500.0,
                "unidade": "mg",
            },
            {
                "timestamp": "2025-06-10T12:00:00",
                "acao": "ajuste",
                "medicamento": "dipirona",
                "classe": "analgesico_nao_opioide",
                "dose": 600.0,
                "unidade": "mg",
            },
        ]
        self.assertEqual(anomaly_detection.detectar_anomalias_prescricoes(eventos), [])

    def test_antimicrobiano_concluido_nao_e_suspensao_precoce(self) -> None:
        eventos = [
            {
                "timestamp": "2025-06-10T08:00:00",
                "acao": "suspensao",
                "medicamento": "ceftriaxona",
                "classe": "antimicrobiano",
                "dose": 1000.0,
                "duracao_prevista_dias": 7,
                "dia_tratamento": 7,
            }
        ]
        self.assertEqual(anomaly_detection.detectar_anomalias_prescricoes(eventos), [])


class AnomaliasDeMovimentacaoTests(AmostrasTemporarias):
    def test_detecta_imobilidade_e_agitacao_noturna(self) -> None:
        df = anomaly_detection.carregar_movimentacao(
            self.caminho("PAC-0006", "movimentacao.csv")
        )
        tipos = {a.tipo for a in anomaly_detection.detectar_anomalias_movimentacao(df)}
        self.assertEqual(tipos, {"imobilidade_prolongada", "agitacao_noturna"})

    def test_padrao_normal_nao_gera_achado(self) -> None:
        df = anomaly_detection.carregar_movimentacao(
            self.caminho("PAC-0002", "movimentacao.csv")
        )
        self.assertEqual(anomaly_detection.detectar_anomalias_movimentacao(df), [])

    def test_atividade_baixa_de_madrugada_nao_e_imobilidade(self) -> None:
        # Dormir a noite nao e imobilidade: a regra so olha horario diurno.
        inicio = datetime(2025, 6, 10, 0, 0)
        df = pd.DataFrame(
            {
                "timestamp": [inicio + timedelta(hours=h) for h in range(6)],
                "indice_atividade": [2.0] * 6,
                "fonte": ["sensor_leito"] * 6,
            }
        )
        self.assertEqual(anomaly_detection.detectar_anomalias_movimentacao(df), [])


# ---------------------------------------------------------------------------
# Fusao e alertas
# ---------------------------------------------------------------------------


def _achado(modalidade: str, tipo: str, severidade: str = "atencao", instante=None) -> Achado:
    return Achado(
        modalidade=modalidade,
        tipo=tipo,
        descricao=f"achado de teste {tipo}",
        severidade=severidade,
        score=0.8,
        instante=instante,
    )


class FusaoMultimodalTests(unittest.TestCase):
    def test_identifica_quadro_respiratorio_com_duas_modalidades(self) -> None:
        achados = [
            _achado("audio", "termo_critico_respiratorio", "critico", "00:03.00"),
            _achado(
                "sinais_vitais",
                "tendencia_queda_saturacao_oxigenio",
                instante="2025-06-10T17:20:00",
            ),
        ]
        sindromes = multimodal_fusion.identificar_sindromes(achados)
        nomes = {s.nome for s in sindromes}
        self.assertIn("deterioracao_respiratoria", nomes)

    def test_uma_modalidade_sozinha_nao_forma_quadro(self) -> None:
        achados = [
            _achado("audio", "termo_critico_respiratorio", "critico"),
            _achado("audio", "fala_entrecortada"),
        ]
        self.assertEqual(multimodal_fusion.identificar_sindromes(achados), [])

    def test_quadro_nunca_e_menos_grave_que_o_pior_achado_absorvido(self) -> None:
        achados = [
            _achado("video", "amplitude_ombro_esquerdo_excedida", "critico"),
            _achado("audio", "termo_critico_fadiga", "atencao"),
        ]
        sindrome = next(
            s
            for s in multimodal_fusion.identificar_sindromes(achados)
            if s.nome == "reabilitacao_insegura"
        )
        # A regra e de atencao, mas ela absorveu um achado critico.
        self.assertEqual(sindrome.severidade, "critico")

    def test_quadro_absorve_todas_as_familias_que_o_sustentam(self) -> None:
        achados = [
            _achado("video", "amplitude_ombro_esquerdo_excedida", "critico"),
            _achado("video", "desvio_postural_tronco"),
            _achado("audio", "termo_critico_fadiga"),
        ]
        sindrome = next(
            s
            for s in multimodal_fusion.identificar_sindromes(achados)
            if s.nome == "reabilitacao_insegura"
        )
        self.assertIn("execucao_exercicio", sindrome.grupos)
        self.assertIn("queixas_criticas", sindrome.grupos)
        self.assertEqual(len(sindrome.achados), 3)

    def test_sem_achado_o_risco_e_zero(self) -> None:
        fusao = multimodal_fusion.fundir("PAC-0000", [])
        self.assertEqual(fusao.risco, 0.0)
        self.assertEqual(fusao.nivel_risco, "sem_achados")

    def test_risco_cresce_com_severidade_e_com_numero_de_modalidades(self) -> None:
        leve = multimodal_fusion.fundir(
            "PAC-0000", [_achado("audio", "fadiga_vocal", "informativo")]
        )
        grave = multimodal_fusion.fundir(
            "PAC-0000",
            [
                _achado("audio", "termo_critico_respiratorio", "critico"),
                _achado("sinais_vitais", "limite_critico_saturacao_oxigenio", "critico"),
            ],
        )
        self.assertLess(leve.risco, grave.risco)
        self.assertLessEqual(grave.risco, 100.0)

    def test_risco_e_sempre_limitado_a_cem(self) -> None:
        muitos = [
            _achado("sinais_vitais", f"limite_critico_canal_{i}", "critico")
            for i in range(50)
        ]
        self.assertLessEqual(multimodal_fusion.fundir("PAC-0000", muitos).risco, 100.0)

    def test_instante_do_quadro_prefere_timestamp_absoluto(self) -> None:
        achados = [
            _achado("audio", "termo_critico_respiratorio", "critico", "00:03.00"),
            _achado(
                "sinais_vitais",
                "tendencia_queda_saturacao_oxigenio",
                instante="2025-06-10T17:20:00",
            ),
        ]
        sindrome = multimodal_fusion.identificar_sindromes(achados)[0]
        self.assertEqual(sindrome.como_achado().instante, "2025-06-10T17:20:00")


class MotorDeAlertasTests(unittest.TestCase):
    def test_achado_informativo_nao_gera_alerta(self) -> None:
        motor = MotorDeAlertas()
        alertas = motor.gerar(
            "PAC-0006", [_achado("sinais_vitais", "desvio_pontual_x", "informativo")]
        )
        self.assertEqual(alertas, [])

    def test_consolida_achados_da_mesma_familia_em_um_alerta(self) -> None:
        motor = MotorDeAlertas()
        achados = [
            _achado("sinais_vitais", "tendencia_alta_frequencia_cardiaca"),
            _achado("sinais_vitais", "tendencia_queda_saturacao_oxigenio"),
            _achado("sinais_vitais", "tendencia_alta_temperatura"),
        ]
        alertas = motor.gerar("PAC-0006", achados)
        self.assertEqual(len(alertas), 1)
        self.assertEqual(len(alertas[0].achados), 3)
        self.assertIn("3 achados", alertas[0].titulo)

    def test_familia_absorvida_nao_gera_alerta_proprio(self) -> None:
        motor = MotorDeAlertas()
        achados = [_achado("sinais_vitais", "tendencia_alta_temperatura")]
        self.assertEqual(
            motor.gerar("PAC-0006", achados, grupos_absorvidos=["tendencia_sinais_vitais"]),
            [],
        )

    def test_severidade_define_a_prioridade(self) -> None:
        motor = MotorDeAlertas()
        alertas = motor.gerar(
            "PAC-0006",
            [
                _achado("video", "invasao_area_critica", "critico"),
                _achado("movimentacao", "imobilidade_prolongada", "atencao"),
            ],
        )
        self.assertEqual(alertas[0].prioridade, "vermelho")
        self.assertEqual(alertas[1].prioridade, "laranja")

    def test_supressao_evita_repetir_o_mesmo_grupo(self) -> None:
        motor = MotorDeAlertas(janela_supressao=timedelta(minutes=15))
        agora = datetime(2025, 6, 10, 17, 0)
        achado = [_achado("sinais_vitais", "limite_critico_temperatura", "critico")]
        self.assertEqual(len(motor.gerar("PAC-0006", achado, agora=agora)), 1)
        self.assertEqual(
            len(motor.gerar("PAC-0006", achado, agora=agora + timedelta(minutes=5))), 0
        )
        self.assertEqual(
            len(motor.gerar("PAC-0006", achado, agora=agora + timedelta(minutes=20))), 1
        )

    def test_alerta_multimodal_avisa_todas_as_equipes_envolvidas(self) -> None:
        motor = MotorDeAlertas()
        achado = Achado(
            modalidade="multimodal",
            tipo="sindrome_deterioracao_respiratoria",
            descricao="quadro de teste",
            severidade="critico",
            score=0.9,
            evidencias={"modalidades": "audio, sinais_vitais", "acao": "avaliar agora"},
        )
        alerta = motor.gerar("PAC-0006", [achado])[0]
        self.assertIn("Medico assistente da consulta", alerta.destino)
        self.assertIn("Equipe de plantao do leito", alerta.destino)
        self.assertEqual(alerta.acao_recomendada, "avaliar agora")

    def test_severidade_invalida_e_rejeitada(self) -> None:
        with self.assertRaises(ValueError):
            Achado("video", "x", "desc", "urgentissimo", 0.5)

    def test_score_e_limitado_entre_zero_e_um(self) -> None:
        self.assertEqual(Achado("video", "x", "d", "atencao", 3.0).score, 1.0)
        self.assertEqual(Achado("video", "x", "d", "atencao", -2.0).score, 0.0)

    def test_agrupamento_usa_o_proprio_tipo_quando_nao_ha_familia(self) -> None:
        self.assertEqual(grupo_do_achado("profissional_ausente"), "profissional_ausente")
        self.assertEqual(
            grupo_do_achado("tendencia_alta_temperatura"), "tendencia_sinais_vitais"
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class PipelineTests(AmostrasTemporarias):
    def test_relatorio_registra_offline_mesmo_com_azure_disponivel(self) -> None:
        with patch("fase4.azure_services.azure_speech_configurada", return_value=True), \
             patch("fase4.azure_services.azure_language_configurada", return_value=True), \
             patch("fase4.azure_services._sdk_speech_disponivel", return_value=True), \
             patch("fase4.azure_services._sdk_language_disponivel", return_value=True):
            resultado = processar_paciente(
                "PAC-0002", raiz_amostras=self.raiz, provedor_azure="offline",
                gravar_relatorios=False, gravar_video_anotado=False,
            )
            texto = relatorio_markdown(resultado)
        self.assertIn("| speech_to_text | `offline` |", texto)
        self.assertIn("| text_analytics | `offline` |", texto)

    def test_localiza_as_amostras_disponiveis(self) -> None:
        amostras = localizar_amostras("PAC-0006", self.raiz)
        for chave in ("sinais_vitais", "prescricoes", "movimentacao", "audio", "keypoints"):
            self.assertIn(chave, amostras)
        self.assertNotIn("keypoints", localizar_amostras("PAC-0002", self.raiz))

    def test_caso_completo_gera_quadros_e_alertas(self) -> None:
        with tempfile.TemporaryDirectory() as saida:
            resultado = processar_paciente(
                "PAC-0006",
                raiz_amostras=self.raiz,
                dir_saida=Path(saida),
                provedor_azure="offline",
                gravar_video_anotado=False,
            )
            self.assertEqual(
                sorted(resultado.modalidades_processadas),
                ["audio", "movimentacao", "prescricoes", "sinais_vitais", "video"],
            )
            self.assertTrue(resultado.fusao.sindromes)
            self.assertTrue(resultado.alertas)
            self.assertEqual(resultado.fusao.nivel_risco, "alto")
            for nome in ("relatorio_monitoramento", "achados", "alertas"):
                self.assertTrue(Path(resultado.arquivos_gerados[nome]).exists())

    def test_fusao_reduz_o_numero_de_notificacoes(self) -> None:
        resultado = processar_paciente(
            "PAC-0006",
            raiz_amostras=self.raiz,
            provedor_azure="offline",
            gravar_relatorios=False,
            gravar_video_anotado=False,
        )
        relevantes = [a for a in resultado.achados if a.severidade != "informativo"]
        self.assertLess(len(resultado.alertas), len(relevantes))

    def test_paciente_estavel_nao_recebe_alerta(self) -> None:
        resultado = processar_paciente(
            "PAC-0002",
            raiz_amostras=self.raiz,
            provedor_azure="offline",
            gravar_relatorios=False,
            gravar_video_anotado=False,
        )
        self.assertEqual(resultado.alertas, [])
        self.assertEqual(resultado.fusao.sindromes, [])
        self.assertIn("video", resultado.modalidades_ausentes)

    def test_modalidade_ausente_nao_interrompe_o_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as vazio:
            resultado = processar_paciente(
                "PAC-9999",
                raiz_amostras=Path(vazio),
                provedor_azure="offline",
                gravar_relatorios=False,
                gravar_video_anotado=False,
            )
        self.assertEqual(resultado.modalidades_processadas, [])
        self.assertEqual(resultado.achados, [])
        self.assertEqual(resultado.alertas, [])
        self.assertEqual(resultado.fusao.nivel_risco, "sem_achados")
        self.assertEqual(resultado.provedores_utilizados()["speech_to_text"], "nao_executado")


if __name__ == "__main__":
    unittest.main()
