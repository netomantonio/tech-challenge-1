"""Isolation Forest no laco de tempo real e robustez a canal ausente / NaN."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from fase4 import anomaly_detection
from fase4.anomaly_detection import DetectorMultivariadoOnline
from fase4.config import LIMIARES_ANOMALIA
from fase4.data import gerar_dados_sinteticos as gerador


class AmostrasTemporarias(unittest.TestCase):
    diretorio: tempfile.TemporaryDirectory
    raiz: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.diretorio = tempfile.TemporaryDirectory()
        cls.raiz = Path(cls.diretorio.name)
        for paciente in gerador.PACIENTES_PADRAO:
            gerador.gerar_amostras_paciente(paciente, seed=42, destino_raiz=cls.raiz)
        cls.df_piora = anomaly_detection.carregar_sinais_vitais(cls.raiz / "PAC-0006" / "sinais_vitais.csv")
        cls.df_estavel = anomaly_detection.carregar_sinais_vitais(cls.raiz / "PAC-0002" / "sinais_vitais.csv")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.diretorio.cleanup()


def _alimentar(detector: DetectorMultivariadoOnline, df: pd.DataFrame):
    canais = [c for c in LIMIARES_ANOMALIA.canais if c in df.columns]
    achados = []
    for _, linha in df.iterrows():
        valores = {c: float(linha[c]) for c in canais}
        achados.extend(detector.atualizar(linha["timestamp"].to_pydatetime(), valores))
    return achados


class IsolationForestOnlineTests(AmostrasTemporarias):
    def test_reproduz_o_episodio_do_lote_com_atraso_de_confirmacao(self) -> None:
        lote, _ = anomaly_detection.detectar_isolation_forest(self.df_piora)
        self.assertEqual(len(lote), 1, "o PAC-0006 deve ter exatamente um episodio multivariado")

        detector = DetectorMultivariadoOnline()
        online = _alimentar(detector, self.df_piora)
        self.assertTrue(detector.pronto)
        self.assertEqual(len(online), 1)
        achado = online[0]
        self.assertEqual(achado.tipo, "anomalia_multivariada")
        self.assertEqual(achado.severidade, "critico")
        self.assertEqual(achado.evidencias["modo"], "tempo_real")

        inicio_lote = datetime.fromisoformat(lote[0].evidencias["inicio"])
        inicio_online = datetime.fromisoformat(achado.evidencias["inicio"])
        tolerancia = timedelta(minutes=LIMIARES_ANOMALIA.minutos_uniao_episodio)
        self.assertLessEqual(abs(inicio_online - inicio_lote), tolerancia)

        # O achado e emitido no minuto em que o episodio completa a duracao minima.
        fim_online = datetime.fromisoformat(achado.evidencias["fim"])
        self.assertEqual(achado.evidencias["minutos"], LIMIARES_ANOMALIA.minutos_minimos_episodio)
        self.assertEqual(
            fim_online - inicio_online,
            timedelta(minutes=LIMIARES_ANOMALIA.minutos_minimos_episodio - 1),
        )
        # Mesmo corte do lote (mesmo basal, mesma semente).
        self.assertAlmostEqual(achado.evidencias["corte_basal"], lote[0].evidencias["corte_basal"], places=3)
        # Os canais mais alterados coincidem.
        self.assertEqual(achado.descricao, lote[0].descricao)

    def test_paciente_estavel_nao_gera_episodio_online(self) -> None:
        detector = DetectorMultivariadoOnline()
        self.assertEqual(_alimentar(detector, self.df_estavel), [])
        self.assertTrue(detector.pronto)

    def test_nao_julga_antes_do_basal(self) -> None:
        detector = DetectorMultivariadoOnline()
        parcial = self.df_piora.iloc[: anomaly_detection.MINUTOS_REFERENCIA - 1]
        self.assertEqual(_alimentar(detector, parcial), [])
        self.assertFalse(detector.pronto)
        self.assertIsNone(detector.pontuar({c: 0.0 for c in LIMIARES_ANOMALIA.canais}))

    def test_basal_curto_demais_desliga_o_detector(self) -> None:
        detector = DetectorMultivariadoOnline(minutos_referencia=10)
        self.assertEqual(_alimentar(detector, self.df_piora), [])
        self.assertFalse(detector.pronto)

    def test_monitorar_em_tempo_real_inclui_o_multivariado(self) -> None:
        achados = [
            a for _, lista in anomaly_detection.monitorar_em_tempo_real(self.df_piora) for a in lista
        ]
        multivariados = [a for a in achados if a.tipo == "anomalia_multivariada"]
        self.assertEqual(len(multivariados), 1)
        univariados = [a for a in achados if a.tipo != "anomalia_multivariada"]
        self.assertEqual(
            [a.tipo for a in univariados],
            [a.tipo for a in anomaly_detection.avaliar_serie(self.df_piora)],
        )
        sem = [
            a for _, lista in anomaly_detection.monitorar_em_tempo_real(
                self.df_piora, incluir_multivariado=False
            ) for a in lista
        ]
        self.assertFalse(any(a.tipo == "anomalia_multivariada" for a in sem))

    def test_multivariado_antecipa_o_limite_clinico(self) -> None:
        instantes = {}
        for instante, lista in anomaly_detection.monitorar_em_tempo_real(self.df_piora):
            for achado in lista:
                if achado.tipo == "anomalia_multivariada":
                    instantes.setdefault("multivariado", instante)
                elif achado.tipo.startswith("limite_critico_"):
                    instantes.setdefault("limite", instante)
        self.assertIn("multivariado", instantes)
        self.assertIn("limite", instantes)
        self.assertLess(instantes["multivariado"], instantes["limite"])


class RobustezCanalAusenteENanTests(AmostrasTemporarias):
    def _df_com_buracos(self) -> pd.DataFrame:
        df = self.df_estavel.drop(columns=["temperatura"]).copy()
        for indice in (5, 60, 119, 300):
            df.loc[indice, "frequencia_cardiaca"] = np.nan
        df.loc[200, ["pressao_sistolica", "pressao_diastolica"]] = np.nan
        return df

    def test_avaliar_serie_ignora_nan_e_canal_ausente(self) -> None:
        achados = anomaly_detection.avaliar_serie(self._df_com_buracos())
        self.assertEqual([a for a in achados if a.severidade != "informativo"], [])
        self.assertFalse(any("temperatura" in a.tipo for a in achados))

    def test_isolation_forest_nao_quebra_com_nan(self) -> None:
        df = self._df_com_buracos()
        achados, enriquecido = anomaly_detection.detectar_isolation_forest(df)
        self.assertEqual(achados, [])
        self.assertIn("score_isolation_forest", enriquecido.columns)
        self.assertFalse(enriquecido["score_isolation_forest"].isna().any())
        # A serie original gravada continua com o buraco: nada foi inventado.
        self.assertTrue(np.isnan(enriquecido.loc[5, "frequencia_cardiaca"]))

    def test_tempo_real_nao_quebra_com_nan(self) -> None:
        df = self._df_com_buracos()
        achados = [a for _, lista in anomaly_detection.monitorar_em_tempo_real(df) for a in lista]
        self.assertEqual([a for a in achados if a.severidade != "informativo"], [])

    def test_detector_univariado_descarta_nan_sem_contar_como_recuperacao(self) -> None:
        detector = anomaly_detection.DetectorSinaisVitais()
        instante = datetime(2025, 6, 10, 8, 0)
        for i in range(130):
            valores = {"frequencia_cardiaca": 80.0 if i % 7 else float("nan")}
            self.assertEqual(detector.atualizar(instante + timedelta(minutes=i), valores), [])
        # O historico so tem leituras validas.
        historico = detector._estados["frequencia_cardiaca"].historico
        self.assertFalse(any(np.isnan(v) for v in historico))


if __name__ == "__main__":
    unittest.main()
