"""Conversao de registros do PhysioNet para o esquema da Fase 4, sem rede."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fase4.data import baixar_physionet as pn
from fase4.data import gerar_dados_sinteticos as gerador
from fase4.validacao_physionet import validar_registro

try:
    import wfdb  # noqa: F401
    TEM_WFDB = True
except ImportError:  # pragma: no cover
    TEM_WFDB = False

HEADER_EXEMPLO = """p044083-2112-05-04-19-50n 18 1 164919 19:50:40 04/05/2112
3314767n.dat 16 1/bpm 0 0 0 0 0 HR
3314767n.dat 16 1/bpm 0 0 0 0 0 PULSE
3314767n.dat 16 1/mmHg 0 0 0 0 0 ABP Sys
3314767n.dat 16 1/mmHg 0 0 0 0 0 ABP Dias
3314767n.dat 16 1/pm 0 0 0 0 0 RESP
3314767n.dat 16 1/% 0 0 0 0 0 SpO2
# comentario ignorado
"""


def _df_1hz(segundos: int = 180, inicio: str = "2112-05-04 19:50:00") -> pd.DataFrame:
    indice = pd.date_range(inicio, periods=segundos, freq="s")
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "HR": rng.normal(80, 1, segundos),
            "PULSE": rng.normal(80, 1, segundos),
            "ABP Sys": rng.normal(120, 2, segundos),
            "ABP Dias": rng.normal(70, 2, segundos),
            "NBP Sys": np.nan,
            "SpO2": rng.normal(97, 0.3, segundos),
            "RESP": rng.normal(16, 0.5, segundos),
        },
        index=indice,
    )
    df.loc[df.index[10], "HR"] = 250.0  # artefato: a mediana do minuto deve ignora-lo
    df.loc[df.index[70:75], "SpO2"] = np.nan  # buraco curto dentro do segundo minuto
    return df


class ResolverRegistroTests(unittest.TestCase):
    def test_deriva_a_pasta_do_nome(self) -> None:
        self.assertEqual(
            pn.resolver_registro("p044083-2112-05-04-19-50n"),
            ("p04/p044083", "p044083-2112-05-04-19-50n"),
        )
        self.assertEqual(
            pn.resolver_registro("p04/p044083/p044083-2112-05-04-19-50n"),
            ("p04/p044083", "p044083-2112-05-04-19-50n"),
        )
        with self.assertRaises(ValueError):
            pn.resolver_registro("registro-estranho")

    def test_arquivos_do_header(self) -> None:
        self.assertEqual(pn.arquivos_do_header(HEADER_EXEMPLO), ["3314767n.dat"])
        self.assertEqual(pn.arquivos_do_header("so-cabecalho 0 1 0\n"), [])


class ConversaoParaSchemaTests(unittest.TestCase):
    def test_reamostra_renomeia_e_limpa(self) -> None:
        tabela, info = pn.converter_para_schema(_df_1hz())
        self.assertEqual(
            list(tabela.columns),
            ["timestamp", "frequencia_cardiaca", "pressao_sistolica", "pressao_diastolica",
             "saturacao_oxigenio", "frequencia_respiratoria"],
        )
        self.assertEqual(info["canais_ausentes"], ["temperatura"])
        self.assertEqual(info["canais"]["pressao_sistolica"]["fonte"], "ABP Sys")
        self.assertEqual(info["canais"]["frequencia_cardiaca"]["fonte"], "HR")
        self.assertEqual(len(tabela), 3)
        self.assertTrue(tabela["timestamp"].str.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$").all())
        self.assertEqual(tabela["timestamp"].iloc[0], "2112-05-04T19:50:00")
        # A mediana por minuto nao e puxada pelo artefato de 250 bpm.
        self.assertLess(tabela["frequencia_cardiaca"].iloc[0], 90)
        self.assertFalse(tabela.isna().any().any())
        self.assertIsInstance(tabela.index, pd.RangeIndex)
        self.assertEqual(info["minutos_exportados"], 3)

    def test_fallback_para_nbp_quando_abp_nao_existe(self) -> None:
        df = _df_1hz(segundos=600)
        df["ABP Sys"] = np.nan
        df["NBP Sys"] = np.nan
        # Pressao nao invasiva: uma medida a cada 5 minutos.
        for i in range(0, 600, 300):
            df.loc[df.index[i], "NBP Sys"] = 118.0 + i / 300
        tabela, info = pn.converter_para_schema(df)
        self.assertEqual(info["canais"]["pressao_sistolica"]["fonte"], "NBP Sys")
        self.assertEqual(info["canais"]["pressao_sistolica"]["ffill_max_min"], pn.LIMITE_FFILL_NBP_MIN)
        self.assertEqual(len(tabela), 10)  # os buracos foram preenchidos pelo ffill longo
        self.assertEqual(tabela["pressao_sistolica"].iloc[4], 118.0)
        self.assertEqual(tabela["pressao_sistolica"].iloc[5], 119.0)

    def test_canal_sem_fonte_e_listado_como_ausente(self) -> None:
        df = _df_1hz().drop(columns=["RESP"])
        tabela, info = pn.converter_para_schema(df)
        self.assertNotIn("frequencia_respiratoria", tabela.columns)
        self.assertIn("frequencia_respiratoria", info["canais_ausentes"])

    def test_buraco_longo_descarta_os_minutos(self) -> None:
        df = _df_1hz(segundos=1200)  # 20 min
        df.loc[df.index[300:900], "HR"] = np.nan  # 10 min sem FC
        df.loc[df.index[300:900], "PULSE"] = np.nan
        tabela, info = pn.converter_para_schema(df)
        self.assertEqual(info["minutos_totais"], 20)
        self.assertEqual(info["minutos_exportados"], 20 - 10 + pn.LIMITE_FFILL_MIN)
        self.assertEqual(info["minutos_descartados"], 10 - pn.LIMITE_FFILL_MIN)

    def test_valor_impossivel_vira_nan(self) -> None:
        df = _df_1hz()
        df.loc[df.index[60:120], "SpO2"] = 0.0  # sensor desconectado grava zero
        tabela, info = pn.converter_para_schema(df)
        self.assertEqual(len(tabela), 2)  # o minuto com SpO2 = 0 foi descartado
        self.assertEqual(info["minutos_descartados"], 1)

    def test_exige_indice_datetime(self) -> None:
        with self.assertRaises(ValueError):
            pn.converter_para_schema(pd.DataFrame({"HR": [1.0, 2.0]}))

    def test_inicio_fora_do_minuto_cheio_gera_minuto_parcial(self) -> None:
        # O registro real comeca as 19:50:40: o primeiro minuto tem 20 amostras.
        tabela, info = pn.converter_para_schema(_df_1hz(inicio="2112-05-04 19:50:40"))
        self.assertEqual(info["minutos_totais"], 4)
        self.assertEqual(tabela["timestamp"].iloc[0], "2112-05-04T19:50:00")


class ValidacaoSemRedeTests(unittest.TestCase):
    """O runner de validacao roda sobre um CSV convertido a partir da amostra sintetica."""

    def test_gera_relatorio_e_arquivos(self) -> None:
        with tempfile.TemporaryDirectory() as temporario:
            raiz = Path(temporario)
            gerador.gerar_amostras_paciente("PAC-0002", seed=42, destino_raiz=raiz / "amostras")
            dados = raiz / "dados" / "registro-teste"
            dados.mkdir(parents=True)
            df = pd.read_csv(raiz / "amostras" / "PAC-0002" / "sinais_vitais.csv")
            df.drop(columns=["temperatura"]).to_csv(dados / "sinais_vitais.csv", index=False)
            (dados / "fonte.json").write_text(json.dumps({
                "dataset": "teste", "slug": "t", "versao": "0", "registro": "registro-teste",
                "canais": {"frequencia_cardiaca": {"fonte": "HR", "unidade": "bpm", "ffill_max_min": 5}},
                "canais_ausentes": ["temperatura"], "fs_hz": 1, "amostras_lidas": 43200,
                "minutos_exportados": 720, "minutos_descartados": 0,
            }), encoding="utf-8")

            resumo = validar_registro("registro-teste", dir_dados=raiz / "dados", dir_saida=raiz / "saida")
            saida = raiz / "saida" / "registro-teste"
            for nome in ("relatorio_physionet.md", "achados.json", "achados.csv", "alertas.json", "sinais_vitais_scores.csv"):
                self.assertTrue((saida / nome).exists(), nome)
            relatorio = (saida / "relatorio_physionet.md").read_text(encoding="utf-8")
            self.assertIn("## Fonte", relatorio)
            self.assertIn("| temperatura | ausente", relatorio)
            self.assertIn("Episódios multivariados", relatorio)
            self.assertIn("não mede sensibilidade", relatorio)
        self.assertEqual(resumo["minutos"], 720)
        self.assertEqual(resumo["alertas"], 0)  # controle estavel continua sem alerta sem a temperatura
        self.assertEqual(resumo["episodios_lote"], 0)
        self.assertEqual(resumo["episodios_online"], 0)


@unittest.skipUnless(TEM_WFDB, "wfdb nao instalado")
class LeituraWfdbTests(unittest.TestCase):
    def test_round_trip_com_wfdb(self) -> None:
        import wfdb

        with tempfile.TemporaryDirectory() as pasta:
            sinais = np.column_stack([
                np.full(120, 80.0), np.full(120, 120.0), np.full(120, 70.0), np.full(120, 97.0), np.full(120, 16.0),
            ])
            wfdb.wrsamp(
                "mini", fs=1, units=["bpm", "mmHg", "mmHg", "%", "pm"],
                sig_name=["HR", "ABP Sys", "ABP Dias", "SpO2", "RESP"],
                p_signal=sinais, fmt=["16"] * 5, write_dir=pasta,
            )
            df, meta = pn.ler_registro(Path(pasta) / "mini", minutos=1)
        self.assertEqual(list(df.columns), ["HR", "ABP Sys", "ABP Dias", "SpO2", "RESP"])
        self.assertEqual(len(df), 60)
        self.assertEqual(meta["fs_hz"], 1.0)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        tabela, _ = pn.converter_para_schema(df)
        self.assertEqual(len(tabela), 1)
        self.assertEqual(tabela["frequencia_cardiaca"].iloc[0], 80.0)


if __name__ == "__main__":
    unittest.main()
