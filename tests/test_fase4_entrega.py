"""Canais de entrega dos alertas: fila JSONL, webhook HTTP e integracao."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from fase4 import anomaly_detection
from fase4.alertas import (
    Achado,
    FilaJsonl,
    MotorDeAlertas,
    WebhookHttp,
    canais_padrao,
    descrever_canais,
)
from fase4.data import gerar_dados_sinteticos as gerador
from fase4.pipeline import processar_paciente
from fase4.receptor_webhook_demo import ReceptorAlertas, criar_servidor


def _achados() -> list[Achado]:
    return [
        Achado("sinais_vitais", "limite_critico_temperatura", "Temperatura acima do limite", "critico", 0.9,
               instante="2025-06-10T18:03:00"),
        Achado("video", "profissional_ausente", "Sem profissional em cena", "atencao", 0.6, instante="00:10.00"),
        Achado("prescricao", "salto_de_dose", "Aumento de 400% na dose", "critico", 0.9, instante="14:05"),
        Achado("audio", "sentimento_negativo", "Sentimento negativo", "informativo", 0.7),
    ]


class FilaJsonlTests(unittest.TestCase):
    def test_grava_um_alerta_por_linha_e_marca_o_canal(self) -> None:
        with tempfile.TemporaryDirectory() as pasta:
            fila = FilaJsonl(Path(pasta) / "sub" / "fila.jsonl")
            motor = MotorDeAlertas(canais=[fila])
            alertas = motor.gerar("PAC-X", _achados(), agora=datetime(2025, 6, 10, tzinfo=timezone.utc))
            self.assertEqual(len(alertas), 3)  # o informativo nao vira alerta
            linhas = fila.caminho.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(linhas), 3)
        registros = [json.loads(l) for l in linhas]
        self.assertEqual([r["alerta_id"] for r in registros], [a.alerta_id for a in alertas])
        self.assertTrue(all("entregue_em" in r for r in registros))
        self.assertTrue(all(r["canais_entrega"] == ["fila_jsonl"] for r in registros))
        self.assertTrue(all(a.canais_entrega == ["fila_jsonl"] for a in alertas))
        self.assertEqual(motor.entregas_falhas, 0)

    def test_sem_canal_o_motor_continua_igual(self) -> None:
        alertas = MotorDeAlertas().gerar("PAC-X", _achados())
        self.assertEqual(len(alertas), 3)
        self.assertTrue(all(a.canais_entrega == [] for a in alertas))


class WebhookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pasta = tempfile.TemporaryDirectory()
        self.arquivo = Path(self.pasta.name) / "recebidos.jsonl"
        self.servidor = criar_servidor("127.0.0.1", 0, self.arquivo, silencioso=True)
        self.thread = threading.Thread(target=self.servidor.serve_forever, daemon=True)
        self.thread.start()
        host, porta = self.servidor.server_address[:2]
        self.url = f"http://{host}:{porta}/alertas"

    def tearDown(self) -> None:
        self.servidor.shutdown()
        self.servidor.server_close()
        self.pasta.cleanup()

    def test_posta_cada_alerta_e_o_receptor_registra(self) -> None:
        motor = MotorDeAlertas(canais=[WebhookHttp(self.url)])
        alertas = motor.gerar("PAC-X", _achados())
        self.assertEqual(ReceptorAlertas.recebidos, 3)
        recebidos = [json.loads(l) for l in self.arquivo.read_text(encoding="utf-8").splitlines()]
        self.assertEqual({r["alerta_id"] for r in recebidos}, {a.alerta_id for a in alertas})
        self.assertTrue(all(a.canais_entrega == ["webhook"] for a in alertas))
        self.assertEqual(motor.entregas_falhas, 0)

    def test_falha_do_webhook_nao_derruba_o_motor(self) -> None:
        quebrado = WebhookHttp("http://127.0.0.1:9/alertas", timeout_s=0.5)
        with tempfile.TemporaryDirectory() as pasta:
            fila = FilaJsonl(Path(pasta) / "fila.jsonl")
            motor = MotorDeAlertas(canais=[quebrado, fila])
            with patch("fase4.alertas.registrar_evento") as auditoria:
                alertas = motor.gerar("PAC-X", _achados())
            self.assertEqual(len(alertas), 3)
            self.assertEqual(motor.entregas_falhas, 3)
            self.assertTrue(all(a.canais_entrega == ["fila_jsonl"] for a in alertas))
            eventos = [c.args[0] for c in auditoria.call_args_list]
        self.assertEqual(eventos.count("entrega_falhou"), 3)
        self.assertEqual(eventos.count("alerta_entregue"), 3)

    def test_canais_padrao_lem_as_variaveis_de_ambiente(self) -> None:
        with tempfile.TemporaryDirectory() as pasta:
            with patch.dict("os.environ", {"FASE4_WEBHOOK_URL": self.url}, clear=False):
                canais = canais_padrao(pasta)
            self.assertEqual([type(c).__name__ for c in canais], ["FilaJsonl", "WebhookHttp"])
            self.assertEqual(canais[0].caminho, Path(pasta) / "fila_alertas.jsonl")
            self.assertEqual(canais[1].url, self.url)
            descricoes = descrever_canais(canais)
            self.assertIn("fila JSONL", descricoes[0])
            self.assertIn(self.url, descricoes[1])

            outra = Path(pasta) / "outra.jsonl"
            with patch.dict("os.environ", {"FASE4_FILA_ALERTAS": str(outra), "FASE4_WEBHOOK_URL": ""}):
                canais = canais_padrao(pasta)
            self.assertEqual(len(canais), 1)
            self.assertEqual(canais[0].caminho, outra)


class IntegracaoComPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.diretorio = tempfile.TemporaryDirectory()
        cls.raiz = Path(cls.diretorio.name)
        gerador.gerar_amostras_paciente("PAC-0006", seed=42, destino_raiz=cls.raiz)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.diretorio.cleanup()

    def test_pipeline_grava_a_fila_na_raiz_da_execucao(self) -> None:
        with tempfile.TemporaryDirectory() as saida, patch.dict("os.environ", {"FASE4_WEBHOOK_URL": ""}):
            resultado = processar_paciente(
                "PAC-0006", raiz_amostras=self.raiz, dir_saida=Path(saida),
                provedor_azure="offline", gravar_video_anotado=False,
            )
            fila = Path(saida) / "fila_alertas.jsonl"
            self.assertTrue(fila.exists())
            linhas = fila.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(linhas), len(resultado.alertas))
            self.assertTrue(all(a.canais_entrega == ["fila_jsonl"] for a in resultado.alertas))
            relatorio = (Path(saida) / "PAC-0006" / "relatorio_monitoramento.md").read_text(encoding="utf-8")
            self.assertIn("Entregue via", relatorio)
            self.assertIn("fila_jsonl", relatorio)

    def test_sem_gravar_relatorios_nao_ha_fila(self) -> None:
        with tempfile.TemporaryDirectory() as saida:
            resultado = processar_paciente(
                "PAC-0006", raiz_amostras=self.raiz, dir_saida=Path(saida),
                provedor_azure="offline", gravar_relatorios=False, gravar_video_anotado=False,
            )
            self.assertFalse((Path(saida) / "fila_alertas.jsonl").exists())
            self.assertTrue(all(a.canais_entrega == [] for a in resultado.alertas))

    def test_tempo_real_entrega_cada_alerta_na_fila(self) -> None:
        df = anomaly_detection.carregar_sinais_vitais(self.raiz / "PAC-0006" / "sinais_vitais.csv")
        with tempfile.TemporaryDirectory() as pasta:
            fila = FilaJsonl(Path(pasta) / "fila.jsonl")
            motor = MotorDeAlertas(canais=[fila])
            total = 0
            for instante, achados in anomaly_detection.monitorar_em_tempo_real(df):
                total += len(motor.gerar("PAC-0006", achados, agora=instante))
            linhas = fila.caminho.read_text(encoding="utf-8").splitlines()
        self.assertGreaterEqual(total, 4)
        self.assertEqual(len(linhas), total)


if __name__ == "__main__":
    unittest.main()
