"""Testes do servico HTTP e da interface local da Fase 3."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from fase3.web_app import AssistantRuntime, create_app


class FakeTrainingManager:
    def __init__(self, active: bool = False) -> None:
        self.ativo = active
        self.started: tuple[str, object] | None = None

    def job(self):
        return None

    def overview(self):
        return {
            "dataset": {"train": 40, "validation": 8, "ready": True},
            "adapters": [],
            "next_version": "qwen2.5-1.5b-v5",
            "promoted": {
                "adapter_path": "resultados/fase3/finetuning/qwen2.5-1.5b-v4/lora_adapter",
                "lora_scale": 0.75,
            },
            "latest_calibration": None,
            "job": None,
        }

    def iniciar_treino(self, config):
        self.started = ("treino", config)
        return {"id": "job-1", "status": "aguardando"}


class Fase3WebTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = AssistantRuntime("fake")
        self.client = TestClient(create_app(self.runtime))

    def test_interface_e_status_estao_disponiveis(self) -> None:
        page = self.client.get("/")
        favicon = self.client.get("/favicon.ico")
        status = self.client.get("/api/status")

        self.assertEqual(page.status_code, 200)
        self.assertEqual(favicon.status_code, 200)
        self.assertEqual(favicon.headers["content-type"], "image/svg+xml")
        self.assertIn("Assistente de Protocolos Clinicos", page.text)
        self.assertIn("Operacoes do modelo", page.text)
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()["backend"], "fake")
        self.assertFalse(status.json()["modelo_carregado"])

    def test_runtime_web_padrao_informa_modelo_local(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            client = TestClient(create_app(AssistantRuntime()))
            status = client.get("/api/status").json()

        self.assertEqual(status["backend"], "local")
        self.assertEqual(status["modelo_base"], "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertIn("qwen2.5-1.5b-v4", status["adapter"])
        self.assertEqual(status["escala_lora"], 0.75)

    def test_lista_somente_prontuarios_sinteticos(self) -> None:
        response = self.client.get("/api/pacientes")

        self.assertEqual(response.status_code, 200)
        pacientes = response.json()
        self.assertEqual(len(pacientes), 6)
        self.assertTrue(all(item["paciente_id"].startswith("PAC-") for item in pacientes))
        self.assertTrue(all("nome" not in item for item in pacientes))

    def test_consulta_serializa_evidencias_e_decisao(self) -> None:
        estado = {
            "paciente_encontrado": True,
            "paciente": {"paciente_id": "PAC-0001"},
            "sugestao": {
                "resposta": "Resposta fundamentada. Fonte: [PROT-006].",
                "fontes": [{"id": "PROT-006", "titulo": "Checklist"}],
                "modo_resposta": "llm",
                "motivo_bloqueio": None,
            },
            "bloqueado": False,
            "exames_pendentes": ["ecocardiograma_basal"],
            "tem_exames_pendentes": True,
            "rota_exames": "com_pendencias",
            "alertas": ["Exame pendente"],
            "etapas_executadas": [
                "buscar_paciente",
                "verificar_exames_pendentes",
                "alertar_exames_pendentes",
                "sugerir_tratamento",
                "checar_seguranca",
                "emitir_alertas",
                "registrar_auditoria",
            ],
        }
        with patch.object(self.runtime, "consultar", return_value=estado):
            response = self.client.post(
                "/api/consultas",
                json={
                    "paciente_id": "PAC-0001",
                    "pergunta": "Posso iniciar a quimioterapia hoje?",
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["modo_resposta"], "llm")
        self.assertEqual(body["rota_exames"], "com_pendencias")
        self.assertEqual(body["fontes"][0]["id"], "PROT-006")
        self.assertEqual(body["alertas"], ["Exame pendente"])
        self.assertEqual(body["etapas_executadas"][-1], "registrar_auditoria")

    def test_rejeita_entrada_invalida(self) -> None:
        response = self.client.post(
            "/api/consultas",
            json={"paciente_id": "PAC 0001", "pergunta": "x"},
        )
        self.assertEqual(response.status_code, 422)

    def test_painel_inicia_treino_local_e_descarrega_inferencia(self) -> None:
        runtime = AssistantRuntime("local")
        jobs = FakeTrainingManager()
        client = TestClient(create_app(runtime, jobs))
        payload = {
            "version": "qwen2.5-1.5b-v5",
            "epochs": 6,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 0.00002,
            "max_length": 512,
            "seed": 42,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
        }

        with patch.object(runtime, "descarregar") as descarregar:
            overview = client.get("/api/treinamento")
            response = client.post("/api/treinamento/iniciar", json=payload)

        self.assertEqual(overview.status_code, 200)
        self.assertEqual(overview.json()["dataset"]["train"], 40)
        self.assertEqual(response.status_code, 200)
        descarregar.assert_called_once()
        self.assertEqual(jobs.started, ("treino", payload))

    def test_consulta_fica_bloqueada_durante_job_de_gpu(self) -> None:
        runtime = AssistantRuntime("local")
        client = TestClient(create_app(runtime, FakeTrainingManager(active=True)))

        response = client.post(
            "/api/consultas",
            json={"paciente_id": "PAC-0001", "pergunta": "Qual a conduta segura?"},
        )

        self.assertEqual(response.status_code, 409)
        self.assertIn("operacao de modelo", response.json()["detail"])

    def test_operacoes_de_modelo_exigem_backend_local(self) -> None:
        response = self.client.get("/api/treinamento")

        self.assertEqual(response.status_code, 409)


if __name__ == "__main__":
    unittest.main()
