"""Testes do orquestrador de treinamento exposto pela interface local."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fase3 import training_service
from fase3.evaluation_contract import current_evaluation_contract
from fase3.training_service import TrainingJobManager, resolver_adapter


class TrainingServiceTests(unittest.TestCase):
    def test_resolver_adapter_rejeita_versao_fora_do_padrao(self) -> None:
        with self.assertRaises(ValueError):
            resolver_adapter("../../modelo")

    def test_overview_encontra_dataset_e_adapters_versionados(self) -> None:
        overview = TrainingJobManager().overview()

        self.assertEqual(overview["dataset"]["train"], 40)
        self.assertEqual(overview["dataset"]["validation"], 8)
        self.assertTrue(overview["dataset"]["ready"])
        self.assertTrue(any(item["version"] == "qwen2.5-1.5b-v4" for item in overview["adapters"]))

    def test_treino_monta_comando_controlado_sem_sobrescrever(self) -> None:
        manager = TrainingJobManager()
        config = {
            "version": "qwen2.5-1.5b-v9999",
            "epochs": 6,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-5,
            "max_length": 512,
            "seed": 42,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
        }

        with patch.object(manager, "_iniciar", return_value={"status": "aguardando"}) as iniciar:
            result = manager.iniciar_treino(config)

        command = iniciar.call_args.args[2]
        self.assertEqual(result["status"], "aguardando")
        self.assertIn("fase3.finetuning.train_lora", command)
        self.assertIn("--gradient-accumulation-steps", command)
        self.assertIn("--lora-dropout", command)

        with self.assertRaises(FileExistsError):
            manager.iniciar_treino({**config, "version": "qwen2.5-1.5b-v4"})

    def test_treino_aceita_modelo_customizado_e_precisao_nf4(self) -> None:
        manager = TrainingJobManager()
        config = {
            "version": "modelo-clinico-v99",
            "model_alias": "modelo-clinico",
            "precision": "nf4",
            "gradient_checkpointing": True,
            "epochs": 1,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 2e-5,
            "max_length": 384,
            "seed": 42,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
        }
        model = {
            "alias": "modelo-clinico",
            "label": "Modelo clinico",
            "source": "organizacao/modelo-clinico",
            "source_type": "huggingface",
            "revision": "abc123",
            "target_modules": [],
        }

        with (
            patch.object(training_service, "get_model", return_value=model),
            patch.object(manager, "_iniciar", return_value={"status": "aguardando"}) as iniciar,
        ):
            manager.iniciar_treino(config)

        command = iniciar.call_args.args[2]
        self.assertIn("organizacao/modelo-clinico", command)
        self.assertEqual(command[command.index("--precision") + 1], "nf4")
        self.assertIn("--gradient-checkpointing", command)
        self.assertIn("abc123", command)

    def test_promocao_exige_calibracao_aprovada_do_mesmo_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            finetuning = root / "finetuning"
            adapter = finetuning / "qwen2.5-1.5b-v1" / "lora_adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            calibration = root / "calibracao.json"
            promoted = root / "promoted.json"
            calibration.write_text(
                json.dumps(
                    {
                        "evaluation_contract": current_evaluation_contract(),
                        "selecionado": {
                            "adapter_path": str(adapter),
                            "lora_scale": 0.75,
                            "aprovado": True,
                        }
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch.object(training_service, "FINETUNING_DIR", finetuning),
                patch.object(training_service, "CALIBRATION_PATH", calibration),
                patch.object(training_service, "PROMOTED_MODEL_CONFIG_PATH", promoted),
            ):
                config = TrainingJobManager().promover("qwen2.5-1.5b-v1")

            self.assertEqual(config["lora_scale"], 0.75)
            self.assertTrue(promoted.exists())

    def test_promocao_rejeita_calibracao_de_contrato_antigo(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            finetuning = root / "finetuning"
            adapter = finetuning / "qwen2.5-1.5b-v1" / "lora_adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            calibration = root / "calibracao.json"
            calibration.write_text(
                json.dumps(
                    {
                        "selecionado": {
                            "adapter_path": str(adapter),
                            "lora_scale": 0.75,
                            "aprovado": True,
                        }
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch.object(training_service, "FINETUNING_DIR", finetuning),
                patch.object(training_service, "CALIBRATION_PATH", calibration),
            ):
                with self.assertRaisesRegex(ValueError, "calibracao esta desatualizada"):
                    TrainingJobManager().promover("qwen2.5-1.5b-v1")


if __name__ == "__main__":
    unittest.main()
