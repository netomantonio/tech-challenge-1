"""Testes do catalogo portatil de modelos-base."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fase3 import model_registry


class ModelRegistryTests(unittest.TestCase):
    def test_modelo_local_precisa_ficar_em_raiz_permitida(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            allowed = root / "models"
            allowed.mkdir()
            model_dir = allowed / "clinical-model"
            model_dir.mkdir()
            registry = root / "registry.json"
            with (
                patch.object(model_registry, "REGISTRY_PATH", registry),
                patch.dict("os.environ", {"FASE3_MODEL_ROOTS": str(allowed)}),
            ):
                model = model_registry.register_model(
                    {
                        "alias": "clinical-model",
                        "label": "Clinical model",
                        "source_type": "local",
                        "source": str(model_dir),
                    }
                )
                self.assertEqual(model_registry.get_model("clinical-model")["source"], str(model_dir))
                self.assertTrue(model_registry.model_installed(model))

    def test_rejeita_repo_id_hugging_face_invalido(self) -> None:
        with self.assertRaises(ValueError):
            model_registry.register_model(
                {
                    "alias": "modelo-invalido",
                    "label": "Modelo invalido",
                    "source_type": "huggingface",
                    "source": "sem-organizacao",
                }
            )


if __name__ == "__main__":
    unittest.main()
