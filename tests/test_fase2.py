from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from src.genetic_optimization import (
    GENE_SPACES,
    QUICK_EXPERIMENTS,
    GAConfig,
    GeneticOptimizer,
    load_cancer_data,
    run_all_experiments,
)
from src.llm_interpretation import (
    FeatureEvidence,
    LLMInterpretation,
    ModelResult,
    build_interpretation_prompt,
    derive_actionable_insights,
    evaluate_interpretation_quality,
    generate_interpretation,
)


DATA_PATH = Path("data/cancer_mama.csv")
FAKE_INTERPRETATION = """**RESUMO DO RESULTADO**
O resultado do modelo indica classificação estimada Maligno, com probabilidade de malignidade de 98,00%.

**EVIDÊNCIAS DO MODELO**
As evidências numéricas fornecidas direcionam a classificação estimada para Maligno.

**INSIGHTS ACIONÁVEIS PARA MÉDICOS**
Recomenda-se revisão clínica por profissional e confirmação conforme protocolo local.
Sugere-se agendamento de exames complementares como mamografia e encaminhamento
para mastologista, respeitando a realidade de acesso da paciente ao sistema de saúde.

**LIMITAÇÕES E SEGURANÇA**
Esta explicação não constitui diagnóstico médico; o dataset é acadêmico e sem validação externa."""


class FakeChatCompletions:
    def create(self, **kwargs):
        self.kwargs = kwargs
        message = type("FakeMessage", (), {"content": FAKE_INTERPRETATION})()
        choice = type("FakeChoice", (), {"message": message})()
        return type("FakeResponse", (), {"choices": [choice]})()


class FakeChat:
    def __init__(self) -> None:
        self.completions = FakeChatCompletions()


class FakeGroqClient:
    def __init__(self) -> None:
        self.chat = FakeChat()


class PhaseTwoContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.output_path = Path(cls.temporary_directory.name)
        cls.features, cls.target = load_cancer_data(DATA_PATH)
        cls.result = run_all_experiments(
            DATA_PATH, cls.output_path, experiment_configs=QUICK_EXPERIMENTS
        )

        import src.api as api

        api.MODEL_PATH = cls.result["artifact_path"]
        api.initialize_model()
        cls.api = api

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_directory.cleanup()

    def test_genes_decode_only_valid_alleles(self) -> None:
        config = GAConfig(
            name="test_mutation",
            population_size=4,
            generations=1,
            crossover_rate=1.0,
            mutation_rate=1.0,
        )
        logger = logging.getLogger("test_ga")
        optimizer = GeneticOptimizer(
            "decision_tree", config, self.features, self.target, logger, cv_splits=2
        )
        mutated = optimizer.mutate(optimizer.random_individual())
        decoded = optimizer.decode(mutated)
        for gene, value in decoded.items():
            self.assertIn(value, GENE_SPACES["decision_tree"][gene])

    def test_smoke_run_persists_model_and_comparison(self) -> None:
        self.assertTrue(self.result["artifact_path"].exists())
        self.assertEqual(self.result["artifact_path"].name, "modelo_serving.joblib")
        self.assertEqual(len(self.result["comparison"]), 6)
        self.assertIn("recall_maligno", self.result["comparison"].columns)

    def test_api_serves_the_recommended_baseline_when_it_outperforms_ga(self) -> None:
        serving_model = self.result["summary"]["serving_model"]
        self.assertEqual(serving_model["source"], "baseline_fase_1")
        self.assertEqual(serving_model["model"], "Regressão Logística")

    def test_api_predicts_using_persisted_artifact(self) -> None:
        payload = self.api.PredictRequest(features=self.features.iloc[0].to_dict())
        response = asyncio.run(self.api.predict(payload))
        self.assertIn(response.diagnosis, {"Maligno", "Benigno"})
        self.assertAlmostEqual(
            response.probability_malignant + response.probability_benign, 1.0, places=6
        )

    def test_api_rejects_incomplete_features(self) -> None:
        payload = self.api.PredictRequest(features={"radius_mean": 10.0})
        with self.assertRaises(HTTPException) as context:
            asyncio.run(self.api.predict(payload))
        self.assertEqual(context.exception.status_code, 422)

    def test_prompt_and_quality_check_are_clinically_constrained(self) -> None:
        result = ModelResult(0, "Maligno", 0.98, 0.02, "Regressão Logística")
        evidence = [FeatureEvidence("radius_worst", 25.0, -2.1, "Maligno")]
        prompt = build_interpretation_prompt(result, evidence)
        self.assertIn("Probabilidade estimada de malignidade: 98.00%", prompt)
        self.assertIn("INSIGHTS ACIONAVEIS PARA MEDICOS", prompt)
        interpretation = asyncio.run(
            generate_interpretation(
                result,
                evidence,
                client=FakeGroqClient(),
                model_name="gpt-test",
            )
        )
        quality = evaluate_interpretation_quality(interpretation.explanation, "Maligno")
        self.assertEqual(interpretation.llm_model, "gpt-test")
        self.assertEqual(quality["score_objetivo"], 1.0)
        self.assertEqual(len(interpretation.insights_acionaveis), 1)

    def test_actionable_insights_translate_numeric_evidence(self) -> None:
        result = ModelResult(0, "Maligno", 0.76, 0.24, "Regressão Logística")
        evidence = [FeatureEvidence("area_worst", 1200.0, -2.4, "Maligno")]
        insights = derive_actionable_insights(result, evidence)
        self.assertEqual(len(insights), 1)
        self.assertIn("area_worst", insights[0].sinal)
        self.assertIn("probabilidade_maligna=76.00%", insights[0].evidencia_numerica)
        self.assertIn("revisão", insights[0].implicacao_para_revisao)

    def test_api_interpret_uses_llm_explanation(self) -> None:
        payload = self.api.PredictRequest(features=self.features.iloc[0].to_dict())
        fake = LLMInterpretation(
            explanation=FAKE_INTERPRETATION,
            llm_model="gpt-test",
            prompt_version="clinical_explanation_v3",
            disclaimer="Nao constitui diagnostico.",
            evidence=[],
            insights_acionaveis=[],
        )

        async def fake_generate(*_args, **_kwargs):
            return fake

        with patch.object(
            self.api, "generate_interpretation", side_effect=fake_generate
        ):
            response = asyncio.run(self.api.interpret(payload, None))
        self.assertEqual(response.llm_model, "gpt-test")
        self.assertEqual(response.quality_checks["score_objetivo"], 1.0)
        self.assertEqual(response.insights_acionaveis, [])

    def test_api_interpret_reports_missing_api_key(self) -> None:
        payload = self.api.PredictRequest(features=self.features.iloc[0].to_dict())
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(HTTPException) as context:
                asyncio.run(self.api.interpret(payload, None))
        self.assertEqual(context.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
