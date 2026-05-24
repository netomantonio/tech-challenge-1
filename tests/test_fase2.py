from __future__ import annotations

import logging
from pathlib import Path
import tempfile
import unittest

from fastapi import HTTPException

from src.genetic_optimization import (
    GENE_SPACES,
    QUICK_EXPERIMENTS,
    GAConfig,
    GeneticOptimizer,
    load_cancer_data,
    run_all_experiments,
)


DATA_PATH = Path("data/cancer_mama.csv")


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
        self.assertEqual(serving_model["model"], "Regressao Logistica")

    def test_api_predicts_using_persisted_artifact(self) -> None:
        payload = self.api.PredictRequest(features=self.features.iloc[0].to_dict())
        response = self.api.predict(payload)
        self.assertIn(response.diagnosis, {"Maligno", "Benigno"})
        self.assertAlmostEqual(
            response.probability_malignant + response.probability_benign, 1.0, places=6
        )

    def test_api_rejects_incomplete_features(self) -> None:
        payload = self.api.PredictRequest(features={"radius_mean": 10.0})
        with self.assertRaises(HTTPException) as context:
            self.api.predict(payload)
        self.assertEqual(context.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()
