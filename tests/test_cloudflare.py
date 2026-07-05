from __future__ import annotations

import asyncio
import json
from pathlib import Path
import unittest

import joblib
import pandas as pd

from src.edge_security import validate_turnstile
from src.model_inference import (
    load_serving_model,
    serving_model_from_joblib_artifact,
)


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


class FakeHttpClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.request_data: dict | None = None

    async def post(self, _url: str, data: dict):
        self.request_data = data
        return FakeResponse(self.payload)


class CloudflareServingTests(unittest.TestCase):
    def test_manifest_matches_joblib_for_all_dataset_rows(self) -> None:
        joblib_artifact = joblib.load(Path("resultados/fase2/modelo_serving.joblib"))
        legacy_model = serving_model_from_joblib_artifact(joblib_artifact)
        edge_model = load_serving_model(Path("src/modelo_serving.json"))
        data = pd.read_csv("data/cancer_mama.csv")

        maximum_difference = 0.0
        mismatches = 0
        for features in data[list(edge_model.feature_names)].to_dict(orient="records"):
            legacy_prediction = legacy_model.predict(features)
            edge_prediction = edge_model.predict(features)
            mismatches += int(legacy_prediction[0] != edge_prediction[0])
            maximum_difference = max(
                maximum_difference,
                abs(legacy_prediction[1] - edge_prediction[1]),
                abs(legacy_prediction[2] - edge_prediction[2]),
            )

        self.assertEqual(mismatches, 0)
        self.assertLessEqual(maximum_difference, 1e-12)

    def test_manifest_contains_expected_algorithm_metadata(self) -> None:
        payload = json.loads(
            Path("src/modelo_serving.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["preprocessing"]["type"], "StandardScaler")
        self.assertEqual(payload["classifier"]["type"], "LogisticRegression")
        self.assertEqual(len(payload["feature_names"]), 30)

    def test_turnstile_validation_does_not_modify_payload(self) -> None:
        client = FakeHttpClient({"success": True})
        valid = asyncio.run(
            validate_turnstile(
                "token-de-teste",
                "segredo-de-teste",
                "192.0.2.1",
                client=client,
            )
        )
        self.assertTrue(valid)
        self.assertEqual(
            client.request_data,
            {
                "secret": "segredo-de-teste",
                "response": "token-de-teste",
                "remoteip": "192.0.2.1",
            },
        )

    def test_turnstile_rejects_empty_token_without_network_request(self) -> None:
        client = FakeHttpClient({"success": True})
        valid = asyncio.run(validate_turnstile("", "segredo", client=client))
        self.assertFalse(valid)
        self.assertIsNone(client.request_data)


if __name__ == "__main__":
    unittest.main()
