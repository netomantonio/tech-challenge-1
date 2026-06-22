"""Exporta o joblib legado para o manifesto portátil usado no Worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib


def _plain_value(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def build_manifest(artifact: dict[str, Any]) -> dict[str, Any]:
    """Extrai somente os parâmetros já treinados, sem recalcular o modelo."""

    pipeline = artifact["model"]
    scaler = pipeline.named_steps["scaler"]
    classifier = pipeline.named_steps["model"]
    return {
        "schema_version": 1,
        "model_key": artifact["model_key"],
        "model_label": artifact["model_label"],
        "source": artifact["source"],
        "feature_names": list(artifact["feature_names"]),
        "target_mapping": artifact["target_mapping"],
        "preprocessing": {
            "type": "StandardScaler",
            "mean": [float(value) for value in scaler.mean_],
            "scale": [float(value) for value in scaler.scale_],
        },
        "classifier": {
            "type": "LogisticRegression",
            "coefficients": [float(value) for value in classifier.coef_[0]],
            "intercept": float(classifier.intercept_[0]),
            "classes": [int(value) for value in classifier.classes_],
        },
        "parameters": artifact["parameters"],
        "holdout_test_metrics": {
            key: _plain_value(value)
            for key, value in artifact["holdout_test_metrics"].items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("resultados/fase2/modelo_serving.joblib"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/modelo_serving.json"),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Falha se o manifesto versionado divergir do joblib.",
    )
    args = parser.parse_args()

    expected = build_manifest(joblib.load(args.source))
    if args.check:
        current = json.loads(args.output.read_text(encoding="utf-8"))
        if current != expected:
            raise SystemExit(
                "O manifesto portátil está desatualizado em relação ao joblib."
            )
        print("Manifesto portátil equivalente ao joblib.")
        return

    args.output.write_text(
        json.dumps(expected, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Manifesto exportado para {args.output}.")


if __name__ == "__main__":
    main()
