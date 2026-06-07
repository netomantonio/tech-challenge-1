"""Avalia a qualidade das interpretacoes geradas pela LLM."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd

from src.llm_interpretation import (
    LLMUnavailableError,
    ModelResult,
    derive_feature_evidence,
    evaluate_interpretation_quality,
    generate_interpretation,
    interpretation_to_dict,
)


def predict_case(artifact: dict, row: pd.Series) -> ModelResult:
    features = row.to_dict()
    model = artifact["model"]
    frame = pd.DataFrame([features])
    prediction = int(model.predict(frame)[0])
    probabilities = model.predict_proba(frame)[0]
    classes = list(model.named_steps["model"].classes_)
    return ModelResult(
        prediction=prediction,
        diagnosis="Maligno" if prediction == 0 else "Benigno",
        probability_malignant=float(probabilities[classes.index(0)]),
        probability_benign=float(probabilities[classes.index(1)]),
        model=str(artifact["model_label"]),
    )


def select_representative_cases(artifact: dict, csv_path: Path) -> list[tuple[str, pd.Series]]:
    data = pd.read_csv(csv_path)
    features = data.drop(columns=["id", "diagnosis"])
    probabilities = artifact["model"].predict_proba(features)
    classes = list(artifact["model"].named_steps["model"].classes_)
    malignant_probability = probabilities[:, classes.index(0)]
    candidates = features.copy()
    candidates["probability_malignant"] = malignant_probability
    selected: list[tuple[str, int]] = [
        ("alto_risco_maligno", int(candidates["probability_malignant"].idxmax())),
        ("alto_risco_benigno", int(candidates["probability_malignant"].idxmin())),
        (
            "caso_limiar",
            int((candidates["probability_malignant"] - 0.5).abs().idxmin()),
        ),
    ]
    for label, target_probability in (
        ("faixa_probabilidade_25", 0.25),
        ("faixa_probabilidade_40", 0.40),
        ("faixa_probabilidade_60", 0.60),
        ("faixa_probabilidade_75", 0.75),
    ):
        selected.append(
            (
                label,
                int((candidates["probability_malignant"] - target_probability).abs().idxmin()),
            )
        )

    seen: set[int] = set()
    cases: list[tuple[str, pd.Series]] = []
    for label, index in selected:
        if index in seen:
            continue
        seen.add(index)
        row = candidates.drop(columns="probability_malignant").loc[index]
        cases.append((label, row))
    return cases


def summarize_quality(dataframe: pd.DataFrame, llm_model: str) -> dict:
    quality_columns = [
        column
        for column in dataframe.columns
        if column not in {"caso", "classificacao", "probabilidade_maligna", "score_objetivo"}
    ]
    failed_by_criterion = {
        column: int((~dataframe[column].astype(bool)).sum()) for column in quality_columns
    }
    return {
        "quantidade_casos": int(len(dataframe)),
        "llm_model": llm_model,
        "score_medio": float(dataframe["score_objetivo"].mean()),
        "score_minimo": float(dataframe["score_objetivo"].min()),
        "score_maximo": float(dataframe["score_objetivo"].max()),
        "criterios_reprovados": failed_by_criterion,
    }


def evaluate_cases(model_path: Path, csv_path: Path, output_dir: Path) -> pd.DataFrame:
    if not os.getenv("GROQ_API_KEY"):
        raise LLMUnavailableError(
            "GROQ_API_KEY nao configurada. "
            "Obtenha sua chave gratuita em console.groq.com/keys"
        )

    artifact = joblib.load(model_path)
    records: list[dict] = []
    interpretations: list[dict] = []
    llm_model = ""
    for case_name, row in select_representative_cases(artifact, csv_path):
        result = predict_case(artifact, row)
        evidence = derive_feature_evidence(artifact, row.to_dict())
        interpretation = generate_interpretation(result, evidence)
        llm_model = interpretation.llm_model
        quality = evaluate_interpretation_quality(
            interpretation.explanation, result.diagnosis
        )
        records.append(
            {
                "caso": case_name,
                "classificacao": result.diagnosis,
                "probabilidade_maligna": result.probability_malignant,
                **quality,
            }
        )
        interpretations.append(
            {
                "caso": case_name,
                "resultado_modelo": result.__dict__,
                **interpretation_to_dict(interpretation),
                "avaliacao_objetiva": quality,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(records)
    dataframe.to_csv(output_dir / "avaliacao_interpretacoes_llm.csv", index=False)
    with (output_dir / "interpretacoes_llm.json").open("w", encoding="utf-8") as file:
        json.dump(interpretations, file, ensure_ascii=False, indent=2)
    summary = summarize_quality(dataframe, llm_model)
    with (output_dir / "resumo_avaliacao_llm.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliacao das interpretacoes geradas pela LLM")
    parser.add_argument("--model", default="resultados/fase2/modelo_serving.joblib")
    parser.add_argument("--data", default="data/cancer_mama.csv")
    parser.add_argument("--output", default="resultados/fase2")
    args = parser.parse_args()
    try:
        result = evaluate_cases(Path(args.model), Path(args.data), Path(args.output))
    except LLMUnavailableError as error:
        raise SystemExit(str(error))
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
