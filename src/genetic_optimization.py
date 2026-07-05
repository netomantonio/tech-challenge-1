"""Genetic hyperparameter optimization for the breast cancer classifiers.

This module extends the Phase 1 notebook without depending on notebook state.
The genetic algorithm evaluates candidates only on the training partition by
cross-validation. The held-out test partition is used after selection to
compare the optimized models against the original baselines and define the
demonstration model exposed by the API.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer


RANDOM_STATE = 42

MODEL_LABELS = {
    "logistic_regression": "Regressao Logistica",
    "knn": "KNN",
    "decision_tree": "Arvore de Decisao",
}

# Each chromosome is a tuple of allele indexes, one index for each gene below.
GENE_SPACES: dict[str, dict[str, tuple[Any, ...]]] = {
    "logistic_regression": {
        "C": (0.001, 0.01, 0.1, 1.0, 10.0, 100.0),
        "penalty": ("l1", "l2"),
        "class_weight": (None, "balanced"),
    },
    "knn": {
        "n_neighbors": tuple(range(1, 26, 2)),
        "weights": ("uniform", "distance"),
        "p": (1, 2),
    },
    "decision_tree": {
        "max_depth": (None, 2, 3, 4, 5, 6, 8, 10),
        "min_samples_split": (2, 4, 6, 8, 10, 15),
        "min_samples_leaf": (1, 2, 3, 4, 5),
        "criterion": ("gini", "entropy"),
        "class_weight": (None, "balanced"),
    },
}

BASELINE_PARAMETERS = {
    "logistic_regression": {
        "C": 1.0,
        "penalty": "l2",
        "class_weight": None,
        "solver": "lbfgs",
    },
    "knn": {
        "n_neighbors": 7,
        "weights": "uniform",
        "p": 2,
    },
    "decision_tree": {
        "max_depth": 4,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "criterion": "gini",
        "class_weight": None,
    },
}


@dataclass(frozen=True)
class GAConfig:
    """Configuration for one genetic algorithm experiment."""

    name: str
    population_size: int
    generations: int
    crossover_rate: float
    mutation_rate: float
    tournament_size: int = 3
    elite_count: int = 1
    seed: int = RANDOM_STATE


DEFAULT_EXPERIMENTS = (
    GAConfig(
        name="E1_pop_pequena_mutacao_baixa",
        population_size=8,
        generations=6,
        crossover_rate=0.80,
        mutation_rate=0.08,
        seed=42,
    ),
    GAConfig(
        name="E2_balanceado",
        population_size=12,
        generations=8,
        crossover_rate=0.85,
        mutation_rate=0.15,
        seed=84,
    ),
    GAConfig(
        name="E3_exploratorio",
        population_size=16,
        generations=10,
        crossover_rate=0.90,
        mutation_rate=0.30,
        seed=126,
    ),
)

QUICK_EXPERIMENTS = (
    GAConfig(
        name="smoke_test",
        population_size=4,
        generations=2,
        crossover_rate=0.80,
        mutation_rate=0.20,
        tournament_size=2,
        seed=42,
    ),
)


@dataclass
class SearchResult:
    """Best candidate and generation trace from one search execution."""

    model_key: str
    config: GAConfig
    best_genotype: tuple[int, ...]
    best_parameters: dict[str, Any]
    best_cv_metrics: dict[str, float]
    history: list[dict[str, Any]]
    evaluated_candidates: int


def configure_logger(output_dir: Path) -> logging.Logger:
    """Configure console and file tracking for optimization runs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cancer_mama_ga")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    close_logger(logger)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )
    file_handler = logging.FileHandler(
        output_dir / "treinamento_ga.log", mode="w", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def close_logger(logger: logging.Logger) -> None:
    """Release file handles so notebook reruns also work on Windows."""

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()


def load_cancer_data(csv_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load and prepare the same tabular prediction task used in Phase 1."""

    data = pd.read_csv(csv_path).drop(columns=["Unnamed: 32"], errors="ignore")
    required = {"diagnosis", "id"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset sem as colunas obrigatorias: {sorted(missing)}")

    target = data["diagnosis"].map({"M": 0, "B": 1})
    if target.isna().any():
        raise ValueError("Foram encontrados rotulos diferentes de M/B em diagnosis.")

    features = data.drop(columns=["diagnosis", "id"])
    return features, target.astype(int)


def build_pipeline(model_key: str, parameters: dict[str, Any]) -> Pipeline:
    """Build a classifier pipeline from decoded genetic parameters."""

    if model_key == "logistic_regression":
        model_parameters = parameters.copy()
        solver = model_parameters.pop("solver", "liblinear")
        model = LogisticRegression(
            solver=solver,
            max_iter=10000,
            random_state=RANDOM_STATE,
            **model_parameters,
        )
        steps = [("scaler", StandardScaler()), ("model", model)]
    elif model_key == "knn":
        model = KNeighborsClassifier(**parameters)
        steps = [("scaler", StandardScaler()), ("model", model)]
    elif model_key == "decision_tree":
        model = DecisionTreeClassifier(random_state=RANDOM_STATE, **parameters)
        steps = [("model", model)]
    else:
        raise KeyError(f"Modelo nao suportado: {model_key}")

    return Pipeline(steps)


def evaluate_on_test(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float | int]:
    """Fit on training data and return clinically relevant held-out metrics."""

    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    model_classes = list(pipeline.named_steps["model"].classes_)
    malignant_index = model_classes.index(0)
    malignant_probability = pipeline.predict_proba(x_test)[:, malignant_index]
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_weighted": precision_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(y_test, predictions, average="weighted", zero_division=0),
        "recall_maligno": recall_score(y_test, predictions, pos_label=0, zero_division=0),
        "f1_maligno": f1_score(y_test, predictions, pos_label=0, zero_division=0),
        "falsos_negativos_maligno": int(matrix[0, 1]),
        "auc_roc_maligno": roc_auc_score((y_test == 0).astype(int), malignant_probability),
    }


class GeneticOptimizer:
    """Simple categorical genetic algorithm with elitism and tournament selection."""

    def __init__(
        self,
        model_key: str,
        config: GAConfig,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        logger: logging.Logger,
        cv_splits: int = 5,
    ) -> None:
        if model_key not in GENE_SPACES:
            raise KeyError(f"Espaco genetico nao encontrado: {model_key}")
        if config.population_size < 2 or config.generations < 1:
            raise ValueError("O AG exige populacao >= 2 e ao menos uma geracao.")
        if config.elite_count >= config.population_size:
            raise ValueError("elite_count deve ser menor que population_size.")

        self.model_key = model_key
        self.config = config
        self.x_train = x_train
        self.y_train = y_train
        self.logger = logger
        self.random = random.Random(config.seed)
        self.gene_names = tuple(GENE_SPACES[model_key].keys())
        self.cv = StratifiedKFold(
            n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE
        )
        self.cache: dict[tuple[int, ...], dict[str, float]] = {}

    def random_individual(self) -> tuple[int, ...]:
        return tuple(
            self.random.randrange(len(GENE_SPACES[self.model_key][gene]))
            for gene in self.gene_names
        )

    def decode(self, individual: tuple[int, ...]) -> dict[str, Any]:
        return {
            gene: GENE_SPACES[self.model_key][gene][allele]
            for gene, allele in zip(self.gene_names, individual)
        }

    def evaluate(self, individual: tuple[int, ...]) -> dict[str, float]:
        if individual in self.cache:
            return self.cache[individual]

        pipeline = build_pipeline(self.model_key, self.decode(individual))
        scoring = {
            "recall_maligno": make_scorer(recall_score, pos_label=0, zero_division=0),
            "f1_maligno": make_scorer(f1_score, pos_label=0, zero_division=0),
            "accuracy": "accuracy",
        }
        validation = cross_validate(
            pipeline,
            self.x_train,
            self.y_train,
            cv=self.cv,
            scoring=scoring,
            n_jobs=1,
        )
        metrics = {
            "recall_maligno": float(validation["test_recall_maligno"].mean()),
            "f1_maligno": float(validation["test_f1_maligno"].mean()),
            "accuracy": float(validation["test_accuracy"].mean()),
        }
        metrics["fitness"] = (
            0.65 * metrics["recall_maligno"]
            + 0.25 * metrics["f1_maligno"]
            + 0.10 * metrics["accuracy"]
        )
        self.cache[individual] = metrics
        return metrics

    @staticmethod
    def _ranking(metrics: dict[str, float]) -> tuple[float, float, float, float]:
        return (
            metrics["fitness"],
            metrics["recall_maligno"],
            metrics["f1_maligno"],
            metrics["accuracy"],
        )

    def tournament_selection(self, population: list[tuple[int, ...]]) -> tuple[int, ...]:
        contenders = self.random.sample(
            population, min(self.config.tournament_size, len(population))
        )
        return max(contenders, key=lambda item: self._ranking(self.evaluate(item)))

    def crossover(
        self, first: tuple[int, ...], second: tuple[int, ...]
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        child_one: list[int] = []
        child_two: list[int] = []
        for allele_one, allele_two in zip(first, second):
            if self.random.random() < 0.5:
                child_one.append(allele_one)
                child_two.append(allele_two)
            else:
                child_one.append(allele_two)
                child_two.append(allele_one)
        return tuple(child_one), tuple(child_two)

    def mutate(self, individual: tuple[int, ...]) -> tuple[int, ...]:
        mutated = list(individual)
        for index, gene in enumerate(self.gene_names):
            if self.random.random() < self.config.mutation_rate:
                mutated[index] = self.random.randrange(len(GENE_SPACES[self.model_key][gene]))
        return tuple(mutated)

    def run(self) -> SearchResult:
        population = [self.random_individual() for _ in range(self.config.population_size)]
        history: list[dict[str, Any]] = []
        overall_best: tuple[int, ...] | None = None
        overall_metrics: dict[str, float] | None = None

        for generation in range(self.config.generations + 1):
            evaluated = [(item, self.evaluate(item)) for item in population]
            ranked = sorted(
                evaluated, key=lambda item: self._ranking(item[1]), reverse=True
            )
            generation_best, generation_metrics = ranked[0]

            if overall_metrics is None or self._ranking(generation_metrics) > self._ranking(
                overall_metrics
            ):
                overall_best = generation_best
                overall_metrics = generation_metrics

            mean_fitness = float(np.mean([item[1]["fitness"] for item in evaluated]))
            history_row = {
                "generation": generation,
                "best_fitness": generation_metrics["fitness"],
                "mean_fitness": mean_fitness,
                "best_recall_maligno": generation_metrics["recall_maligno"],
                "best_f1_maligno": generation_metrics["f1_maligno"],
                "best_parameters": json.dumps(
                    self.decode(generation_best), sort_keys=True, ensure_ascii=False
                ),
            }
            history.append(history_row)
            self.logger.info(
                "experiment=%s model=%s generation=%d best_fitness=%.5f "
                "mean_fitness=%.5f recall_maligno=%.5f candidates=%d",
                self.config.name,
                self.model_key,
                generation,
                generation_metrics["fitness"],
                mean_fitness,
                generation_metrics["recall_maligno"],
                len(self.cache),
            )

            if generation == self.config.generations:
                break

            next_population = [
                item[0] for item in ranked[: self.config.elite_count]
            ]
            while len(next_population) < self.config.population_size:
                parent_one = self.tournament_selection(population)
                parent_two = self.tournament_selection(population)
                if self.random.random() < self.config.crossover_rate:
                    children = self.crossover(parent_one, parent_two)
                else:
                    children = (parent_one, parent_two)
                for child in children:
                    if len(next_population) < self.config.population_size:
                        next_population.append(self.mutate(child))
            population = next_population

        assert overall_best is not None and overall_metrics is not None
        return SearchResult(
            model_key=self.model_key,
            config=self.config,
            best_genotype=overall_best,
            best_parameters=self.decode(overall_best),
            best_cv_metrics=overall_metrics,
            history=history,
            evaluated_candidates=len(self.cache),
        )


def _result_rank(result: SearchResult) -> tuple[float, float, float, float]:
    metrics = result.best_cv_metrics
    return (
        metrics["fitness"],
        metrics["recall_maligno"],
        metrics["f1_maligno"],
        metrics["accuracy"],
    )


def _comparison_row(
    kind: str,
    model_key: str,
    experiment: str,
    parameters: dict[str, Any],
    metrics: dict[str, float | int],
    cv_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "tipo": kind,
        "modelo": MODEL_LABELS[model_key],
        "experimento": experiment,
        "parametros": json.dumps(parameters, ensure_ascii=False, sort_keys=True),
        **metrics,
    }
    if cv_metrics:
        row.update({f"cv_{key}": value for key, value in cv_metrics.items()})
    return row


def _holdout_rank(row: dict[str, Any]) -> tuple[float, int, float, float, float]:
    """Rank final candidates with malignant recall as the primary concern."""

    return (
        float(row["recall_maligno"]),
        -int(row["falsos_negativos_maligno"]),
        float(row["f1_maligno"]),
        float(row["accuracy"]),
        float(row["auc_roc_maligno"]),
    )


def run_all_experiments(
    csv_path: str | Path,
    output_dir: str | Path,
    experiment_configs: Iterable[GAConfig] = DEFAULT_EXPERIMENTS,
) -> dict[str, Any]:
    """Run baselines, all GA configurations, final comparisons and persistence."""

    output_path = Path(output_dir)
    logger = configure_logger(output_path)
    features, target = load_cancer_data(csv_path)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    logger.info(
        "dataset_loaded samples=%d features=%d train=%d test=%d",
        len(features),
        features.shape[1],
        len(x_train),
        len(x_test),
    )

    baseline_rows: list[dict[str, Any]] = []
    for model_key, parameters in BASELINE_PARAMETERS.items():
        metrics = evaluate_on_test(
            build_pipeline(model_key, parameters), x_train, y_train, x_test, y_test
        )
        baseline_rows.append(
            _comparison_row("baseline_fase_1", model_key, "original", parameters, metrics)
        )

    results: list[SearchResult] = []
    experiment_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    configs = tuple(experiment_configs)
    for config in configs:
        for model_index, model_key in enumerate(GENE_SPACES):
            seeded_config = GAConfig(**{**asdict(config), "seed": config.seed + model_index})
            search = GeneticOptimizer(
                model_key=model_key,
                config=seeded_config,
                x_train=x_train,
                y_train=y_train,
                logger=logger,
            ).run()
            results.append(search)
            test_metrics = evaluate_on_test(
                build_pipeline(model_key, search.best_parameters),
                x_train,
                y_train,
                x_test,
                y_test,
            )
            experiment_rows.append(
                _comparison_row(
                    "otimizado_ag",
                    model_key,
                    search.config.name,
                    search.best_parameters,
                    test_metrics,
                    search.best_cv_metrics,
                )
            )
            for item in search.history:
                history_rows.append(
                    {
                        "modelo": MODEL_LABELS[model_key],
                        "experimento": search.config.name,
                        **item,
                    }
                )

    selected_by_model = {
        model_key: max(
            (item for item in results if item.model_key == model_key), key=_result_rank
        )
        for model_key in GENE_SPACES
    }
    selected_rows = []
    for model_key, selected in selected_by_model.items():
        selected_rows.append(
            next(
                row
                for row in experiment_rows
                if row["modelo"] == MODEL_LABELS[model_key]
                and row["experimento"] == selected.config.name
            )
        )
    final_comparison = pd.DataFrame(baseline_rows + selected_rows)
    all_experiments = pd.DataFrame(experiment_rows)
    history = pd.DataFrame(history_rows)

    best_optimized = max(selected_by_model.values(), key=_result_rank)
    best_optimized_test_metrics = evaluate_on_test(
        build_pipeline(best_optimized.model_key, best_optimized.best_parameters),
        x_train,
        y_train,
        x_test,
        y_test,
    )

    recommended_row = max(baseline_rows + selected_rows, key=_holdout_rank)
    recommended_model_key = next(
        key for key, label in MODEL_LABELS.items() if label == recommended_row["modelo"]
    )
    recommended_parameters = json.loads(recommended_row["parametros"])
    serving_pipeline = build_pipeline(recommended_model_key, recommended_parameters)
    # The API is demonstrative: after the reported comparison, fit its recommended
    # candidate on the full available dataset for serving.
    serving_pipeline.fit(features, target)
    artifact_path = output_path / "modelo_serving.joblib"
    joblib.dump(
        {
            "model": serving_pipeline,
            "feature_names": list(features.columns),
            "model_key": recommended_model_key,
            "model_label": MODEL_LABELS[recommended_model_key],
            "source": recommended_row["tipo"],
            "parameters": recommended_parameters,
            "holdout_test_metrics": {
                key: recommended_row[key]
                for key in (
                    "accuracy",
                    "precision_weighted",
                    "recall_weighted",
                    "f1_weighted",
                    "recall_maligno",
                    "f1_maligno",
                    "falsos_negativos_maligno",
                    "auc_roc_maligno",
                )
            },
            "target_mapping": {"M": 0, "B": 1},
        },
        artifact_path,
    )

    all_experiments.to_csv(output_path / "experimentos_ga.csv", index=False)
    final_comparison.to_csv(output_path / "comparacao_baseline_otimizados.csv", index=False)
    history.to_csv(output_path / "historico_geracoes.csv", index=False)
    summary = {
        "random_state": RANDOM_STATE,
        "fitness_formula": (
            "0.65 * recall_maligno + 0.25 * f1_maligno + 0.10 * accuracy"
        ),
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "experiments": [asdict(item) for item in configs],
        "selected_models": {
            model_key: {
                "experiment": search.config.name,
                "parameters": search.best_parameters,
                "cv_metrics": search.best_cv_metrics,
            }
            for model_key, search in selected_by_model.items()
        },
        "best_optimized": {
            "model": MODEL_LABELS[best_optimized.model_key],
            "experiment": best_optimized.config.name,
            "parameters": best_optimized.best_parameters,
            "cv_metrics": best_optimized.best_cv_metrics,
            "test_metrics": best_optimized_test_metrics,
        },
        "serving_model": {
            "model": recommended_row["modelo"],
            "source": recommended_row["tipo"],
            "experiment": recommended_row["experimento"],
            "parameters": recommended_parameters,
            "selection_basis": (
                "Melhor resultado observado na comparacao final, priorizando "
                "recall maligno, falsos negativos e F1 maligno."
            ),
            "test_metrics": {
                key: recommended_row[key]
                for key in (
                    "accuracy",
                    "precision_weighted",
                    "recall_weighted",
                    "f1_weighted",
                    "recall_maligno",
                    "f1_maligno",
                    "falsos_negativos_maligno",
                    "auc_roc_maligno",
                )
            },
            "artifact": str(artifact_path),
        },
    }
    with (output_path / "resumo_execucao.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    logger.info(
        "best_optimized model=%s experiment=%s fitness=%.5f",
        best_optimized.model_key,
        best_optimized.config.name,
        best_optimized.best_cv_metrics["fitness"],
    )
    logger.info(
        "serving_model model=%s source=%s artifact=%s",
        recommended_model_key,
        recommended_row["tipo"],
        artifact_path,
    )
    close_logger(logger)

    return {
        "comparison": final_comparison,
        "experiments": all_experiments,
        "history": history,
        "summary": summary,
        "artifact_path": artifact_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Otimizacao genetica - cancer de mama")
    parser.add_argument("--data", default="data/cancer_mama.csv", help="Caminho do CSV.")
    parser.add_argument(
        "--output", default="resultados/fase2", help="Diretorio de artefatos gerados."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Executa somente uma configuracao curta para teste de fumaça.",
    )
    args = parser.parse_args()
    configurations = QUICK_EXPERIMENTS if args.quick else DEFAULT_EXPERIMENTS
    result = run_all_experiments(args.data, args.output, configurations)
    printable = result["comparison"].drop(columns=["parametros"])
    print("\nCOMPARACAO FINAL (teste reservado)")
    print(printable.round(4).to_string(index=False))
    print(f"\nModelo disponibilizado pela API em: {result['artifact_path']}")


if __name__ == "__main__":
    main()
