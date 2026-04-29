"""
Script para download dos datasets utilizados no projeto.

Datasets:
1. Cancer de Mama Wisconsin - baixado do Kaggle via kagglehub

Fontes originais:
- Cancer de Mama: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _data_dir() -> Path:
    return Path(__file__).resolve().parent


def baixar_cancer_mama() -> pd.DataFrame:
    """
    Baixa o dataset "Breast Cancer Wisconsin (Diagnostic)" do Kaggle via kagglehub
    e salva como `data/cancer_mama.csv`.

    Observacao: o Kaggle costuma incluir uma coluna vazia (ex.: "Unnamed: 32").
    Esta funcao remove quaisquer colunas "Unnamed:*" para manter compatibilidade.
    """
    print("Baixando dataset de Cancer de Mama Wisconsin (Kaggle)...")

    try:
        import kagglehub
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Nao foi possivel importar `kagglehub`. "
            "Instale/garanta no ambiente (ver requirements.txt). "
            f"Erro original: {e}"
        ) from e

    try:
        dataset_dir = Path(kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data"))
    except Exception as e:
        raise RuntimeError(
            "Falha ao baixar o dataset via Kaggle. "
            "Verifique se voce tem credenciais do Kaggle configuradas (kaggle.json). "
            f"Erro original: {e}"
        ) from e

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Diretorio do dataset nao encontrado: {dataset_dir}")

    # Preferir o arquivo padrao do dataset. Se nao existir, escolher um unico CSV disponivel.
    candidato = dataset_dir / "data.csv"
    if candidato.exists():
        csv_path = candidato
    else:
        csvs = sorted(p for p in dataset_dir.rglob("*.csv") if p.is_file())
        if len(csvs) == 1:
            csv_path = csvs[0]
        else:
            raise FileNotFoundError(
                "Nao foi possivel identificar o CSV do dataset. "
                f"CSVs encontrados ({len(csvs)}): {[str(p) for p in csvs]}"
            )

    df = pd.read_csv(csv_path)

    colunas_unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if colunas_unnamed:
        df = df.drop(columns=colunas_unnamed)

    out_path = _data_dir() / "cancer_mama.csv"
    df.to_csv(out_path, index=False)

    print(f"Dataset salvo em: {out_path}")
    print(f"Formato: {df.shape[0]} amostras, {df.shape[1]} colunas")
    return df


if __name__ == "__main__":
    print("=" * 50)
    print("Download dos Datasets do Projeto")
    print("=" * 50)
    print()

    baixar_cancer_mama()
    print()

    print("Dataset baixado com sucesso!")

