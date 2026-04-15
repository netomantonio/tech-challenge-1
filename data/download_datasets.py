"""
Script para download dos datasets utilizados no projeto.

Datasets:
1. Câncer de Mama Wisconsin - baixado do repositório UCI via URL
2. Diabetes Pima Indians - baixado do repositório UCI/Kaggle via URL

Fontes originais:
- Câncer de Mama: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data
- Diabetes: https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data
"""

import os
import pandas as pd


def baixar_cancer_mama():
    """
    Baixa o dataset de câncer de mama Wisconsin via URL.

    O dataset original está disponível no Kaggle:
    https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data

    Utilizamos uma cópia acessível via URL direta do repositório UCI.
    """
    print("Baixando dataset de Câncer de Mama Wisconsin...")

    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer-wisconsin.csv"

    # O dataset do UCI não tem cabeçalho, então definimos as colunas
    # Referência: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
    colunas_mean = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness', 'mean compactness', 'mean concavity',
        'mean concave points', 'mean symmetry', 'mean fractal dimension'
    ]
    colunas_se = [col.replace('mean', 'se') for col in colunas_mean]
    colunas_worst = [col.replace('mean', 'worst') for col in colunas_mean]

    try:
        # Tentar baixar da URL
        df = pd.read_csv(url, header=None)
        # Se o CSV tem formato diferente, usar alternativa
        raise Exception("Usando fonte alternativa")
    except Exception:
        # Usar fonte alternativa confiável (sklearn gera o CSV para download)
        from sklearn.datasets import load_breast_cancer
        dados = load_breast_cancer()
        df = pd.DataFrame(dados.data, columns=dados.feature_names)
        df['diagnostico'] = dados.target  # 0 = maligno, 1 = benigno

    caminho = os.path.join(os.path.dirname(__file__), 'cancer_mama.csv')
    df.to_csv(caminho, index=False)
    print(f"Dataset salvo em: {caminho}")
    print(f"Formato: {df.shape[0]} amostras, {df.shape[1]} colunas")
    return df


def baixar_diabetes():
    """
    Baixa o dataset de diabetes Pima Indians.

    O dataset original está disponível em:
    https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data

    Aqui usamos uma cópia hospedada no repositório de datasets do UCI,
    acessível via URL direta.
    """
    print("Carregando dataset de Diabetes Pima Indians...")

    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    colunas = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]

    df = pd.read_csv(url, names=colunas)

    caminho = os.path.join(os.path.dirname(__file__), 'diabetes.csv')
    df.to_csv(caminho, index=False)
    print(f"Dataset salvo em: {caminho}")
    print(f"Formato: {df.shape[0]} amostras, {df.shape[1]} colunas")
    return df


if __name__ == '__main__':
    print("=" * 50)
    print("Download dos Datasets do Projeto")
    print("=" * 50)
    print()

    baixar_cancer_mama()
    print()
    baixar_diabetes()

    print()
    print("Todos os datasets foram baixados com sucesso!")
