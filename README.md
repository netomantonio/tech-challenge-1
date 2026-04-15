# Tech Challenge - Fase 1 | Sistema de Diagnóstico Médico com Machine Learning

## FIAP Pós Tech - AI for Devs

### Descrição do Projeto

Sistema inteligente de suporte ao diagnóstico médico, desenvolvido como parte do Tech Challenge da Fase 1 da pós-graduação FIAP. O projeto utiliza algoritmos de Machine Learning para classificar exames médicos, auxiliando na detecção de doenças.

### Datasets Utilizados

1. **Câncer de Mama Wisconsin** - Classificação de tumores como malignos ou benignos
   - Fonte: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data
   - 569 amostras, 30 features

2. **Diabetes Pima Indians** - Diagnóstico de diabetes
   - Fonte: https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data
   - 768 amostras, 8 features

> **Observação:** os arquivos `.csv` dos datasets **não são versionados** no repositório. Eles são baixados automaticamente ao executar `python data/download_datasets.py` (passo obrigatório antes de rodar os notebooks).

### Modelos Implementados

| Dataset | Modelos | Métrica Principal |
|---------|---------|-------------------|
| Câncer de Mama | Regressão Logística, KNN, Árvore de Decisão | Recall |
| Diabetes | SVM, KNN, Árvore de Decisão | F1-Score |

### Estrutura do Projeto

```
tech-challenge/
├── README.md                      # Este arquivo
├── Dockerfile                     # Container Docker
├── requirements.txt               # Dependências Python
├── data/
│   ├── download_datasets.py       # Script para baixar os datasets
│   ├── cancer_mama.csv            # ⚠️ Não versionado — gerado pelo script
│   └── diabetes.csv               # ⚠️ Não versionado — gerado pelo script
├── src/
│   ├── __init__.py
│   └── utils.py                   # Funções auxiliares (visualização, métricas)
├── notebooks/
│   ├── 01_cancer_mama.ipynb       # Pipeline completo - Câncer de Mama
│   └── 02_diabetes.ipynb          # Pipeline completo - Diabetes
└── resultados/                    # ⚠️ Não versionado — gráficos gerados pelos notebooks
```

> Itens marcados como **não versionados** estão no `.gitignore` por serem reproduzíveis: os `.csv` são baixados pelo `download_datasets.py` e os arquivos em `resultados/` são exportados pelos próprios notebooks durante a execução.

### Pré-requisitos

- Python 3.11+
- pip

### Instalação e Execução

#### Opção 1: Execução Local

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Baixar os datasets
python data/download_datasets.py

# 3. Abrir o Jupyter Notebook
jupyter notebook
```

Depois, navegue até a pasta `notebooks/` e abra os notebooks na ordem:
1. `01_cancer_mama.ipynb`
2. `02_diabetes.ipynb`

#### Opção 2: Execução com Docker

```bash
# 1. Construir a imagem
docker build -t tech-challenge-fase1 .

# 2. Executar o container
docker run -p 8888:8888 tech-challenge-fase1
```

Acesse o Jupyter Notebook pelo link exibido no terminal (geralmente http://localhost:8888).

### Tecnologias Utilizadas

- **Python 3.11**
- **pandas** - Manipulação de dados
- **numpy** - Operações numéricas
- **scikit-learn** - Modelos de ML, pré-processamento e métricas
- **matplotlib / seaborn** - Visualização de dados
- **shap** - Interpretabilidade dos modelos
- **jupyter** - Notebooks interativos

### Pipeline de Cada Notebook

1. **Exploração de Dados (EDA)** - Estatísticas descritivas, distribuições, visualizações
2. **Pré-processamento** - Limpeza, tratamento de valores ausentes, escalonamento, correlação
3. **Modelagem** - Treinamento de 3 modelos de classificação por dataset
4. **Avaliação** - Classification report, matriz de confusão, curva ROC, validação cruzada
5. **Interpretação** - Feature importance e SHAP values
6. **Discussão Crítica** - Aplicabilidade prática e limitações

### Autores

Alunos da FIAP Pós Tech - AI for Devs
