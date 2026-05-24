# Tech Challenge - Fase 1 | Diagnóstico de Câncer de Mama com Machine Learning

## FIAP Pós Tech - AI for Devs

### Descrição do Projeto

Sistema de suporte ao diagnóstico médico, desenvolvido como parte do Tech Challenge da Fase 1 da pós-graduação FIAP. O projeto utiliza algoritmos de Machine Learning para classificar tumores de mama como malignos ou benignos, com base em características extraídas de imagens digitalizadas de aspiração por agulha fina (FNA).

### Dataset Utilizado

**Câncer de Mama Wisconsin (Diagnostic)**
- Fonte: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data
- 569 amostras, 30 features numéricas
- Classes (diagnóstico): Maligno = 0 e Benigno = 1

> **Observação:** o arquivo `.csv` do dataset **não é versionado** no repositório. Ele é baixado automaticamente ao executar `python data/download_datasets.py` (passo obrigatório antes de rodar o notebook).

### Modelos Implementados

| Modelos | Métrica Principal |
|---------|-------------------|
| Regressão Logística, KNN, Árvore de Decisão | Recall (classe Maligno) |

A métrica de Recall para a classe Maligno foi escolhida como principal porque, no contexto clínico, um falso negativo (tumor maligno classificado como benigno) é o erro mais crítico.

### Estrutura do Projeto

```
tech-challenge/
├── README.md                      # Este arquivo
├── Dockerfile                     # Container Docker
├── requirements.txt               # Dependências Python
├── data/
│   ├── download_datasets.py       # Script para baixar o dataset
│   └── cancer_mama.csv            # Não versionado — gerado pelo script
├── src/
│   ├── __init__.py
│   └── utils.py                   # Funções auxiliares (visualização, métricas)
├── notebooks/
│   └── 01_cancer_mama.ipynb       # Pipeline completo - Câncer de Mama
└── resultados/                    # Não versionado — gráficos gerados pelo notebook
```

> Itens marcados como **não versionados** estão no `.gitignore` por serem reproduzíveis: o `.csv` é baixado pelo `download_datasets.py` e os arquivos em `resultados/` são exportados pelo notebook durante a execução.

### Pré-requisitos

- Python 3.11+
- pip

> Observação: para este projeto, **não é necessário** ter conta/credenciais do Kaggle para baixar o dataset via `kagglehub`.

### Instalação e Execução

#### Opção 1: Execução Local

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Baixar o dataset
python data/download_datasets.py

# 3. Abrir o Jupyter Notebook
jupyter notebook
```

Depois, navegue até a pasta `notebooks/` e abra `01_cancer_mama.ipynb`.

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
- **kagglehub** - Download do dataset via Kaggle
- **jupyter / notebook** - Notebooks interativos

### Pipeline do Notebook

1. **Exploração de Dados (EDA)** - Estatísticas descritivas, distribuições, visualizações
2. **Pré-processamento** - Tradução de colunas, remoção de features sem valor preditivo, escalonamento, análise de correlação
3. **Modelagem** - Treinamento de 3 modelos de classificação com seleção automática de hiperparâmetros (K no KNN) e comparação de escaladores
4. **Avaliação** - Classification report, matriz de confusão, curva ROC, validação cruzada estratificada
5. **Interpretação** - Feature importance (Árvore de Decisão) e SHAP values
6. **Discussão Crítica** - Aplicabilidade prática e limitações

### Autores

Antonio Miranda, Elaine, Marcos Mol, Lucas da Costa, Ricardo Loureiro - AI for Devs (9IADT)

## Tech Challenge - Fase 2 | Otimização Genética e Escalabilidade

A continuação do Projeto 1 usa o mesmo dataset de câncer de mama e os mesmos
três modelos para implementar otimização de hiperparâmetros via algoritmo
genético. A métrica prioritária continua sendo o `recall` da classe Maligno.

### Entregáveis da Fase 2

| Arquivo | Finalidade |
| --- | --- |
| `notebooks/03_otimizacao_genetica_cancer_mama.ipynb` | Notebook principal com três experimentos do AG e comparação com a Fase 1 |
| `src/genetic_optimization.py` | Implementação dos genes, seleção, cruzamento, mutação, fitness, logging e persistência do modelo |
| `src/api.py` | API de inferência com endpoints de saúde e métricas Prometheus |
| `relatorio_tecnico_fase2.md` | Resultados obtidos e análise crítica da comparação |
| `docs/arquitetura_fase2.md` | Arquitetura, decisões, observabilidade e execução |
| `deploy/k8s/` | `Deployment`, `Service` e `HorizontalPodAutoscaler` |
| `Dockerfile.api` | Container da API escalável |

### Execução da Otimização

```bash
pip install -r requirements.txt
python data/download_datasets.py
python -m src.genetic_optimization --data data/cancer_mama.csv --output resultados/fase2
```

Também é possível executar diretamente o notebook
`notebooks/03_otimizacao_genetica_cancer_mama.ipynb`.

Os resultados gerados incluem tabelas CSV, histórico de fitness, log de
treinamento e `resultados/fase2/modelo_ga_campeao.joblib`. A pasta
`resultados/` permanece ignorada pelo Git por conter artefatos reproduzíveis.

### API, Métricas e Autoscaling

Após gerar o modelo:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
docker build -f Dockerfile.api -t tech-challenge-fase2-api:latest .
kubectl apply -k deploy/k8s
```

A API fornece `POST /predict`, `GET /health`, `GET /health/live`,
`GET /health/ready` e `GET /metrics`. O HPA mantém de 2 a 10 réplicas da API,
escalando quando a utilização média de CPU ultrapassa 60%.
