# Tech Challenge - Diagnóstico de Câncer de Mama

## FIAP Pós Tech - AI for Devs

Este repositório reúne o projeto original de diagnóstico de câncer de mama
desenvolvido na Fase 1 e sua continuação na Fase 2, voltada à otimização de
hiperparâmetros e à demonstração de escalabilidade.

O projeto usa o **Wisconsin Breast Cancer Diagnostic Dataset** para classificar
tumores como:

- `0`: Maligno;
- `1`: Benigno.

A métrica prioritária é o **recall da classe Maligno**, pois um falso negativo
pode atrasar investigação e tratamento.

## Fase 1 - Modelo de Diagnóstico

### Objetivo

Na Fase 1 foi desenvolvido o pipeline base em
`notebooks/01_cancer_mama.ipynb`, cobrindo:

1. Exploração do dataset e análise do balanceamento das classes.
2. Pré-processamento, remoção de `id`, divisão estratificada e escalonamento.
3. Treinamento de Regressão Logística, KNN e Árvore de Decisão.
4. Avaliação com matriz de confusão, curva ROC e validação cruzada.
5. Interpretação com importância de features e SHAP.

### Dataset

| Item | Informação |
| --- | --- |
| Fonte | [Kaggle - Breast Cancer Wisconsin](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data) |
| Amostras | 569 |
| Features preditivas | 30 variáveis numéricas |
| Alvo | Maligno (`0`) ou Benigno (`1`) |

O arquivo `data/cancer_mama.csv` não é versionado. Ele pode ser gerado com:

```bash
python data/download_datasets.py
```

### Resultados da Fase 1

| Modelo | Accuracy | Recall Maligno | F1 Maligno | Falsos Negativos |
| --- | ---: | ---: | ---: | ---: |
| Regressão Logística | **0.9825** | **0.9762** | **0.9762** | **1** |
| KNN (`K=7`) | 0.9737 | 0.9286 | 0.9630 | 3 |
| Árvore de Decisão | 0.9386 | 0.9286 | 0.9176 | 3 |

A Regressão Logística foi o melhor baseline observado, principalmente por
deixar passar apenas um caso maligno no conjunto de teste.

### Artefatos da Fase 1

- Notebook: `notebooks/01_cancer_mama.ipynb`
- Relatório técnico: `relatorio_tecnico_01_cancer_mama.md`
- Relatório técnico em PDF: `relatorio_tecnico_01_cancer_mama.pdf`

## Fase 2 - Otimização e Escalabilidade

A Fase 2 estende o pipeline original com algoritmo genético para otimização
dos três modelos, comparação com os baselines, logging, monitoramento e uma
API mínima para sustentar a configuração de autoscaling.

### Entregas Atendidas

| Requisito | Implementação |
| --- | --- |
| Algoritmo genético | `src/genetic_optimization.py` |
| Codificação, seleção, cruzamento e mutação | Espaços genéticos discretos, torneio, cruzamento uniforme e mutação por gene |
| Função fitness | `0.65 * recall_maligno + 0.25 * f1_maligno + 0.10 * accuracy` |
| Três experimentos | Executados no notebook `03_otimizacao_genetica_cancer_mama.ipynb` |
| Comparação com originais | Tabelas e gráficos no notebook e em `resultados/fase2/` |
| Monitoramento e logging | Logs de treinamento e API com métricas Prometheus |
| Escalabilidade automática | API containerizada e `HorizontalPodAutoscaler` em `deploy/k8s/` |
| Arquitetura e decisões | `docs/arquitetura_fase2.md` e `relatorio_tecnico_fase2.md` |

## Estrutura do Repositório

```text
data/
  download_datasets.py
deploy/k8s/
  deployment.yaml
  hpa.yaml
  kustomization.yaml
  service.yaml
docs/
  arquitetura_fase2.md
notebooks/
  01_cancer_mama.ipynb
  03_otimizacao_genetica_cancer_mama.ipynb
resultados/fase2/
  comparacao_baseline_otimizados.csv
  experimentos_ga.csv
  historico_geracoes.csv
  modelo_serving.joblib
  resumo_execucao.json
  treinamento_ga.log
src/
  api.py
  genetic_optimization.py
  utils.py
tests/
  test_fase2.py
Dockerfile
requirements.txt
relatorio_tecnico_01_cancer_mama.md
relatorio_tecnico_01_cancer_mama.pdf
relatorio_tecnico_fase2.md
```

### Experimentos Genéticos

| Experimento | População | Gerações | Cruzamento | Mutação |
| --- | ---: | ---: | ---: | ---: |
| `E1_pop_pequena_mutacao_baixa` | 8 | 6 | 0.80 | 0.08 |
| `E2_balanceado` | 12 | 8 | 0.85 | 0.15 |
| `E3_exploratorio` | 16 | 10 | 0.90 | 0.30 |

Cada configuração foi executada para Regressão Logística, KNN e Árvore de
Decisão. A busca usa apenas o conjunto de treino com validação cruzada
estratificada; o teste reservado é usado na comparação final.

### Resultado da Fase 2

| Modelo | Versão | Accuracy | Recall Maligno | F1 Maligno | FN Maligno |
| --- | --- | ---: | ---: | ---: | ---: |
| Regressão Logística | Original | **0.9825** | **0.9762** | **0.9762** | **1** |
| Regressão Logística | AG - E2 | 0.9561 | **0.9762** | 0.9425 | **1** |
| KNN | Original | **0.9737** | **0.9286** | **0.9630** | **3** |
| KNN | AG - E1 | 0.9649 | **0.9286** | 0.9512 | **3** |
| Árvore de Decisão | Original | **0.9386** | **0.9286** | **0.9176** | **3** |
| Árvore de Decisão | AG - E2 | 0.9035 | 0.9048 | 0.8736 | 4 |

O algoritmo genético encontrou configurações competitivas, mas não superou a
Regressão Logística original no teste reservado. Por isso, a API demonstrativa
utiliza o baseline logístico como modelo recomendado, em vez de publicar uma
variante otimizada inferior.

## Execução Local

```bash
python -m pip install -r requirements.txt
python data/download_datasets.py
```

Para consultar ou reexecutar a Fase 1, abra:

```text
notebooks/01_cancer_mama.ipynb
```

Para executar a Fase 2 e iniciar a API:

```bash
python -m src.genetic_optimization --data data/cancer_mama.csv --output resultados/fase2
python src/api.py
```

O notebook principal da Fase 2 é
`notebooks/03_otimizacao_genetica_cancer_mama.ipynb`.

Com a API em execução:

- Swagger UI: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`
- Métricas Prometheus: `http://127.0.0.1:8000/metrics`

## Container e Autoscaling

```bash
docker build -t tech-challenge-fase2-api:latest .
kubectl apply -k deploy/k8s
kubectl get deployment,service,hpa
```

O HPA mantém entre 2 e 10 réplicas da API e escala com alvo de 60% de
utilização média de CPU. O cluster deve possuir `metrics-server`.

## Limitações

Este projeto é acadêmico. O dataset possui apenas 569 amostras, não houve
validação externa, e a API não deve ser usada como diagnóstico clínico
autônomo.

## Autores

Antonio Miranda, Elaine, Marcos Mol, Lucas da Costa, Ricardo Loureiro - AI for Devs (9IADT)
