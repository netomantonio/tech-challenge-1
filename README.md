# Tech Challenge - Diagnóstico de Câncer de Mama

## FIAP Pós Tech - AI for Devs

Este repositório reúne o projeto original de diagnóstico de câncer de mama
desenvolvido na Fase 1 e sua continuação na Fase 2, voltada à otimização de
hiperparâmetros, escalabilidade e interpretação de resultados com LLM.

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

## Fase 2 - Otimização, Escalabilidade e LLM

A Fase 2 estende o pipeline original com algoritmo genético para otimização
dos três modelos, comparação com os baselines, logging, monitoramento e uma
API mínima para sustentar a configuração de autoscaling. A API também integra
uma LLM via Groq para explicar resultados em linguagem natural sob restrições de segurança
para contexto médico.

### Entregas Atendidas

| Requisito | Implementação |
| --- | --- |
| Algoritmo genético | `src/genetic_optimization.py` |
| Codificação, seleção, cruzamento e mutação | Espaços genéticos discretos, torneio, cruzamento uniforme e mutação por gene |
| Função fitness | `0.65 * recall_maligno + 0.25 * f1_maligno + 0.10 * accuracy` |
| Três experimentos | Executados no notebook `02_otimizacao_genetica_cancer_mama.ipynb` |
| Comparação com originais | Tabelas e gráficos no notebook e em `resultados/fase2/` |
| Monitoramento e logging | Logs de treinamento e API com métricas Prometheus |
| Escalabilidade automática | API containerizada e `HorizontalPodAutoscaler` em `deploy/k8s/` |
| Arquitetura e decisões | `docs/arquitetura_fase2.md` e `relatorio_tecnico_fase2.md` |
| Integração com LLM | `src/llm_interpretation.py` e endpoint `POST /interpret` |
| Prompt engineering | Instruções clínicas versionadas em `clinical_explanation_v3` |
| Avaliação da interpretação | `src/evaluate_llm.py` e notebook `03_interpretacao_llm_cancer_mama.ipynb` |

## Estrutura do Repositório

```text
cloudflare/api/        # toolchain e configuração do Python Worker
  pyproject.toml
  wrangler.jsonc
data/
  download_datasets.py
deploy/cloudflare/      # pipeline de deploy do Worker e testes (Cloudflare Builds)
  deploy-worker.sh
  test-worker.sh
deploy/k8s/
  deployment.yaml
  hpa.yaml
  kustomization.yaml
  service.yaml
docs/
  arquitetura_fase2.md
  cloudflare-deploy.md  # infraestrutura de nuvem e procedimentos de deploy
  frontend.md           # documentação do frontend
frontend/               # aplicação React (Cloudflare Pages)
  functions/[[path]].ts # Pages Function: proxy via Service Binding
  public/               # _headers (CSP) e _routes.json
  src/                  # App.tsx, api.ts, features.ts, types.ts, styles.css
  wrangler.jsonc
notebooks/
  01_cancer_mama.ipynb
  02_otimizacao_genetica_cancer_mama.ipynb
  03_interpretacao_llm_cancer_mama.ipynb
resultados/fase2/
  comparacao_baseline_otimizados.csv
  experimentos_ga.csv
  historico_geracoes.csv
  modelo_serving.joblib
  resumo_execucao.json
  treinamento_ga.log
  avaliacao_interpretacoes_llm.csv  # gerado com GROQ_API_KEY
  interpretacoes_llm.json           # gerado com GROQ_API_KEY
  resumo_avaliacao_llm.json         # gerado com GROQ_API_KEY
scripts/
  export_serving_model.py # exporta o manifesto JSON servido na borda
src/
  api.py
  edge_security.py        # validação do Turnstile na borda
  evaluate_llm.py
  genetic_optimization.py
  llm_interpretation.py
  model_inference.py      # inferência a partir do manifesto JSON
  modelo_serving.json     # manifesto do modelo empacotado no Worker
  utils.py
  worker.py               # entrypoint do Python Worker (ASGI + segurança)
tests/
  test_fase2.py
  test_cloudflare.py      # equivalência do manifesto e validação do Turnstile
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

Essa escolha também define o escopo operacional da interpretação por LLM: os
três modelos continuam documentados e comparados na Fase 2, mas o endpoint
`POST /interpret` explica o diagnóstico produzido pelo modelo final
selecionado para serving. Como a Regressão Logística foi o melhor modelo
observado para publicação, ela é o único pipeline serializado em
`modelo_serving.joblib`.

### Interpretação com LLM

O endpoint `POST /interpret` utiliza a API do Groq com o modelo
configurado no ambiente. A LLM recebe a classificação, as probabilidades e as
cinco evidências locais mais relevantes do modelo, sem identificadores, e
devolve uma explicação estruturada para revisão profissional.

O retorno da API mostra `model: "Regressão Logística"` porque a camada de
serving trabalha com um único modelo recomendado após a comparação entre
Regressão Logística, KNN e Árvore de Decisão. Assim, a LLM interpreta o
resultado que seria efetivamente entregue em produção, e não uma votação ou
comparação simultânea entre os três modelos.

O prompt `clinical_explanation_v3` exige quatro seções (`Resumo do Resultado`,
`Evidências do Modelo`, `Insights Acionáveis para Médicos` e `Limitações e
Segurança`). Além do texto livre, o sistema persiste `insights_acionaveis` em
estrutura própria, relacionando sinal, evidência numérica, implicação para
revisão e cautela.

A versão v3 reforça três cuidados específicos do enunciado: (1) contexto de
saúde da mulher, situando os achados em cuidados tipicamente associados a
esse contexto sem presumir dados não fornecidos; (2) sensibilidade de gênero,
proibindo termos alarmistas ou sentenciosos; e (3) privacidade e
confidencialidade, vetando qualquer identificador pessoal da paciente. A
rubrica de avaliação (`evaluate_interpretation_quality`) ganhou o critério
`sensibilidade_cultural_e_genero`, que reprova respostas com linguagem
estigmatizante.

Para gerar interpretações reais:

```powershell
$env:GROQ_API_KEY="sua-chave"      # console.groq.com/keys
$env:GROQ_LLM_MODEL="openai/gpt-oss-120b"  # opcional
python -m src.evaluate_llm
```

Esse comando avalia casos determinísticos de alto risco, baixo risco, limiar e
faixas adicionais de probabilidade. Ele salva a rubrica objetiva em
`resultados/fase2/avaliacao_interpretacoes_llm.csv`, as interpretações em
`resultados/fase2/interpretacoes_llm.json` e o consolidado em
`resultados/fase2/resumo_avaliacao_llm.json`.
Sem `GROQ_API_KEY`, o fluxo local, o prompt e os testes funcionam, mas não
há resposta real de LLM para reportar.

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

Os notebooks da Fase 2 são:

- `notebooks/02_otimizacao_genetica_cancer_mama.ipynb`: otimização genética;
- `notebooks/03_interpretacao_llm_cancer_mama.ipynb`: interpretação com GPT e avaliação.

## Aplicação na Cloudflare

O repositório inclui uma implantação integral e isolada na Cloudflare, com
frontend React no **Pages**, API em **Python Workers**, artefatos no **R2** e
proteção por **Turnstile**, **Rate Limiting** e **Service Bindings**.

### Acesso ao ambiente

| Recurso | Endereço |
| --- | --- |
| **Aplicação web** | https://cancer-mama-web-preview.pages.dev |
| **API (Worker)** | https://cancer-mama-api-preview.antonio-5b5.workers.dev |
| Health check | https://cancer-mama-api-preview.antonio-5b5.workers.dev/health |
| Swagger UI | https://cancer-mama-api-preview.antonio-5b5.workers.dev/docs |

Abra a **aplicação web**, preencha as 30 medições (ou use "Preencher exemplo
acadêmico"), e escolha *Classificar* ou *Classificar e interpretar*. A
interpretação é protegida por Turnstile e exibida em Markdown formatado.

### Infraestrutura de nuvem

| Serviço | Recurso | Papel |
| --- | --- | --- |
| Cloudflare Pages | `cancer-mama-web-preview` | Frontend estático + Pages Functions. |
| Python Workers | `cancer-mama-api-preview` | FastAPI via ASGI (Pyodide): inferência e LLM. |
| R2 | `cancer-mama-artifacts-preview` | Artefatos imutáveis por commit (`builds/<sha>/`). |
| Service Binding | `API` | Liga a Pages Function ao Worker sem internet pública. |
| Turnstile + Rate Limiting | — | Desafio anti-bot e limites em `/predict` e `/interpret`. |

A implantação não usa D1, não persiste entradas ou resultados clínicos e não
altera os algoritmos de treinamento. O `joblib` original continua versionado;
o Worker usa um manifesto JSON matematicamente equivalente, validado nas 569
amostras do dataset.

### Documentação detalhada

- **Frontend** (stack, estrutura, fluxo de dados, build e testes):
  [docs/frontend.md](docs/frontend.md).
- **Infraestrutura e deploy** (serviços de nuvem, secrets, deploy via CLI ou
  Cloudflare Builds e critérios de validação):
  [docs/cloudflare-deploy.md](docs/cloudflare-deploy.md).

### Procedimento rápido de deploy

Publicação direta via `wrangler` autenticado (detalhes e o fluxo por GitHub
Builds em [docs/cloudflare-deploy.md](docs/cloudflare-deploy.md)):

```bash
# API (Python Worker) + secrets
cd cloudflare/api && uv run pywrangler deploy
printf '%s' "$GROQ_API_KEY" | wrangler secret put GROQ_API_KEY

# Frontend (Pages)
cd frontend && npm ci
VITE_TURNSTILE_SITE_KEY=1x00000000000000000000AA npm run build
wrangler pages deploy --branch feat/cloudflare-fullstack
```

Com a API em execução:

- **Swagger UI (documentação interativa): `http://127.0.0.1:8000/docs`**
- Health check: `http://127.0.0.1:8000/health`
- Métricas Prometheus: `http://127.0.0.1:8000/metrics`

Com `GROQ_API_KEY` configurada, a Swagger UI também permite testar
`POST /interpret`.

> A raiz `http://127.0.0.1:8000/` não tem rota definida e retorna
> `{"detail":"Not Found"}` — isso é esperado. Use o Swagger UI (`/docs`) ou
> as rotas abaixo.

### Como chamar cada endpoint

Todas as 30 features abaixo são obrigatórias e devem usar exatamente os
nomes originais do dataset Wisconsin (`radius_mean`, `texture_mean`, ...,
`fractal_dimension_worst`).

**`GET /health/live`** — liveness do processo, sempre responde se a API
está no ar:

```bash
curl http://127.0.0.1:8000/health/live
```

**`GET /health` (ou `/health/ready`)** — readiness; falha com `503` se o
modelo não carregou:

```bash
curl http://127.0.0.1:8000/health
```

**`GET /metrics`** — métricas no formato Prometheus:

```bash
curl http://127.0.0.1:8000/metrics
```

**`POST /predict`** — classificação e probabilidades a partir das 30
features:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "radius_mean": 17.99, "texture_mean": 10.38, "perimeter_mean": 122.8,
      "area_mean": 1001.0, "smoothness_mean": 0.1184, "compactness_mean": 0.2776,
      "concavity_mean": 0.3001, "concave points_mean": 0.1471, "symmetry_mean": 0.2419,
      "fractal_dimension_mean": 0.07871, "radius_se": 1.095, "texture_se": 0.9053,
      "perimeter_se": 8.589, "area_se": 153.4, "smoothness_se": 0.006399,
      "compactness_se": 0.04904, "concavity_se": 0.05373, "concave points_se": 0.01587,
      "symmetry_se": 0.03003, "fractal_dimension_se": 0.006193, "radius_worst": 25.38,
      "texture_worst": 17.33, "perimeter_worst": 184.6, "area_worst": 2019.0,
      "smoothness_worst": 0.1622, "compactness_worst": 0.6656, "concavity_worst": 0.7119,
      "concave points_worst": 0.2654, "symmetry_worst": 0.4601, "fractal_dimension_worst": 0.1189
    }
  }'
```

Resposta esperada:

```json
{
  "prediction": 0,
  "diagnosis": "Maligno",
  "probability_malignant": 1.0,
  "probability_benign": 0.0,
  "model": "Regressao Logistica"
}
```

**`POST /interpret`** — mesmo corpo de `/predict`, mas exige
`GROQ_API_KEY` configurada no ambiente da API e devolve explicação em
linguagem natural, evidências e insights acionáveis:

```bash
curl -X POST http://127.0.0.1:8000/interpret \
  -H "Content-Type: application/json" \
  -d @requisicao.json
```

Onde `requisicao.json` tem o mesmo formato `{"features": {...}}` usado em
`/predict`. Você pode gerar esse arquivo a partir do dataset com:

```bash
python -c "import json,pandas as pd; d=pd.read_csv('data/cancer_mama.csv').drop(columns=['id','diagnosis']); print(json.dumps({'features': d.iloc[0].to_dict()}))" > requisicao.json
```

Sem `GROQ_API_KEY`, `/interpret` responde `503` informando que a LLM não
está disponível, enquanto `/predict` continua funcionando normalmente.

## Container e Autoscaling

```bash
docker build -t tech-challenge-fase2-api:latest .
# somente para habilitar POST /interpret no cluster:
kubectl create secret generic cancer-mama-llm-secrets --from-literal=groq-api-key="$GROQ_API_KEY"
kubectl apply -k deploy/k8s
kubectl get deployment,service,hpa
```

Sem o Secret da chave, a API continua disponivel para `/predict`, enquanto
`/interpret` informa que a LLM nao esta configurada.

O HPA mantém entre 2 e 10 réplicas da API e escala com alvo de 60% de
utilização média de CPU. O cluster deve possuir `metrics-server`.

## Limitações

Este projeto é acadêmico. O dataset possui apenas 569 amostras, não houve
validação externa, e a API não deve ser usada como diagnóstico clínico
autônomo.

## Autores

Antonio Miranda, Elaine, Marcos Mol, Lucas da Costa, Ricardo Loureiro - AI for Devs (9IADT)
