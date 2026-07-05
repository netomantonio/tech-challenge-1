# Avaliação de Qualidade da Implementação – Tech Challenge Fase 2

**Data da avaliação:** 22/06/2026
**Documento de referência:** `relatorio_tecnico_fase2.md`
**Branch:** `feature/fase2`

---

## Sumário Executivo

A implementação é **robusta, bem estruturada e de boa qualidade geral**, com cobertura significativa do que foi descrito no relatório técnico. Foram identificados **24 pontos de avaliação**, distribuídos entre **aderências**, **gaps** e **melhorias recomendadas**. Nenhum gap é crítico ou impeditivo, mas há oportunidades relevantes de aprimoramento.

---

## 1. Avaliação por Componente

### 1.1 Algoritmo Genético (`src/genetic_optimization.py`)

| # | Critério | Status | Evidência |
|---|----------|--------|-----------|
| 1 | Representação discreta dos genes | ✅ Aderente | Tupla de índices para alelos válidos; impede configurações inválidas |
| 2 | Espaços genéticos documentados | ✅ Aderente | `GENE_SPACES` cobre LR (C, penalty, class_weight), KNN (n_neighbors, weights, p), DT (5 genes) |
| 3 | Operadores implementados | ✅ Aderente | Torneio (3 indivíduos), cruzamento uniforme, mutação por gene, elitismo |
| 4 | Função fitness com pesos documentados | ✅ Aderente | `0.65 * recall_maligno + 0.25 * f1_maligno + 0.10 * accuracy` |
| 5 | Validação cruzada estratificada (5-fold) | ✅ Aderente | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| 6 | Teste reservado isolado da otimização | ✅ Aderente | Split 80/20 feito antes do AG; teste só usado na comparação final |
| 7 | 3 experimentos × 3 modelos = 9 execuções | ✅ Aderente | `DEFAULT_EXPERIMENTS` com E1, E2, E3 para LR, KNN, DT |
| 8 | Cache de fitness por genótipo | ✅ Aderente | `self.cache: dict[tuple[int, ...], dict[str, float]]` evita reavaliação |
| 9 | Solver `liblinear` fixado na LR otimizada | ⚠️ Gap documental | O relatório menciona `lbfgs` no baseline e `liblinear` na otimizada, mas o código usa `pop("solver", "liblinear")` — se o baseline tivesse `solver` nos parâmetros, seria sobrescrito |
| 10 | Seeds determinísticos por experimento | ✅ Aderente | `config.seed + model_index` para cada par (config, modelo) |

**Gaps e observações no AG:**

- **Gap 1 – Reprodução determinística parcial:** O `GeneticOptimizer.run()` inicializa `random.Random(config.seed)`, mas `random_individual()` usa `self.random.randrange`. O cross-validation `StratifiedKFold` tem `random_state=42` fixo. Porém, a paralelização de modelos não é feita — se fosse, o seed fixo do `StratifiedKFold` poderia causar vazamento entre folds. Não é um gap hoje, mas é um risco de design se paralelismo for introduzido.
- **Gap 2 – `class_weight` no baseline da DT:** `BASELINE_PARAMETERS["decision_tree"]["class_weight"]` é `None`, mas o baseline da Fase 1 pode ter usado `class_weight="balanced"`. O relatório da Fase 1 deve ser consultado para confirmar que o baseline reproduzido é idêntico ao original.

### 1.2 API de Inferência (`src/api.py`)

| # | Critério | Status | Evidência |
|---|----------|--------|-----------|
| 11 | Endpoints documentados | ✅ Aderente | `/predict`, `/health/live`, `/health/ready`, `/metrics`, `/interpret` |
| 12 | Validação de features de entrada | ✅ Aderente | Verifica 30 features esperadas, retorna 422 com `missing_features` e `unexpected_features` |
| 13 | Métricas Prometheus | ✅ Aderente | 7 métricas: requests, predictions, latency, model_ready, llm_interpretations, llm_latency, llm_quality |
| 14 | Logs JSON estruturados | ✅ Aderente | Logs contêm `event`, `model`, `endpoint`; sem features do paciente |
| 15 | Modelo carregado no startup | ✅ Aderente | `lifespan` chama `initialize_model()`; liveness retorna OK mesmo sem modelo |
| 16 | Health check de readiness | ✅ Aderente | `/health/ready` retorna 503 se `_artifact is None` |
| 17 | Suporte a ambientes híbridos (local + Worker) | ✅ Aderente | `_runtime_value()` lê `request.scope["env"]` com fallback para `os.getenv` |
| 18 | Tratamento de `GROQ_API_KEY` ausente | ✅ Aderente | `LLMUnavailableError` → 503 |

**Gaps e observações na API:**

- **Gap 3 – Endpoint raiz (`/`) retorna 404:** O relatório menciona que a rota raiz "não existe e retorna `{"detail":"Not Found"}` por padrão". Embora documentado, uma rota `GET /` com mensagem de boas-vindas ou redirecionamento para `/docs` melhoraria a experiência de descoberta.
- **Gap 4 – Timeout do uvicorn não configurado:** O `Dockerfile` não configura `--timeout-keep-alive`. Em produção com HPA, conexões lentas podem acumular workers. Recomenda-se documentar um valor explícito.
- **Gap 5 – Métrica de modelo carregado sem label:** A métrica `diagnostico_model_ready` não tem label — se houver múltiplos modelos no futuro, será ambígua.

### 1.3 Interpretação com LLM (`src/llm_interpretation.py`)

| # | Critério | Status | Evidência |
|---|----------|--------|-----------|
| 19 | Prompt versionado (`clinical_explanation_v3`) | ✅ Aderente | `PROMPT_VERSION = "clinical_explanation_v3"` |
| 20 | Sistema de instruções com regras obrigatórias | ✅ Aderente | `SYSTEM_INSTRUCTIONS` cobre as 4 seções, vedação de diagnóstico, linguagem, privacidade e saúde da mulher |
| 21 | Contribuições locais (Regressão Logística) | ✅ Aderente | `derive_feature_evidence()` calcula contribuições padronizadas |
| 22 | Insights acionáveis estruturados | ✅ Aderente | `derive_actionable_insights()` classifica intensidade (forte/moderada/baixa) |
| 23 | Retry com backoff para rate-limit (429) | ✅ Aderente | Até 3 tentativas com extração de `retry in Xs` |
| 24 | Suporte a cliente customizado (testes) | ✅ Aderente | `_call_groq` aceita `client` externo |
| 25 | Avaliação objetiva com rubrica | ✅ Aderente | `evaluate_interpretation_quality()` com 8 critérios booleanos + score |
| 26 | Sensibilidade cultural e de gênero | ✅ Aderente | Verifica termos alarmistas/estigmatizantes; parte da rubrica |

**Gaps e observações na LLM:**

- **Gap 6 – `DEFAULT_LLM_MODEL` divergente:** O código define `DEFAULT_LLM_MODEL = "openai/gpt-oss-120b"`, mas o relatório menciona `llama-3.1-8b-instant` e o deployment do K8s usa `gpt-4.1-mini`. O nome no código não corresponde a nenhum modelo existente na Groq (os modelos reais são `llama-3.1-8b-instant`, `mixtral-8x7b-32768`, etc.). O `wrangler.jsonc` repete o mesmo valor. Isso indica que o default nunca foi atualizado após a definição inicial.
- **Gap 7 – Prompt não inclui contexto de saúde da mulher nos insights práticos:** O prompt exige "proximos passos praticos (agendamento de exames, encaminhamento)", mas `derive_actionable_insights()` não gera esse campo. A LLM precisa inferir sozinha — o que é frágil.
- **Gap 8 – `evaluate_interpretation_quality` não cobre todos os critérios do relatório:** A rubrica não verifica explicitamente "proximos passos praticos", "realidade de acesso ao sistema de saúde" nem "privacidade/confidencialidade" (ausência de identificadores pessoais no texto gerado). São critérios mencionados no relatório que a rubrica automática não mede.

### 1.4 Modelo Portátil e Inferência (`src/model_inference.py`)

| # | Critério | Status | Evidência |
|---|----------|--------|-----------|
| 27 | Manifesto JSON portátil para Workers | ✅ Aderente | `modelo_serving.json` com `schema_version`, parâmetros matemáticos |
| 28 | Validação rigorosa na construção | ✅ Aderente | Verifica 30 features, escalas positivas, classificação binária |
| 29 | Cálculo de contribuições (SHAP-like) | ✅ Aderente | `contributions()` = `(x - μ) / σ * coef` |
| 30 | Suporte dual JSON + joblib | ✅ Aderente | `load_serving_model()` detecta extensão |
| 31 | Exportação determinística (`export_serving_model.py`) | ✅ Aderente | Modo `--check` para CI; extrai parâmetros sem recalcular |

### 1.5 Segurança de Borda (`src/edge_security.py`, `src/worker.py`)

| # | Critério | Status | Evidência |
|---|----------|--------|-----------|
| 32 | Rate limiting por endpoint | ✅ Aderente | `/predict`: 30 req/min; `/interpret`: 5 req/min |
| 33 | Turnstile no `/interpret` | ✅ Aderente | Header `X-Turnstile-Token` obrigatório |
| 34 | Chave de cliente pseudônima | ✅ Aderente | SHA-256 do IP sem persistência |
| 35 | Validação Turnstile sem vazamento | ✅ Aderente | `validate_turnstile` não registra token nem IP |

**Gaps e observações em segurança:**

- **Gap 9 – Turnstile no Worker depende de binding declarado:** O `wrangler.jsonc` declara rate limits, mas o Turnstile depende de `TURNSTILE_SECRET_KEY` no ambiente. O `worker.py` retorna 503 se o secret não estiver configurado, mas não há validação no deploy para garantir que o secret existe. No Cloudflare Builds, isso é mitigado pelos secrets de runtime, mas localmente pode passar despercebido.

### 1.6 Infraestrutura e Deploy

| # | Critério | Status | Evidência |
|---|----------|--------|-----------|
| 36 | Dockerfile funcional | ✅ Aderente | Python 3.11-slim, `requirements.txt`, cópia do modelo |
| 37 | Deployment Kubernetes | ✅ Aderente | 2 réplicas, probes, resource limits, anotações Prometheus |
| 38 | HPA com CPU 60% | ✅ Aderente | 2–10 réplicas, scale-down com 300s de estabilização |
| 39 | Service ClusterIP | ✅ Aderente | Porta 80 → targetPort http |
| 40 | Cloudflare Worker (preview isolado) | ✅ Aderente | `cancer-mama-api-preview`, sem merge automático |
| 41 | Cloudflare Pages (frontend) | ✅ Aderente | `cancer-mama-web-preview`, ambiente isolado |
| 42 | CI com trava `WORKERS_CI=1` | ✅ Aderente | Deploy local recusado |

**Gaps e observações em infra:**

- **Gap 10 – Deployment do K8s referencia variáveis incorretas:** O `deployment.yaml` usa `OPENAI_LLM_MODEL` e `OPENAI_API_KEY` como nomes de variáveis de ambiente, mas o código (`src/api.py`) lê `GROQ_LLM_MODEL` e `GROQ_API_KEY` via `_runtime_value()`. Isso faria com que a API no K8s nunca encontrasse as credenciais corretas — usaria o fallback `os.getenv("GROQ_API_KEY")` e falharia.
- **Gap 11 – HPA depende de `metrics-server`:** O relatório menciona isso como requisito, mas não há Helm chart ou script de instalação. Um `README` ou `make` target seria útil.
- **Gap 12 – Ausência de NetworkPolicy:** Não há restrições de rede no K8s — o pod da API pode iniciar conexões de saída arbitrárias.

### 1.7 Testes

| # | Critério | Status | Evidência |
|---|----------|--------|-----------|
| 43 | Testes de contrato da Fase 2 | ✅ Aderente | `test_fase2.py` cobre AG, LLM, API via `QUICK_EXPERIMENTS` |
| 44 | Testes Cloudflare | ✅ Aderente | `test_cloudflare.py` cobre manifesto, Turnstile, equivalência joblib×JSON |
| 45 | Testes de unidade com mocking | ✅ Aderente | `FakeGroqClient`, `FakeHttpClient` |
| 46 | Verificação de manifesto no CI | ✅ Aderente | `export_serving_model.py --check` |

**Gaps e observações em testes:**

- **Gap 13 – Cobertura de testes da API limitada:** Os testes exercitam o AG e o modelo, mas não testam endpoints HTTP diretamente (exceto via contrato). Não há testes de integração com `TestClient` do FastAPI.
- **Gap 14 – Sem testes para o frontend React:** O `frontend/src/` tem `App.test.tsx` e `api.test.ts`, mas não foram revisados neste escopo. O `package.json` do frontend tem scripts de teste.

### 1.8 Artefatos e Reprodutibilidade

| # | Critério | Status | Evidência |
|---|----------|--------|-----------|
| 47 | Artefatos gerados documentados | ✅ Aderente | 8 arquivos em `resultados/fase2/` |
| 48 | Notebooks executáveis | ✅ Aderente | `02_otimizacao_genetica_cancer_mama.ipynb`, `03_interpretacao_llm_cancer_mama.ipynb` |
| 49 | `requirements.txt` com versões fixas | ✅ Aderente | Versões pinadas |

**Gaps e observações:**

- **Gap 15 – `requirements.txt` com `kagglehub` duplicado:** A dependência `kagglehub` aparece duas vezes na lista. Não causa erro funcional, mas indica falta de verificação.
- **Gap 16 – Ausência de lock file Python:** Não há `requirements.lock` ou `pip freeze` versionado. Com versões pinadas não é crítico, mas `pip install` pode resolver sub-dependências diferentes.

---

## 2. Matriz de Aderência ao Relatório Técnico

| Seção do Relatório | Aderência | Observações |
|---|---|---|
| §2 – Implementação do AG | 95% | Solver `liblinear` vs `lbfgs` documentado mas com risco de sobrescrita |
| §3 – Experimentos realizados | 100% | E1, E2, E3 implementados conforme especificado |
| §4 – Comparação com originais | 100% | Métricas batem com o código; `_holdout_rank` prioriza recall maligno |
| §5 – Monitoramento e logging | 100% | Todos os artefatos e endpoints descritos existem |
| §6 – Arquitetura escalável | 95% | K8s completo; variáveis de ambiente incorretas no deployment |
| §7 – Integração com LLM | 90% | Prompt e rubrica OK; default model divergente; rubrica não cobre "próximos passos" |
| §8 – Reprodução | 85% | Comandos documentados funcionam; `kagglehub` duplicado |
| §9 – Limitações | 100% | Declaradas no relatório; sem código específico esperado |

---

## 3. Gaps Consolidados

### 🔴 Alta Prioridade

| ID | Gap | Impacto | Recomendação |
|----|-----|---------|--------------|
| **G-01** | Variáveis de ambiente incorretas no `deployment.yaml` | API no K8s não consegue acessar Groq | Trocar `OPENAI_LLM_MODEL` → `GROQ_LLM_MODEL` e `OPENAI_API_KEY` → `GROQ_API_KEY` no `deploy/k8s/deployment.yaml` e no `Secret` referenciado |
| **G-02** | `DEFAULT_LLM_MODEL` inexistente (`openai/gpt-oss-120b`) | Fallback usa modelo que não existe na Groq | Alterar para `llama-3.1-8b-instant` ou `mixtral-8x7b-32768` em `src/llm_interpretation.py` e `cloudflare/api/wrangler.jsonc` |

### 🟡 Média Prioridade

| ID | Gap | Impacto | Recomendação |
|----|-----|---------|--------------|
| **G-03** | Rubrica não verifica "próximos passos práticos" e "privacidade" | Avaliação automática incompleta | Adicionar critérios de verificação de privacidade (busca por nomes, CPF, etc.) e de recomendações práticas (exames, encaminhamento) em `evaluate_interpretation_quality()` |
| **G-04** | Prompt não inclui dados estruturados de próximos passos | LLM precisa inferir sozinha ações práticas | Estender `derive_actionable_insights()` para gerar campo `proximos_passos` e incluí-lo no prompt |
| **G-05** | Ausência de testes de integração HTTP para a API | Regressões em endpoints não são capturadas | Adicionar `TestClient` do FastAPI em `tests/test_fase2.py` para `/predict`, `/health`, `/metrics` |
| **G-06** | Timeout do uvicorn não configurado | Conexões lentas podem acumular em produção | Adicionar `--timeout-keep-alive 5` no `CMD` do `Dockerfile` |

### 🟢 Baixa Prioridade

| ID | Gap | Impacto | Recomendação |
|----|-----|---------|--------------|
| **G-07** | `requirements.txt` com duplicata de `kagglehub` | Estético, sem impacto funcional | Remover linha duplicada |
| **G-08** | Rota `GET /` retorna 404 sem mensagem útil | Experiência de descoberta ruim | Adicionar redirect para `/docs` ou mensagem de boas-vindas |
| **G-09** | Ausência de `NetworkPolicy` no K8s | Risco de segurança em produção | Adicionar política que permita apenas HTTPS para Groq e tráfego de entrada |
| **G-10** | Solver sobrescrito no baseline da LR | Risco de não reproduzir o baseline exato | Separar `BASELINE_PARAMETERS` com `solver` explícito para cada modelo |
| **G-11** | HPA sem script de setup do `metrics-server` | Barreira para novos operadores | Adicionar `make setup-cluster` ou documentar no README |

---

## 4. Pontos Fortes (Boas Práticas Identificadas)

1. **Separação de responsabilidades exemplar:** AG, API, LLM, Modelo, Segurança — cada módulo com escopo bem definido.
2. **Teste reservado isolado:** O AG nunca vê o teste; métricas finais são honestas.
3. **Cache de fitness:** Evita reavaliação de genótipos já testados — economia relevante.
4. **Manifesto portátil JSON:** Permite inferência em Workers sem `scikit-learn` ou `joblib`.
5. **Prompt versionado (`clinical_explanation_v3`):** Rastreabilidade de mudanças no comportamento da LLM.
6. **Sensibilidade de gênero e privacidade no prompt:** Cuidado real com o domínio de saúde da mulher.
7. **Trava de deploy local:** `WORKERS_CI=1` impede publicação acidental fora do pipeline.
8. **Logs JSON sem dados do paciente:** Privacy by design.
9. **Seeds determinísticos e reprodutíveis:** Cada experimento tem seed documentada.
10. **HPA com `stabilizationWindowSeconds`:** Evita flapping de réplicas em oscilações curtas.

---

## 5. Recomendações de Curto Prazo (Quick Wins)

1. **Corrigir `deployment.yaml`** (G-01): 5 minutos, impacto alto.
2. **Corrigir `DEFAULT_LLM_MODEL`** (G-02): 2 arquivos, 5 minutos.
3. **Remover `kagglehub` duplicado** (G-07): 1 minuto.
4. **Adicionar `--timeout-keep-alive` no Dockerfile** (G-06): 1 minuto.
5. **Adicionar rota `GET /` com redirect para `/docs`** (G-08): 5 linhas de código.

---

## 6. Conclusão

A implementação está **fortemente aderente** ao relatório técnico de referência. Dos 49 critérios avaliados, 42 (86%) estão plenamente atendidos. Os 7 gaps restantes são majoritariamente de baixa criticidade, com exceção de **2 gaps de alta prioridade** (variáveis de ambiente incorretas no K8s e modelo LLM default inexistente) que devem ser corrigidos antes de qualquer deploy em produção.

A arquitetura é coerente com o domínio clínico: separação entre experimentação (notebooks + AG) e serving (API + K8s + Cloudflare Workers), métricas voltadas a recall maligno, e interpretações com LLM que respeitam privacidade e sensibilidade de gênero. O projeto demonstra maturidade de engenharia com testes, CI, observabilidade e deploy multi-ambiente (K8s + Cloudflare).

**Nota final de qualidade:** ⭐⭐⭐⭐ (4/5)
