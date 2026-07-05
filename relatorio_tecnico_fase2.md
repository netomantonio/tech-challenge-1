# Relatorio tecnico - Tech Challenge 2

## Otimizacao de modelos de diagnostico por algoritmos geneticos

## 1. Objetivo e base da continuacao

Este trabalho estende o notebook `01_cancer_mama.ipynb`, que classifica
tumores como malignos (`0`) ou benignos (`1`) a partir do Wisconsin Breast
Cancer Diagnostic Dataset. Foram mantidos:

- 569 amostras e 30 features numericas;
- remocao das colunas `id` e `diagnosis` antes da modelagem;
- divisao treino/teste estratificada de 80/20, com `random_state=42`;
- Regressao Logistica, KNN e Arvore de Decisao como modelos avaliados;
- `recall` da classe Maligno como metrica clinica prioritaria.

O objetivo novo e otimizar hiperparametros desses modelos via algoritmo
genetico (AG), comparar o resultado contra os modelos originais e fornecer
uma arquitetura de inferencia monitoravel e escalavel.

## 2. Implementacao do algoritmo genetico

O AG foi implementado em `src/genetic_optimization.py`, sem bibliotecas
especializadas de otimizacao. Um cromossomo e uma tupla de indices que aponta
para valores permitidos de hiperparametros.

| Modelo | Genes otimizados |
| --- | --- |
| Regressao Logistica | `C`, `penalty`, `class_weight` |
| KNN | `n_neighbors`, `weights`, `p` |
| Arvore de Decisao | `max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion`, `class_weight` |

Operadores implementados:

| Operador | Estrategia |
| --- | --- |
| Selecao | Torneio com tres individuos |
| Cruzamento | Uniforme, selecionando cada gene de um dos pais |
| Mutacao | Troca aleatoria de alelo por gene conforme taxa configurada |
| Elitismo | Preservacao do melhor individuo da geracao |

A funcao fitness prioriza o risco clinico de falso negativo:

```text
fitness = 0.65 * recall_maligno + 0.25 * f1_maligno + 0.10 * accuracy
```

Para impedir vazamento de dados, a fitness usa apenas a particao de treino,
com validacao cruzada estratificada de 5 folds. O teste com 114 amostras e
usado somente depois da selecao para relatar o desempenho final.

## 3. Experimentos realizados

As tres configuracoes exigidas foram aplicadas individualmente aos tres
modelos, gerando nove execucoes:

| Experimento | Populacao | Geracoes | Cruzamento | Mutacao |
| --- | ---: | ---: | ---: | ---: |
| E1 - populacao pequena / mutacao baixa | 8 | 6 | 0,80 | 0,08 |
| E2 - balanceado | 12 | 8 | 0,85 | 0,15 |
| E3 - exploratorio | 16 | 10 | 0,90 | 0,30 |

Melhores fitness por execucao, calculadas em validacao cruzada:

| Modelo | E1 | E2 | E3 | Configuracao selecionada |
| --- | ---: | ---: | ---: | --- |
| Regressao Logistica | 0,9686 | **0,9753** | **0,9753** | E2, por ordem deterministica de selecao |
| KNN | **0,9530** | **0,9530** | **0,9530** | E1, por ordem deterministica de selecao |
| Arvore de Decisao | 0,9209 | **0,9275** | 0,9259 | E2 |

Parametros escolhidos:

| Modelo | Parametros selecionados pelo AG |
| --- | --- |
| Regressao Logistica | `C=1.0`, `penalty=l2`, `class_weight=balanced` |
| KNN | `n_neighbors=3`, `weights=distance`, `p=1` |
| Arvore de Decisao | `max_depth=5`, `min_samples_split=6`, `min_samples_leaf=4`, `criterion=gini`, `class_weight=None` |

## 4. Comparacao final com os modelos originais

Resultados sobre o mesmo teste reservado da Fase 1:

| Modelo | Versao | Acuracia | Recall Maligno | F1 Maligno | FN Maligno | AUC-ROC Maligno |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Regressao Logistica | Original | **0,9825** | **0,9762** | **0,9762** | **1** | **0,9954** |
| Regressao Logistica | AG - E2 | 0,9561 | **0,9762** | 0,9425 | **1** | **0,9954** |
| KNN | Original | **0,9737** | **0,9286** | **0,9630** | **3** | **0,9884** |
| KNN | AG - E1 | 0,9649 | **0,9286** | 0,9512 | **3** | 0,9716 |
| Arvore de Decisao | Original | **0,9386** | **0,9286** | **0,9176** | **3** | 0,9342 |
| Arvore de Decisao | AG - E2 | 0,9035 | 0,9048 | 0,8736 | 4 | **0,9358** |

### Interpretacao critica

O AG funcionou como metodo de busca e identificou configuracoes com maior
fitness na validacao cruzada. Entretanto, essa melhoria de validacao nao se
converteu em melhoria no teste reservado:

- a Regressao Logistica otimizada preservou o melhor `recall` maligno
  (0,9762) e um unico falso negativo, mas reduziu acuracia e F1 por produzir
  mais falsos positivos;
- o KNN otimizado manteve recall e falsos negativos do original, mas teve
  queda em F1 e AUC;
- a Arvore otimizada piorou o recall no teste.

Portanto, para uma recomendacao baseada neste teste, a **Regressao Logistica
original permanece o melhor modelo observado**. O melhor individuo otimizado
continua registrado nos artefatos do experimento, mas a API publica o baseline
logistico, evitando servir uma configuracao inferior. Como essa decisao usa o
teste reservado apos a comparacao, suas metricas nao devem ser lidas como nova
estimativa imparcial de desempenho em producao.

Essa decisao tambem delimita a interpretacao com LLM. O trabalho compara os
tres modelos exigidos pela Fase 2, mas o servico operacional expõe somente o
pipeline recomendado em `modelo_serving.joblib`. Portanto, o endpoint
`POST /interpret` explica o diagnostico produzido pelo modelo final
selecionado para serving, que neste experimento foi a Regressao Logistica.
Interpretar KNN e Arvore no mesmo endpoint exigiria outro contrato de API e
estrategias de explicabilidade diferentes para cada familia de modelo.

## 5. Monitoramento e logging

O treinamento produz automaticamente:

| Arquivo gerado em `resultados/fase2/` | Finalidade |
| --- | --- |
| `treinamento_ga.log` | Registro de fitness, recall e candidatos por geracao |
| `experimentos_ga.csv` | Resultado das nove buscas |
| `historico_geracoes.csv` | Curvas de evolucao do AG |
| `comparacao_baseline_otimizados.csv` | Comparacao final no teste |
| `resumo_execucao.json` | Parametros e metricas do campeao |
| `modelo_serving.joblib` | Pipeline recomendado e serializado para inferencia |

A API em `src/api.py` disponibiliza:

- `POST /predict`, com classe e probabilidades;
- `GET /health/live` e `GET /health/ready`;
- `GET /metrics`, compativel com Prometheus;
- logs JSON por predicao, sem registrar as features recebidas.

As metricas medem volume/status de requisicoes, distribuicao de predicoes,
latencia de inferencia e disponibilidade do modelo.

## 6. Arquitetura escalavel

A camada de serving possui container proprio (`Dockerfile`) e manifestos
Kubernetes em `deploy/k8s/`:

| Recurso | Configuracao |
| --- | --- |
| `Deployment` | 2 replicas iniciais, probes e limites de recursos |
| `Service` | Exposicao interna HTTP da API |
| `HorizontalPodAutoscaler` | 2 a 10 replicas, alvo de 60% de CPU |

O scale-down aguarda 300 segundos para reduzir oscilacoes de replicas. O
cluster precisa do `metrics-server` para o HPA de CPU; Prometheus pode coletar
`/metrics` para dashboards e alertas.

## 7. Integracao com LLM para interpretacao

A camada de interpretacao foi implementada em `src/llm_interpretation.py` e
integrada a API pelo endpoint `POST /interpret`. A integracao usa uma LLM
pre-treinada por meio da API do Groq, com modelo configuravel por ambiente.

O endpoint interpreta o resultado do modelo recomendado para serving. Embora
Regressao Logistica, KNN e Arvore de Decisao sejam otimizados e comparados, a
API publica apenas o campeao operacional da comparacao final. Por isso, as
respostas de `/interpret` retornam `model: "Regressao Logistica"` quando esse
e o artefato selecionado em `modelo_serving.joblib`.

A LLM nao recebe identificador do paciente. O contexto enviado contem:

- classe prevista e probabilidades de malignidade/benignidade;
- as cinco contribuicoes locais de maior magnitude da Regressao Logistica;
- insights acionaveis estruturados derivados dessas contribuicoes;
- instrucoes fixas para produzir texto orientado a revisao profissional.

### 7.1 Prompt engineering

O prompt `clinical_explanation_v3` obriga quatro secoes:

1. `RESUMO DO RESULTADO`;
2. `EVIDENCIAS DO MODELO`;
3. `INSIGHTS ACIONAVEIS PARA MEDICOS`;
4. `LIMITACOES E SEGURANCA`.

As instrucoes impedem que a resposta declare diagnostico confirmado ou
prescreva tratamento. Tambem exigem a mencao da natureza academica do modelo,
ausencia de validacao externa e necessidade de revisao clinica. Alem do texto
livre, cada resposta persiste `insights_acionaveis` com sinal observado,
evidencia numerica, implicacao para revisao e cautela de interpretacao.

A versao v3 acrescenta tres regras especificas para o contexto de saude da
mulher: (1) situar os achados em cuidados tipicos desse contexto (ex.:
acompanhamento ginecologico/mastologico) sem presumir dados nao fornecidos;
(2) vedar linguagem alarmista, sentenciosa ou estigmatizante (ex.: "sentenca
de morte", "fatal", "sem cura"); e (3) proibir qualquer identificador pessoal
da paciente no texto gerado, reforcando privacidade e confidencialidade. As
instrucoes tambem passaram a exigir que os insights acionaveis incluam
proximos passos praticos (agendamento de exames, encaminhamento) pensando na
realidade de acesso da paciente ao sistema de saude, sem prescrever conduta.

### 7.2 Avaliacao de qualidade

O modulo `src/evaluate_llm.py` seleciona casos deterministicos:

| Caso | Objetivo |
| --- | --- |
| Alto risco maligno | Avaliar clareza em resultado de alerta |
| Alto risco benigno | Verificar se a resposta evita falsa seguranca |
| Caso proximo do limiar | Avaliar comunicacao de maior incerteza |
| Faixas de probabilidade | Verificar consistencia em perfis intermediarios |

Cada interpretacao e avaliada por uma rubrica objetiva que verifica classe
prevista, probabilidade, secoes exigidas, insights acionaveis, limitacao
explicita, orientacao de revisao profissional, ausencia de prescricao e
sensibilidade cultural/de genero (`sensibilidade_cultural_e_genero`, que
reprova linguagem alarmista ou estigmatizante). A rubrica normaliza acentos,
Markdown e variacoes simples de titulo para reduzir falsos negativos de
formatacao. O endpoint tambem expoe contagem, latencia e distribuicao dessa
pontuacao em `/metrics`.

Com `GROQ_API_KEY` configurada, o comando abaixo regenera as tabelas reais:

```bash
python -m src.evaluate_llm
```

Esse comando gera `avaliacao_interpretacoes_llm.csv`, `interpretacoes_llm.json`
e `resumo_avaliacao_llm.json` em `resultados/fase2/`. Na ultima execucao, com
o prompt `clinical_explanation_v3`, os 7 casos representativos atingiram
`score_objetivo = 1.0` (media, minimo e maximo), sem nenhum criterio
reprovado. A avaliacao automatica deve ser complementada por avaliacao humana
especializada antes de qualquer uso clinico.

## 8. Reproducao

```bash
pip install -r requirements.txt
python data/download_datasets.py
python -m src.genetic_optimization --data data/cancer_mama.csv --output resultados/fase2
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Os notebooks executaveis encontram-se em
`notebooks/02_otimizacao_genetica_cancer_mama.ipynb` e
`notebooks/03_interpretacao_llm_cancer_mama.ipynb`. A documentacao detalhada
da arquitetura encontra-se em `docs/arquitetura_fase2.md`.

## 9. Limitacoes

- O numero de amostras e baixo para concluir desempenho clinico.
- Nao foi feita validacao com dados externos.
- Multiplas buscas aumentam risco de selecionar configuracoes especificas
  demais ao conjunto de treino, ainda que o teste permaneça separado.
- O sistema e academico e nao substitui avaliacao medica.
