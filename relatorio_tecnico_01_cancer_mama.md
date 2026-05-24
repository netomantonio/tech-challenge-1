# Relatório técnico - Notebook 01: Diagnóstico de Câncer de Mama

## 1. Contexto e objetivo

O notebook `01_cancer_mama.ipynb` desenvolve um pipeline de Machine Learning para classificar tumores de mama como **malignos** ou **benignos** usando o dataset Wisconsin Breast Cancer Diagnostic.

O conjunto possui **569 amostras**, **30 variáveis numéricas** extraídas de imagens de aspiração por agulha fina (FNA), além das colunas `id` e `diagnosis`. A variável alvo foi codificada como:

- `0`: Maligno
- `1`: Benigno

Como o problema está inserido em um contexto médico, a métrica mais importante é o **recall da classe Maligno**. O erro mais crítico é o falso negativo: um tumor maligno previsto como benigno.

## 2. Estratégias de pré-processamento

### 2.1 Carregamento e verificação inicial

O dataset foi carregado a partir de `data/cancer_mama.csv`, gerado pelo script `data/download_datasets.py`. A inspeção inicial mostrou:

- **569 registros**;
- **32 colunas originais**;
- **30 features numéricas**;
- **nenhum valor nulo**.

Como não havia dados ausentes, não foi necessário aplicar imputação. Isso reduz risco de introduzir viés por preenchimento artificial.

### 2.2 Conversão da variável alvo

A coluna original `diagnosis` usa os rótulos `M` e `B`. Ela foi convertida para uma variável numérica chamada `diagnostico`:

- `M` foi mapeado para `0`, representando tumor maligno;
- `B` foi mapeado para `1`, representando tumor benigno.

Depois da conversão, a coluna textual `diagnosis` foi removida para evitar duplicidade da variável alvo.

### 2.3 Tradução e organização das features

As colunas foram traduzidas para português para deixar o notebook mais legível. Exemplos:

- `radius_mean` virou `raio_media`;
- `texture_worst` virou `textura_pior`;
- `concave points_worst` virou `pontos_concavos_pior`;
- `area_se` virou `area_ep`.

Essa etapa não altera os dados, apenas melhora a interpretação das análises e gráficos.

### 2.4 Remoção de variável sem valor preditivo

A coluna `id` foi removida antes da modelagem. Ela identifica cada amostra, mas não representa uma característica clínica ou morfológica do tumor. Mantê-la poderia induzir o modelo a aprender ruído ou padrões artificiais.

Após essa remoção, o conjunto de entrada ficou com **30 features preditivas**.

### 2.5 Análise do balanceamento das classes

A distribuição das classes foi:

| Classe | Quantidade | Proporção |
|---|---:|---:|
| Maligno | 212 | 37,3% |
| Benigno | 357 | 62,7% |

Existe predominância da classe benigna, mas o desbalanceamento não é extremo. Por isso, o notebook priorizou uma divisão estratificada e métricas específicas para malignidade, sem aplicar oversampling ou undersampling.

### 2.6 Divisão treino e teste

Os dados foram divididos em treino e teste com proporção **80/20**, usando `stratify=y` para preservar a distribuição das classes:

| Conjunto | Amostras | Maligno | Benigno |
|---|---:|---:|---:|
| Treino | 455 | 170 (37,4%) | 285 (62,6%) |
| Teste | 114 | 42 (36,8%) | 72 (63,2%) |

A estratificação é importante porque evita que o conjunto de teste fique com uma proporção de malignos muito diferente da base original.

### 2.7 Escalonamento das variáveis

As features possuem escalas muito diferentes. Por exemplo, medidas de área têm valores muito maiores que medidas de suavidade ou dimensão fractal. Isso afeta principalmente modelos baseados em distância, como KNN, e também melhora a estabilidade da Regressão Logística.

Foram comparados dois escaladores com validação cruzada de 5 folds usando Regressão Logística:

| Escalador | F1-score médio | Desvio padrão |
|---|---:|---:|
| StandardScaler | 0,9780 | 0,0099 |
| MinMaxScaler | 0,9712 | 0,0114 |

O **StandardScaler** foi escolhido por apresentar desempenho ligeiramente superior. Ele padroniza as variáveis para média 0 e desvio padrão 1.

Um ponto importante do pipeline é que o escalador foi ajustado apenas no conjunto de treino:

- `fit_transform` no treino;
- `transform` no teste.

Isso evita vazamento de dados, pois informações estatísticas do teste não entram no treinamento.

### 2.8 Correlação entre variáveis

A análise de correlação encontrou **21 pares de features com correlação acima de 0,9**, principalmente entre medidas de raio, perímetro e área. Exemplos:

- `raio_media` e `perimetro_media`: 0,9979;
- `raio_pior` e `perimetro_pior`: 0,9937;
- `raio_media` e `area_media`: 0,9874.

Essas correlações indicam redundância informacional. O notebook optou por manter as 30 features, mas essa observação é importante para interpretar modelos lineares e explicações por SHAP: variáveis correlacionadas devem ser lidas como grupos de sinais relacionados, não como causas isoladas.

## 3. Modelos usados e justificativa

### 3.1 Regressão Logística

A Regressão Logística foi usada como modelo base por ser simples, rápida e adequada para classificação binária com dados tabulares numéricos.

Ela é uma boa escolha inicial porque:

- tem baixo custo computacional;
- produz probabilidades de classe;
- é menos propensa a overfitting em bases pequenas quando comparada a modelos mais flexíveis;
- permite interpretação com coeficientes e SHAP.

No notebook, foi configurada com `max_iter=10000` para garantir convergência.

### 3.2 KNN - K-Nearest Neighbors

O KNN foi usado para testar uma abordagem baseada em similaridade entre amostras. Ele classifica um novo caso observando os vizinhos mais próximos no espaço das features.

Esse modelo foi incluído porque:

- não assume uma fronteira linear;
- pode capturar padrões locais;
- é intuitivo para comparação com a Regressão Logística.

Como o KNN depende diretamente de distâncias, o escalonamento é indispensável. O melhor valor de `K` foi escolhido por validação cruzada, priorizando o recall da classe Maligno e usando o F1-score ponderado como critério auxiliar.

O melhor valor encontrado foi:

- `K = 7`;
- recall maligno médio na validação: **0,9412**;
- F1-score ponderado médio: **0,9691**.

### 3.3 Árvore de Decisão

A Árvore de Decisão foi usada por sua interpretabilidade. Ela cria regras de decisão em formato hierárquico, facilitando a inspeção dos critérios usados pelo modelo.

Ela foi incluída porque:

- captura relações não lineares;
- permite visualizar regras de classificação;
- fornece importância das variáveis;
- serve como contraponto interpretável aos outros modelos.

A profundidade máxima foi escolhida por validação cruzada. O melhor valor encontrado foi:

- `max_depth = 4`;
- F1-score médio na validação: **0,9382**.

Controlar a profundidade é essencial para reduzir overfitting, especialmente em um dataset com apenas 569 amostras.

## 4. Resultados dos modelos

### 4.1 Métricas no conjunto de teste

| Modelo | Acurácia | Precisão weighted | Recall weighted | F1 weighted | Recall Maligno | F1 Maligno | Falsos negativos maligno | AUC-ROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Regressão Logística | 0,9825 | 0,9825 | 0,9825 | 0,9825 | 0,9762 | 0,9762 | 1 | 0,9954 |
| KNN (K=7) | 0,9737 | 0,9747 | 0,9737 | 0,9735 | 0,9286 | 0,9630 | 3 | 0,9884 |
| Árvore de Decisão | 0,9386 | 0,9390 | 0,9386 | 0,9387 | 0,9286 | 0,9176 | 3 | 0,9342 |

### 4.2 Matrizes de confusão

As matrizes abaixo seguem a ordem:

- linhas: classe real;
- colunas: classe prevista;
- ordem das classes: `Maligno`, `Benigno`.

**Regressão Logística**

| Real \ Previsto | Maligno | Benigno |
|---|---:|---:|
| Maligno | 41 | 1 |
| Benigno | 1 | 71 |

**KNN (K=7)**

| Real \ Previsto | Maligno | Benigno |
|---|---:|---:|
| Maligno | 39 | 3 |
| Benigno | 0 | 72 |

**Árvore de Decisão**

| Real \ Previsto | Maligno | Benigno |
|---|---:|---:|
| Maligno | 39 | 3 |
| Benigno | 4 | 68 |

### 4.3 Validação cruzada

Na validação cruzada estratificada com 5 folds, usando recall da classe Maligno:

| Modelo | Recall Maligno médio | Desvio padrão |
|---|---:|---:|
| Regressão Logística | 0,9647 | 0,0288 |
| KNN (K=7) | 0,9235 | 0,0399 |
| Árvore de Decisão | 0,9118 | 0,0416 |

A Regressão Logística também foi o modelo mais estável nessa avaliação.

## 5. Interpretação dos dados e dos resultados

### 5.1 Comparação geral

A **Regressão Logística** foi o melhor modelo no contexto do notebook. Ela apresentou:

- maior acurácia;
- maior F1-score ponderado;
- maior recall para a classe Maligno;
- menor número de falsos negativos;
- maior AUC-ROC.

O principal ponto é que ela deixou passar apenas **1 caso maligno** no conjunto de teste. Como falsos negativos são o erro mais grave neste problema, esse resultado torna a Regressão Logística a escolha mais adequada entre os modelos avaliados.

O **KNN** teve desempenho alto, mas perdeu 3 casos malignos. Ele não classificou nenhum caso benigno como maligno, o que mostra alta precisão para a classe Maligno, mas sua sensibilidade foi menor que a da Regressão Logística.

A **Árvore de Decisão** teve o menor desempenho geral. Apesar disso, foi útil para explicar quais variáveis mais contribuíram para as decisões e para visualizar regras de classificação.

### 5.2 Interpretação das variáveis relevantes

Na Árvore de Decisão, as features mais importantes foram:

| Posição | Feature | Importância |
|---:|---|---:|
| 1 | `raio_pior` | 0,7335 |
| 2 | `pontos_concavos_pior` | 0,1220 |
| 3 | `textura_ep` | 0,0458 |
| 4 | `textura_pior` | 0,0323 |
| 5 | `concavidade_pior` | 0,0172 |

A variável `raio_pior` dominou a explicação da árvore, indicando que medidas extremas de tamanho do tumor foram muito importantes para separar as classes.

Os gráficos SHAP reforçaram essa leitura:

- na Árvore de Decisão, `raio_pior` apareceu como a variável mais dominante;
- na Regressão Logística, a influência foi mais distribuída entre `textura_pior`, `raio_ep`, `area_pior`, `raio_pior`, `pontos_concavos_pior`, `area_ep` e `perimetro_pior`;
- valores altos em medidas de tamanho, área, textura e pontos côncavos tenderam a contribuir para a classe Maligno.

Essa interpretação é coerente com a natureza das features: tumores malignos tendem a apresentar maior irregularidade, maior concavidade e medidas geométricas mais elevadas. Ainda assim, a explicação é estatística e depende do dataset; não deve ser tratada como causalidade médica.

### 5.3 Leitura clínica do erro

Em aplicações médicas, a acurácia isolada pode ser enganosa. O tipo de erro importa.

Neste problema:

- falso negativo: tumor maligno previsto como benigno;
- falso positivo: tumor benigno previsto como maligno.

O falso negativo é mais crítico porque pode atrasar diagnóstico e tratamento. Por isso, o recall maligno foi usado como métrica principal.

Nesse critério, a Regressão Logística foi superior:

- identificou 41 dos 42 casos malignos no teste;
- teve recall maligno de **0,9762**;
- apresentou apenas **1 falso negativo maligno**.

## 6. Limitações

Apesar dos resultados elevados, o projeto possui limitações importantes:

- o dataset tem apenas 569 amostras;
- os dados são estruturados e limpos, diferentes de muitos ambientes clínicos reais;
- não houve validação externa com dados de outros hospitais ou populações;
- as variáveis foram previamente extraídas das imagens, ou seja, o modelo não analisa imagens brutas;
- a alta correlação entre features exige cuidado na interpretação individual de variáveis;
- o resultado não deve ser usado como diagnóstico autônomo.

O modelo pode ser visto como ferramenta experimental de apoio, mas a decisão final deve permanecer com o médico.

## 7. Conclusão

O notebook mostra que modelos clássicos de Machine Learning conseguem bom desempenho no problema de classificação de tumores de mama usando features numéricas do dataset Wisconsin.

A **Regressão Logística** foi o melhor modelo no conjunto de critérios avaliados, principalmente por combinar alto desempenho geral, maior recall da classe Maligno, menor número de falsos negativos e boa interpretabilidade.

O **KNN** apresentou desempenho competitivo, mas menor sensibilidade para malignidade. A **Árvore de Decisão** teve desempenho inferior, porém contribuiu para explicar os principais fatores usados na classificação.

Do ponto de vista técnico, o pipeline está bem estruturado: remove variável sem valor preditivo, preserva a proporção das classes, evita vazamento de dados no escalonamento, compara modelos com validação cruzada e interpreta os resultados com feature importance e SHAP. Do ponto de vista prático, os resultados devem ser entendidos como acadêmicos e não como evidência suficiente para uso clínico real.
