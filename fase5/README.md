# Guardiã AI - Fase 5

Apoio ao profissional no atendimento de mulheres em investigação de câncer de
mama, com espaço no mesmo atendimento para um relato relacionado à segurança.
O ML classifica medidas citológicas do Wisconsin; o relato não entra no
classificador nem gera escore de violência. RAG e LLM ajudam a consultar
orientações e organizar a revisão humana. A decisão final é sempre do profissional.

## Grupo e entrega

| Integrante | RM |
| --- | --- |
| _preencher_ | _preencher_ |
| _preencher_ | _preencher_ |

- Repositório: https://github.com/netomantonio/tech-challenge-1
- Vídeo (YouTube/Vimeo, até 15 min): _preencher após a publicação_
- Relatório técnico: `output/pdf/relatorio_tecnico_fase5.pdf`
- Roteiro do vídeo: `docs/script_video_demonstracao_fase5.txt`

## Executar

Na raiz, com Python 3.11 ou superior:

```powershell
pip install -r requirements-fase5.txt
python -m fase5.train      # treina, compara os modelos e gera os exemplos
python -m fase5.app        # interface em http://127.0.0.1:8050, API em /docs
```

Na interface, escolher um dos casos de exemplo (conjunto de teste público),
editar a pergunta e o relato sintético e clicar em **Analisar**. O resultado
mostra a estimativa do modelo, as medidas que mais influenciaram, a explicação
da LLM com citações clicáveis e as fontes consultadas. No fim, o profissional
registra se concorda, discorda ou considera inconclusivo; a decisão vai para a
auditoria. Pela API, `POST /analisar` recebe o campo `entrada` de um objeto de
`resultados/fase5/exemplos.json` e `POST /revisao` registra a decisão.

O padrão `demo` não chama LLM e se identifica na resposta. Para a LLM real,
preencher no `.env` da raiz (ver `.env.example`):

```
GROQ_API_KEY=<sua-chave>
FASE5_LLM_BACKEND=groq
FASE5_LLM_MODEL=openai/gpt-oss-20b
```

Os valores do `.env` prevalecem sobre os do terminal. Não há fallback
silencioso: se o provedor falhar, a API devolve 503. No modo Groq, medidas,
pergunta e relato vão para o provedor; usar apenas dados públicos ou sintéticos.

Scripts de análise (geram os arquivos de `resultados/fase5/`):

| Comando | O que faz |
| --- | --- |
| `python -m fase5.explore` | exploração e testes de preparo (`exploracao.json`, `figuras/`) |
| `python -m fase5.rag_eval` | compara configurações do RAG (`avaliacao_rag.json`) |
| `python -m fase5.compare_llm` | compara modelos do Groq no fluxo real (`comparacao_llm.json`) |
| `python -m fase5.validate_llm` | executa um caso com LLM real e guarda a resposta (`validacao_llm.json`) |
| `python -m fase5.report` | gera o relatório em PDF a partir desses arquivos e roda os testes |

## Arquitetura

```
Interface web / API (FastAPI)
  -> LangGraph: modelo -> recuperação -> explicação -> auditoria
       modelo:      regressão logística + contribuições locais
       recuperação: TF-IDF sobre 12 documentos públicos (knowledge.json)
       explicação:  LLM via Groq (ou demo sem LLM)
       auditoria:   JSONL sem relato nem medidas brutas
  -> profissional revisa e registra a decisão (POST /revisao)
```

| Arquivo | Papel |
| --- | --- |
| `train.py` | treino, comparação, métricas e seleção dos exemplos |
| `explore.py` | exploração dos dados e teste das decisões de preparo |
| `rag.py` | pré-processamento e recuperação; `rag_eval.py` avalia as variantes |
| `service.py` | grafo LangGraph, prompt, extração de citações e auditoria |
| `app.py` e `web/` | API e interface (HTML/JS puro, mesmo padrão das Fases 3 e 4) |
| `compare_llm.py`, `validate_llm.py`, `smoke.py` | verificações com LLM real e no Docker |

## Dados e preparo

569 amostras, 30 variáveis, 212 malignas e 357 benignas; sem valores ausentes e
sem duplicatas. As medidas vêm de imagens de punção aspirativa (PAAF) e
descrevem núcleos celulares: são dados de exame, não sintomas. A exploração
completa está no notebook da Fase 1 (`notebooks/01_cancer_mama.ipynb`); o
`explore.py` repete o que justifica o preparo:

- escalas muito diferentes (área chega a 4254, dimensão fractal a 0,03), por
  isso a padronização dentro do pipeline;
- 21 pares com |r| > 0,9 (raio, perímetro e área), o que afeta a leitura da
  importância das variáveis;
- outliers em 171 linhas, 67% malignas: são o sinal que queremos detectar, não
  foram removidos.

Testamos o preparo por CV no treino (recall maligno): sem padronização 0,906,
StandardScaler 0,965, log1p + StandardScaler 0,971 e sem as variáveis muito
correlacionadas 0,953. A diferença do log1p equivale a um caso entre ~170;
mantivemos o StandardScaler.

## Modelos e métricas

Divisão estratificada 455/114, seed 42. Seleção por recall maligno médio em CV
de cinco folds no treino, com F1 como desempate; o teste só entra na avaliação final.

| Modelo | Recall CV | Accuracy | Precision maligno | Recall maligno | F1 maligno | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Regressão logística | 0,9647 | 0,9825 | 0,9762 | 0,9762 | 0,9762 | 1 |
| Random Forest | 0,9529 | 0,9561 | 0,9512 | 0,9286 | 0,9398 | 3 |

Falso negativo pode atrasar a investigação; falso positivo gera ansiedade e
exames adicionais. Por isso o recall maligno orienta a escolha, sem esconder
precision e F1. O único falso negativo do teste é um dos casos da interface.

Explicabilidade: a importância global por permutação (queda de recall) coloca
worst smoothness, worst texture e worst symmetry no topo. Isso não coincide com
as variáveis que mais separam as classes isoladamente (worst concave points,
worst perimeter), porque variáveis correlacionadas se substituem quando uma é
permutada. A explicação local troca cada medida pela mediana do treino e mostra
as cinco maiores variações; em casos com probabilidade perto de 0 ou 1 essas
variações ficam quase nulas. Nenhuma das duas é causal.

## RAG e indicação das fontes

`knowledge.json` tem 12 sínteses didáticas escritas a partir de páginas oficiais
(INCA, Ministério da Saúde, UCI, Lei Maria da Penha, Leis 10.778, 12.732,
14.149 e 14.847, Ligue 180, MDS), cada uma com ID, URL e data de consulta.
Não são os protocolos completos das instituições.

A primeira versão (TF-IDF sem stopwords, limiar 0,01, três documentos) devolvia
a base inteira para quase toda pergunta. Em 21 perguntas de atendimento e 5 fora
do escopo:

| Configuração | Acerto top-3 | Precisão | Abstenção fora do escopo | Docs/pergunta |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0,76 | 0,27 | 0,00 | 3,0 |
| stopwords + palavras-chave + prefixo 6 (atual) | 1,00 | 0,83 | 0,60 | 1,4 |

Duas perguntas fora do escopo ainda recuperam um documento por casamento
acidental ("alta", "previsão"), e "medidas" traz também o documento de medidas
protetivas. É o limite de uma busca lexical.

A resposta da API traz as fontes recuperadas (trecho, URL, score) e, separadas,
as fontes que a LLM citou com `[id]`. Citação de ID não recuperado aparece como
citação inválida na interface e na auditoria.

## LLM

O `.env.example` sugeria `llama-3.1-8b-instant`, que saiu da lista do Groq.
Comparamos os modelos disponíveis nos quatro casos de exemplo e numa pergunta
fora do escopo (`comparacao_llm.json`): `gpt-oss-20b` e `gpt-oss-120b` ficaram
equivalentes (citações só de fontes recuperadas, nenhuma inversão de classe);
o 20b ficou como padrão pela latência (~1,4 s contra ~3,7 s). O `qwen3.8-27b`
levou mais de 30 s por resposta.

A comparação revelou três problemas, já corrigidos: os gpt-oss liam "classe 0"
como benigno (agora o rótulo vai por extenso); o 20b esgotava o limite padrão de
tokens raciocinando e devolvia texto vazio ou cortado (limite maior, raciocínio
baixo e resposta truncada tratada como erro); e as citações vinham como `【id】`.

Mesmo citando fontes, a LLM ainda acrescenta orientações gerais que não estão
nelas. O prompt restringe diagnóstico, prescrição e decisões de segurança, mas
não garante obediência. A revisão profissional é obrigatória.

## Auditoria e responsabilidade

`resultados/fase5/auditoria.jsonl`: identificador, data UTC, hashes de entrada e
resposta, modelo, versão, resultado, contribuições, fontes recuperadas e citadas,
citações inválidas, provedor e versão do prompt; e, em linhas separadas, a
decisão do profissional. Não armazena relato livre nem medidas brutas. Os hashes
permitem conferir um registro apresentado depois, mas não reconstruir o
atendimento, e não são anonimização forte. API local sem autenticação: não expor
como serviço de produção.

## Docker e testes

```powershell
docker build -f fase5/Dockerfile -t guardia-fase5 .
docker run --rm -p 127.0.0.1:8050:8050 guardia-fase5
python -m unittest discover -s tests -p 'test_fase5.py' -v
```

Para Groq no Docker: `-e FASE5_LLM_BACKEND=groq -e GROQ_API_KEY` com a chave
definida no host. O treino roda no build. Os logs ficam no container; montar um
volume com permissão do usuário guardia para persistir.

Verificado: 9 testes passando; imagem Docker construída e, dentro do container,
`GET /`, `GET /health` e `POST /analisar` responderam 200 em modo demo
(`validacao_docker.json`); LLM real validada com `openai/gpt-oss-20b`
(`validacao_llm.json`, resposta completa salva).

Pendente: gravar e publicar o vídeo e preencher integrantes e link acima.

Fonte do dataset: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
