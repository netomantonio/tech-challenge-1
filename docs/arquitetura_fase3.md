# Arquitetura e decisoes de implementacao - Tech Challenge 3

## 1. Objetivo

A Fase 3 avanca do modelo de diagnostico das Fases 1/2 para um **assistente
virtual medico** treinado com dados proprios (ficticios) do hospital,
capaz de responder duvidas clinicas de medicos e sugerir consultas a
protocolos internos, com fluxos de decisao automatizados e seguros
coordenados por LangChain/LangGraph.

Este modulo (`fase3/`) e isolado do restante do repositorio: nao altera
nada das Fases 1/2, mantendo o pipeline de diagnostico de cancer de mama
intacto.

## 2. Componentes

| Componente | Responsabilidade |
| --- | --- |
| `fase3/data/build_finetuning_dataset.py` | Preprocessing, anonimizacao (regex) e curadoria do dataset de fine-tuning |
| `fase3/data/protocolos_hospital.json` | Protocolos internos, FAQs e modelos de documento ficticios (PT) |
| `fase3/data/pacientes_sinteticos.json` | Prontuarios ficticios usados para semear o mock de EHR e gerar exemplos de treino |
| `fase3/data/sample_medquad.jsonl` | Amostra real do MedQuAD/PubMedQA (com atribuicao de fonte por linha) |
| `fase3/finetuning/train_lora.py` | Fine-tuning LoRA/PEFT de um LLM causal sobre o dataset combinado |
| `fase3/llm_backend.py` | LLM plugavel compativel com LangChain: `groq`, `local` (adapter LoRA) e `fake` (testes) |
| `fase3/retrieval.py` | Retrieval BM25 sobre os protocolos internos |
| `fase3/ehr_tools.py` | Mock de EHR estruturado (SQLite) semeado a partir dos prontuarios ficticios |
| `fase3/guardrails.py` | Bloqueio de prescricao direta, checagem de PII e disclaimer obrigatorio |
| `fase3/logging_utils.py` | Log de auditoria estruturado (stdout JSON + `resultados/fase3/auditoria.jsonl`) |
| `fase3/assistant_chain.py` | Pipeline LangChain (retrieval + LLM + guardrails + explainability) |
| `fase3/clinical_flow_graph.py` | Fluxo de decisao LangGraph (exames pendentes, sugestao, alertas) |
| `fase3/evaluate_assistant.py` | Avaliacao deterministica sobre casos representativos |
| `fase3/cli_demo.py` | Demo de ponta a ponta usada na gravacao do video |
| `notebooks/04_assistente_medico_fase3.ipynb` | Relatorio executavel do pipeline completo |

Fluxo geral:

```mermaid
flowchart TD
    PROT[("protocolos_hospital.json")] --> BUILD
    PAC[("pacientes_sinteticos.json")] --> BUILD
    MEDQUAD[("sample_medquad.jsonl\nMedQuAD/PubMedQA")] --> BUILD

    subgraph BUILD["build_finetuning_dataset.py"]
        direction TB
        ANON["Anonimizacao\n(regex: CPF, telefone, e-mail, nome)"] --> CURA["Curadoria\n(dedup + filtro de tamanho)"]
    end

    BUILD --> DATASET[("finetuning_train.jsonl\nfinetuning_val.jsonl")]
    DATASET --> LORA["train_lora.py\nLoRA/PEFT sobre modelo causal pequeno"]
    LORA --> ADAPTER[("lora_adapter/\n+ training_summary.json")]

    ADAPTER -.-> BACKEND
    GROQAPI["Groq API"] -.-> BACKEND

    subgraph ASSISTENTE["fase3.assistant_chain (LangChain)"]
        BACKEND["llm_backend.get_llm()\ngroq | local | fake"]
        RETR["retrieval.py\nBM25 sobre protocolos"]
        EHR["ehr_tools.py\nSQLite mock de prontuario"]
        CHAIN["prompt | llm | StrOutputParser"]
        GUARD["guardrails.py\nbloqueio + disclaimer"]
        RETR --> CHAIN
        EHR --> CHAIN
        BACKEND --> CHAIN
        CHAIN --> GUARD
    end

    subgraph GRAFO["clinical_flow_graph (LangGraph)"]
        direction LR
        BP["buscar_paciente"] -->|encontrado| VEP["verificar_exames_pendentes"]
        BP -->|nao encontrado| AUD
        VEP --> ST["sugerir_tratamento"]
        ST --> CS["checar_seguranca"]
        CS --> EA["emitir_alertas"]
        EA --> AUD["registrar_auditoria"]
    end

    ASSISTENTE --> ST
    GRAFO --> LOG[("resultados/fase3/auditoria.jsonl")]
```

## 3. Fine-tuning LoRA/PEFT

### 3.1 Dataset

`fase3/data/build_finetuning_dataset.py` combina tres fontes em um formato
unico `{instruction, input, output, source_type, source_id}`:

| Fonte | Exemplos | Observacao |
| --- | --- | --- |
| Protocolos internos (`PROT-001`..`PROT-012`) | Explicacao de protocolo + FAQs internas | Ficticios, escritos para este projeto |
| Prontuarios sinteticos | Perguntas sobre exames pendentes por paciente | Identificacao apenas por codigo (`PAC-000x`), nunca nome |
| MedQuAD (CancerGov, foco Breast Cancer) | 12 pares de pergunta/resposta | Dominio publico (orgao do governo dos EUA), via `abachaa/MedQuAD` |
| PubMedQA | 8 pares de pergunta/resposta (long answer) | Resumos de abstracts do PubMed, via `pubmedqa/pubmedqa` |
| Exemplos clínicos alinhados | 18 pares com paráfrases | Preservação de números, fontes, validação médica e seis intenções clínicas |

Cada texto passa por `anonimizar_texto()` (regex para CPF, telefone,
e-mail e rotulos de nome) antes de entrar no dataset — defensivo mesmo
sobre fontes ja sinteticas, simulando o que rodaria sobre dados reais do
hospital. A curadoria (`curar()`) remove duplicatas exatas e exemplos fora
da faixa de 20 a 1500 caracteres de resposta. O resultado atual (**57
exemplos**) e dividido de forma deterministica em treino (**48**) e
validacao (**9**).

### 3.2 Treinamento

`fase3/finetuning/train_lora.py` usa `transformers` + `peft` (LoRA) +
`datasets`, 100% CPU. As dependencias pesadas ficam em
`requirements-fase3.txt`, fora do `requirements.txt` principal, para nao
afetar a CI das Fases 1/2.

O treino usa response-only loss: system prompt, pergunta, contexto e
padding recebem label `-100`. Quatro experimentos reais foram preservados:

| Modelo | Treino/val. efetivos | Loss val. | Perplexidade | Resultado de geração |
| --- | ---: | ---: | ---: | --- |
| `distilgpt2` | 33 / 6 | 4,7692 | 117,826 | Repetitivo e sem capacidade instrucional em português |
| Qwen2.5-0.5B v1 | 30 / 5 | 2,6581 | 14,269 | Melhor formato, mas fontes/valores inconsistentes |
| Qwen2.5-0.5B v2 | 75 / 7 | 2,2172 | 9,182 | Alterou `BI-RADS 4` para 5 e `7/10` para `7/9` |
| **Qwen2.5-1.5B** | **60 / 7** | **2,1101** | **8,249** | Selecionado com escala LoRA 0,1 e grounding obrigatório |

Os adapters e resumos ficam em `resultados/fase3/finetuning/`. A redução
de loss não é tratada como evidência suficiente: cada versão também foi
testada em geração clínica. O histórico completo está no relatório técnico.

## 4. Assistente clinico (LangChain) e explainability

`fase3.assistant_chain.responder_pergunta_clinica()`:

1. Recupera ate 3 protocolos relevantes via BM25 (`fase3/retrieval.py`,
   sem embeddings, 100% offline);
2. Busca o contexto estruturado do paciente no mock de EHR SQLite
   (`fase3/ehr_tools.py`), semeado a partir de `pacientes_sinteticos.json`;
3. Produz um plano factual autorizado com os dados recuperados;
4. Monta uma chain LCEL (`prompt | llm | StrOutputParser`) com o Qwen +
   adapter LoRA (`fase3/llm_backend.py`);
5. Aplica guardrails de seguranca (`fase3/guardrails.py`);
6. Valida fonte, números, aderência, repetição e adequação clínica;
7. Repara somente citações ou usa fallback fundamentado quando a geração
   semântica é inadequada;
8. Registra auditoria com fontes, motivos, reparo e fallback.

```mermaid
sequenceDiagram
    actor Medico
    participant CHAIN as assistant_chain.py
    participant RETR as retrieval.py (BM25)
    participant EHR as ehr_tools.py (SQLite)
    participant LLM as llm_backend.py
    participant GROUND as grounding clinico
    participant GUARD as guardrails.py
    participant LOG as logging_utils.py

    Medico->>+CHAIN: responder_pergunta_clinica(pergunta, paciente_id)
    CHAIN->>+RETR: buscar_protocolos(pergunta)
    RETR-->>-CHAIN: documentos + fontes
    CHAIN->>+EHR: get_paciente(paciente_id)
    EHR-->>-CHAIN: exames pendentes, alertas ativos
    CHAIN->>+LLM: chain.invoke(plano factual + contexto compacto)
    LLM-->>-CHAIN: resposta bruta
    CHAIN->>+GUARD: aplicar_guardrails(resposta)
    GUARD-->>-CHAIN: resposta segura + bloqueado + motivo
    CHAIN->>+GROUND: validar fontes, numeros e adequacao
    GROUND-->>-CHAIN: aceitar, reparar citacao ou fallback
    CHAIN->>LOG: registrar_interacao(fontes, motivos, fallback)
    CHAIN-->>-Medico: resposta, fontes, bloqueado
```

Toda resposta segura recebe um disclaimer fixo informando que exige
validacao de um medico responsavel. Respostas com prescricao direta
(regex sobre verbos de acao + dose/via) ou com PII detectada nunca chegam
ao usuario — sao substituidas por uma mensagem padronizada e o motivo do
bloqueio fica registrado no log de auditoria.

## 5. Fluxo de decisao com LangGraph

`fase3.clinical_flow_graph` implementa o fluxo pedido no desafio: ao
receber informacoes de um paciente, o sistema verifica exames pendentes,
sugere conduta e emite alertas para a equipe medica.

```mermaid
flowchart TD
    START(("inicio")) --> BP["buscar_paciente"]
    BP -->|paciente encontrado| VEP["verificar_exames_pendentes"]
    BP -->|nao encontrado| AUD["registrar_auditoria"]
    VEP --> ST["sugerir_tratamento\n(assistant_chain)"]
    ST --> CS["checar_seguranca\n(traduz bloqueio em alerta)"]
    CS --> EA["emitir_alertas\n(exames pendentes + alertas clinicos ativos)"]
    EA --> AUD
    AUD --> FIM(("fim"))
```

A bifurcacao apos `buscar_paciente` e uma decisao real: se o codigo do
paciente nao existe no prontuario, o fluxo encerra com seguranca antes de
qualquer sugestao de conduta, em vez de arriscar uma resposta sem
contexto.

## 6. Seguranca, validacao e observabilidade

| Mecanismo | Implementacao |
| --- | --- |
| Nunca prescrever diretamente | `guardrails.contem_prescricao_direta()` bloqueia e retorna mensagem padrao |
| Validacao humana obrigatoria | Disclaimer fixo em toda resposta nao bloqueada |
| Ausencia de PII na resposta | `detectar_pii()` (CPF, telefone, e-mail, rotulo de nome) |
| Explainability | Toda resposta carrega a lista de protocolos usados (`id` + `titulo`) |
| Grounding clínico | Reprova fonte/valor inventado, evasão, repetição e inadequação por cenário |
| Fallback seguro | Resposta determinística baseada somente no EHR e protocolo recuperado |
| Logging de auditoria | `logging_utils.registrar_interacao()` — stdout JSON + `resultados/fase3/auditoria.jsonl` |
| Avaliacao objetiva | `evaluate_assistant.py`, rubrica deterministica (sem LLM-juiz), mesma filosofia do `src/evaluate_llm.py` da Fase 2 |

## 7. Execucao

Na raiz do repositorio:

```bash
python -m pip install -r requirements.txt
python -m fase3.data.build_finetuning_dataset

# fine-tuning (opcional, dependencias pesadas):
python -m pip install -r requirements-fase3.txt
python -m fase3.finetuning.train_lora \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir resultados/fase3/finetuning/qwen2.5-1.5b \
  --epochs 3 --learning-rate 0.00005 --max-length 256 --clinical-repeat 3

# demo usando a LLM customizada local (adapter LoRA treinado):
python -m fase3.cli_demo --paciente-id PAC-0001 \
  --pergunta "Posso iniciar a quimioterapia hoje?" \
  --backend local \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-path resultados/fase3/finetuning/qwen2.5-1.5b/lora_adapter

# demo de ponta a ponta:
python -m fase3.cli_demo --paciente-id PAC-0001 --pergunta "Posso iniciar a quimioterapia hoje?"
# sem GROQ_API_KEY, use --backend fake

# avaliacao do assistente:
python -m fase3.evaluate_assistant --backend local --lora-scale 0.1

# testes automatizados:
python -m unittest discover -s tests -v
```

Ou abra e execute `notebooks/04_assistente_medico_fase3.ipynb`.

## 8. Limitacoes e decisoes de producao

- Todos os protocolos, prontuarios e pacientes sao ficticios, criados para
  este projeto academico; nao refletem um hospital real.
- O modelo selecionado é Qwen2.5-1.5B, mas a avaliação final teve 0% de
  aceitação direta da LLM e 100% de fallback. O pipeline é seguro; a
  geração bruta ainda exige mais dados clínicos e um modelo maior.
- O retrieval usa BM25 (lexico) em vez de embeddings semanticos, escolha
  deliberada para manter o pipeline 100% offline e reproduzivel; um ganho
  de qualidade viria de um retriever semantico em producao.
- Os guardrails sao baseados em regras (regex), nao em um classificador
  treinado; cobrem os casos pedidos no desafio mas podem ter falsos
  negativos em frases fora do padrao esperado.
- Assim como na Fase 2, nenhuma sugestao do assistente substitui avaliacao
  clinica presencial ou decisao de um medico responsavel.
