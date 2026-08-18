# Relatório técnico — Tech Challenge Fase 3

## Assistente Virtual Médico: fine-tuning, LangChain e LangGraph

## 1. Objetivo e escopo

A Fase 3 implementa um assistente virtual médico treinado com dados
fictícios do hospital, capaz de consultar prontuários estruturados,
recuperar protocolos internos e apoiar dúvidas clínicas. O módulo fica em
`fase3/` e não altera o pipeline de diagnóstico das Fases 1 e 2.

Este relatório documenta também os experimentos que corrigiram o ponto de
revisão: **o pipeline LoRA era real, mas a qualidade da LLM customizada era
inadequada**. Os resultados desfavoráveis foram mantidos nesta análise para
não confundir queda de loss com qualidade clínica.

## 2. Dataset, anonimização e curadoria

### 2.1 Fontes

`fase3/data/build_finetuning_dataset.py` converte todas as fontes para o
formato `{instruction, input, output, source_type, source_id}`.

| Fonte | Quantidade bruta | Uso |
| --- | ---: | --- |
| Protocolos internos | 12 documentos | Oncologia, cirurgia, enfermagem, segurança, FAQs e modelos fictícios |
| Prontuários sintéticos | 6 pacientes | Contexto estruturado, exames pendentes e alertas clínicos |
| Exemplos do assistente | 18 pares clínicos | Formato de resposta, preservação numérica, fonte e validação médica |
| MedQuAD | 12 pares | Dados públicos sobre câncer de mama, com atribuição por registro |
| PubMedQA | 8 pares | Resumos públicos de abstracts, com atribuição por registro |

Os 18 pares clínicos incluem paráfrases para seis intenções: exames
pré-quimioterapia, BI-RADS 4/biópsia, febre/taquicardia/sepse, dor
pós-operatória, checklist de exames e paciente inexistente. As perguntas da
avaliação final não são cópias literais dessas instruções.

### 2.2 Proteção de dados e curadoria

- Normalização de espaços e decomposição de FAQs em pares pergunta/resposta.
- Anonimização defensiva de CPF, telefone, e-mail e rótulos de nome.
- Uso exclusivo de códigos fictícios `PAC-000x`, sem nomes reais.
- Remoção de duplicatas por hash da instrução + resposta.
- Exclusão de respostas fora da faixa de 20 a 1.500 caracteres.
- Split determinístico para reprodutibilidade.

O dataset final tem **57 exemplos curados: 48 de treino e 9 de validação**.
Por padrão, o treino clínico não mistura MedQuAD/PubMedQA, porque os testes
mostraram degradação de domínio e de idioma. Depois desse filtro, cada
experimento registra no `training_summary.json` a quantidade efetivamente
usada, o oversampling clínico e os hashes SHA-256 dos splits.

## 3. Pipeline de treinamento LoRA/PEFT

`fase3/finetuning/train_lora.py` usa PyTorch, Transformers, Datasets e PEFT.
O treino foi executado em CPU e possui as seguintes garantias:

- LoRA real, com adapter `adapter_model.safetensors` carregável pelo PEFT.
- Loss calculada **somente nos tokens da resposta**; system prompt,
  pergunta, contexto e padding recebem label `-100`.
- Chat template do tokenizer para modelos instrucionais.
- Seed de treino e de dados igual a 42.
- Avaliação ao fim de cada época.
- Registro de loss, perplexidade, duração, versão do PyTorch, dispositivo,
  módulos LoRA, hiperparâmetros e hashes do dataset.
- Para Qwen, LoRA em `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
  `up_proj` e `down_proj`, com `r=8`, `alpha=16` e `dropout=0,05`.

## 4. Histórico completo dos modelos e motivo das mudanças

### 4.1 Comparação quantitativa

| Experimento | Modelo-base | Dataset efetivo treino/val. | Épocas | LR | Parâmetros LoRA | Loss treino final | Loss val. final | Perplexidade val. | Duração |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Smoke inicial | `distilgpt2` | 33 / 6 | 3 | 5e-4 | não registrado no resumo antigo | 4,6014 | 4,7692 | 117,826 | 43,76 s |
| Qwen 0,5B v1 | `Qwen2.5-0.5B-Instruct` | 30 / 5 | 4 | 5e-5 | 4.399.104 (0,8826%) | 2,5861 | 2,6581 | 14,269 | 377,80 s |
| Qwen 0,5B v2 | `Qwen2.5-0.5B-Instruct` | 75 / 7 | 5 | 1e-4 | 4.399.104 (0,8826%) | 1,2592 | 2,2172 | 9,182 | 457,67 s |
| **Modelo selecionado** | **`Qwen2.5-1.5B-Instruct`** | **60 / 7** | **3** | **5e-5** | **9.232.384 (0,5945%)** | **2,0048** | **2,1101** | **8,249** | **593,06 s** |

Notas importantes:

- “Loss treino final” é a média informada pelo Trainer para toda a execução.
  As médias por época ficam nos resumos de cada experimento.
- A comparação entre modelos deve considerar também a geração. Loss menor
  não garante preservação de números, idioma ou conduta clínica.
- O Qwen 0,5B v2 usou oversampling 4; o Qwen 1,5B usou oversampling 3.
- Os adapters Qwen 0,5B v2 e 1,5B possuem, respectivamente, cerca de 17,6 MB
  e 37,0 MB.

### 4.2 Experimento 1 — DistilGPT-2

O `distilgpt2` foi usado inicialmente como smoke test de baixo custo. A
loss de validação caiu de 4,8405 para 4,7692, provando que o código treinava
um adapter real. Porém o modelo não é instrucional nem foi desenvolvido
para português. As respostas eram repetitivas, sem estrutura clínica e
chegaram a inventar IDs como `PROT-013`.

**Decisão:** manter apenas como evidência histórica de smoke test. Não usar
como backend padrão nem como evidência de qualidade clínica.

### 4.3 Experimento 2 — Qwen2.5-0.5B-Instruct v1

A primeira substituição adotou um modelo instrucional multilíngue e mudou o
treino para response-only loss. A perplexidade de validação caiu de
aproximadamente 117,8 para 14,269. Mesmo assim, a geração ainda apresentava
recusas evasivas, termos corrompidos, fontes omitidas e valores inventados.

Exemplos observados: resposta sem conduta para exames pendentes, “BI-RATS”
em vez de BI-RADS e classificação de dor incompatível com `7/10`.

**Decisão:** ampliar o dataset clínico, incluir paráfrases e aumentar a
capacidade de ajuste do LoRA.

### 4.4 Experimento 3 — Qwen2.5-0.5B-Instruct v2

O dataset passou de 45 para 57 exemplos curados; o treino efetivo usou 75
exemplos após oversampling clínico. A loss média de treino por época caiu
de 2,5634 para 0,4909 e a perplexidade de validação chegou a 9,182. Houve
leve piora da validação depois da terceira época, sinal de início de
sobreajuste.

O teste de geração mostrou por que loss isolada não bastava: o modelo
alterou `BI-RADS 4` para `BI-RADS 5`, `7/10` para `7/9` e misturou português
com outros idiomas.

**Decisão:** não promover o v2 e testar um modelo-base com maior capacidade.

### 4.5 Experimento 4 — Qwen2.5-1.5B-Instruct

O modelo de 1,5B apresentou a melhor validação. As losses foram:

| Época | Loss média de treino | Loss de validação |
| ---: | ---: | ---: |
| 1 | 2,3752 | 2,3172 |
| 2 | 1,9048 | 2,1559 |
| 3 | 1,7343 | 2,1101 |

Não houve inversão da curva de validação nas três épocas. Entretanto, o
adapter em intensidade total ainda sobrepunha conhecimento do modelo-base.
Em testes manuais, chegou a afirmar que uma paciente com exames pendentes
estava apta a iniciar tratamento e alterou fontes.

Foi então calibrada a contribuição do adapter para `lora_scale=0,1`. Essa
escala preserva mais da capacidade linguística do modelo-base e mantém o
adapter ativo. O valor é configurável por `--lora-scale` ou pela variável
`FASE3_LOCAL_LORA_SCALE` e é registrado na avaliação.

**Decisão:** selecionar Qwen2.5-1.5B-Instruct + adapter LoRA + escala 0,1,
mas exigir grounding determinístico antes de exibir qualquer resposta.

## 5. Assistente, RAG e barreira de grounding

`fase3.assistant_chain.responder_pergunta_clinica()` executa:

1. Retrieval BM25 com normalização de acentos, sinônimos clínicos e
   reranking por intenção.
2. Consulta ao mock de EHR em SQLite.
3. Construção de um plano factual autorizado a partir do prontuário e do
   protocolo recuperado.
4. Redação pelo Qwen + LoRA via chain LCEL
   (`prompt | llm | StrOutputParser`).
5. Guardrail contra PII e prescrição direta.
6. Validação de grounding: fonte ausente/inválida, número inventado,
   caracteres inesperados, baixa aderência, repetição, resposta evasiva e
   critérios clínicos específicos do caso.
7. Reparo apenas de citação quando o conteúdo está correto; fallback
   determinístico fundamentado quando há desvio semântico ou numérico.
8. Disclaimer de validação médica e log de auditoria.

O fallback não é apresentado como melhoria da LLM. Ele é uma barreira de
segurança do produto, e sua taxa é reportada separadamente.

## 6. Avaliação final

### 6.1 Rubrica

`fase3/evaluate_assistant.py` avalia seis perguntas parafraseadas e exige:

- fontes citadas, existentes e limitadas aos documentos recuperados;
- ausência de protocolo ou número inventado;
- conteúdo clínico esperado e adequação específica ao caso;
- ausência de resposta evasiva e repetição excessiva;
- ausência de PII e de prescrição direta;
- disclaimer e resposta não vazia.

Os cenários cobrem exames pendentes antes da quimioterapia, BI-RADS 4,
sepse, dor pós-operatória `7/10`, paciente inexistente e checklist geral.

### 6.2 Resultado do backend local real

Com `Qwen/Qwen2.5-1.5B-Instruct`, adapter
`resultados/fase3/finetuning/qwen2.5-1.5b/lora_adapter`, escala 0,1 e
geração determinística:

| Métrica | Resultado |
| --- | ---: |
| Casos | 6 |
| Score objetivo final do pipeline | 1,000 |
| Score de segurança final | 1,000 |
| Score de qualidade final | 1,000 |
| Taxa de fallback de grounding | 1,000 |
| Taxa de reparo apenas de citação | 0,000 |
| Taxa de saída da LLM aceita sem fallback | 0,000 |

Interpretação correta: **o pipeline final entregou respostas fundamentadas,
mas a LLM bruta ainda não atingiu qualidade clínica suficiente nesse
conjunto**. Os seis casos foram protegidos pelo fallback. Assim, a correção
elimina o risco de mostrar uma resposta inadequada e torna a limitação
mensurável, mas não autoriza uso clínico autônomo.

Artefatos:

- `resultados/fase3/avaliacao_assistente.json`
- `resultados/fase3/avaliacao_assistente.csv`
- `resultados/fase3/resumo_avaliacao_assistente.json`
- `resultados/fase3/finetuning/*/training_summary.json`

## 7. Execução reprodutível

```bash
python -m pip install -r requirements-fase3.txt
python -m fase3.data.build_finetuning_dataset

python -m fase3.finetuning.train_lora \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir resultados/fase3/finetuning/qwen2.5-1.5b \
  --epochs 3 --learning-rate 0.00005 --max-length 256 \
  --clinical-repeat 3

python -m fase3.evaluate_assistant \
  --backend local \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-path resultados/fase3/finetuning/qwen2.5-1.5b/lora_adapter \
  --lora-scale 0.1 --max-new-tokens 100

python -m unittest discover -s tests -v
```

## 8. Limitações e próximos passos

- Todos os protocolos e prontuários são fictícios; não houve validação
  clínica externa.
- O dataset clínico é pequeno. A taxa de aceitação bruta de 0% mostra que
  são necessários mais exemplos revisados por especialistas.
- Próximo experimento recomendado: modelo instrucional maior, treinamento
  em GPU, conjunto clínico independente de teste, early stopping e busca de
  hiperparâmetros.
- A escala LoRA 0,1 foi calibrada nos experimentos locais e deve ser
  revalidada após qualquer mudança de modelo ou dataset.
- BM25 e regras determinísticas são úteis para reprodutibilidade, mas não
  substituem validação médica, avaliação semântica independente e testes de
  segurança mais amplos.
- Nenhuma resposta deve ser usada como diagnóstico, prescrição ou conduta
  autônoma. A decisão final pertence ao médico responsável.
