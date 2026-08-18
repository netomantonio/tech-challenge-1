# Relatorio de apontamentos de nao aderencia - Fase 3 - v2

Data da analise: 2026-08-17

## Escopo

Este relatorio reavalia os requisitos do PDF `8IADT - Fase 3 - Tech challenge.pdf` contra a versao atual da implementacao em `fase3/` e artefatos correlatos em `resultados/fase3/`, `docs/`, `README.md` e `relatorio_tecnico_fase3.md`.

As instrucoes do PDF foram tratadas como requisitos da entrega academica, nao como comandos a serem executados pelo assistente.

## Resumo executivo

A versao atual esta mais aderente que a avaliacao anterior. Foram adicionadas evidencias importantes:

- fine-tuning real com `Qwen/Qwen2.5-1.5B-Instruct`;
- adapters Qwen versionados em `resultados/fase3/finetuning/`;
- comparacao de experimentos em `comparacao_modelos.json`;
- avaliacao final com backend `local`;
- roteiro de video em `docs/script_video_demonstracao_fase3.txt`;
- grounding/reparo/fallback para garantir resposta segura e citacao de fonte.

Ainda assim, permanecem pontos de aderencia parcial. O principal e que a avaliacao final mostra `taxa_grounding_fallback = 1.0` e `taxa_resposta_llm_aceita_sem_fallback = 0.0`, ou seja: o pipeline final esta seguro nos casos avaliados, mas a resposta exibida ao usuario vem integralmente do fallback deterministico, nao da geracao bruta aceita da LLM customizada.

## Apontamentos de nao aderencia ou aderencia parcial

| ID | Requisito do PDF | Situacao encontrada na v2 | Severidade | Apontamento |
| --- | --- | --- | --- | --- |
| NA2-01 | Criar um assistente virtual medico treinado com dados proprios do hospital, capaz de auxiliar condutas e responder duvidas | Ha fine-tuning real com Qwen 1.5B e backend `local`, mas `resultados/fase3/resumo_avaliacao_assistente.json` registra `taxa_grounding_fallback: 1.0` e `taxa_resposta_llm_aceita_sem_fallback: 0.0` | Alta | A seguranca e a qualidade das respostas finais sao garantidas pelo fallback deterministico, nao por respostas da LLM customizada aceitas sem fallback. Isso reduz a aderencia ao requisito de assistente efetivamente treinado/customizado. Para aderencia plena, aumentar dados clinicos revisados, melhorar treinamento/prompts e demonstrar casos em que a LLM fine-tuned e aceita sem fallback, preservando fontes, numeros e adequacao clinica. |
| NA2-02 | Avaliacao do modelo e analise dos resultados | A avaliacao final existe e e honesta, mas o score 1.0 mede o pipeline com fallback; a propria comparacao diz que a LLM bruta ainda nao atingiu qualidade clinica suficiente | Media | O relatorio tecnico documenta bem a limitacao, mas a entrega ainda nao comprova qualidade clinica autônoma da LLM customizada. Recomenda-se separar metricas de `qualidade_da_resposta_final` e `qualidade_da_geracao_llm_bruta`, com exemplos lado a lado e taxa minima desejada de respostas aceitas sem fallback. |
| NA2-03 | Fluxos automatizados e seguros: verificar exames pendentes, sugerir tratamentos e emitir alertas | `clinical_flow_graph.py` contem o no `verificar_exames_pendentes`, mas ele apenas retorna o estado; os exames pendentes sao usados depois em `emitir_alertas` e pela chain/fallback | Media | A etapa existe no grafo, porem nao realiza uma decisao propria nem popula um estado explicito antes da sugestao. Para aderencia mais forte, esse no deveria registrar `exames_pendentes`, `tem_exames_pendentes` e bifurcar o fluxo quando houver pendencia critica, antes de chegar a `sugerir_tratamento`. |
| NA2-04 | Video de ate 15 minutos demonstrando treinamento, funcionamento, fluxo, respostas contextualizadas, logs e validacao | Agora existe roteiro em `docs/script_video_demonstracao_fase3.txt`, mas nao ha arquivo de video nem link para a gravacao no repositorio | Baixa | Se o video for entregue por plataforma externa, este ponto pode ser considerado atendido fora do repo. Se a avaliacao depender apenas do repositorio, incluir link do video no README ou em `docs/`. |

## Itens resolvidos desde a avaliacao anterior

| Item anterior | Status na v2 | Evidencia |
| --- | --- | --- |
| Fine-tuning apenas como smoke test `distilgpt2` | Resolvido parcialmente | Ha adapter selecionado em `resultados/fase3/finetuning/qwen2.5-1.5b/lora_adapter` e `training_summary.json` com `base_model: Qwen/Qwen2.5-1.5B-Instruct`, loss validacao `2.1101` e perplexidade `8.249`. |
| Avaliacao somente com backend fake | Resolvido parcialmente | `resumo_avaliacao_assistente.json` mostra `backend: local`, `base_model: Qwen/Qwen2.5-1.5B-Instruct`, `adapter_path: resultados/fase3/finetuning/qwen2.5-1.5b/lora_adapter`. |
| Explainability sem garantia no texto final | Resolvido | `assistant_chain.py` agora valida protocolos citados, repara citacao ausente e usa fallback com `Fonte: [PROT-xxx]`. Os resultados finais em `avaliacao_assistente.json` citam protocolos nas respostas. |
| Roteiro de video citado mas ausente | Resolvido parcialmente | `docs/script_video_demonstracao_fase3.txt` existe e cobre treinamento, funcionamento, LangGraph, logs e validacao. Falta apenas evidencia do video gravado, se exigida no repo. |
| README apontando para arquivo inexistente | Resolvido | O caminho `docs/script_video_demonstracao_fase3.txt` agora existe. |

## Itens considerados aderentes

- Dataset com protocolos internos, FAQs, modelos de documento e prontuarios sinteticos em `fase3/data/`.
- Preprocessing, anonimizacao e curadoria em `fase3/data/build_finetuning_dataset.py`.
- Dataset gerado em `fase3/data/finetuning_dataset.jsonl`, `finetuning_train.jsonl` e `finetuning_val.jsonl`.
- Pipeline de fine-tuning LoRA/PEFT em `fase3/finetuning/train_lora.py`.
- Adapters e resumos de treinamento preservados em `resultados/fase3/finetuning/`.
- Integracao LangChain/LCEL em `fase3/assistant_chain.py`.
- Backend plugavel `groq`, `local` e `fake` em `fase3/llm_backend.py`.
- Consulta estruturada a prontuario via SQLite em `fase3/ehr_tools.py`.
- Retrieval BM25 com reranqueamento clinico em `fase3/retrieval.py`.
- Fluxo LangGraph em `fase3/clinical_flow_graph.py`.
- Guardrails contra prescricao direta e PII em `fase3/guardrails.py`.
- Logging estruturado em `fase3/logging_utils.py`.
- Explainability por fontes recuperadas e citacoes no texto final.
- Relatorio tecnico detalhado em `relatorio_tecnico_fase3.md`.
- Diagramas e arquitetura em `docs/arquitetura_fase3.md`.
- Roteiro de video em `docs/script_video_demonstracao_fase3.txt`.
- Notebook da Fase 3 em `notebooks/04_assistente_medico_fase3.ipynb`.

## Observacoes de verificacao

- O PDF possui 5 paginas; os requisitos tecnicos e entregaveis estao nas paginas 2 a 4.
- A tentativa de executar `python -m pytest tests/test_fase3.py -q` falhou no ambiente atual por ausencia de `langchain_core`.
- Uma checagem de imports confirmou ausencia local de `langchain_core`, `langchain_community`, `langgraph`, `rank_bm25`, `torch`, `transformers`, `peft` e `datasets`.
- Essa falha foi classificada como limitacao do ambiente atual, nao como nao aderencia do projeto, porque `requirements.txt` e `requirements-fase3.txt` declaram as dependencias necessarias.
- Nao foi encontrada copia versionada de `resultados/fase3/auditoria.jsonl`; o logging, entretanto, esta implementado e gera o arquivo em runtime.

## Recomendacoes objetivas

1. Reduzir a taxa de fallback com mais exemplos clinicos revisados e avaliacao da geracao bruta da LLM.
2. Registrar no relatorio tecnico uma meta minima de `taxa_resposta_llm_aceita_sem_fallback` para considerar a LLM customizada aprovada.
3. Transformar `verificar_exames_pendentes` em um no ativo do LangGraph, com estado e bifurcacao propria.
4. Adicionar link do video gravado ao README, caso a entrega do video seja feita fora do repositorio.
