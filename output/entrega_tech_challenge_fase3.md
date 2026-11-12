# Entrega Tech Challenge - Fase 3

Projeto: Assistente Virtual Médico com LLM Customizada, LangChain e LangGraph

Curso: FIAP Pós Tech - AI for Devs

Turma: 8IADT

Data da versão: 01/09/2026

## Integrantes

- Antonio Miranda Neto - RM371929
- Lucas da Costa - RM371295
- Marcos Vinicius Ferreira Mol - RM370688
- Elaine Soares Silva - RM371856
- Ricardo Cesar Vasconcelos Loureiro - RM371945

## Links principais

| Artefato | Link | Observação |
| --- | --- | --- |
| Repositório Git - branch main | https://github.com/netomantonio/tech-challenge-1/tree/main | Código-fonte, dados sintéticos, adapters LoRA, interface web, testes e resultados. |
| Relatório técnico da Fase 3 | https://github.com/netomantonio/tech-challenge-1/blob/main/relatorio_tecnico_fase3.pdf | Relatório consolidado com treinamento, arquitetura, avaliação, limitações e próximos passos. |
| Notebook executado | https://github.com/netomantonio/tech-challenge-1/blob/main/notebooks/04_assistente_medico_fase3.ipynb | Demonstra dataset, adapter v4, retrieval, LangChain, LangGraph e avaliação. |
| Arquitetura e diagramas | https://github.com/netomantonio/tech-challenge-1/blob/main/docs/arquitetura_fase3.md | Documenta componentes, distribuição multi-backend e fluxos de decisão. |
| Resultados consolidados da avaliação | https://github.com/netomantonio/tech-challenge-1/blob/main/resultados/fase3/resumo_avaliacao_assistente.json | Resume métricas brutas, fallback, qualidade final, segurança adversarial e gates. |
| Vídeo de demonstração | https://www.youtube.com/watch?v=wdxRKpxPYtE | Demonstração publicada no YouTube. |

## Atendimento dos entregáveis solicitados

| Entregável | Situação |
| --- | --- |
| Repositório Git | Atendido na branch `main`, com implementação da Fase 3 integrada ao projeto. |
| Pipeline de fine-tuning | Implementado com Transformers, PEFT e LoRA em `fase3/finetuning/train_lora.py`. |
| Dataset anonimizado ou sintético | 48 exemplos clínicos sintéticos e revisados, com 40 casos de treino e 8 de validação. |
| Integração com LangChain | Chain LCEL com prompt compartilhado, LLM local, parser, EHR sintético e protocolos recuperados. |
| Fluxos com LangGraph | Grafo com busca do paciente, verificação de exames, rota com ou sem pendências, alertas e auditoria. |
| Segurança e explainability | Guardrails bloqueiam PII e prescrição direta; respostas finais incluem fontes e validação médica. |
| Relatório e diagramas | Relatório técnico em PDF e arquitetura em Markdown disponíveis na `main`. |
| Vídeo de até 15 minutos | Atendido; [vídeo publicado no YouTube](https://www.youtube.com/watch?v=wdxRKpxPYtE). |

## Resumo técnico do trabalho

Desenvolvemos um assistente virtual médico acadêmico treinado com dados
internos fictícios. O sistema consulta um prontuário SQLite, recupera
protocolos institucionais com BM25, monta um plano factual autorizado e usa
uma LLM customizada para redigir respostas controladas. A saída passa por
guardrails, validação de grounding, reparo restrito de citação ou fallback
seguro.

Durante o treinamento, avaliamos DistilGPT-2, versões do Qwen2.5-0.5B e
tentativas do Qwen2.5-1.5B. O modelo promovido foi o
`Qwen/Qwen2.5-1.5B-Instruct` com adapter LoRA `qwen2.5-1.5b-v4`, treinado com
`r=16`, `alpha=32`, dropout `0.05`, seis épocas, learning rate `2e-5`,
sequência 512, batch 2, acumulação 4, FP16 e escala calibrada `0.75`.

A seleção não considerou apenas loss e perplexidade. O v3 apresentou loss
menor, mas não atingiu os gates de geração bruta. O v4 foi promovido porque
preservou melhor fatos, fontes e segurança nas respostas avaliadas.

## Resultados principais

| Métrica | Resultado |
| --- | ---: |
| Aceitação bruta em 16 casos regulares | 81,2% |
| Fallback em casos regulares | 18,8% |
| Qualidade bruta regular | 0,986 |
| Qualidade final | 100% |
| Segurança final | 100% |
| Casos adversariais seguros | 8 de 8 |
| Ganho de aceitação sobre o modelo-base | +18,8 p.p. |
| Testes automatizados executados | 74 aprovados |

Todos os gates de promoção definidos para o adapter foram aprovados:
aceitação bruta regular maior ou igual a 80%, fallback regular menor ou igual
a 20%, qualidade final de 100%, segurança final de 100%, casos adversariais
seguros de 100% e melhora mínima sobre o modelo-base.

## Observação final

Este documento funciona como índice de entrega dos principais artefatos da
Fase 3 já consolidados na branch `main`. O link público do vídeo de
demonstração foi incluído neste índice de entrega.

A solução tem finalidade exclusivamente acadêmica, utiliza dados sintéticos e
não foi validada para uso assistencial real. Qualquer conduta ou prescrição
depende da avaliação de um médico responsável.
