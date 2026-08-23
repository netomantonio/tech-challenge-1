# Relatório de aderência final - Fase 3

Data da auditoria: 2026-08-23

## Escopo

Neste relatório, comparamos os requisitos acadêmicos do PDF
`8IADT - Fase 3 - Tech challenge.pdf` com a versão atual do nosso projeto.
Mantivemos o relatório v2 apenas como registro histórico da evolução do
trabalho.

Consideramos a solução tecnicamente aderente quando todos os requisitos de
código, dados, treinamento, integração, segurança, avaliação e documentação
estão cobertos. A gravação e a publicação do vídeo continuam como uma etapa
externa que será concluída pela equipe.

## Resultado executivo

Todos os gates técnicos que definimos para promover o adapter foram
aprovados:

| Gate | Meta | Resultado | Status |
| --- | ---: | ---: | --- |
| Aceitação bruta em casos regulares inéditos | >= 80% | 81,2% | Atendido |
| Fallback em casos regulares | <= 20% | 18,8% | Atendido |
| Qualidade da resposta final | 100% | 100% | Atendido |
| Segurança final | 100% | 100% | Atendido |
| Casos adversariais seguros | 100% | 100% | Atendido |
| Melhora do adapter sobre o base | >= 10 p.p. ou 0,10 score | +18,8 p.p. | Atendido |

## Fechamento dos apontamentos v2

| ID | Situação final | Evidência |
| --- | --- | --- |
| `NA2-01` | Resolvido | Adapter `qwen2.5-1.5b-v4`, escala `0.75`, alcançou 81,2% de aceitação bruta e superou o base em 18,8 p.p.; o fallback não responde pela maioria dos casos. |
| `NA2-02` | Resolvido | Avaliação separa resposta bruta e final, mede segurança, reparo e fallback, usa 16 casos regulares + 8 adversariais inéditos e falha por código quando os gates não passam. |
| `NA2-03` | Resolvido | O estado LangGraph contém exames e rota; há bifurcação real `com_pendencias` / `sem_pendencias`, alerta em nó próprio e encerramento antecipado para paciente inexistente. |
| `NA2-04` | Pendente externo | O roteiro está atualizado, mas a gravação, publicação e inclusão do link do vídeo serão feitas posteriormente pela equipe. |

## Auditoria dos requisitos técnicos

| Requisito | Status | Evidência principal |
| --- | --- | --- |
| Dados próprios do hospital | Atendido | 48 casos clínicos revisados, protocolos e prontuários sintéticos em `fase3/data/` |
| Preprocessing, anonimização e curadoria | Atendido | `build_finetuning_dataset.py` e testes de PII/deduplicação |
| Fine-tuning real | Atendido | `train_lora.py` e adapter PEFT íntegro em `qwen2.5-1.5b-v4/lora_adapter` |
| Avaliação e análise dos resultados | Atendido | JSON/CSV bruto e final, baseline, calibração e comparação de modelos |
| Integração LangChain | Atendido | LCEL em `assistant_chain.py`, com prompt compartilhado |
| Consulta a dados estruturados | Atendido | EHR SQLite em `ehr_tools.py` |
| Respostas contextualizadas | Atendido | Busca usa pergunta, diagnóstico, exames, observações e alertas do paciente |
| Fluxo automatizado LangGraph | Atendido | Rotas reais de exames em `clinical_flow_graph.py` |
| Nunca prescrever sem validação humana | Atendido | Guardrail, disclaimer e testes adversariais |
| Logging para auditoria | Atendido | Eventos incluem exames, rota, alertas, fontes e modo de resposta |
| Explainability | Atendido | Resposta final fundamentada e fontes com ID/título |
| Modularização em Python | Atendido | Pacote `fase3/` separado por responsabilidades |
| Relatório técnico e arquitetura | Atendido | `relatorio_tecnico_fase3.md` e `docs/arquitetura_fase3.md` |

## Dados e ausência de vazamento

- 48 exemplos clínicos: 40 de treino e 8 de validação.
- Oito famílias, com um caso de validação por família.
- 24 casos exclusivos de avaliação: 16 regulares e 8 adversariais.
- Nenhuma pergunta de avaliação aparece nos splits de fine-tuning.
- MedQuAD/PubMedQA não entram no adapter promovido.
- Dados clínicos permanecem sintéticos ou anonimizados.

## Modelo promovido

| Item | Valor |
| --- | --- |
| Base | `Qwen/Qwen2.5-1.5B-Instruct` |
| Adapter | `resultados/fase3/finetuning/qwen2.5-1.5b-v4/lora_adapter` |
| LoRA | `r=16`, `alpha=32`, dropout `0.05` |
| Treino | 6 épocas, LR `2e-5`, sequência 512 |
| Lote | batch 2, acumulação 4 |
| Hardware | RTX 3060 12 GB, FP16 |
| Seed | 42 |
| Escala | `0.75` |

O resumo de treinamento registra hashes dos splits, prompt e adapter. As
losses reavaliadas foram 0,486915 no treino e 0,435758 na validação.

## Artefatos finais

- `resultados/fase3/avaliacao_assistente.json` e `.csv`;
- `resultados/fase3/resumo_avaliacao_assistente.json`;
- `resultados/fase3/avaliacao_assistente_base.json` e `.csv`;
- `resultados/fase3/calibracao_adapter.json`;
- `resultados/fase3/finetuning/comparacao_modelos.json`;
- `resultados/fase3/finetuning/qwen2.5-1.5b-v4/training_summary.json`.

## Verificação executada

- Dataset regenerado: 48 exemplos, split 40/8.
- Testes específicos da Fase 3 e da interface web: 41 aprovados.
- Suíte completa com `python -m unittest discover -s tests -v`: 56 aprovados.
- Compilação de `fase3/`: concluída sem erros.
- `git diff --check`: concluído sem erros de whitespace.

## Conclusão

Concluímos que nossa implementação atende aos requisitos técnicos do PDF.
A única pendência para completar a entrega é a gravação e a publicação do
vídeo de demonstração. Reforçamos que o sistema permanece acadêmico, utiliza
dados sintéticos e depende obrigatoriamente de validação médica humana.
