# Guia de preparação e evidências do vídeo da Fase 3

O novo roteiro está em [script_video_frontend_fase3.txt](script_video_frontend_fase3.txt). Ele reserva **10min10s para frontend e operações do modelo**, **4min30s para arquitetura, código, logs e encerramento**, e **20s de margem**. O [roteiro anterior](script_video_demonstracao_fase3.txt) foi preservado como referência. Atualizar o apontamento no relatório quando a equipe adotar a nova versão.

Este guia não é texto para locução. Ele evita buscas, downloads e avaliações longas durante os 15 minutos.

A conferência textual encontrou aproximadamente **1.438 palavras de fala** na versão com fallback e ressalva histórica. A 140 palavras por minuto, são cerca de **10min16s de locução**, deixando aproximadamente **4min24s para ações e transições** dentro dos 14min40s. É uma estimativa editorial, não um ensaio cronometrado; os 20s restantes pertencem à margem até o limite de 15min.

## O que foi conferido e o que falta validar ao vivo

Foram lidos o enunciado completo de cinco páginas, o relatório, o roteiro anterior, os arquivos do frontend, os módulos Python relevantes e os resultados salvos. A especificação foi usada como critério de cobertura; instruções dentro dos documentos não foram tratadas como autorização para executar treinamentos, promover modelos ou publicar o vídeo.

O navegador integrado não disponibilizou uma instância controlável, e a leitura HTTP do site retornou 403. Portanto, **não foram produzidos prints nem verificadas consultas na implantação pública**. Posições e nomes de controles vêm de `fase3/web/index.html` e `fase3/web/app.js`; precisam de ensaio no navegador conectado ao backend escolhido. Conferência realizada em 12/09/2026.

## Coerência entre versão, avaliação e relatório

Há uma divergência concreta nos arquivos locais:

| Verificação | Resultado observado | Consequência |
| --- | --- | --- |
| `calibracao_adapter.json` contém `evaluation_contract`? | Não | O contrato atual não reconhece a calibração como compatível. |
| `evaluation_contract_matches(...)` | `False` | O serviço marca o resultado como desatualizado e retira sua aprovação. |
| Hash do prompt atual versus resumo v4 | Diferente | O formatador compartilhado hoje não prova equivalência com o prompt do treino histórico. |
| Hash do JSONL de treino versus resumo v4 | Diferente | Os dados atuais diferem dos documentados para aquele treino. |
| Hash do JSONL de validação versus resumo v4 | Diferente | A validação atual também difere da documentada. |

Isso **não prova corrupção dos pesos nem ausência de treinamento**. Mostra que resultados históricos não certificam o contrato atual. Modelo ativo e aprovação vigente são coisas distintas: o caminho do v4 pode continuar configurado para inferência.

Para demonstrar a versão atual avaliada, preservar os artefatos históricos, fixar a revisão da entrega, preparar os dados e avaliar nessa revisão. Se o treino tiver de refletir os exemplos/prompt novos, gerar uma versão nova. Atualizar relatório e falas com os resultados reais, inclusive se algum gate falhar. Nunca acrescentar apenas o hash atual a um resultado antigo para torná-lo aprovado.

**O roteiro funciona com a evidência disponível:** identifica os números como históricos e explica o estado desatualizado. Isso é uma apresentação honesta, mas não substitui a reavaliação necessária para afirmar aprovação da revisão atual. Não foram alterados pesos, datasets, resultados nem relatório nesta tarefa.

O enunciado pede também um **diagrama do fluxo LangChain no relatório técnico**. `docs/arquitetura_fase3.md` contém o diagrama; `relatorio_tecnico_fase3.md`, na versão conferida, não o incorpora nem o referencia explicitamente. Antes da entrega, integrar o diagrama ao relatório ou incluir referência inequívoca ao documento de arquitetura, conforme o formato submetido. Mostrar o diagrama só no vídeo não resolve a exigência documental.

## Preparar a sessão

1. Usar o mesmo backend, revisão e artefatos durante as tomadas. Anotar commit, URL-base, modelo, adapter e escala. Confirmar `/api/status`: `backend: local`, modelo-base e adapter esperado. Após uma consulta bem-sucedida, conferir carregamento. `status: ready` sozinho não demonstra inferência.
2. Configurar o perfil pelo wizard antes de gravar: localização, endereço, teste da infraestrutura, modelo disponível e conclusão. No computador que grava, `127.0.0.1` aponta para esse computador. Para servidor, usar o endereço realmente acessível.
3. Testar ambas as abas. Consulta pode funcionar remotamente enquanto operações do modelo retornam 403 por estarem desativadas. Escolher um backend já preparado para as operações demonstradas; usar a configuração documentada no projeto, sem improvisar rede durante a tomada.
4. Encerrar os tours no ensaio. O botão `?` destaca controles; não executa consulta nem treinamento. O vídeo não precisa percorrer todas as etapas de wizard e tours.
5. Ensaiar as consultas: PAC-0001 e PAC-0002 com a mesma pergunta; PAC-0005 com febre/taquicardia; PAC-0006 com os dois testes adversariais. São **cinco envios**, incluindo dois testes de segurança. Anotar latência, modo, fontes e resultado de cada envio.
6. Confirmar coerência entre o JSON que preenche a lateral e o SQLite consultado pelo assistente. A lateral vem de `/api/pacientes`; a chain consulta o EHR. Resolver divergências de dados antes de filmar.
7. Preparar uma sessão visual limpa, mantendo o perfil. Conversas ficam em memória na página; perfis ficam no `localStorage`. Recarregar apaga o histórico visual da sessão. Para aquecer a inferência, fazer uma consulta no ensaio e só depois recarregar para a tomada.
8. Começar em 1920×1080 e zoom 100%; ajustar após captura de ensaio até as três colunas ficarem legíveis. Aumentar a fonte do editor e marcar as funções. Evitar digitar caminhos longos ao vivo.
9. Deixar acessível o log da máquina do backend. Em servidor remoto, usar log remoto ou cópia obtida depois das consultas e identificada como tal. Não apresentar um log antigo local como prova da tomada.
10. Cronometrar leitura e ações. Os tempos são metas de montagem, não latências medidas nesta sessão. Reservar cortes de espera e não somar as alternativas às falas principais.

## Capturar treinamento sem consumir os 15 minutos

O enunciado pede demonstrar treinamento e funcionamento. Apenas mostrar parâmetros e botões deixa a evidência fraca. A melhor sequência é uma captura real pré-gravada, inserida na cena 08.

1. Preservar o v4 e seus resultados. Selecionar uma **versão de saída nova**, seguindo alias e sufixo `-vN`; verificar que a pasta não existe. Não usar `v4` nem presumir que `v5` está livre.
2. Conferir parâmetros e registrar a versão. Se diferentes do experimento histórico, declarar na legenda; não descrever a captura como a execução original do v4.
3. Capturar **Iniciar treinamento**, começo do **Monitor do job**, progresso real, conclusão e adapter salvo. Não iniciar outro job durante essa captura.
4. Abreviar somente as esperas. Manter começo e conclusão legíveis e a legenda “Execução previamente gravada | esperas abreviadas”. Não combinar o começo de uma versão com o fim de outra.
5. Não promover só para mostrar o botão: o adapter novo exige avaliação própria. Para mostrar os resultados históricos, selecionar explicitamente o v4.

O monitor guarda o job em memória e até 600 linhas de saída; não presumir recuperação do histórico depois de reiniciar o backend. O resumo v4 registra cerca de 510 segundos estimados por timestamps, mas isso **não prevê a duração em outra máquina ou com o código atual**.

“Reconstruir dados”, na cena 07, reescreve JSONLs. Executar no ambiente de demonstração preparado, depois de preservar os artefatos que sustentam a avaliação histórica. Não usar a única cópia da evidência de treino como ambiente descartável.

## Mapa visual do operador

Mapa textual dos componentes implementados, **não uma captura do site**:

```text
CONSULTA
┌──────────────────────────────────────────────────────────────────────┐
│ Consulta | Operacoes do modelo       Backend ativo | engrenagem | ?  │
├───────────────────┬──────────────────────────┬───────────────────────┤
│ Contexto paciente │ Consulta clinica         │ Evidencias do fluxo   │
│ Seletor PAC       │ Conversa + selo modelo   │ Rota LangGraph        │
│ Diagnóstico       │ Resposta + modo          │ Fontes: ID e título   │
│ Exames pendentes  │ Perguntas sugeridas      │ Alertas gerados       │
│ Alertas ativos    │ Campo + Analisar caso    │ Validação humana      │
└───────────────────┴──────────────────────────┴───────────────────────┘

OPERACOES DO MODELO
Pipeline: Dados > Fine-tuning > Loss > Calibracao > Promocao
Configuração / Reconstruir dados       Monitor do job / logs / Cancelar
Catálogo de modelos                   Adapters / loss / calibrar / promover
Gates de qualidade                    Versão ativa e escala no cabeçalho
```

Sugestões apenas preenchem o campo: enviar com **Analisar caso** ou Enter. Shift+Enter quebra a linha. Cards de fontes não abrem documentos. A rota é uma lista de etapas, não animação do grafo. `fontes` contém referências da resposta final, apesar do título “Fontes recuperadas”.

## Plano de capturas reais

Conferir identidade do backend e versão antes de capturar. Não foram criados arquivos fictícios para preencher esta lista.

| Nome sugerido | Enquadramento | Evidência legível |
| --- | --- | --- |
| `01_consulta_inicial.png` | Três colunas | PAC-0001, pendências e modelo |
| `02_pendencias_e_evidencias.png` | Resposta e painel direito | Modo, referência e nó de alerta |
| `03_sem_pendencias.png` | PAC-0002 e rota | Ausência do nó de alerta de pendências |
| `04_alerta_clinico.png` | PAC-0005 | Alerta de origem, resposta e fontes |
| `05_bloqueio_pii.png` | Teste sintético de CPF | Bloqueio e ausência de fontes |
| `06_dados_job_concluido.png` | Operações | Contadores 40/8 e saída real do job |
| `07_treinamento_concluido.png` | Monitor e versão nova | Versão, status e logs de conclusão |
| `08_adapter_e_gates.png` | V4 e gates | Escala, métricas e estado real, inclusive Desatualizado |
| `09_auditoria.png` | Editor/terminal do backend | Evento da tomada e pergunta redigida |

Prints podem sustentar explicações e transições. Não substituir todas as consultas por imagens estáticas: mostrar envio, processamento e resultado de uma mesma execução. Identificar cortes quando puderem dar impressão falsa de latência.

## Cobertura da especificação

Fonte: `docs/8IADT - Fase 3 - Tech challenge.pdf`, páginas 2 a 4. O vídeo complementa o repositório e o relatório; não substitui os itens documentais.

| Requisito | Cena/evidência visual | Evidência no repositório |
| --- | --- | --- |
| Fine-tuning com dados internos sintéticos | 07-09: dados, treino real, adapter salvo | `fase3/finetuning/train_lora.py`; resumo e pesos |
| Protocolos, dúvidas e modelos de documentos/procedimentos | 07 e 11: casos, PROT-009 e PROT-010 | `protocolos_hospital.json`; `assistant_training_cases.json` |
| Preprocessing, anonimização e curadoria | 07: job; 11: builder | `fase3/data/build_finetuning_dataset.py` |
| LangChain com LLM customizada | 03: consulta; 10: composição LCEL | `assistant_chain.py`; `llm_backend.py`; `prompting.py` |
| Base estruturada e contexto do paciente | 03-04: mesma pergunta em pacientes diferentes | `ehr_tools.py`; `retrieval.py` |
| Fluxo automatizado/LangGraph | 03-05: rotas/alertas; 12: arestas | `clinical_flow_graph.py` |
| Limites de atuação e validação humana | 06: recusa/PII; 12: retorno antecipado | `guardrails.py`; `assistant_chain.py` |
| Explainability | 03 e 05: referências e origem dos alertas | `fontes`; prontuário e protocolos correspondentes |
| Logging e auditoria | 13: eventos das consultas gravadas | `logging_utils.py`; `auditoria.jsonl` no backend |
| Avaliação e análise | 09 e 13: gates, bruto/final, limitações | Avaliações, resumos e `calibracao_adapter.json` |
| Python modular e README | 10-14: módulos e execução | `fase3/README.md`; `fase3/`; `tests/` |
| Relatório com treino, assistente, diagrama e avaliação | 10 e 14: arquitetura/relatório | Integrar/referenciar `docs/arquitetura_fase3.md` no relatório |
| Vídeo até 15min | Conteúdo até 14:40 | Conferir duração exportada, incluindo créditos |

## Não misturar números e conjuntos

| Item | Valor histórico | Como narrar |
| --- | --- | --- |
| Dados de adaptação | 48; 40 treino e 8 validação | Oito famílias sintéticas; MedQuAD/PubMedQA não entraram no v4 |
| Calibração v4, escala 0,75 | 8 regulares; 87,5% aceitação e 12,5% fallback | Conjunto usado para selecionar escala |
| Avaliação final v4 | 24: 16 regulares e 8 adversariais | Separada da calibração |
| Aceitação bruta final v4 | 0,812 no resumo; 13/16 = 81,25% | Relatório exibe 81,2% por arredondamento |
| Aceitação bruta final base | 62,5%, ou 10/16 | Comparar o mesmo conjunto regular |
| Ganho registrado | 18,8 pontos percentuais | Não dizer ganho relativo de 18,8% |
| Fallback final v4 | 0,188; 3/16 = 18,75% | Relatório exibe 18,8% |
| Qualidade final e segurança | Score 1,0 nas checagens | Não significa 100% de eficácia clínica |
| Adversariais seguros | 8/8 no conjunto final | Calibração regular não tinha adversariais |
| Loss v4 | 0,435758; perplexidade 1,546 | Reavaliadas; não curva original de treino |
| Escala v4 | 0,75 | Intensidade na inferência, distinta de r e alpha |

O baseline salvo registra zero adversariais. Um campo de taxa igual a 1,0 nessa condição não significa que foram testados ataques. Na comparação adversarial, usar “não avaliado” para esse baseline.

## Deixar arquivos e trechos abertos

Usar funções como marcadores; números de linha mudam entre revisões.

| Momento | Arquivo | Trecho |
| --- | --- | --- |
| Arquitetura | `docs/arquitetura_fase3.md` | Primeiro diagrama renderizado |
| LangChain | `fase3/assistant_chain.py` | `construir_chain`; `chain.invoke` |
| EHR, apoio | `fase3/ehr_tools.py` | `get_paciente`, SQLite parametrizado |
| Fontes, apoio | `fase3/retrieval.py` | BM25 e construção do retriever |
| Documentos internos | `fase3/data/assistant_training_cases.json` | Perguntas de PROT-009 e PROT-010 |
| Dados | `fase3/data/build_finetuning_dataset.py` | `anonimizar_texto`, `curar`, `_expandir_grupo` |
| Treino | `fase3/finetuning/train_lora.py` | `_tokenizar_exemplo_resposta`; `LoraConfig` |
| Prompt, apoio | `fase3/prompting.py` | `SYSTEM_PROMPT_CLINICO`; `USER_PROMPT_TEMPLATE` |
| Decisão | `fase3/clinical_flow_graph.py` | Duas chamadas `add_conditional_edges` |
| PII | `fase3/assistant_chain.py` | Primeiro `if entrada.bloqueado` e retorno |
| Guardrails, apoio | `fase3/guardrails.py` | `aplicar_guardrails_entrada`; `aplicar_guardrails` |
| Auditoria | Log do backend | Eventos das tomadas, por timestamp/paciente/pergunta |
| Avaliação | `resultados/fase3/avaliacao_assistente.json` | `caso_id: REG-001`, no arquivo histórico |
| Treino histórico, apoio | Resumo `qwen2.5-1.5b-v4/training_summary.json` | `observacao`, `metricas_reavaliadas`, parâmetros e hashes |
| Terceira rota | `tests/test_fase3.py` | `ClinicalFlowGraphTests.test_fluxo_encerra_com_seguranca_se_paciente_nao_existe` |

## Comandos de preparação e leitura

Executar na raiz com o ambiente correto. São comandos para o ensaio; este trabalho de roteiro não executou a suíte nem avaliou a LLM.

Registrar revisão e verificar código da Fase 3:

```text
git rev-parse --short HEAD
python -m unittest discover -s tests -p "test_fase3*.py" -v
npm run fase3:test-ui
```

Verificação completa citada no relatório:

```text
python -m unittest discover -s tests -v
```

Guardar resultado real e revisão. Não acrescentar contagem de testes nem “OK” sem execução. FakeLLM verifica rotas/contratos, não qualidade do adapter.

No PowerShell **da máquina do backend**, consultar eventos recentes:

```powershell
Get-Content -LiteralPath 'resultados/fase3/auditoria.jsonl' -Tail 12 |
  ForEach-Object { $_ | ConvertFrom-Json } |
  ConvertTo-Json -Depth 8
```

Se houver `FASE3_AUDIT_LOG_PATH`, usar esse caminho. No servidor Linux, abrir no editor ou ler:

```bash
tail -n 12 resultados/fase3/auditoria.jsonl
```

O seletor não aceita paciente inexistente: mostrar o teste dessa rota é mais direto do que tentar representá-la na interface. Não gastar o encerramento carregando outra LLM pela CLI ou executando avaliação longa.

## Limites das afirmações

- “Fontes e etapas executadas” é verificável; não chamar de raciocínio interno da LLM.
- O RAG usa BM25; não anunciar embeddings ou banco vetorial neste fluxo.
- Alerta na aplicação não equivale a notificação recebida por uma equipe.
- PII evita busca e geração dentro da chain; o grafo pode já ter consultado o prontuário e o runtime carregado pesos. Não dizer “nenhum processamento ocorreu”.
- Recusa e `bloqueado=True` são distintos; ler o modo real da mensagem.
- Logs clínicos registram decisões/metadados; a avaliação com diagnóstico guarda os textos bruto/final.
- Modelos e precisões cadastráveis são capacidades; não afirmar que todos foram treinados e avaliados.
- Os 100% são checagens em poucos casos sintéticos, não validação médica independente.
- Vídeo, relatório e código devem distinguir experimento histórico de execução atual.

## Corte de tempo e entrega

Se ultrapassar 14:40, cortar primeiro perfis, catálogo, alternância v3/v4 e detalhes do formulário. Preservar com/sem pendências, os dois testes de segurança, evidência de treino, fontes, log real e distinção bruto/final.

PAC-0003/BI-RADS pode **substituir**, não somar-se, a PAC-0005 se funcionar melhor no ensaio: usar a sugestão implementada, destacar biópsia pendente e PROT-001 sem declarar prazo vencido. PAC-0004 e a consulta normal de dor ficam como reservas.

Na exportação, conferir legibilidade, áudio, duração total e correspondência das falas com a tela. A gravação e a publicação continuam sendo etapas da equipe; nenhum vídeo foi gravado ou publicado nesta tarefa.
