# Relatório técnico - Tech Challenge Fase 3

## 1. Escopo

Nesta fase, desenvolvemos um assistente virtual médico acadêmico treinado
com dados internos fictícios. Nossa solução consulta um prontuário
estruturado, recupera protocolos institucionais, utiliza uma LLM
customizada para redigir a resposta, aplica guardrails e organiza as
decisões por meio do LangGraph.

Nenhum dado ou resultado deste projeto foi validado para uso assistencial
real. Toda conduta exige avaliação do médico responsável.

## 2. Dados

Organizamos o arquivo `fase3/data/assistant_training_cases.json` com 48
casos revisados, sendo seis exemplos para cada família clínica:

1. quimioterapia com pendências;
2. checklist pré-tratamento;
3. BI-RADS e biópsia;
4. sepse;
5. dor pós-operatória;
6. paciente ausente;
7. tentativa de prescrição;
8. consultas gerais a protocolos.

No script `build_finetuning_dataset.py`, aplicamos anonimização, curadoria,
deduplicação e divisão determinística dos dados. O conjunto final contém 40
casos de treino e 8 de validação, com um exemplo de validação por família.
Mantivemos MedQuAD e PubMedQA somente como referência histórica, sem
utilizá-los no adapter promovido.

Para a avaliação final, utilizamos outro arquivo com 16 casos regulares e 8
casos adversariais inéditos. Também criamos testes automatizados para evitar
sobreposição de perguntas e vazamento de informações pessoais entre os
conjuntos de treino, validação e avaliação.

## 3. Alinhamento de prompt

Utilizamos `fase3/prompting.py` como fonte única do system prompt e do
formato das mensagens. Dessa maneira, mantivemos os mesmos campos no treino
e na inferência:

- contexto do paciente;
- fontes autorizadas;
- pergunta médica;
- plano factual autorizado.

Definimos a LLM como uma redatora controlada: os valores clínicos e os IDs
dos protocolos devem ser preservados. Optamos por uma decodificação
determinística e retiramos a penalidade de repetição, pois observamos que
ela prejudicava a reprodução fiel do plano factual.

## 4. Treinamento e seleção

Durante o desenvolvimento, executamos diferentes treinamentos até chegar ao
adapter promovido. A tabela abaixo resume os resultados e a decisão tomada
em cada etapa:

| Experimento | Modelo-base | Loss de validação | Perplexidade | Resultado e decisão |
| --- | --- | ---: | ---: | --- |
| Smoke inicial | `distilgpt2` | 4,7692 | 117,826 | Validou o pipeline, mas gerou texto repetitivo e inadequado ao domínio clínico. |
| Qwen 0,5B v1 | `Qwen2.5-0.5B-Instruct` | 2,6581 | 14,269 | Melhorou a estrutura, porém omitiu fontes e apresentou valores inconsistentes. |
| Qwen 0,5B v2 | `Qwen2.5-0.5B-Instruct` | 2,2172 | 9,182 | Reduziu a loss, mas alterou BI-RADS e valores da escala de dor; não foi promovido. |
| Qwen 1,5B v3 | `Qwen2.5-1.5B-Instruct` | 0,313368 | 1,368 | Obteve a menor loss, mas não atingiu os gates de geração bruta. |
| **Qwen 1,5B v4** | **`Qwen2.5-1.5B-Instruct`** | **0,435758**¹ | **1,546**¹ | **Superou os gates de geração e segurança e foi promovido.** |

¹ Métricas reavaliadas deterministicamente com o adapter salvo.

Essa sequência de experimentos mostrou que a menor loss não representa,
necessariamente, a melhor resposta clínica. Embora o v3 tenha alcançado uma
loss menor que o v4, ele não preservou a qualidade esperada nos testes de
geração. Por esse motivo, nossa escolha considerou conjuntamente a geração
bruta, o uso de fallback, a preservação dos fatos e a segurança.

A configuração final foi:

| Configuração | Valor |
| --- | --- |
| Modelo-base | `Qwen/Qwen2.5-1.5B-Instruct` |
| Adapter | `qwen2.5-1.5b-v4/lora_adapter` |
| LoRA | `r=16`, `alpha=32`, dropout `0.05` |
| Épocas / LR | 6 / `2e-5` |
| Sequência | 512 |
| Batch / acumulação | 2 / 4 |
| Precisão / GPU | FP16 / RTX 3060 12 GB |
| Seed | 42 |
| Escala promovida | `0.75` |

O arquivo textual original do resumo foi interrompido depois do salvamento
dos pesos do v4. Para não reconstruirmos informações que não estavam mais
disponíveis, verificamos a integridade dos 392 tensores, calculamos o hash
SHA-256 do `adapter_model.safetensors` e reavaliamos as losses de maneira
determinística com `fase3.finetuning.evaluate_adapter_loss`:

| Split | Exemplos | Loss reavaliada | Perplexidade reavaliada |
| --- | ---: | ---: | ---: |
| Treino | 40 | 0,486915 | 1,627 |
| Validação | 8 | 0,435758 | 1,546 |

No resumo reconstruído, deixamos explícito que essas métricas foram
reavaliadas. O histórico por época não foi estimado ou inventado. Em um
próximo treinamento, pretendemos preservar também o `trainer_state.json` e
o histórico completo de logs do Trainer.

## 5. Pipeline clínico

No método `responder_pergunta_clinica()`, implementamos as seguintes etapas:

1. consulta ao EHR sintético;
2. BM25 com pergunta, diagnóstico, pendências, observações e alertas;
3. geração de plano factual fundamentado;
4. redação pela LLM via LCEL;
5. guardrails contra PII e prescrição direta;
6. validação de fontes, números, repetição e adequação clínica;
7. reparo apenas de citação ou fallback seguro;
8. disclaimer e auditoria estruturada.

O retorno informa o `modo_resposta`: `llm`, `citacao_reparada`, `fallback`
ou `bloqueada`. A opção `incluir_diagnostico=True` expõe a geração bruta
somente para avaliação. No uso normal, apresentamos apenas o conteúdo que
passou pelas validações.

## 6. LangGraph

No LangGraph, definimos um estado com `exames_pendentes`,
`tem_exames_pendentes` e `rota_exames`. Depois da busca do paciente:

- paciente inexistente encerra sem chamar a LLM;
- sem pendências segue diretamente para sugestão;
- com pendências passa por `alertar_exames_pendentes` e depois recebe uma
  sugestão contextualizada.

Ao final, registramos a rota, os exames, os alertas, as fontes, o eventual
bloqueio e o modo de resposta. Também removemos a emissão duplicada do
alerta de exames.

## 7. Avaliação final

Comparamos as escalas `0.25`, `0.5`, `0.75` e `1.0` nos oito casos de
validação. Após essa calibração, promovemos a escala `0.75` e a avaliamos
uma única vez no conjunto inédito de 24 casos.

| Métrica | Modelo-base | Adapter promovido | Gate |
| --- | ---: | ---: | ---: |
| Aceitação bruta regular | 62,5% | **81,2%** | >= 80% |
| Fallback regular | 25,0% | **18,8%** | <= 20% |
| Qualidade bruta regular | 0,945 | **0,986** | informativa |
| Qualidade final | 1,000 | **1,000** | 1,000 |
| Segurança final | 1,000 | **1,000** | 1,000 |
| Adversariais seguros | n/a | **100%** | 100% |

Obtivemos um ganho de 18,8 pontos percentuais de aceitação em relação ao
modelo-base, superando o gate de promoção de 10 pontos. Para não atribuir ao
LoRA uma qualidade produzida pelo fallback, avaliamos e armazenamos
separadamente as respostas brutas e as respostas finais.

Artefatos principais:

- `resultados/fase3/avaliacao_assistente.json` e `.csv`;
- `resultados/fase3/resumo_avaliacao_assistente.json`;
- `resultados/fase3/calibracao_adapter.json`;
- `resultados/fase3/finetuning/comparacao_modelos.json`;
- `resultados/fase3/finetuning/qwen2.5-1.5b-v4/training_summary.json`.

## 8. Reprodutibilidade

```bash
python -m pip install -r requirements.txt -r requirements-fase3.txt
python -m fase3.data.build_finetuning_dataset
python -m fase3.calibrate_adapter \
  --adapter-path resultados/fase3/finetuning/qwen2.5-1.5b-v4/lora_adapter \
  --scales 0.25,0.5,0.75,1.0 --enforce-gates
python -m unittest discover -s tests -v
```

## 9. Limitações e próximos passos

Mesmo com os resultados obtidos, reconhecemos as seguintes limitações:

- utilizamos somente dados sintéticos e anonimizados;
- o dataset de fine-tuning possui 48 exemplos;
- a avaliação final contém 24 casos e ainda não substitui uma avaliação
  clínica conduzida por especialistas;
- o histórico original por época do treinamento v4 não foi preservado;
- a solução não foi validada para uso assistencial real.

Como próximos passos, pretendemos ampliar o conjunto de dados revisados,
incluir uma avaliação cega realizada por profissionais da área da saúde,
preservar todos os logs do Trainer e avaliar modelos instrucionais maiores.

O único entregável externo ainda pendente é a gravação e a publicação do
vídeo de até 15 minutos. O roteiro está disponível em
`docs/script_video_demonstracao_fase3.txt`.
