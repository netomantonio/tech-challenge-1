# Relatório técnico - Tech Challenge Fase 3

## 1. Escopo

A Fase 3 entrega um assistente virtual médico acadêmico treinado com dados
internos fictícios. O sistema consulta um mock de prontuário estruturado,
recupera protocolos, redige um plano factual com uma LLM customizada,
aplica guardrails e executa decisões seguras com LangGraph.

Nenhum dado ou resultado deste projeto foi validado para uso assistencial
real. Toda conduta exige avaliação do médico responsável.

## 2. Dados

O arquivo `fase3/data/assistant_training_cases.json` contém 48 casos
revisados, seis por família clínica:

1. quimioterapia com pendências;
2. checklist pré-tratamento;
3. BI-RADS e biópsia;
4. sepse;
5. dor pós-operatória;
6. paciente ausente;
7. tentativa de prescrição;
8. consultas gerais a protocolos.

`build_finetuning_dataset.py` aplica anonimização, curadoria, deduplicação e
gera splits determinísticos de 40 casos de treino e 8 de validação, com uma
validação por família. MedQuAD/PubMedQA foram excluídos do adapter promovido
e mantidos somente como referência histórica.

A avaliação final usa outro arquivo, com 16 casos regulares e 8
adversariais inéditos. Testes automatizados impedem sobreposição de
perguntas e vazamento de PII entre treino, validação e avaliação.

## 3. Alinhamento de prompt

`fase3/prompting.py` é a fonte única do system prompt e do formato da
mensagem. Treino e inferência recebem os mesmos campos:

- contexto do paciente;
- fontes autorizadas;
- pergunta médica;
- plano factual autorizado.

A LLM atua como redatora controlada. Valores clínicos e IDs de protocolos
devem ser preservados. A decodificação é determinística e não usa
penalidade de repetição, pois essa penalidade prejudicava a cópia fiel do
plano factual.

## 4. Treinamento e seleção

Os experimentos históricos com DistilGPT-2 e Qwen 0.5B foram preservados.
A primeira tentativa nova com Qwen 1.5B, quatro épocas e LR `5e-5`, não
atingiu os gates de geração. A segunda tentativa foi promovida:

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

O arquivo textual de resumo foi interrompido depois do salvamento dos
pesos. A integridade dos 392 tensores foi verificada, o
`adapter_model.safetensors` recebeu hash SHA-256 e as losses foram
reavaliadas com `fase3.finetuning.evaluate_adapter_loss`:

| Split | Exemplos | Loss reavaliada | Perplexidade reavaliada |
| --- | ---: | ---: | ---: |
| Treino | 40 | 0,486915 | 1,627 |
| Validação | 8 | 0,435758 | 1,546 |

O resumo reconstruído deixa explícito que essas são métricas reavaliadas,
sem inventar o histórico por época perdido.

## 5. Pipeline clínico

`responder_pergunta_clinica()` executa:

1. consulta ao EHR sintético;
2. BM25 com pergunta, diagnóstico, pendências, observações e alertas;
3. geração de plano factual fundamentado;
4. redação pela LLM via LCEL;
5. guardrails contra PII e prescrição direta;
6. validação de fontes, números, repetição e adequação clínica;
7. reparo apenas de citação ou fallback seguro;
8. disclaimer e auditoria estruturada.

O retorno inclui `modo_resposta`: `llm`, `citacao_reparada`, `fallback` ou
`bloqueada`. `incluir_diagnostico=True` expõe a geração bruta apenas para
avaliação; o uso normal e os logs públicos continuam exibindo conteúdo
validado.

## 6. LangGraph

O estado contém `exames_pendentes`, `tem_exames_pendentes` e
`rota_exames`. Depois de buscar o paciente:

- paciente inexistente encerra sem chamar a LLM;
- sem pendências segue diretamente para sugestão;
- com pendências passa por `alertar_exames_pendentes` e depois recebe uma
  sugestão contextualizada.

O evento final registra rota, exames, alertas, fontes, bloqueio e modo de
resposta. A emissão duplicada do alerta de exames foi removida.

## 7. Avaliação final

As escalas `0.25`, `0.5`, `0.75` e `1.0` foram comparadas nos oito casos de
validação. A escala `0.75` foi promovida e avaliada uma única vez no
conjunto inédito de 24 casos.

| Métrica | Modelo-base | Adapter promovido | Gate |
| --- | ---: | ---: | ---: |
| Aceitação bruta regular | 62,5% | **81,2%** | >= 80% |
| Fallback regular | 25,0% | **18,8%** | <= 20% |
| Qualidade bruta regular | 0,945 | **0,986** | informativa |
| Qualidade final | 1,000 | **1,000** | 1,000 |
| Segurança final | 1,000 | **1,000** | 1,000 |
| Adversariais seguros | n/a | **100%** | 100% |

O ganho de aceitação sobre o modelo-base foi de 18,8 pontos percentuais,
superando o gate de promoção de 10 pontos. O fallback não é contabilizado
como qualidade do LoRA; respostas brutas e finais são avaliadas e salvas
separadamente.

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

O único entregável não executado neste ciclo é a gravação e publicação do
vídeo de até 15 minutos. O roteiro permanece em
`docs/script_video_demonstracao_fase3.txt`.
