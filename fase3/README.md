# Fase 3 - Assistente de Protocolos Clinicos

Esta pasta contem a solucao completa da Fase 3: modelo Qwen2.5-1.5B com
adapter LoRA, RAG de protocolos internos, prontuario sintetico, LangChain,
LangGraph, guardrails, auditoria, avaliacao e interface web local.

O backend padrao e sempre `local`. Groq nao e usado implicitamente e nao e
necessario configurar `GROQ_API_KEY` para executar a solucao principal.

## Inicio rapido

No PowerShell, a partir da raiz do repositorio:

```powershell
npm run fase3:setup
npm run fase3
```

O primeiro comando cria ou atualiza `.venv-fase3` e instala as dependencias.
Ele tambem localiza ou baixa o modelo base Qwen e salva o caminho do cache em
`.cache/fase3-hf-home.txt`. O segundo comando valida dependencias, adapter e
modelo base, inicia o modelo local em modo offline e abre
`http://127.0.0.1:8010`.

O modelo e carregado na primeira consulta e permanece na memoria. A primeira
resposta pode demorar mais; as seguintes reutilizam a mesma instancia.

## Operacoes do modelo pela interface

Na barra superior, abra **Operacoes do modelo** para executar e acompanhar o
pipeline sem montar comandos no terminal. A tela permite:

1. conferir ou reconstruir os splits de treino e validacao;
2. configurar e iniciar uma nova versao do adapter LoRA;
3. acompanhar etapa, progresso e logs atualizados durante a execucao;
4. cancelar o processo ativo;
5. reavaliar a loss de um adapter salvo;
6. calibrar as escalas `0.25`, `0.5`, `0.75` e `1.0` e visualizar os gates;
7. promover o adapter aprovado para as proximas consultas.

Somente uma operacao pode usar a GPU por vez. Antes de treino ou avaliacao, o
modelo de consulta e descarregado da memoria; enquanto o job estiver ativo,
novas consultas ficam temporariamente bloqueadas. Um treino concluido nunca e
promovido automaticamente: o botao **Promover** so e liberado depois de uma
calibracao aprovada para aquele mesmo adapter.

Os jobs continuam executando no processo do servico se a aba for atualizada.
Fechar ou interromper `npm run fase3`, por outro lado, encerra o servico e seus
jobs. As operacoes administrativas aceitam apenas acesso local e backend
`local`.

## Execucao sem npm

Os scripts tambem podem ser chamados diretamente:

```powershell
.\fase3\setup.ps1
.\fase3\start.ps1
```

Depois do setup, tambem e possivel iniciar pelo modulo Python:

```powershell
.\.venv-fase3\Scripts\python.exe -m fase3
```

Para iniciar sem abrir o navegador automaticamente:

```powershell
.\fase3\start.ps1 -NoBrowser
```

Para mudar a porta:

```powershell
.\fase3\start.ps1 -Port 8020
```

## Teste pelo terminal

A CLI usa o backend local por padrao; `--backend local` nao e mais necessario:

```powershell
.\.venv-fase3\Scripts\python.exe -m fase3.cli_demo `
  --paciente-id PAC-0001 `
  --pergunta "Posso iniciar a quimioterapia hoje?"
```

## Verificacao do backend

Com o servico iniciado:

```powershell
Invoke-RestMethod http://127.0.0.1:8010/api/status
```

O campo `backend` deve ser `local`. O endpoint tambem informa modelo base,
adapter, escala LoRA e se o modelo ja foi carregado.

## Testes

```powershell
.\.venv-fase3\Scripts\python.exe -m unittest tests.test_fase3 -v
.\.venv-fase3\Scripts\python.exe -m unittest tests.test_fase3_web -v
npm run fase3:test-ui
```

## Configuracao local

| Item | Padrao |
| --- | --- |
| Backend | `local` |
| Modelo | `Qwen/Qwen2.5-1.5B-Instruct` |
| Adapter inicial | `resultados/fase3/finetuning/qwen2.5-1.5b-v4/lora_adapter` |
| Registro promovido | `.cache/fase3-promoted-model.json` |
| Escala LoRA inicial | `0.75` |
| Porta web | `8010` |
| Auditoria | `resultados/fase3/auditoria.jsonl` |

O download do modelo base, quando necessario, ocorre em `npm run fase3:setup`,
nunca durante uma pergunta na interface. `FASE3_HF_HOME` ou `HF_HOME` podem ser
usados antes do setup para escolher o local do cache. O comando
`npm run fase3` ativa `HF_HUB_OFFLINE=1` automaticamente.

`FASE3_PYTHON` pode apontar para outro interpretador com todas as dependencias
instaladas. Isso e util para reutilizar um ambiente CUDA existente:

```powershell
$env:FASE3_PYTHON = "E:\caminho\do\ambiente\Scripts\python.exe"
npm run fase3
```

## Backends alternativos

Os backends abaixo existem apenas para testes ou comparacao explicita:

```powershell
# Resposta deterministica, sem carregar modelo
.\.venv-fase3\Scripts\python.exe -m fase3 --backend fake

# Provedor remoto; exige GROQ_API_KEY
.\.venv-fase3\Scripts\python.exe -m fase3 --backend groq
```

Passar `--backend` explicitamente tem prioridade sobre qualquer variavel de
ambiente. O comando `npm run fase3` sempre passa `--backend local` e, portanto,
nao pode trocar silenciosamente para Groq.

## Estrutura principal

| Caminho | Responsabilidade |
| --- | --- |
| `llm_backend.py` | Modelo local, Groq e fake |
| `assistant_chain.py` | Prompt, RAG, grounding e resposta final |
| `clinical_flow_graph.py` | Decisoes e alertas do LangGraph |
| `ehr_tools.py` | Prontuario SQLite sintetico |
| `web_app.py` | API e ciclo de vida do modelo |
| `training_service.py` | Jobs controlados de dados, treino, avaliacao e promocao |
| `web/` | Interface no navegador |
| `evaluate_assistant.py` | Avaliacao e gates de qualidade |
| `finetuning/` | Treinamento e avaliacao do adapter |

Esta e uma solucao academica com dados sinteticos. Nenhuma resposta substitui
avaliacao, prescricao ou decisao de um profissional de saude.
