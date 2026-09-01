# Fase 3 - Assistente de Protocolos Clinicos

Solucao academica com modelo local configuravel, adapter LoRA/QLoRA, RAG de
protocolos, prontuario sintetico, LangChain, LangGraph, guardrails, auditoria,
avaliacao e interface web. O backend padrao e `local`; Groq somente e usado
quando escolhido explicitamente.

## Inicio rapido

Windows, na raiz do repositorio:

```powershell
npm run fase3:setup
npm run fase3
```

Linux ou macOS:

```bash
bash fase3/setup.sh
bash fase3/start.sh
```

O setup cria `.venv-fase3`, instala as dependencias e prepara o preset
`qwen2.5-1.5b`. A execucao valida o modelo e o adapter promovido antes de abrir
`http://127.0.0.1:8010`. O processo de inferencia usa `HF_HUB_OFFLINE=1`, por
isso uma consulta nunca inicia um download inesperado.

### Atualizacao e reinicio no Linux

Servidores Linux podem manter sua configuracao em um arquivo local e executar
todo o ciclo de atualizacao com um comando:

```bash
cp fase3/service.env.example fase3/service.env
# Edite fase3/service.env uma unica vez.
bash fase3/update_restart.sh
```

O script valida se o repositorio esta limpo, busca a branch configurada, encerra
somente o PID registrado em `.logs/fase3-backend.pid`, executa
`git pull --ff-only`, inicia o backend com `nohup` e aguarda
`/api/capabilities`. O terminal fica livre depois da inicializacao; logs e PID
ficam em `.logs/`. Se os arquivos de requisitos mudarem entre as revisoes, as
dependencias sao atualizadas automaticamente com o mesmo Python do servico.

Para o backend Lightning usado pelo grupo, os principais valores sao:

```dotenv
FASE3_GIT_BRANCH=main
FASE3_HOST=0.0.0.0
FASE3_PORT=8000
FASE3_ALLOWED_ORIGINS=https://assistente-protocolos-fase3.pages.dev
FASE3_ALLOW_REMOTE_TRAINING=1
FASE3_INSTANCE_NAME="Backend Lightning"
```

Se o Studio nao possuir `.venv-fase3`, o script detecta o Python ativo. Tambem
e possivel fixa-lo em `FASE3_PYTHON` dentro de `service.env`.

Para instalar apenas o ambiente e escolher o modelo depois pela interface:

```powershell
./fase3/setup.ps1 -SkipModel
```

## Frontend e backends da equipe

O frontend pode ser aberto pelo proprio backend ou publicado como site
estatico. Cada navegador guarda seus perfis em `localStorage` e chama
diretamente o backend ativo. Cloudflare Pages nao recebe consultas, modelos ou
dados de treinamento.

Na primeira abertura, o wizard exige uma conexao valida e um modelo disponivel.
Os perfis podem ser adicionados, editados, testados e removidos pelo botao de
configuracao. Conversas sao isoladas por `backend_id + paciente_id`.

Na etapa **Endereco**, o wizard monta o comando `python -m fase3` em tempo real.
E possivel definir nome da instancia, endereco de escuta, porta, abertura da
interface local e permissao de treinamento por outras maquinas. O comando ja
inclui a origem do frontend e mostra um alerta quando a configuracao expoe o
backend ou as operacoes de modelo na rede.

### Mesma maquina

Use `http://127.0.0.1:8010`. O backend permanece restrito ao loopback:

```powershell
./fase3/start.ps1
```

Chrome e Edge podem solicitar permissao de acesso a rede local quando o
frontend veio de uma origem HTTPS.

### Outra maquina na LAN ou VPN

Na maquina que possui o modelo:

```powershell
./fase3/start.ps1 -HostAddress 0.0.0.0 `
  -AllowedOrigin "https://assistente-protocolos-fase3.pages.dev"
```

No wizard, informe `http://192.168.x.x:8010`, um hostname `.local` ou um IP
privado acessivel por VPN. A porta 8010 precisa estar liberada no firewall. Um
`localhost` informado no computador de outro integrante continua apontando
para o computador desse integrante, nunca para a maquina que executa o modelo.

Consultas remotas sao permitidas. Operacoes de modelo vindas de outra maquina
ficam desativadas por padrao. Para habilita-las explicitamente:

```powershell
./fase3/start.ps1 -HostAddress 0.0.0.0 `
  -AllowedOrigin "https://assistente-protocolos-fase3.pages.dev" `
  -AllowRemoteTraining
```

Esse modo nao possui autenticacao e exibe um aviso destacado no terminal. Use
somente em uma rede ou VPN confiavel.

### Servidor HTTPS ou Tunnel

Enderecos publicos precisam usar HTTPS. Cloudflare Tunnel pode publicar o
backend sem DNS proprio; configure a URL HTTPS resultante no wizard e inclua a
origem Pages em `FASE3_ALLOWED_ORIGINS`. Sem rota LAN, VPN ou Tunnel, o
navegador nao consegue acessar uma maquina privada.

## Cloudflare Pages

O build estatico e separado do backend:

```powershell
npm run fase3:web:build
npm run fase3:web:preview
```

O preview abre em `http://localhost:8788`. Para publicar o projeto
`assistente-protocolos-fase3`:

```powershell
npx wrangler login
npm run fase3:web:create
npm run fase3:web:deploy
```

O artefato fica em `fase3/web/dist` e inclui CSP, headers de seguranca e SPA
fallback. Nao existe Worker proxy nem backend central nesse deploy.

Frontend publicado: <https://assistente-protocolos-fase3.pages.dev>. Os scripts
`start.ps1` e `start.sh` ja autorizam essa origem por padrao. Para substituir
ou acrescentar origens, use `-AllowedOrigin` no PowerShell ou
`FASE3_ALLOWED_ORIGINS` no ambiente.

## Modelos e treinamento

Na tela **Operacoes do modelo** e possivel:

1. cadastrar repo ID/revisao do Hugging Face ou diretorio permitido;
2. instalar e acompanhar o download por job e logs;
3. escolher CPU/GPU, FP32, FP16, BF16 ou QLoRA NF4 conforme capacidades;
4. treinar LoRA com versao `{alias}-vN`;
5. avaliar loss, calibrar escalas, conferir gates e promover o adapter.

Os campos editaveis possuem ajuda contextual pelo botao `?`. Os textos explicam
funcao, relacoes entre parametros e impacto esperado em memoria, tempo,
estabilidade, reproducibilidade e qualidade. A ajuda funciona por hover, foco
de teclado ou clique/toque e pode ser fechada com `Esc`.

Qwen2.5 0.5B e 1.5B sao presets, nao limitacoes. Modelos precisam ser
compativeis com `AutoModelForCausalLM`, tokenizer e PEFT. Para modelos locais,
configure raizes com `FASE3_MODEL_ROOTS` separadas pelo delimitador de caminhos
do sistema. `trust_remote_code` exige opt-in no cadastro e revisao fixa.

NF4 exige GPU CUDA e `bitsandbytes`, instalado por `setup.ps1`/`setup.sh` a
partir de `requirements-fase3.txt`; a opcao fica indisponivel quando o backend
nao reporta esse recurso. Depois de atualizar as dependencias, reinicie o
backend para atualizar as capacidades. Nenhum download, treino ou promocao e iniciado pelos
tours ou pelo wizard sem clique explicito.

## Diagnostico e API

```powershell
./.venv-fase3/Scripts/python.exe -m fase3.manage doctor
Invoke-RestMethod http://127.0.0.1:8010/api/capabilities
```

`/api/capabilities` informa versao da API, CPU, memoria, GPU/VRAM, precisao,
modelos, adapter promovido, raizes permitidas e permissao de treino remoto.
O terminal mostra URL loopback, IPs privados, origens CORS e estado remoto.

Variaveis principais:

| Variavel | Funcao |
| --- | --- |
| `FASE3_ALLOWED_ORIGINS` | Origens CORS separadas por virgula |
| `FASE3_ALLOW_REMOTE_TRAINING=1` | Libera operacoes remotas sem autenticacao |
| `FASE3_MODEL_ROOTS` | Raizes aceitas para modelos locais |
| `FASE3_HF_HOME` ou `HF_HOME` | Cache do Hugging Face |
| `FASE3_PYTHON` | Interpretador alternativo com CUDA/dependencias |
| `FASE3_INSTANCE_NAME` | Nome exibido no seletor de backend |

O backend responde preflight CORS e Private Network Access com
`Access-Control-Allow-Private-Network: true` quando solicitado. Nao use
`no-cors`: respostas opacas nao permitem validar a API.

## Testes

```powershell
./.venv-fase3/Scripts/python.exe -m unittest discover -s tests -v
npm run fase3:test-ui
npm run fase3:web:build
```

## Estrutura

| Caminho | Responsabilidade |
| --- | --- |
| `web_app.py` | API, CORS/PNA, capacidades e ciclo de vida do modelo |
| `model_registry.py` | Presets e modelos customizados |
| `model_manager.py`, `manage.py` | Instalacao e diagnostico portatil |
| `training_service.py` | Jobs, logs, avaliacao e promocao |
| `finetuning/train_lora.py` | LoRA/QLoRA generico e manifesto do adapter |
| `web/` | Frontend, wizard, perfis, tours e build Pages |
| `clinical_flow_graph.py` | Rotas e alertas LangGraph |
| `assistant_chain.py` | Prompt, RAG, grounding e resposta final |

Os prontuarios e dados sao sinteticos. Nenhuma resposta substitui avaliacao,
prescricao ou decisao de um profissional de saude.
