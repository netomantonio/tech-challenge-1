# Cloudflare: infraestrutura de nuvem e deploy

Este projeto roda integralmente na borda da Cloudflare, usando Pages, Python
Workers, R2, Turnstile, Rate Limiting e Service Bindings. É uma demonstração
isolada: não há D1, contas de usuário, histórico de consultas nem persistência
de features ou respostas clínicas.

## Ambiente ao vivo

| Recurso | Endereço |
| --- | --- |
| **Aplicação web (Pages)** | https://cancer-mama-web-preview.pages.dev |
| **API (Worker)** | https://cancer-mama-api-preview.antonio-5b5.workers.dev |
| Health check | https://cancer-mama-api-preview.antonio-5b5.workers.dev/health |
| Swagger UI | https://cancer-mama-api-preview.antonio-5b5.workers.dev/docs |

> O caminho de uso normal é a aplicação web. A URL do Worker é exposta apenas
> para verificação; em produção o acesso se dá pela Pages Function via Service
> Binding interno.

## Arquitetura

```text
Navegador
   │
   ▼
Cloudflare Pages (React) ── arquivos estáticos
   │
   ▼
Pages Function ── Service Binding "API" ──► Python Worker ── HTTPS ──► Groq (LLM)
                                              │
                                              ├─ Rate Limiting (/predict, /interpret)
                                              └─ Turnstile (/interpret)

Deploy ── artefatos imutáveis por commit ──► R2 privado (builds/<commit-sha>/)
```

### Serviços de nuvem utilizados

| Serviço Cloudflare | Recurso | Papel |
| --- | --- | --- |
| **Pages** | `cancer-mama-web-preview` | Hospeda o frontend estático e as Pages Functions. |
| **Workers (Python)** | `cancer-mama-api-preview` | Executa o FastAPI pelo adaptador ASGI (Pyodide). Inferência + LLM. |
| **R2** | `cancer-mama-artifacts-preview` | Cópias imutáveis do modelo, relatórios e resultados por commit. Fora do caminho crítico. |
| **Service Bindings** | binding `API` | Liga a Pages Function ao Worker sem trafegar pela internet pública. |
| **Turnstile** | widget | Desafio anti-bot exigido em `POST /interpret`. |
| **Rate Limiting** | `PREDICT_RATE_LIMITER`, `INTERPRET_RATE_LIMITER` | Limita `/predict` (30/60s) e `/interpret` (5/60s) por chave pseudônima do cliente. |
| **Observability** | logs do Worker | `wrangler tail` e painel, com `head_sampling_rate` total. |

O modelo é carregado do manifesto [`src/modelo_serving.json`](../src/modelo_serving.json),
empacotado junto ao Worker e matematicamente equivalente ao `joblib` original,
validado nas 569 amostras do dataset. O R2 guarda apenas cópias versionadas.

### Arquivos de configuração

| Arquivo | Função |
| --- | --- |
| [`cloudflare/api/wrangler.jsonc`](../cloudflare/api/wrangler.jsonc) | Worker: nome, `main`, flags Python, vars, rate limiters, observability. |
| [`cloudflare/api/pyproject.toml`](../cloudflare/api/pyproject.toml) | Dependências do Worker (FastAPI, httpx, prometheus-client) e toolchain. |
| [`frontend/wrangler.jsonc`](../frontend/wrangler.jsonc) | Pages: nome, diretório de saída e Service Binding `API`. |
| [`deploy/cloudflare/deploy-worker.sh`](../deploy/cloudflare/deploy-worker.sh) | Pipeline de deploy do Worker + envio de artefatos ao R2 (usado pelo Cloudflare Builds). |
| [`deploy/cloudflare/test-worker.sh`](../deploy/cloudflare/test-worker.sh) | Testes e empacotamento `--dry-run` do Worker. |

## Secrets e variáveis

| Nome | Onde | Tipo | Descrição |
| --- | --- | --- | --- |
| `GROQ_API_KEY` | Worker | secret | Chave da API Groq (interpretação LLM). Sem ela, `/interpret` responde `503`. |
| `TURNSTILE_SECRET_KEY` | Worker | secret | Chave **secreta** do Turnstile. Sem ela, `/interpret` responde `503`. |
| `GROQ_LLM_MODEL` | Worker | var | Modelo do Groq (padrão `openai/gpt-oss-120b`). |
| `ENVIRONMENT` | Worker | var | Identifica o ambiente (`preview`). |
| `VITE_TURNSTILE_SITE_KEY` | Pages | var de build | Chave **pública** do Turnstile, embutida no bundle. |

Para preview/demonstração são usadas as chaves de **teste** oficiais do
Cloudflare (site `1x00000000000000000000AA`, secret
`1x0000000000000000000000000000000AA`), que sempre validam. Em produção, gere
um widget Turnstile real e autorize os hostnames do Pages e do Worker.

---

## Procedimento A — Deploy via CLI (`wrangler`)

Publicação direta a partir de uma máquina com `wrangler` autenticado. É o
caminho mais rápido para subir ou atualizar o ambiente preview.

### Pré-requisitos

- [`wrangler`](https://developers.cloudflare.com/workers/wrangler/) autenticado
  (`wrangler login` ou `CLOUDFLARE_API_TOKEN`) com permissão de Workers, Pages e R2.
- [`uv`](https://docs.astral.sh/uv/) (toolchain do Python Worker) e Node.js 20+.

### 1. Worker (API)

```bash
# bucket R2 de artefatos (idempotente)
wrangler r2 bucket create cancer-mama-artifacts-preview

# deploy do Python Worker
cd cloudflare/api
uv run pywrangler deploy

# secrets de runtime (lidos do stdin, sem ecoar)
printf '%s' "$GROQ_API_KEY"        | wrangler secret put GROQ_API_KEY
printf '%s' "$TURNSTILE_SECRET_KEY" | wrangler secret put TURNSTILE_SECRET_KEY
```

### 2. Artefatos no R2 (opcional, espelha o pipeline)

```bash
sha="$(git rev-parse HEAD)"
while IFS= read -r -d '' file; do
  wrangler r2 object put "cancer-mama-artifacts-preview/builds/$sha/$file" --file "$file"
done < <(git ls-files -z src/modelo_serving.json resultados/fase2 docs \
            'relatorio_tecnico_*.md' 'relatorio_tecnico_*.pdf')
```

### 3. Frontend (Pages)

```bash
cd frontend
npm ci
VITE_TURNSTILE_SITE_KEY=1x00000000000000000000AA npm run build

# cria o projeto na primeira vez (idempotente)
wrangler pages project create cancer-mama-web-preview \
  --production-branch feat/cloudflare-fullstack

# publica (lê o Service Binding de frontend/wrangler.jsonc)
wrangler pages deploy --branch feat/cloudflare-fullstack
```

---

## Procedimento B — Deploy via Cloudflare Builds (GitHub)

Fluxo contínuo: cada commit na branch gera artefatos imutáveis e publica
automaticamente. É a forma recomendada para um ambiente sustentado.

### Worker

1. Em **Workers & Pages → Create application**, importe o repositório GitHub e
   autorize o GitHub App da Cloudflare.
2. Nome `cancer-mama-api-preview`, branch `feat/cloudflare-fullstack`, diretório
   raiz = raiz do repositório.
3. Comando de build: `npm ci && npm run test:worker`
4. Comando de deploy: `npm run deploy:worker`
5. Secrets de runtime: `GROQ_API_KEY` e `TURNSTILE_SECRET_KEY`.
6. Caminhos observados:

   ```text
   cloudflare/api/*
   deploy/cloudflare/*
   src/*
   tests/*
   scripts/*
   resultados/fase2/*
   package.json
   package-lock.json
   requirements.txt
   ```

O script de deploy tem uma trava (`WORKERS_CI=1`) que o impede de publicar fora
do Cloudflare Builds. Na primeira execução, o pipeline cria o bucket
`cancer-mama-artifacts-preview`, envia os artefatos e publica o Worker. O token
automático do Builds precisa de escrita em Workers e R2.

> **Atenção ao dataset:** `data/cancer_mama.csv` não é versionado. O
> `test:worker` baixa o dataset do Kaggle quando ele não existe, o que exige
> credenciais Kaggle (`KAGGLE_USERNAME`/`KAGGLE_KEY`) no ambiente do Builds.
> Sem isso, o build falha. Alternativas: versionar o CSV, configurar as
> credenciais, ou desacoplar `test:worker` do dataset.

### Pages

1. Crie um segundo projeto no Cloudflare Pages a partir do mesmo repositório,
   nome `cancer-mama-web-preview`, branch `feat/cloudflare-fullstack`.
2. Diretório raiz `frontend`; build `npm ci && npm test && npm run build`;
   saída `dist`.
3. Variável de build `VITE_TURNSTILE_SITE_KEY` com a site key pública.
4. Confirme o Service Binding `API` apontando para `cancer-mama-api-preview`
   (também declarado em [`frontend/wrangler.jsonc`](../frontend/wrangler.jsonc)).
5. Caminhos observados: `frontend/*`.

No Turnstile, autorize os hostnames do Pages e do Worker. O endpoint
`/interpret` exige `X-Turnstile-Token`; `/predict` não exige desafio.

---

## Validação local sem publicar

```bash
npm ci
npm run test:worker        # testes + empacotamento --dry-run do Worker

cd frontend
npm ci
npm test
npm run build
```

Worker localmente (sem publicar):

```bash
cp cloudflare/api/.dev.vars.example cloudflare/api/.dev.vars
cp frontend/.env.example frontend/.env.local
cd cloudflare/api
uv run pywrangler dev
```

## Critérios de verificação do preview

- `GET /health` retorna `{"status":"ready"}`.
- `POST /predict` responde com classificação e probabilidades.
- `POST /interpret` sem Turnstile retorna `403`; sem secret configurado, `503`;
  com Turnstile válido, `200` com a interpretação do LLM.
- A aplicação web carrega, classifica e renderiza a interpretação em Markdown.
- O bucket R2 contém `builds/<commit-sha>/src/modelo_serving.json` e os demais
  artefatos versionados.

> **Rate limiting:** os bindings `PREDICT_RATE_LIMITER` e `INTERPRET_RATE_LIMITER`
> estão configurados. A Workers Rate Limiting API é aproximada e
> *eventually consistent*, portanto o bloqueio (`429`) pode não ocorrer de forma
> determinística na N-ésima requisição em rajadas rápidas de um único cliente.
