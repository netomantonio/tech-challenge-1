# Frontend (React + Cloudflare Pages)

Interface web da demonstração de diagnóstico de câncer de mama. Entrega o
formulário das 30 features do dataset Wisconsin, aciona a classificação e
apresenta a interpretação assistida por LLM de forma amigável.

- **Ambiente ao vivo:** https://cancer-mama-web-preview.pages.dev
- **Código:** [`frontend/`](../frontend)

## Stack

| Item | Tecnologia |
| --- | --- |
| Biblioteca de UI | React 19 |
| Bundler / dev server | Vite 8 |
| Linguagem | TypeScript 5.9 |
| Testes | Vitest 4 + Testing Library (ambiente `jsdom`) |
| Proteção anti-bot | `@marsidev/react-turnstile` (Cloudflare Turnstile) |
| Renderização de Markdown | `react-markdown` 10 + `remark-gfm` 4 |
| Hospedagem | Cloudflare Pages + Pages Functions |

## Estrutura

```text
frontend/
  functions/
    [[path]].ts        # Pages Function: proxy para o Worker via Service Binding
    proxy.test.ts
  public/
    _headers           # CSP e cabeçalhos de segurança
    _routes.json       # rotas que invocam a Function (resto é estático)
  src/
    App.tsx            # componente principal (formulário, resultado, interpretação)
    api.ts             # cliente fetch para /predict e /interpret
    features.ts        # definição das 30 features e exemplo acadêmico
    types.ts           # tipos das respostas da API + type guard
    main.tsx           # bootstrap do React
    styles.css         # estilos (inclui a renderização do Markdown)
    test/setup.ts      # setup do Testing Library
  wrangler.jsonc       # configuração do projeto Pages e do Service Binding
  vite.config.ts       # plugin React + configuração do Vitest
```

## Como os dados fluem

O frontend nunca fala direto com o Worker. As chamadas usam **caminhos
relativos** (`/predict`, `/interpret`), capturadas pela Pages Function e
encaminhadas ao Worker por um **Service Binding** interno:

```text
Navegador
   │  fetch("/predict" | "/interpret")  (caminho relativo)
   ▼
Pages Function  frontend/functions/[[path]].ts
   │  context.env.API.fetch(request)    (Service Binding, sem sair para a internet)
   ▼
Python Worker  cancer-mama-api-preview
```

- O cliente em [`src/api.ts`](../frontend/src/api.ts) envia `{"features": {...}}`
  e, no caso de `/interpret`, adiciona o cabeçalho `X-Turnstile-Token`.
- A Function em [`functions/[[path]].ts`](../frontend/functions/[[path]].ts)
  reforça cabeçalhos de segurança na resposta (`X-Content-Type-Options`,
  `Referrer-Policy`, `Cache-Control: no-store`) e devolve `503` amigável se o
  Worker estiver indisponível.
- O arquivo [`public/_routes.json`](../frontend/public/_routes.json) define
  quais rotas invocam a Function (`/predict`, `/interpret`, `/health`,
  `/metrics`, `/docs`, `/openapi.json`). Todo o restante é servido como estático.

## Componentes da interface

- **Formulário de 30 medições** — geradas a partir de
  [`src/features.ts`](../frontend/src/features.ts): 10 medições × 3 grupos
  (`Médias`, `Erro padrão`, `Piores valores`). O botão "Preencher exemplo
  acadêmico" usa um caso real do dataset.
- **Classificar** — chama `POST /predict` (não exige Turnstile).
- **Classificar e interpretar** — exige o desafio Turnstile e chama
  `POST /interpret`.
- **Painel de resultado** — mostra o diagnóstico, as barras de probabilidade e,
  na interpretação, o texto do LLM renderizado em Markdown (títulos, tabela de
  evidências, listas), além das evidências do modelo e dos insights acionáveis.

### Probabilidades em contexto clínico

As probabilidades são **truncadas (floor)**, nunca arredondadas para cima. Uma
probabilidade `0.9999999987` é exibida como `99,99%` — e não `100,00%` — para
não transmitir uma certeza que o modelo não tem. Probabilidades ínfimas porém
não nulas aparecem como `< 0,01%` em vez de `0,00%`. A barra de progresso usa
o mesmo valor truncado, com ponto decimal (CSS válido). Veja `formatPercentage`
e `fillWidth` em [`src/App.tsx`](../frontend/src/App.tsx).

### Renderização do Markdown

A explicação do LLM vem em Markdown (GitHub Flavored). É renderizada com
`react-markdown` + `remark-gfm`, que por padrão **não** interpreta HTML cru e
sanitiza URLs perigosas (`javascript:` etc.) — apropriado para conteúdo gerado
por LLM. Os estilos ficam sob o seletor `.markdown-body` em
[`src/styles.css`](../frontend/src/styles.css). Tabelas largas ganham scroll
horizontal via wrapper `.table-scroll`.

## Variáveis de ambiente

| Variável | Momento | Descrição |
| --- | --- | --- |
| `VITE_TURNSTILE_SITE_KEY` | build | Chave **pública** do widget Turnstile. Sem ela, a interpretação fica desabilitada na UI (a classificação simples continua). |

Para desenvolvimento e preview são usadas as chaves de **teste** oficiais do
Cloudflare (`1x00000000000000000000AA`), que sempre validam. Veja
[`frontend/.env.example`](../frontend/.env.example).

## Desenvolvimento local

```bash
cd frontend
npm ci
cp .env.example .env.local      # site key de teste do Turnstile
npm run dev                     # http://localhost:5173
```

> No `npm run dev`, o Vite serve apenas o estático/SPA. Para exercitar o proxy
> e o Service Binding de ponta a ponta, rode o Worker localmente
> (`uv run pywrangler dev` em `cloudflare/api`) ou use o ambiente publicado.

## Testes e build

```bash
cd frontend
npm test          # Vitest (unidade + interação)
npm run build     # tsc -b && vite build  →  dist/
```

A suíte cobre: a montagem do formulário com as 30 medições, o envio mínimo a
`/predict`, a validação de formulário incompleto, o caminho completo de
`/interpret` com Turnstile e o truncamento clínico das probabilidades.

## Deploy

O frontend é publicado no Cloudflare Pages. O passo a passo (CLI direto e
Cloudflare Builds conectado ao GitHub) está em
[cloudflare-deploy.md](cloudflare-deploy.md).
