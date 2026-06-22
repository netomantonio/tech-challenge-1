# Implantação na Cloudflare pelo GitHub

Este projeto está preparado para publicar uma demonstração isolada usando
Cloudflare Pages, Python Workers, R2, Turnstile, Rate Limiting e Service
Bindings. Nenhuma publicação deve ser feita diretamente por uma máquina local.

## Arquitetura

```text
Navegador
   |
   v
Cloudflare Pages (React)
   |
   v
Pages Function -- Service Binding --> Python Worker -- HTTPS --> Groq
                                         |
                                         +-- Rate Limiting
                                         +-- Turnstile

Cloudflare Builds -- artefatos imutáveis por commit --> R2 privado
```

- O Pages entrega os arquivos estáticos e encaminha somente as rotas existentes
  da API pelo Service Binding `API`.
- O Worker executa o FastAPI pelo adaptador ASGI oficial.
- O modelo é carregado do manifesto `src/modelo_serving.json`, empacotado junto
  ao Worker. O R2 não participa do caminho crítico de inferência.
- O R2 guarda cópias imutáveis do modelo, relatórios e resultados sob
  `builds/<commit-sha>/`.
- Não há D1, contas de usuário, histórico de consultas nem armazenamento das
  features ou respostas clínicas.

## Fluxo de branches

A implementação vive em `feat/cloudflare-fullstack`, criada a partir de
`feature/fase2`. Os projetos descritos abaixo são ambientes isolados de preview.
Não faça merge, rebase, promoção ou alteração de outra branch como parte desta
configuração. A promoção posterior é uma decisão manual do responsável pelo
repositório.

## Configurar o Worker no Cloudflare Builds

No painel da Cloudflare:

1. Acesse **Workers & Pages**, selecione **Create application** e importe o
   repositório GitHub `netomantonio/tech-challenge-1`.
2. Autorize o GitHub App da Cloudflare para esse repositório.
3. Use o nome `cancer-mama-api-preview` e a branch
   `feat/cloudflare-fullstack`.
4. Deixe o diretório raiz como a raiz do repositório.
5. Configure o comando de build:

   ```bash
   npm ci && npm run test:worker
   ```

6. Configure o comando de deploy:

   ```bash
   npm run deploy:worker
   ```

7. Cadastre estes secrets de runtime no Worker:

   - `GROQ_API_KEY`: chave da API Groq;
   - `TURNSTILE_SECRET_KEY`: chave secreta do widget Turnstile.

8. Configure os caminhos observados pelo build para incluir:

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

O comando de deploy possui uma trava que exige `WORKERS_CI=1`. Assim, ele falha
quando executado fora do Cloudflare Builds. Durante a primeira execução, o
pipeline cria idempotentemente o bucket `cancer-mama-artifacts-preview`, envia
os artefatos e publica o Worker. O token automático do Cloudflare Builds precisa
ter permissão de escrita em Workers e R2.

## Configurar o Pages

Crie um segundo projeto, desta vez no Cloudflare Pages:

1. Importe o mesmo repositório e use o nome
   `cancer-mama-web-preview`.
2. Selecione `feat/cloudflare-fullstack` para o ambiente isolado.
3. Use `frontend` como diretório raiz.
4. Configure o comando de build:

   ```bash
   npm ci && npm test && npm run build
   ```

5. Configure `dist` como diretório de saída.
6. Cadastre `VITE_TURNSTILE_SITE_KEY` como variável de build com a chave
   pública do Turnstile.
7. Confirme no ambiente de preview o Service Binding `API` apontando para
   `cancer-mama-api-preview`. Ele também está declarado em
   `frontend/wrangler.jsonc`.
8. Limite os caminhos observados a `frontend/*`.

No Turnstile, autorize os hostnames do Pages e do Worker gerados pela
Cloudflare. O endpoint `/interpret` exige o cabeçalho `X-Turnstile-Token`; o
endpoint `/predict` não exige desafio.

## Validação local sem publicação

Os comandos abaixo não publicam recursos:

```bash
npm ci
npm run test:worker

cd frontend
npm ci
npm test
npm run build
```

Para testar o Worker localmente:

```bash
cp cloudflare/api/.dev.vars.example cloudflare/api/.dev.vars
cp frontend/.env.example frontend/.env.local
cd cloudflare/api
uv run pywrangler dev
```

Os exemplos usam as chaves oficiais de teste do Turnstile. Preencha apenas a
chave Groq local se quiser validar uma interpretação real. Os arquivos copiados
são ignorados pelo Git.

Não execute `npm run deploy:worker` localmente. Além de não ser necessário, o
script recusa a execução quando não está no Cloudflare Builds.

## Critérios de verificação do preview

- O GitHub deve mostrar checks verdes do Worker e do Pages.
- `GET /health` deve retornar `ready`.
- `POST /predict` deve responder com a classificação e probabilidades.
- `POST /interpret` sem Turnstile deve retornar `403`; sem secret configurado,
  deve retornar `503`.
- A 31ª chamada de `/predict` e a 6ª de `/interpret` dentro de um minuto, na
  mesma chave de limitação, devem retornar `429`.
- O bucket R2 deve conter `builds/<commit-sha>/src/modelo_serving.json` e os
  demais artefatos versionados.
