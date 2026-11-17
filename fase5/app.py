"""API e interface web local da Guardiã AI.

    python -m fase5.app      # interface em http://127.0.0.1:8050, API em /docs

Mesmo padrão das Fases 3 e 4: FastAPI servindo uma página em HTML/JS puro de
fase5/web/, com toda a lógica no backend.
"""
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from fase5.rag import load_docs
from fase5.service import DEFAULT_MODEL, load_graph, record_review
from fase5.train import OUTPUT

WEB = Path(__file__).with_name('web')
AUDIT = OUTPUT / 'auditoria.jsonl'


class Atendimento(BaseModel):
    model_config = ConfigDict(extra='forbid')
    features: dict[str, float]
    relato: str = Field(default='', max_length=6000)
    pergunta: str = Field(default='Como apoiar o atendimento?', min_length=3, max_length=1000)


class Revisao(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(min_length=36, max_length=36)
    decisao: Literal['concordo', 'discordo', 'inconclusivo']


@asynccontextmanager
async def lifespan(app):
    app.state.graph = load_graph()
    yield


app = FastAPI(title='Guardiã AI — Fase 5', lifespan=lifespan,
              description='Dados públicos ou sintéticos. Apoio ao profissional; revisão humana obrigatória. Não inserir identificadores pessoais. Modo demo não utiliza LLM.')
app.mount('/static', StaticFiles(directory=WEB), name='static')


@app.get('/', include_in_schema=False)
def index():
    # Sem cache: a pagina referencia os estaticos com versao (?v=) e precisa vir sempre atualizada.
    return FileResponse(WEB / 'index.html', headers={'Cache-Control': 'no-store'})


@app.get('/health')
def health():
    return {'status': 'ok', 'revisao_humana_obrigatoria': True}


@app.get('/info')
def info():
    backend = os.environ.get('FASE5_LLM_BACKEND', 'demo')
    return {'backend': backend,
            'modelo_llm': (os.environ.get('FASE5_LLM_MODEL') or DEFAULT_MODEL) if backend == 'groq' else None,
            'base_conhecimento': [{k: d[k] for k in ('id', 'contexto', 'titulo', 'url')} for d in load_docs()]}


@app.get('/exemplos')
def exemplos():
    path = OUTPUT / 'exemplos.json'
    if not path.exists():
        raise HTTPException(status_code=503, detail='Execute python -m fase5.train para gerar os exemplos.')
    return json.loads(path.read_text(encoding='utf-8'))


@app.post('/analisar')
def analisar(entrada: Atendimento):
    try:
        return app.state.graph.invoke({'entrada': entrada.model_dump()})['resultado']
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (RuntimeError, httpx.HTTPError):
        logging.getLogger(__name__).warning('Falha de provedor ou processamento; sem conteúdo do atendimento nos logs.')
        raise HTTPException(status_code=503, detail='Análise indisponível. Confira a configuração do provedor; não houve fallback silencioso.')


@app.post('/revisao')
def revisao(entrada: Revisao):
    """Registra a decisão do profissional na auditoria. A IA não decide pelo profissional."""
    try:
        return record_review(AUDIT, entrada.id, entrada.decisao)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('fase5.app:app', host='127.0.0.1', port=8050)
