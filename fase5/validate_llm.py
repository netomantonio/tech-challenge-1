"""Validação opcional real, com exemplo público e relato sintético apenas.

Roda o caso limítrofe com relato de segurança no backend Groq e guarda a resposta
completa, para que o relatório mostre uma saída real da LLM e não um texto escrito à mão.
"""
import json
import os
from datetime import datetime, timezone

from fase5.service import load_env, load_graph
from fase5.train import OUTPUT


def main():
    load_env()
    os.environ['FASE5_LLM_BACKEND'] = 'groq'
    exemplos = json.loads((OUTPUT / 'exemplos.json').read_text(encoding='utf-8'))
    caso = next(e for e in exemplos if 'limítrofe' in e['nome'])
    result = load_graph().invoke({'entrada': caso['entrada']})['resultado']
    evidence = {'executado_em': datetime.now(timezone.utc).isoformat(),
                'caso': caso['nome'], 'pergunta': caso['entrada']['pergunta'], 'relato': caso['entrada']['relato'],
                'provedor': result['provedor_efetivo'],
                'probabilidade_maligno': round(result['ml']['probabilidade_maligno'], 4),
                'fontes': [d['id'] for d in result['fontes']],
                'fontes_citadas': result['fontes_citadas'], 'citacoes_invalidas': result['citacoes_invalidas'],
                'auditoria': result['auditoria_registrada'],
                'resposta_nao_vazia': bool(result['explicacao']), 'resposta': result['explicacao']}
    (OUTPUT / 'validacao_llm.json').write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in evidence.items() if k != 'resposta'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
