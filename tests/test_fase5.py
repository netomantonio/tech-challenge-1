import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
from fastapi.testclient import TestClient

from fase5.train import train
from fase5.rag import retrieve
from fase5.service import build_graph, citations, predict, explain
from fase5.app import app


class GuardiaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.metrics = train(cls.root)
        cls.artifact = joblib.load(cls.root / 'modelo.joblib')
        cls.examples = json.loads((cls.root / 'exemplos.json').read_text(encoding='utf-8'))
        cls.entry = cls.examples[0]['entrada']

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_models_and_holdout(self):
        self.assertEqual(len(self.metrics['models']), 2)
        self.assertEqual(self.metrics['train'] + self.metrics['test'], 569)
        for result in self.metrics['models'].values():
            self.assertGreater(result['recall_maligno'], .8)
            self.assertEqual(sum(map(sum, result['matriz_confusao_ordem_0_1'])), 114)

    def test_examples_cover_boundary(self):
        probs = [e['probabilidade_maligno'] for e in self.examples]
        self.assertTrue(any(.2 < p < .8 for p in probs), probs)
        self.assertEqual({e['classe_real'] for e in self.examples}, {0, 1})

    def test_invalid_features(self):
        for values in ({}, {**self.entry['features'], 'extra': 2},
                       {**self.entry['features'], 'mean radius': float('nan')},
                       {**self.entry['features'], 'mean radius': -1}):
            with self.assertRaises(ValueError):
                predict(values, self.artifact)

    def test_retrieval_relevance_and_abstention(self):
        ids = lambda q: [d['id'] for d in retrieve(q)]
        # Casos que falhavam na primeira versão do RAG.
        self.assertIn('seguranca-maria-da-penha', ids('Tenho medo do meu companheiro, ele me ameaça'))
        self.assertIn('seguranca-ligue180', ids('Para qual número ela pode ligar para pedir ajuda?'))
        self.assertEqual(ids('Receita de bolo de cenoura'), [])
        self.assertEqual(ids('zzzyyyxxx'), [])
        self.assertLess(len(ids('Com que idade fazer mamografia de rotina?')), 3)
        self.assertEqual(ids('Com que idade fazer mamografia de rotina?')[0], 'saude-deteccao-precoce')

    def test_citations_split_valid_and_invalid(self):
        sources = [{'id': 'saude-deteccao-precoce'}, {'id': 'seguranca-ligue180'}]
        text = 'A mamografia [saude-deteccao-precoce]. Ligue [seguranca-ligue180, lei-inventada].'
        self.assertEqual(citations(text, sources), (['saude-deteccao-precoce', 'seguranca-ligue180'], ['lei-inventada']))
        # Formatos observados nos modelos gpt-oss: colchete lenticular e hífen não separável.
        self.assertEqual(citations('Ver 【saude‑deteccao‑precoce】.', sources)[0], ['saude-deteccao-precoce'])

    def test_audit_privacy_and_real_graph(self):
        entry = {**self.entry, 'relato': 'SEGREDO TESTE violência privacidade'}
        path = self.root / 'audit.jsonl'
        result = build_graph(self.artifact, path).invoke({'entrada': entry})['resultado']
        self.assertTrue(result['auditoria_registrada'])
        self.assertEqual(result['provedor_efetivo'], 'demo-sem-llm')
        self.assertTrue(result['revisao_humana_obrigatoria'])
        self.assertEqual(len(result['ml']['contribuicoes']), 5)
        self.assertNotIn('SEGREDO', path.read_text(encoding='utf-8'))
        self.assertEqual(json.loads(path.read_text(encoding='utf-8').splitlines()[-1])['id'], result['id'])

    def test_llm_receives_retrieved_context(self):
        def fake(payload, backend):
            self.assertTrue(payload['fontes'])
            self.assertIn('ml', payload)
            return 'Explicação de teste [saude-deteccao-precoce] [fonte-falsa]', 'teste-simulado'
        result = build_graph(self.artifact, self.root / 'fake.jsonl', generator=fake).invoke(
            {'entrada': {**self.entry, 'pergunta': 'mamografia de rotina rastreamento'}})['resultado']
        self.assertEqual(result['provedor_efetivo'], 'teste-simulado')
        self.assertEqual(result['fontes_citadas'], ['saude-deteccao-precoce'])
        self.assertEqual(result['citacoes_invalidas'], ['fonte-falsa'])

    def test_no_silent_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                explain({}, 'groq')

    def test_api_complete_validation_and_review(self):
        audit = self.root / 'api.jsonl'
        graph = build_graph(self.artifact, audit)
        with patch('fase5.app.load_graph', return_value=graph), patch('fase5.app.AUDIT', audit), \
                TestClient(app) as client:
            self.assertEqual(client.get('/health').status_code, 200)
            self.assertIn('Guardiã AI', client.get('/').text)
            self.assertTrue(client.get('/info').json()['base_conhecimento'])
            response = client.post('/analisar', json=self.entry)
            self.assertEqual(response.status_code, 200, response.text)
            analysis = response.json()
            self.assertTrue(analysis['auditoria_registrada'])
            self.assertEqual(client.post('/analisar', json={'features': {}}).status_code, 422)
            self.assertEqual(client.post('/analisar', json={**self.entry, 'pergunta': 'x' * 1001}).status_code, 422)
            review = client.post('/revisao', json={'id': analysis['id'], 'decisao': 'concordo'})
            self.assertEqual(review.status_code, 200, review.text)
            self.assertEqual(json.loads(audit.read_text(encoding='utf-8').splitlines()[-1])['decisao'], 'concordo')
            unknown = client.post('/revisao', json={'id': '0' * 36, 'decisao': 'concordo'})
            self.assertEqual(unknown.status_code, 404)
            self.assertEqual(client.post('/revisao', json={'id': analysis['id'], 'decisao': 'talvez'}).status_code, 422)


if __name__ == '__main__':
    unittest.main()
