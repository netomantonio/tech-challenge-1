"""Recuperação lexical sobre a base de conhecimento da Guardiã AI.

A primeira versão usava TF-IDF de palavras sem stopwords e limiar 0,01. Na pratica
quase toda pergunta trazia a base inteira, porque palavras como "com" e "de" bastavam
para gerar similaridade. Esta versão remove stopwords em português, indexa palavras-chave
cadastradas junto com cada documento (vocabulário coloquial dos relatos) e reduz cada
termo a um prefixo para aproximar flexões (ameaça, ameaçou). As configurações abaixo
são comparadas em fase5/rag_eval.py; a padrão foi escolhida por essa avaliação.
"""
import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

KNOWLEDGE = Path(__file__).with_name('knowledge.json')

# Lista curta escrita a mao (sem acento, pois o texto e normalizado antes).
STOPWORDS = frozenset('''
a ao aos aquela aquele aqueles as ate com como da das de dela dele deles depois do dos e ela ele eles em
entre era essa esse esta este estou eu foi for ha isso isto ja la lhe mais mas me mesmo meu minha muito
na nas nem no nos nossa o os ou para pela pelo per pode por porque qual quando que quem se sem ser seu
sua so sobre sua tambem te tem tenho ter teu tu um uma umas uns voce voces vou esta estao sao era sido
estar fazer faz fiz ficar fico pra pro num numa dessa desse nesta neste nessa nesse ai aqui ali entao
hoje ontem agora sempre nunca ainda bem bom boa muito pouco cada outro outra todo toda todos todas
qualquer algum alguma alguem nada tudo coisa vez vezes dia dias ano anos
como pode posso devo deve preciso precisa gostaria quero queria saber informacao informacoes duvida
'''.split())


def normalize(text):
    text = unicodedata.normalize('NFKD', text.lower())
    return ''.join(c for c in text if not unicodedata.combining(c))


@dataclass(frozen=True)
class Config:
    nome: str
    stopwords: bool = True
    palavras_chave: bool = True
    prefixo: int = 0          # 0 = sem truncamento
    ngram: tuple = (1, 1)
    limiar: float = .08       # similaridade minima absoluta
    relativo: float = .35     # descarta documentos abaixo de 35% do melhor score
    k: int = 3

    def tokens(self, text):
        words = re.findall(r'[a-z0-9]+', normalize(text))
        if self.stopwords:
            words = [w for w in words if w not in STOPWORDS and len(w) > 1]
        if self.prefixo:
            words = [w[:self.prefixo] for w in words]
        return words


BASELINE = Config('v1 (palavras, sem stopwords)', stopwords=False, palavras_chave=False,
                  ngram=(1, 2), limiar=.01, relativo=0)
CONFIGS = [
    BASELINE,
    Config('stopwords', palavras_chave=False),
    Config('stopwords + palavras-chave'),
    Config('stopwords + palavras-chave + prefixo 5', prefixo=5),
    Config('stopwords + palavras-chave + prefixo 6', prefixo=6),
]
PADRAO = CONFIGS[4]


def load_docs(path=None):
    return json.loads(Path(path or KNOWLEDGE).read_text(encoding='utf-8'))


class Retriever:
    def __init__(self, docs, config=PADRAO):
        self.docs, self.config = docs, config
        corpus = [d['titulo'] + ' ' + d['texto'] + (' ' + d.get('palavras_chave', '') if config.palavras_chave else '')
                  for d in docs]
        self.vectorizer = TfidfVectorizer(analyzer=lambda t: self._analyze(t), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(corpus)

    def _analyze(self, text):
        words = self.config.tokens(text)
        low, high = self.config.ngram
        return [' '.join(words[i:i + n]) for n in range(low, high + 1) for i in range(len(words) - n + 1)]

    def scores(self, query):
        return (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()

    def search(self, query):
        scores = self.scores(query)
        best = scores.max() if len(scores) else 0
        found = []
        for i in np.argsort(-scores)[:self.config.k]:
            if scores[i] >= self.config.limiar and scores[i] >= self.config.relativo * best:
                found.append({**self.docs[i], 'score': round(float(scores[i]), 4)})
        return found


@lru_cache(maxsize=4)
def _default(path):
    return Retriever(load_docs(path))


def retrieve(query, path=None):
    """Devolve ate tres documentos com score; lista vazia quando nada e relevante."""
    return _default(str(path or KNOWLEDGE)).search(query)
