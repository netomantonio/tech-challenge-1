"""Avaliacao da recuperacao: compara configuracoes do RAG num conjunto de perguntas.

As perguntas foram escritas antes do ajuste dos parametros, em linguagem de atendimento
(inclusive coloquial). Para cada uma anotamos os documentos considerados relevantes.
As perguntas fora do escopo devem terminar sem documento (abstencao).
O conjunto e pequeno e foi escrito pela propria equipe; o resultado mostra tendencia,
nao desempenho garantido em perguntas reais.
"""
import json

from fase5.rag import CONFIGS, PADRAO, Retriever, load_docs
from fase5.train import OUTPUT

PERGUNTAS = [
    ('Quais sinais na mama exigem encaminhamento urgente?', {'saude-sinais-alerta'}),
    ('Ela sentiu um caroço endurecido no seio que está crescendo', {'saude-sinais-alerta'}),
    ('Saiu secreção com sangue do mamilo', {'saude-sinais-alerta'}),
    ('Com que idade a mulher deve fazer mamografia de rotina?', {'saude-deteccao-precoce'}),
    ('O resultado do modelo confirma que é câncer?', {'saude-diagnostico-biopsia', 'saude-deteccao-precoce'}),
    ('Qual o próximo passo para confirmar o diagnóstico?', {'saude-diagnostico-biopsia'}),
    ('O que significa worst concave points nas medidas?', {'saude-dados-citologia'}),
    ('De onde vêm as medidas usadas pelo modelo?', {'saude-dados-citologia'}),
    ('Quanto tempo ela pode esperar para começar o tratamento pelo SUS?', {'saude-prazos-sus'}),
    ('Em quantos dias os exames devem sair quando há suspeita de câncer?', {'saude-prazos-sus'}),
    ('Histórico familiar de câncer de mama aumenta o risco?', {'saude-sinais-alerta'}),
    ('Tenho medo do meu companheiro, ele me ameaça', {'seguranca-maria-da-penha', 'seguranca-medidas-protetivas',
                                                      'seguranca-avaliacao-risco'}),
    ('Meu marido controla meu dinheiro e me humilha', {'seguranca-maria-da-penha'}),
    ('Ele me empurrou e bateu em mim ontem', {'seguranca-maria-da-penha'}),
    ('Como ela pode pedir medida protetiva contra o ex?', {'seguranca-medidas-protetivas'}),
    ('Existe formulário para avaliar o risco de violência?', {'seguranca-avaliacao-risco'}),
    ('O serviço de saúde precisa notificar casos de violência?', {'seguranca-notificacao'}),
    ('Para qual número ela pode ligar para pedir ajuda?', {'seguranca-ligue180'}),
    ('Ela não tem para onde ir com os filhos se sair de casa', {'seguranca-acolhimento'}),
    ('Ela quer conversar sozinha, sem o acompanhante na sala', {'seguranca-privacidade'}),
    ('Prefere ser atendida em sala reservada por vergonha', {'seguranca-privacidade'}),
]
FORA_DO_ESCOPO = [
    'Gestante com pressão alta',
    'Qual a dose de insulina para diabetes?',
    'Receita de bolo de cenoura',
    'Previsão do tempo para amanhã',
    'Como renovar a carteira de motorista?',
]


def evaluate(config, docs):
    retriever = Retriever(docs, config)
    hits, precision, rr = [], [], []
    for question, relevant in PERGUNTAS:
        ids = [d['id'] for d in retriever.search(question)]
        hits.append(any(i in relevant for i in ids))
        precision.append(sum(i in relevant for i in ids) / len(ids) if ids else 0)
        rank = next((n for n, i in enumerate(ids, 1) if i in relevant), None)
        rr.append(1 / rank if rank else 0)
    abst = [not retriever.search(q) for q in FORA_DO_ESCOPO]
    returned = [len(retriever.search(q)) for q, _ in PERGUNTAS]
    return {'configuracao': config.nome, 'acerto_top3': sum(hits) / len(hits),
            'mrr': sum(rr) / len(rr), 'precisao_media': sum(precision) / len(precision),
            'abstencao_fora_escopo': sum(abst) / len(abst),
            'documentos_por_pergunta': sum(returned) / len(returned),
            'falhas': [q for (q, _), h in zip(PERGUNTAS, hits) if not h]}


def main():
    docs = load_docs()
    results = [evaluate(c, docs) for c in CONFIGS]
    report = {'perguntas_no_escopo': len(PERGUNTAS), 'perguntas_fora_escopo': len(FORA_DO_ESCOPO),
              'documentos_na_base': len(docs), 'padrao': PADRAO.nome, 'resultados': results}
    (OUTPUT / 'avaliacao_rag.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    for r in results:
        print(f"{r['configuracao']:<45} top3={r['acerto_top3']:.2f} mrr={r['mrr']:.2f} "
              f"prec={r['precisao_media']:.2f} abst={r['abstencao_fora_escopo']:.2f} "
              f"docs={r['documentos_por_pergunta']:.1f} falhas={r['falhas']}")


if __name__ == '__main__':
    main()
