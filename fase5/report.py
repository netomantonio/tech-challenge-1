"""Gera o relatório técnico a partir dos resultados realmente executados.

Lê metricas.json, exploracao.json, avaliacao_rag.json, comparacao_llm.json e
validacao_*.json; roda os testes da Fase 5 para informar o resultado atual.
"""
import json
import re
import unittest
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

from fase5.rag import load_docs
from fase5.train import OUTPUT

ROOT = Path(__file__).resolve().parents[1]


def br(value, digits=4):
    return f'{value:.{digits}f}'.replace('.', ',')


def _load(name):
    path = OUTPUT / name
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def _run_tests():
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_fase5.py')
    result = unittest.TestResult()
    suite.run(result)
    return result.testsRun, len(result.failures) + len(result.errors)


def generate():
    metrics, eda = _load('metricas.json'), _load('exploracao.json')
    rag, llm = _load('avaliacao_rag.json'), _load('comparacao_llm.json')
    val_llm, val_docker = _load('validacao_llm.json'), _load('validacao_docker.json')
    docs = load_docs()
    total_tests, failed = _run_tests()

    target = ROOT / 'output/pdf/relatorio_tecnico_fase5.pdf'
    target.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles['BodyText'].leading = 15
    small = ParagraphStyle('small', parent=styles['BodyText'], fontSize=8.5, leading=11)
    code = ParagraphStyle('code', parent=styles['Code'], fontSize=7, leading=8.6)
    story = []

    def text(value, style='BodyText'):
        story.append(Paragraph(escape(value), styles[style] if isinstance(style, str) else style))
        story.append(Spacer(1, 7))

    def table(rows, widths):
        t = Table(rows, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dce9ef')),
                               ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                               ('FONTSIZE', (0, 0), (-1, -1), 8),
                               ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                               ('LINEBELOW', (0, 0), (-1, -1), .25, colors.HexColor('#c9c8c3')),
                               ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
        story.append(t)
        story.append(Spacer(1, 8))

    def figure(name, width):
        path = OUTPUT / 'figuras' / name
        if path.exists():
            img = Image(str(path))
            img.drawWidth, img.drawHeight = width, width * img.imageHeight / img.imageWidth
            story.append(img)
            story.append(Spacer(1, 6))

    text('Guardiã AI', 'Title')
    text('Tech Challenge - Fase 5 | Relatório técnico | 11/11/2026', 'Heading2')

    text('1. Problema e escopo', 'Heading1')
    text('Escolhemos como cenário principal o acompanhamento de mulheres em investigação de câncer de mama, '
         'com espaço no mesmo atendimento para um relato relacionado à segurança. O fluxo é: entrada dos dados, '
         'classificação por Machine Learning a partir de medidas citológicas, consulta a uma base de documentos '
         'públicos (RAG), explicação por LLM e apresentação ao profissional, que registra a própria decisão. '
         'O sistema não diagnostica, não prescreve e não decide medidas de proteção. O relato nunca entra no '
         'classificador e nenhum escore de violência é calculado: o relato serve apenas para buscar orientações.')

    text('2. Dados, exploração e preparo', 'Heading1')
    r = eda['resumo']
    text(f"Usamos o Wisconsin Diagnostic Breast Cancer (scikit-learn/UCI): {r['amostras']} amostras, "
         f"{r['variaveis']} variáveis, {r['classes']['maligno_0']} malignas e {r['classes']['benigno_1']} benignas "
         f"({br(r['proporcao_maligno'] * 100, 1)}% malignas). Não há valores ausentes nem linhas duplicadas. "
         'As medidas descrevem núcleos celulares de imagens de punção aspirativa (PAAF): são dados de exame, '
         'não sintomas. A exploração completa foi feita no notebook da Fase 1 (notebooks/01_cancer_mama.ipynb); '
         'aqui repetimos, em fase5/explore.py, apenas o que sustenta as decisões de preparo.')
    big, tiny = eda['maiores_escalas'][0], eda['menores_escalas'][-1]
    text(f"Escalas: {big['variavel']} vai até {big['max']:.0f}, enquanto {tiny['variavel']} não passa de "
         f"{br(tiny['max'], 3)}. Sem padronização, a regressão logística é dominada pelas variáveis de área. "
         f"Correlação: {eda['pares_correlacao_acima_0_9']} pares têm |r| > 0,9; raio, perímetro e área chegam "
         'a r = 0,998, porque medem quase a mesma coisa. Isso afeta a explicabilidade (seção 4). '
         f"Outliers pela regra do IQR aparecem em {eda['outliers_iqr']['linhas_com_algum_outlier']} linhas, "
         f"das quais {eda['outliers_iqr']['proporcao_maligno_nessas_linhas'] * 100:.0f}% são malignas: os valores "
         'extremos são justamente o sinal que queremos detectar, então não foram removidos.')
    figure('distribuicao_top6.png', 15.5 * cm)
    text('Testamos as decisões de preparo por validação cruzada estratificada (5 folds) só no treino, com '
         'recall maligno da regressão logística:', small)
    table([['Variante de preparo', 'Recall maligno CV', 'Desvio']] +
          [[p['variante'], br(p['recall_maligno_cv']), br(p['desvio'])] for p in eda['preparo_testado_cv_treino']],
          [10 * cm, 3.2 * cm, 2.2 * cm])
    text('A padronização é necessária (0,906 sem ela). O log1p deu 0,9706 contra 0,9647: a diferença '
         'equivale a um caso maligno entre cerca de 170 do treino, dentro do desvio da variante atual. Mantivemos '
         'o StandardScaler, que é o pipeline já validado nas fases anteriores. Remover as variáveis muito '
         'correlacionadas piorou o recall e aumentou o desvio, então mantivemos as 30. A divisão é estratificada '
         '(455 treino e 114 teste, seed 42) e o scaler fica dentro do pipeline, ajustado em cada fold.')

    text('3. Treinamento e avaliação', 'Heading1')
    text('Comparamos regressão logística (max_iter=3000) e Random Forest (200 árvores). A escolha usa o maior '
         'recall maligno em validação cruzada no treino, com F1 como desempate; o teste não participa da escolha. '
         'Priorizamos recall porque o falso negativo (maligno classificado como benigno) pode atrasar a '
         'investigação; o falso positivo também tem custo (ansiedade, exames adicionais), por isso precision '
         'e F1 continuam na tabela. Accuracy sozinha esconderia esse desequilíbrio de custos.')
    rows = [['Modelo', 'Recall CV', 'Accuracy', 'Precision*', 'Recall*', 'F1*', 'FN*']]
    for name, res in metrics['models'].items():
        rows.append(['Regressão logística' if name == 'regressao_logistica' else 'Random Forest',
                     br(res['cv_recall_maligno']),
                     *[br(res[k]) for k in ['accuracy', 'precision_maligno', 'recall_maligno', 'f1_maligno']],
                     str(res['falsos_negativos_malignos'])])
    table(rows, [3.6 * cm, 2 * cm, 2 * cm, 2 * cm, 2 * cm, 2 * cm, 1.2 * cm])
    text('* Classe maligna (0), no teste separado. Probabilidades não calibradas.', small)
    lr = metrics['models']['regressao_logistica']
    text(f"Modelo selecionado: {metrics['selected']}. Matriz de confusão no teste (linhas real, colunas predito; "
         f"ordem maligno, benigno): {lr['matriz_confusao_ordem_0_1']}. O único falso negativo do teste foi "
         'incluído como caso de demonstração na interface, para discutir o erro mais grave do modelo.')

    text('4. Explicabilidade', 'Heading1')
    imp = sorted(metrics['importance_global_recall_drop'].items(), key=lambda kv: -kv[1])[:6]
    text('Global: importância por permutação, medida como queda do recall maligno no teste (10 repetições).')
    table([['Variável', 'Queda média de recall']] + [[k, br(v)] for k, v in imp], [7 * cm, 4 * cm])
    text('O ranking não coincide com a separação individual das classes da seção 2 (worst concave points e '
         'worst perimeter lideram pelo d de Cohen). A explicação está na correlação: quando permutamos '
         'worst radius, o modelo ainda tem perímetro e área, que carregam a mesma informação; variáveis mais '
         'independentes, como textura e suavidade, aparecem com mais peso. As quedas são pequenas e com apenas '
         '42 casos malignos no teste cada caso vale 0,024 de recall, então o ranking é instável.')
    text('Local: para cada atendimento, substituímos uma medida por vez pela mediana do treino e registramos a '
         'variação na probabilidade de maligno; a interface mostra as cinco maiores, com valor e mediana. '
         'As contribuições não somam a predição e não são causais. Em casos com probabilidade saturada '
         '(próxima de 0 ou 1) trocar uma medida isolada quase não muda o resultado e as contribuições ficam '
         'perto de zero; por isso os exemplos da demonstração incluem um caso limítrofe '
         f"(probabilidade {br([e for e in _load('exemplos.json') if 'limítrofe' in e['nome']][0]['probabilidade_maligno'], 2)}).")

    text('5. RAG: base de conhecimento e recuperação', 'Heading1')
    ctx = {c: sum(d['contexto'] == c for d in docs) for c in ('saude', 'seguranca')}
    text(f"A base (fase5/knowledge.json) tem {len(docs)} sínteses didáticas escritas pela equipe a partir de "
         f"páginas públicas oficiais ({ctx['saude']} de saúde e {ctx['seguranca']} de segurança): INCA, Ministério "
         'da Saúde, UCI, Lei Maria da Penha, Leis 10.778/2003, 12.732/2012, 14.149/2021 e 14.847/2024, '
         'Ligue 180 e o serviço de acolhimento do MDS. Cada documento tem ID, URL e data de consulta. São resumos, '
         'não os protocolos completos das instituições.')
    text('A primeira versão usava TF-IDF de palavras, sem stopwords, com limiar 0,01 e apenas 3 documentos: '
         'quase toda pergunta trazia a base inteira, porque palavras como "com" e "de" bastavam para gerar '
         'similaridade. Criamos um conjunto de avaliação com '
         f"{rag['perguntas_no_escopo']} perguntas de atendimento (cada uma com os documentos relevantes anotados) "
         f"e {rag['perguntas_fora_escopo']} perguntas fora do escopo, e comparamos as variantes:")
    table([['Configuração', 'Acerto top-3', 'MRR', 'Precisão', 'Abstenção fora', 'Docs/pergunta']] +
          [[x['configuracao'], br(x['acerto_top3'], 2), br(x['mrr'], 2), br(x['precisao_media'], 2),
            br(x['abstencao_fora_escopo'], 2), br(x['documentos_por_pergunta'], 1)] for x in rag['resultados']],
          [6.2 * cm, 2 * cm, 1.4 * cm, 1.7 * cm, 2.3 * cm, 2.2 * cm])
    text('Só remover stopwords piorou o acerto: relatos usam palavras do dia a dia ("medo", "bateu", "marido") '
         'que não aparecem no texto formal das leis. A melhora veio de cadastrar palavras-chave coloquiais '
         'em cada documento e de reduzir cada termo a um prefixo de 6 letras (ameaça e ameaçou viram "ameaca"). '
         'Adotamos a última linha. Além do limiar absoluto (0,08), descartamos documentos abaixo de 35% do '
         'melhor score, para não completar a resposta com documentos fracos.')
    text('Limitações observadas: o prefixo aumenta casamentos acidentais. "Gestante com pressão alta" recupera '
         'o documento de sinais de alerta por causa de "alta densidade mamária", e "previsão do tempo" casa '
         'com "previstas". Na pergunta "o que significam as medidas que mais influenciaram", a palavra "medidas" '
         'traz também o documento de medidas protetivas. Não ajustamos a base para esconder esses casos: eles '
         'mostram o limite de uma busca lexical, que uma busca semântica (embeddings) reduziria. O conjunto de '
         'avaliação foi escrito pela própria equipe e é pequeno, então os números indicam tendência.')
    text('Indicação da fonte usada: a API devolve as fontes recuperadas (ID, título, trecho, URL, data, score) e, '
         'separadamente, as fontes que a LLM declarou ter usado, extraídas das citações [id] da resposta. Citação '
         'de um ID que não foi recuperado aparece como "citação inválida" na interface e na auditoria.')

    text('6. Integração com LLM', 'Heading1')
    text('LangGraph organiza o fluxo em quatro nós: modelo, recuperação, explicação e auditoria. A LLM recebe o '
         'resultado do ML (com o rótulo por extenso), as contribuições locais, os documentos recuperados e a '
         'pergunta. O prompt trata entrada e documentos como dados, proíbe diagnóstico, prescrição e decisão de '
         'segurança, exige citação [id] e pede para declarar informação insuficiente quando as fontes não '
         'cobrem a pergunta. O backend demo não chama LLM e se identifica na resposta; falha do provedor gera '
         'erro 503, sem fallback silencioso.')
    text('Escolha do modelo: o .env.example sugeria llama-3.1-8b-instant, que não estava mais disponível no Groq '
         'em novembro de 2026. Rodamos os quatro casos de demonstração e uma pergunta fora do escopo em cada '
         'modelo de chat disponível (fase5/compare_llm.py), no mesmo fluxo da aplicação:')
    rows = [['Modelo', 'Válidas', 'Citou fonte', 'Cit. inválidas', 'Absteve-se', 'Classe invertida', 'Latência']]
    for name, m in llm['modelos'].items():
        s = m['resumo']
        rows.append([name, s['respostas_validas'], s['citou_fonte_quando_havia'], str(s['citacoes_invalidas']),
                     s['reconheceu_insuficiencia'], str(s['classe_invertida']), br(s['latencia_media_s'], 1) + ' s'])
    table(rows, [3.7 * cm, 1.4 * cm, 1.8 * cm, 2 * cm, 1.9 * cm, 2.3 * cm, 1.7 * cm])
    text('As verificações são automáticas e grosseiras (expressões regulares); por isso lemos as respostas, que '
         'estão salvas em resultados/fase5/comparacao_llm.json. Na leitura, o gpt-oss-120b também recusou a '
         'pergunta sobre insulina ("não é possível responder à questão com base nos documentos disponíveis"), o que a expressão regular '
         'não captou. gpt-oss-20b e gpt-oss-120b ficaram equivalentes nesse conjunto; mantivemos o 20b como '
         'padrão pela menor latência, e o 120b fica como alternativa por variável de ambiente. O qwen3.8-27b '
         'levou mais de 30 s por resposta, o que inviabiliza o uso durante o atendimento.')
    text('A comparação revelou três problemas que corrigimos: (1) na primeira rodada, os dois gpt-oss leram '
         '"classe 0" como benigno, quando 0 é maligno; passamos a enviar o rótulo por extenso e a convenção no '
         'prompt, e a inversão não se repetiu; (2) o gpt-oss-20b gastava o limite padrão de 2048 tokens '
         'raciocinando e devolvia texto vazio ou cortado, sem aviso; aumentamos o limite, usamos esforço de '
         'raciocínio baixo e passamos a tratar resposta truncada como erro; (3) os gpt-oss citam '
         'o ID entre colchetes lenticulares (U+3010 e U+3011) e com hífen não separável, que o extrator de citações não reconhecia.')
    if val_llm:
        text(f"Exemplo real ({val_llm['provedor']}, caso \"{val_llm['caso']}\", probabilidade de maligno "
             f"{br(val_llm['probabilidade_maligno'], 2)}; fontes recuperadas {', '.join(val_llm['fontes'])}; citadas "
             f"{', '.join(val_llm['fontes_citadas']) or 'nenhuma'}). Trecho inicial da resposta:", small)
        excerpt = re.sub('[‐-―]', '-', re.sub('[  ]', ' ', val_llm['resposta']))
        excerpt = excerpt.split('**Acolhimento')[0].strip()
        story.append(Preformatted(excerpt.replace('**', ''), code, maxLineLength=118))
        story.append(Spacer(1, 6))
        text('Na mesma resposta, a seção de acolhimento traz orientações gerais (por exemplo, instrumentos de '
             'triagem e serviços de apoio) que não estão nas duas fontes recuperadas, embora a própria resposta '
             'registre isso em Limitações. Citar a fonte não garante que cada frase venha dela: a revisão humana '
             'continua obrigatória.', small)

    text('7. Aplicação, responsabilidade e auditoria', 'Heading1')
    text('A aplicação é uma interface web (FastAPI servindo HTML/JS em fase5/web, porta 8050) com a API em /docs. '
         'O profissional escolhe um caso de exemplo, edita a pergunta e o relato sintético e vê a estimativa, as '
         'contribuições, a explicação com citações clicáveis e as fontes com trecho e link oficial. O aviso de '
         'apoio ao profissional fica fixo no topo. No fim, o profissional registra "concordo", "discordo" ou '
         '"inconclusivo" (POST /revisao); a decisão vai para a auditoria com o ID da análise.')
    text('A auditoria (JSONL) registra ID, data UTC, hash da entrada e da resposta, modelo e versão, resultado, '
         'contribuições, fontes recuperadas e citadas, citações inválidas, provedor e versão do prompt. Não '
         'guarda o relato nem as medidas brutas; os hashes permitem conferir um registro apresentado depois, '
         'mas não reconstruir o atendimento, e não são anonimização forte.')
    text(f"Verificações: {total_tests - failed} de {total_tests} testes automatizados passaram na geração deste "
         'relatório (treino e holdout, exemplos com caso limítrofe, validação de entrada, relevância e abstenção do '
         'RAG, extração de citações, privacidade da auditoria, contexto enviado à LLM, ausência de fallback e API '
         'completa com revisão humana). Teste com LLM simulada não comprova chamada externa; a chamada real está '
         'na seção 6.' + (f" Docker: imagem construída em {val_docker['data_local']}; dentro do container, "
                          + ('GET /, ' if val_docker.get('get_interface') == 200 else '')
                          + 'GET /health e POST /analisar responderam 200 com auditoria, em modo demo.'
                          if val_docker else ''))

    text('8. Limitações', 'Heading1')
    text('Dataset pequeno e antigo, sem dados demográficos: não há como avaliar equidade por grupo nem validação '
         'externa. Probabilidades não calibradas. A explicação local não é causal e fica perto de zero em casos '
         'saturados. A base de conhecimento tem 12 resumos, não protocolos completos, e a busca lexical erra em '
         'sinônimos e casamentos acidentais. A LLM pode acrescentar orientações gerais que não estão nas fontes, e '
         'o prompt não garante imunidade a prompt injection. No modo Groq, medidas, pergunta e relato são enviados '
         'ao provedor; por isso só usamos dados públicos ou sintéticos. A API não tem autenticação e não deve ser '
         'exposta como serviço. Nada foi validado por profissionais de saúde ou da rede de proteção.')

    story.append(PageBreak())
    text('9. Fontes', 'Heading1')
    text('Dataset: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html', small)
    seen = set()
    for doc in docs:
        if doc['url'] not in seen:
            seen.add(doc['url'])
            text(f"{doc['titulo']} - {doc['url']} (consulta: {doc['consultado_em']}).", small)

    def footer(canvas, doc):
        canvas.setFont('Helvetica', 8)
        canvas.drawString(2 * cm, 1.2 * cm, 'Guardiã AI | Protótipo acadêmico | Revisão humana obrigatória')
        canvas.drawRightString(19 * cm, 1.2 * cm, str(doc.page))

    SimpleDocTemplate(str(target), pagesize=(21 * cm, 29.7 * cm), rightMargin=2 * cm,
                      leftMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=2 * cm).build(
                          story, onFirstPage=footer, onLaterPages=footer)
    print(target, f'testes: {total_tests - failed}/{total_tests}')


if __name__ == '__main__':
    generate()
