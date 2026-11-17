"""Treino reproduzível: seleção por CV no treino, teste separado e explicabilidade."""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import cross_validate, train_test_split, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUTPUT = Path(__file__).resolve().parents[1] / 'resultados' / 'fase5'


def train(output=OUTPUT):
    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=.2, random_state=42, stratify=data.target)
    candidates = {
        'regressao_logistica': make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=42)),
        'random_forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1),
    }
    scores, results = {}, {}
    scorer = make_scorer(recall_score, pos_label=0)
    for name, model in candidates.items():
        cv = cross_validate(model, x_train, y_train,
                            cv=StratifiedKFold(5, shuffle=True, random_state=42),
                            scoring={'recall_maligno': scorer, 'f1_maligno': make_scorer(f1_score, pos_label=0)})
        scores[name] = (float(cv['test_recall_maligno'].mean()), float(cv['test_f1_maligno'].mean()))
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        results[name] = {
            'cv_recall_maligno': scores[name][0], 'cv_f1_maligno': scores[name][1],
            'accuracy': float(accuracy_score(y_test, pred)),
            'precision_maligno': float(precision_score(y_test, pred, pos_label=0)),
            'recall_maligno': float(recall_score(y_test, pred, pos_label=0)),
            'f1_maligno': float(f1_score(y_test, pred, pos_label=0)),
            'matriz_confusao_ordem_0_1': confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
            'falsos_negativos_malignos': int(((y_test == 0) & (pred == 1)).sum()),
        }
    selected = max(scores, key=scores.get)
    model = candidates[selected]
    importance = permutation_importance(model, x_test, y_test, scoring=scorer, n_repeats=10, random_state=42)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    artifact = {'model': model, 'name': selected, 'features': data.feature_names.tolist(),
                'reference': np.median(x_train, axis=0), 'version': 'wisconsin-split42-v1'}
    joblib.dump(artifact, output / 'modelo.joblib')
    report = {'dataset': 'sklearn Wisconsin Diagnostic Breast Cancer', 'samples': len(data.target),
              'train': len(y_train), 'test': len(y_test), 'features': len(data.feature_names),
              'missing': int(np.isnan(data.data).sum()), 'classes_0_maligno_1_benigno': np.bincount(data.target).tolist(),
              'selection': 'Maior recall maligno CV no treino; desempate por F1 CV. Teste não seleciona modelo.',
              'selected': selected, 'models': results,
              'importance_global_recall_drop': dict(zip(data.feature_names, importance.importances_mean.tolist()))}
    (output / 'metricas.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    examples = select_examples(model, x_test, y_test, data.feature_names)
    (output / 'exemplos.json').write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding='utf-8')
    return report


def select_examples(model, x_test, y_test, names):
    """Casos do teste para a demonstracao, sem identificadores pessoais.

    A primeira versao usava as duas primeiras linhas do teste, que davam probabilidade
    1,0 e 0,0: as contribuicoes locais ficavam quase nulas. Agora escolhemos um caso
    claro de cada classe, o caso mais proximo da fronteira e, se houver, um falso
    negativo, para discutir o erro mais grave do modelo. Relatos e perguntas sao sinteticos.
    """
    prob = model.predict_proba(x_test)[:, list(model.classes_).index(0)]
    pred = model.predict(x_test)
    chosen = [
        ('Maligno claro', int(np.argmax(np.where(y_test == 0, prob, -1))),
         'Relato sintético: paciente refere nódulo endurecido na mama, percebido há dois meses.',
         'Quais os próximos passos para confirmar o diagnóstico e em quantos dias os exames devem sair?'),
        ('Caso limítrofe com relato de segurança', int(np.argmin(np.abs(prob - .5))),
         'Relato sintético: conta que o companheiro controla o dinheiro dela e a ameaça quando sai para '
         'os exames. Pede para conversar sem o acompanhante na sala.',
         'Como acolher este relato e quais orientações existem para o atendimento?'),
        ('Benigno claro', int(np.argmin(np.where(y_test == 1, prob, 2))), '',
         'O resultado do modelo exclui câncer?'),
    ]
    missed = np.where((y_test == 0) & (pred == 1))[0]
    if len(missed):
        chosen.append(('Erro do modelo: falso negativo no teste', int(missed[0]), '',
                       'O que significam as medidas que mais influenciaram este resultado?'))
    return [{'nome': nome, 'classe_real': int(y_test[i]), 'probabilidade_maligno': round(float(prob[i]), 4),
             'entrada': {'features': dict(zip(names, x_test[i].tolist())), 'relato': relato, 'pergunta': pergunta}}
            for nome, i, relato, pergunta in chosen]


if __name__ == '__main__':
    print(json.dumps(train(), indent=2, ensure_ascii=False))
