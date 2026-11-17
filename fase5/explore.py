"""Exploracao e preparo dos dados da Fase 5.

A exploracao completa do Wisconsin esta no notebook da Fase 1
(notebooks/01_cancer_mama.ipynb). Aqui repetimos so o que justifica as decisoes de
preparo da Guardia AI e testamos essas decisoes por validacao cruzada no treino:
escala, transformacao log e remocao de variaveis muito correlacionadas.

    python -m fase5.explore     # gera resultados/fase5/exploracao.json e figuras/
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from fase5.train import OUTPUT

# Mesmas cores nas figuras e na interface: laranja = maligno, azul = benigno.
COR = {0: '#eb6834', 1: '#2a78d6'}
ROTULO = {0: 'Maligno (0)', 1: 'Benigno (1)'}
TINTA, TINTA_SEC, GRADE = '#0b0b0b', '#52514e', '#e4e3df'


def _estilo(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(GRADE)
    ax.tick_params(colors=TINTA_SEC, labelsize=8)
    ax.grid(axis='y', color=GRADE, linewidth=.6)
    ax.set_axisbelow(True)


def cohen_d(a, b):
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / pooled


def drop_correlated(x, limite=.95):
    """Mantem a primeira variavel de cada par com |r| acima do limite (ordem do dataset)."""
    corr = np.abs(np.corrcoef(x, rowvar=False))
    keep = []
    for j in range(x.shape[1]):
        if all(corr[j, k] <= limite for k in keep):
            keep.append(j)
    return keep


def explore(output=OUTPUT):
    data = load_breast_cancer()
    x, y, names = data.data, data.target, list(data.feature_names)
    x_train, _, y_train, _ = train_test_split(x, y, test_size=.2, random_state=42, stratify=y)

    # 1. Qualidade e balanceamento
    _, counts = np.unique(x, axis=0, return_counts=True)
    resumo = {'amostras': int(len(y)), 'variaveis': len(names), 'ausentes': int(np.isnan(x).sum()),
              'linhas_duplicadas': int((counts > 1).sum()),
              'classes': {'maligno_0': int((y == 0).sum()), 'benigno_1': int((y == 1).sum())},
              'proporcao_maligno': round(float((y == 0).mean()), 4)}

    # 2. Escalas: area chega a milhares, suavidade fica perto de 0,1
    escala = sorted(({'variavel': n, 'min': float(x[:, j].min()), 'max': float(x[:, j].max()),
                      'assimetria': float(((x[:, j] - x[:, j].mean()) ** 3).mean() / x[:, j].std() ** 3)}
                     for j, n in enumerate(names)), key=lambda r: -r['max'])

    # 3. Separacao entre classes (d de Cohen, maligno - benigno)
    efeito = sorted(({'variavel': n, 'd_cohen': round(float(cohen_d(x[y == 0, j], x[y == 1, j])), 3)}
                     for j, n in enumerate(names)), key=lambda r: -abs(r['d_cohen']))

    # 4. Correlacao: raio, perimetro e area medem quase a mesma coisa
    corr = np.corrcoef(x, rowvar=False)
    pares = sorted(({'a': names[i], 'b': names[j], 'r': round(float(corr[i, j]), 3)}
                    for i in range(len(names)) for j in range(i + 1, len(names)) if abs(corr[i, j]) > .9),
                   key=lambda r: -abs(r['r']))

    # 5. Outliers pela regra do IQR, sem remover: concentram-se nos casos malignos
    q1, q3 = np.percentile(x, [25, 75], axis=0)
    fora = (x < q1 - 1.5 * (q3 - q1)) | (x > q3 + 1.5 * (q3 - q1))
    linhas = fora.any(axis=1)
    outliers = {'linhas_com_algum_outlier': int(linhas.sum()),
                'proporcao_maligno_nessas_linhas': round(float((y[linhas] == 0).mean()), 4),
                'por_variavel_top5': sorted(({'variavel': n, 'quantidade': int(fora[:, j].sum())}
                                             for j, n in enumerate(names)), key=lambda r: -r['quantidade'])[:5]}

    # 6. Decisoes de preparo testadas por CV no treino (recall maligno, regressao logistica)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    scorer = make_scorer(recall_score, pos_label=0)
    lr = lambda: LogisticRegression(max_iter=5000, random_state=42)
    keep = drop_correlated(x_train)
    variantes = {
        'sem padronizacao': (make_pipeline(lr()), x_train),
        'StandardScaler (usada no modelo)': (make_pipeline(StandardScaler(), lr()), x_train),
        'log1p + StandardScaler': (make_pipeline(FunctionTransformer(np.log1p), StandardScaler(), lr()), x_train),
        f'StandardScaler sem correlacionadas (|r|>0,95, restam {len(keep)})':
            (make_pipeline(StandardScaler(), lr()), x_train[:, keep]),
    }
    preparo = []
    for nome, (modelo, dados) in variantes.items():
        s = cross_val_score(modelo, dados, y_train, cv=cv, scoring=scorer)
        preparo.append({'variante': nome, 'recall_maligno_cv': round(float(s.mean()), 4),
                        'desvio': round(float(s.std()), 4)})

    output.mkdir(parents=True, exist_ok=True)
    report = {'fonte': 'sklearn.datasets.load_breast_cancer (UCI Breast Cancer Wisconsin Diagnostic)',
              'exploracao_completa': 'notebooks/01_cancer_mama.ipynb (Fase 1)',
              'resumo': resumo, 'maiores_escalas': escala[:3], 'menores_escalas': escala[-3:],
              'separacao_classes_top10': efeito[:10], 'pares_correlacao_acima_0_9': len(pares),
              'pares_mais_correlacionados': pares[:8], 'outliers_iqr': outliers,
              'preparo_testado_cv_treino': preparo}
    (output / 'exploracao.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    _figuras(output / 'figuras', x, y, names, efeito, corr)
    return report


def _figuras(pasta, x, y, names, efeito, corr):
    pasta.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': TINTA, 'axes.labelcolor': TINTA_SEC})

    # Distribuicao das seis variaveis que mais separam as classes (escala padronizada)
    top = [e['variavel'] for e in efeito[:6]]
    fig, axes = plt.subplots(2, 3, figsize=(9, 5))
    for ax, nome in zip(axes.ravel(), top):
        col = x[:, names.index(nome)]
        bins = np.linspace(col.min(), col.max(), 26)
        for classe in (1, 0):
            ax.hist(col[y == classe], bins=bins, color=COR[classe], alpha=.6, label=ROTULO[classe],
                    edgecolor='white', linewidth=.5)
        ax.set_title(nome, fontsize=9, color=TINTA)
        _estilo(ax)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle('Variáveis com maior separação entre classes (d de Cohen)', fontsize=10, color=TINTA)
    fig.tight_layout()
    fig.savefig(pasta / 'distribuicao_top6.png', dpi=150)
    plt.close(fig)

    # Matriz de correlacao com mapa divergente azul - cinza - laranja
    cmap = LinearSegmentedColormap.from_list('div', [COR[1], '#f0efec', COR[0]])
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=6)
    ax.set_yticks(range(len(names)), names, fontsize=6)
    ax.tick_params(colors=TINTA_SEC, length=0)
    fig.colorbar(im, ax=ax, fraction=.04, label='correlação de Pearson')
    ax.set_title('Correlação entre as 30 medidas', fontsize=10, color=TINTA)
    fig.tight_layout()
    fig.savefig(pasta / 'correlacao.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    print(json.dumps(explore(), indent=2, ensure_ascii=False))
