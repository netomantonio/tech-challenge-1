"""
Módulo de funções auxiliares para o projeto de diagnóstico médico.

Contém funções reutilizáveis para visualização, avaliação de modelos
e comparação de resultados entre diferentes algoritmos de ML.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)


# Configuração padrão dos gráficos
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
sns.set_style('whitegrid')


def _salvar_se_necessario(fig, salvar_em):
    """Salva o gráfico se um caminho for fornecido."""
    if salvar_em:
        fig.savefig(salvar_em, dpi=150, bbox_inches='tight')
        print(f"  → Gráfico salvo em: {salvar_em}")


def plotar_distribuicao_classes(y, nomes_classes, titulo='Distribuição das Classes', salvar_em=None):
    """
    Plota gráfico de barras mostrando a distribuição das classes.

    Parâmetros:
        y: array com os rótulos das classes
        nomes_classes: lista com os nomes das classes
        titulo: título do gráfico
        salvar_em: caminho para salvar o gráfico (opcional)
    """
    valores, contagens = np.unique(y, return_counts=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    barras = ax.bar(nomes_classes, contagens, color=['#e74c3c', '#2ecc71'])

    # Adicionar valores nas barras
    for barra, contagem in zip(barras, contagens):
        ax.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 5,
                f'{contagem} ({contagem/len(y)*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold')

    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.set_ylabel('Quantidade de Amostras')
    plt.tight_layout()
    _salvar_se_necessario(fig, salvar_em)
    plt.show()


def plotar_histogramas(df, colunas, titulo='Distribuição das Features', salvar_em=None):
    """
    Plota histogramas para as colunas selecionadas.

    Parâmetros:
        df: DataFrame com os dados
        colunas: lista de colunas para plotar
        titulo: título geral
        salvar_em: caminho para salvar o gráfico (opcional)
    """
    n_colunas = len(colunas)
    n_linhas = (n_colunas + 2) // 3  # 3 gráficos por linha
    fig, axes = plt.subplots(n_linhas, 3, figsize=(15, 4 * n_linhas))
    axes = axes.flatten()

    for i, coluna in enumerate(colunas):
        axes[i].hist(df[coluna], bins=30, color='#3498db', edgecolor='black', alpha=0.7)
        axes[i].set_title(coluna, fontsize=10)
        axes[i].set_ylabel('Frequência')

    # Esconder eixos extras
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(titulo, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    _salvar_se_necessario(fig, salvar_em)
    plt.show()


def plotar_correlacao(df, titulo='Mapa de Correlação', salvar_em=None):
    """
    Plota heatmap de correlação entre as features.

    Parâmetros:
        df: DataFrame com os dados
        titulo: título do gráfico
        salvar_em: caminho para salvar o gráfico (opcional)
    """
    correlacao = df.corr()

    fig, ax = plt.subplots(figsize=(14, 10))
    mascara = np.triu(np.ones_like(correlacao, dtype=bool))

    sns.heatmap(correlacao, mask=mascara, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, linewidths=0.5,
                ax=ax, annot_kws={'size': 8})

    ax.set_title(titulo, fontsize=14, fontweight='bold')
    plt.tight_layout()
    _salvar_se_necessario(fig, salvar_em)
    plt.show()

    return correlacao


def plotar_matriz_confusao(y_real, y_pred, nomes_classes, titulo='Matriz de Confusão', salvar_em=None):
    """
    Plota a matriz de confusão.

    Parâmetros:
        y_real: valores reais
        y_pred: valores preditos
        nomes_classes: nomes das classes
        titulo: título do gráfico
        salvar_em: caminho para salvar o gráfico (opcional)
    """
    cm = confusion_matrix(y_real, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=nomes_classes, yticklabels=nomes_classes, ax=ax)
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.set_ylabel('Valor Real')
    ax.set_xlabel('Valor Predito')
    plt.tight_layout()
    _salvar_se_necessario(fig, salvar_em)
    plt.show()


def plotar_curva_roc(y_real, y_prob, nome_modelo, ax=None):
    """
    Plota a curva ROC para um modelo.

    Parâmetros:
        y_real: valores reais
        y_prob: probabilidades preditas (classe positiva)
        nome_modelo: nome do modelo para a legenda
        ax: eixo matplotlib (opcional)
    """
    fpr, tpr, _ = roc_curve(y_real, y_prob)
    roc_auc = auc(fpr, tpr)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, linewidth=2, label=f'{nome_modelo} (AUC = {roc_auc:.3f})')
    return roc_auc


def plotar_curvas_roc_comparativas(y_real, resultados, titulo='Comparação das Curvas ROC', salvar_em=None):
    """
    Plota curvas ROC de múltiplos modelos no mesmo gráfico.

    Parâmetros:
        y_real: valores reais
        resultados: dict {nome_modelo: y_probabilidades}
        titulo: título do gráfico
        salvar_em: caminho para salvar o gráfico (opcional)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for nome, y_prob in resultados.items():
        plotar_curva_roc(y_real, y_prob, nome, ax=ax)

    # Linha diagonal (classificador aleatório)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aleatório (AUC = 0.500)')
    ax.set_xlabel('Taxa de Falsos Positivos')
    ax.set_ylabel('Taxa de Verdadeiros Positivos')
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    plt.tight_layout()
    _salvar_se_necessario(fig, salvar_em)
    plt.show()


def avaliar_modelo(y_real, y_pred, nome_modelo='Modelo'):
    """
    Calcula as métricas de avaliação de um modelo.

    Parâmetros:
        y_real: valores reais
        y_pred: valores preditos
        nome_modelo: nome do modelo

    Retorna:
        dict com accuracy, precision, recall e f1-score
    """
    metricas = {
        'Modelo': nome_modelo,
        'Acurácia': accuracy_score(y_real, y_pred),
        'Precisão': precision_score(y_real, y_pred, average='weighted'),
        'Recall': recall_score(y_real, y_pred, average='weighted'),
        'F1-Score': f1_score(y_real, y_pred, average='weighted')
    }
    return metricas


def comparar_modelos(lista_metricas):
    """
    Cria uma tabela comparativa com as métricas de vários modelos.

    Parâmetros:
        lista_metricas: lista de dicts retornados por avaliar_modelo()

    Retorna:
        DataFrame com a comparação
    """
    df_comparacao = pd.DataFrame(lista_metricas)
    df_comparacao = df_comparacao.set_index('Modelo')

    # Formatar como porcentagem
    df_formatado = df_comparacao.style.format('{:.4f}').highlight_max(
        axis=0, color='lightgreen'
    )

    return df_comparacao, df_formatado


def exibir_classification_report(y_real, y_pred, nomes_classes):
    """
    Exibe o classification report de forma formatada.

    Parâmetros:
        y_real: valores reais
        y_pred: valores preditos
        nomes_classes: nomes das classes
    """
    print(classification_report(y_real, y_pred, target_names=nomes_classes))
