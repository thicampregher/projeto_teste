import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
from scipy.stats import (
    levene, shapiro,
    ttest_ind,
    ttest_rel,
    f_oneway,
    wilcoxon,
    mannwhitneyu,
    friedmanchisquare,
    kruskal,
)

def composicao_histograma_boxplot(dataframe, coluna, intervalos='auto'):
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, 
        ncols=1, 
        sharex=True,
        gridspec_kw={'height_ratios': (0.20, 0.80), 'hspace': 0.02}
    )

    sns.boxplot(data=dataframe, 
        x=coluna, 
        showmeans=True, 
        meanline=True, 
        meanprops={'color': 'C1', 'linewidth': 1.5, 'linestyle': '--'},
        medianprops={'color': 'C2', 'linewidth': 1.5, 'linestyle': '--'},
        ax=ax1
    )

    sns.histplot(data=dataframe, 
        x=coluna, 
        kde=True, 
        bins=intervalos, 
        ax=ax2
    )

    for ax in (ax1, ax2):
        ax.grid(True, linestyle='--', color='gray', alpha=0.4)
        ax.set_axisbelow(True)

    ax2.axvline(dataframe[coluna].mean(), color='C1', linestyle='--', label='Media')
    ax2.axvline(dataframe[coluna].median(), color='C2', linestyle='--', label='Mediana')
    ax2.axvline(dataframe[coluna].mode()[0], color='C3', linestyle='--', label='Moda')
    
    ax2.legend()
    
    plt.show()


def analise_shapiro(dataframe, alpha=0.05):
    print('Teste de Shapiro-Wilk')
    print('---------------------')
    for coluna in dataframe.columns:
        estatistica_sw, valor_p_sw = shapiro(dataframe[coluna], nan_policy='omit')
        print(f'Estatistica_SW: {estatistica_sw:.3f}')
        if valor_p_sw > alpha:
            print(f'"{coluna}" segue uma distribuição normal (valor-p: {valor_p_sw:.3f})')
        else:
            print(f'"{coluna}" não segue uma distribuição normal (valor-p: {valor_p_sw:.3f})')
        

def analise_levene(dataframe, alpha=0.05, center='mean'):
    print('Teste de Levene')
    print('---------------------')

    estatistica_levene, valor_p_levene = levene(*[dataframe[coluna] for coluna in dataframe.columns], center=center, nan_policy='omit')

    print(f'Estatistica_levene: {estatistica_levene:.3f}')
    if valor_p_levene > alpha:
        print(f'"Variâncias iguais(valor-p: {valor_p_levene:.3f})')
    else:
        print(f'Ao menos uma variância é diferente (valor-p: {valor_p_levene:.3f})')


def analises_shapiro_levene(dataframe, alpha=0.05, center='mean'):
    analise_shapiro(dataframe, alpha)

    print()

    analise_levene(dataframe, alpha, center)

def analise_ttest_ind(
    dataframe,
    alpha=0.05,
    variancias_iguais=True,
    alternativa='two-sided'
):
    estatistica_ttest_ind, valor_p_ttest_ind = ttest_ind(
        *[dataframe[coluna] for coluna in dataframe.columns],
        equal_var=variancias_iguais,
        alternative=alternativa,
        nan_policy='omit'
    )
    
    print('Teste dt de Student')
    print('-------------------')
    print(f'Estatistica_ttest_ind: {estatistica_ttest_ind:.3f}')
    if valor_p_ttest_ind > alpha:
        print(f'Não rejeita a hipótese nula. (valor-p: {valor_p_ttest_ind:.3f})')
    else:
        print(f'Rejeita a hipótese nula. (valor-p: {valor_p_ttest_ind:.3f})')


def analise_ttest_rel(
    dataframe,
    alpha=0.05,
    alternativa='two-sided'
):
    estatistica_ttest_rel, valor_p_ttest_rel = ttest_rel(
        *[dataframe[coluna] for coluna in dataframe.columns],
        alternative=alternativa,
        nan_policy='omit'
    )

    print('Teste dt de Student')
    print('-------------------')
    print(f'Estatistica_ttest_rel: {estatistica_ttest_rel:.3f}')
    if valor_p_ttest_rel > alpha:
        print(f'Não rejeita a hipótese nula. (valor-p: {valor_p_ttest_rel:.3f})')
    else:
        print(f'Rejeita a hipótese nula. (valor-p: {valor_p_ttest_rel:.3f})')


def analise_anova_one_way(
    dataframe,
    alpha=0.05
):
    estatistica_f, valor_p_f = f_oneway(
        *[dataframe[coluna] for coluna in dataframe.columns],
        nan_policy='omit'
    )

    print('Teste ANOVA one way')
    print('-------------------')
    print(f'Estatistica_f: {estatistica_f:.3f}')
    if valor_p_f > alpha:
        print(f'Não rejeita a hipótese nula. (valor-p: {valor_p_f:.3f})')
    else:
        print(f'Rejeita a hipótese nula. (valor-p: {valor_p_f:.3f})')


def analise_wilcoxon(
    dataframe,
    alpha=0.05,
    alternativa='two-sided'
):
    estatistica_wilcoxon, valor_p_wilcoxon = wilcoxon(
        *[dataframe[coluna] for coluna in dataframe.columns],
        nan_policy='omit',
        alternative=alternativa
    )

    print('Teste Wilcoxon')
    print('--------------')
    print(f'Estatistica_wilcoxon: {estatistica_wilcoxon:.3f}')
    if valor_p_wilcoxon > alpha:
        print(f'Não rejeita a hipótese nula. (valor-p: {valor_p_wilcoxon:.3f})')
    else:
        print(f'Rejeita a hipótese nula. (valor-p: {valor_p_wilcoxon:.3f})')


def analise_mannwhitneyu(
    dataframe,
    alpha=0.05,
    alternativa='two-sided'
):
    estatistica_mw, valor_p_mw = mannwhitneyu(
        *[dataframe[coluna] for coluna in dataframe.columns],
        nan_policy='omit',
        alternative=alternativa
    )

    print('Teste Mann-Whitney')
    print('------------------')
    print(f'Estatistica_mw: {estatistica_mw:.3f}')
    if valor_p_mw > alpha:
        print(f'Não rejeita a hipótese nula. (valor-p: {valor_p_mw:.3f})')
    else:
        print(f'Rejeita a hipótese nula. (valor-p: {valor_p_mw:.3f})')


def analise_friedman(
    dataframe,
    alpha=0.05
):
    estatistica_friedman, valor_p_friedman = friedmanchisquare(
        *[dataframe[coluna] for coluna in dataframe.columns],
        nan_policy='omit'
    )

    print('Teste de Friedman')
    print('------------------')
    print(f'Estatistica_friedman: {estatistica_friedman:.3f}')
    if valor_p_friedman > alpha:
        print(f'Não rejeita a hipótese nula. (valor-p: {valor_p_friedman:.3f})')
    else:
        print(f'Rejeita a hipótese nula. (valor-p: {valor_p_friedman:.3f})')


def analise_kruskal(
    dataframe,
    alpha=0.05
):
    estatistica_kruskal, valor_p_kruskal = kruskal(
        *[dataframe[coluna] for coluna in dataframe.columns],
        nan_policy='omit'
    )

    print('Teste de kruskal')
    print('----------------')
    print(f'Estatistica_friedman: {estatistica_kruskal:.3f}')
    if valor_p_kruskal > alpha:
        print(f'Não rejeita a hipótese nula. (valor-p: {valor_p_kruskal:.3f})')
    else:
        print(f'Rejeita a hipótese nula. (valor-p: {valor_p_kruskal:.3f})')