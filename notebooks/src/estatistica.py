from scipy.stats import (
    shapiro,
    levene, 
    ttest_ind,
    ttest_rel,
    mannwhitneyu,
)


def analise_levene(dataframe, alpha=0.05, center='mean'):
    print('Teste de Levene')
    print('---------------------')

    estatistica_levene, valor_p_levene = levene(*[dataframe[coluna] for coluna in dataframe.columns], center=center, nan_policy='omit')

    print(f'Estatistica_levene: {estatistica_levene:.3f}')
    if valor_p_levene > alpha:
        print(f'"Variâncias iguais(valor-p: {valor_p_levene:.3f})')
    else:
        print(f'Ao menos uma variância é diferente (valor-p: {valor_p_levene:.3f})')


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


def remove_outliers(dados, largura_bigodes=1.5):
    q1 = dados.quantile(0.25)
    q3 = dados.quantile(0.75)
    iqr = q3 - q1 # intervalor interquartil
    return dados[(dados >= q1 - largura_bigodes * iqr) & (dados <= q3 + largura_bigodes * iqr)]