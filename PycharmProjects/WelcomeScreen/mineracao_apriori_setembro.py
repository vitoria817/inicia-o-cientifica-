# -*- coding: utf-8 -*-
"""
SETEMBRO (4º mês) — Mineração de regras de associação (Apriori)
Iniciação Científica: Cruzamento CSC 2.0 x SDSS DR15

Segue a etapa 3 da metodologia do artigo:
"Aplicação do algoritmo Apriori (biblioteca mlxtend do Python) com
suporte mínimo de 1% e confiança mínima de 60%. Regras ordenadas por lift."

Pré-requisito: base "limpa" (etapa 2 - Exploração e limpeza), ou seja,
removendo os 3.385 pares identificados em Agosto como inconsistentes
(match_grade = 'D' com probabilidade baixa E separação grande).
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ------------------------------------------------------------
# 1. Carregar dados e recriar as categorias definidas em Agosto
# ------------------------------------------------------------
df = pd.read_csv('CSC2_SDSSDR15.csv')

def cat_sep(x):
    if x < 1:
        return 'separacao=<1"'
    elif x <= 2:
        return 'separacao=1-2"'
    else:
        return 'separacao=>2"'

def cat_prob(x):
    if x >= 0.99:
        return 'probabilidade=alta'
    elif x >= 0.9:
        return 'probabilidade=media'
    else:
        return 'probabilidade=baixa'

df['sep_categoria'] = df['CSC2-SDSSDR15_separation'].apply(cat_sep)
df['prob_categoria'] = df['Match_probability'].apply(cat_prob)

total_antes = len(df)

# ------------------------------------------------------------
# 2. LIMPEZA: remover os 3.385 casos inconsistentes de Agosto
#    (match_grade = D, mas probabilidade baixa E separação grande)
# ------------------------------------------------------------
mask_inconsistente = (
    (df['Match_grade'] == 'D') &
    (df['prob_categoria'] == 'probabilidade=baixa') &
    (df['sep_categoria'] == 'separacao=>2"')
)
n_removidos = mask_inconsistente.sum()
df_limpo = df[~mask_inconsistente].copy()

print(f"Total original: {total_antes}")
print(f"Registros inconsistentes removidos: {n_removidos}")
print(f"Total após limpeza: {len(df_limpo)}")
print()

# ------------------------------------------------------------
# 3. Montar as "transações" (cada par = uma transação com itens)
#    Itens usados: sep_categoria, prob_categoria, tipo SDSS,
#    match_type e match_grade — os mesmos discretizados no artigo
#    ("separação angular, probabilidade de match, tipos morfológicos")
# ------------------------------------------------------------
transacoes = []
for _, row in df_limpo.iterrows():
    itens = [
        row['sep_categoria'],
        row['prob_categoria'],
        f"tipo_sdss={row['SDSSDR15_Type']}",
        f"match_type={row['Match_type']}",
        f"match_grade={row['Match_grade']}",
    ]
    transacoes.append(itens)

te = TransactionEncoder()
te_ary = te.fit(transacoes).transform(transacoes)
df_onehot = pd.DataFrame(te_ary, columns=te.columns_)

# ------------------------------------------------------------
# 4. Apriori — suporte mínimo 1%, exatamente como o artigo pede
# ------------------------------------------------------------
itemsets_frequentes = apriori(df_onehot, min_support=0.01, use_colnames=True)
itemsets_frequentes = itemsets_frequentes.sort_values('support', ascending=False)

# ------------------------------------------------------------
# 5. Regras de associação — confiança mínima 60%, ordenadas por lift
# ------------------------------------------------------------
regras = association_rules(itemsets_frequentes, metric="confidence",
                            min_threshold=0.6, num_itemsets=len(df_onehot))
regras = regras.sort_values('lift', ascending=False).reset_index(drop=True)

# Formatar antecedents/consequents como texto legível
regras['antecedente'] = regras['antecedents'].apply(lambda x: ', '.join(sorted(x)))
regras['consequente'] = regras['consequents'].apply(lambda x: ', '.join(sorted(x)))

colunas_finais = ['antecedente', 'consequente', 'support', 'confidence', 'lift']
regras_final = regras[colunas_finais].rename(columns={
    'support': 'suporte', 'confidence': 'confianca', 'lift': 'lift'
})

print(f"Total de itemsets frequentes (suporte >= 1%): {len(itemsets_frequentes)}")
print(f"Total de regras (confiança >= 60%): {len(regras_final)}")
print()
print("=== TOP 20 REGRAS POR LIFT ===")
print(regras_final.head(20).to_string(index=False))

# ------------------------------------------------------------
# 6. Salvar resultados
# ------------------------------------------------------------
regras_final.to_csv('regras_associacao_setembro.csv', index=False)
itemsets_frequentes.to_csv('itemsets_frequentes_setembro.csv', index=False)
print("\nArquivos salvos: regras_associacao_setembro.csv, itemsets_frequentes_setembro.csv")
