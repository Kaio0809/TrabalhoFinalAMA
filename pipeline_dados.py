import pandas as pd, numpy as np, re 

'''
Criação do Dataset Usasdo
'''

# Leitura dos datasets
#data = pd.read_csv('TrabalhoFinalAMA\datasets\data.csv')
#movies = pd.read_csv("TrabalhoFinalAMA\datasets\data.csv")
#data['names'] = data['Movie Name']

#filmes = pd.merge(data, movies, on='names')
# Merge dos dois dataset, para pegar os filmes que estão nos dois.

#filmes.to_csv("filmes.csv")
# colocar o dataset filmes em um arquivo csv e salvar

dataset_filmes_usado = pd.read_csv("filmes.csv")

dataset_filmes_usado = dataset_filmes_usado[dataset_filmes_usado['Year of Release'] >= 2010]
#Pegar filmes lançados a partir de 2010

fatores_ipca = {2010: 2.03, 2011: 1.92,2012: 1.81,2013: 1.71,2014: 1.61,2015: 1.45,2016: 1.31, 2017: 1.23,2018: 1.19,2019: 1.14,2020: 1.09,2021: 1.00,2022: 0.94,2023: 0.90,2024: 0.86,2025: 1.00}
# Fatores de Correção da inflação

dataset_filmes_usado["fator_ipca"] = dataset_filmes_usado["Year of Release"].map(fatores_ipca)
dataset_filmes_usado["lucro"] = (dataset_filmes_usado["revenue"] - dataset_filmes_usado["budget_x"]) / dataset_filmes_usado["fator_ipca"]
dataset_filmes_usado["budget_x"] = dataset_filmes_usado["budget_x"] / dataset_filmes_usado["fator_ipca"]

dataset_filmes_usado = dataset_filmes_usado.dropna(subset=["revenue"])

dataset_filmes_usado['diretor'] = dataset_filmes_usado['Director'].apply(lambda x: re.sub(r"[\[\]']", "", str(x)))
dados = dataset_filmes_usado.copy()

def tratar_crew(x):
    if isinstance(x, list):
        return [i.strip() for i in x if isinstance(i, str) and i.strip()]
    elif isinstance(x, str):
        return [i.strip() for i in x.strip("[]").split(",") if i.strip()]
    else:
        return []

dados['crew'] = dados['crew'].fillna("").apply(tratar_crew)
dados['diretor'] = dados['diretor'].fillna("").apply(tratar_crew)

lista_de_atores = sorted(set(a for sublist in dados['crew'] for a in sublist))
lista_de_diretores = sorted(set(d for sublist in dados['diretor'] for d in sublist))

def limpar_nome(nome):
    return nome.strip(' "\'')

lista_de_atores = [limpar_nome(n) for n in lista_de_atores]
lista_de_diretores = [limpar_nome(n) for n in lista_de_diretores]

df_atores = dados[['crew', 'Movie Rating']].copy()
df_atores['crew'] = df_atores['crew'].astype(str).str.split(',')
df_atores = df_atores.explode('crew')
df_atores['crew'] = df_atores['crew'].str.strip()

atores_notas = df_atores.groupby('crew')['Movie Rating'].mean().to_dict()

df_diretores = dados[['diretor', 'Movie Rating']].copy()
df_diretores['diretor'] = df_diretores['diretor'].astype(str).str.split(',')
df_diretores = df_diretores.explode('diretor')
df_diretores['diretor'] = df_diretores['diretor'].str.strip()

diretores_notas = df_diretores.groupby('diretor')['Movie Rating'].mean().to_dict()

caracteres_a_remover = "[]{}'\" "

notas_limpo_atores = {k.strip(caracteres_a_remover): v for k, v in atores_notas.items()}
notas_limpo_dir = {k.strip(caracteres_a_remover): v for k, v in diretores_notas.items()}

atores_notas = notas_limpo_atores
diretores_notas = notas_limpo_dir

def calcular_nota_elenco(lista_atores, notas_dict):
    notas = [notas_dict[ator] for ator in lista_atores if ator in notas_dict]
    return [np.mean(notas) if notas else 0, np.median(notas) if notas else 0]

dados['media_elenco'] = dados['crew'].apply(lambda lista: calcular_nota_elenco(lista, atores_notas)[0])
dados['media_direcao'] = dados['diretor'].apply(lambda lista: calcular_nota_elenco(lista, diretores_notas)[0])
dados['mediana_elenco'] = dados['crew'].apply(lambda lista: calcular_nota_elenco(lista, atores_notas)[1])
dados['mediana_direcao'] = dados['diretor'].apply(lambda lista: calcular_nota_elenco(lista, diretores_notas)[1])


dataset_filmes_usado = dados.copy()

filmes_usados = dataset_filmes_usado.copy()
lista_de_generos = []

filmes_usados['genre'] = filmes_usados['genre'].fillna("").astype(str)
filmes_usados['genre'] = filmes_usados['genre'].apply(lambda x: [g.strip() for g in x.split(',') if g.strip()])

lista_de_generos =list(set(g for sublist in filmes_usados['genre'] for g in sublist))

for g in lista_de_generos:
    filmes_usados[g] = filmes_usados['genre'].apply(lambda lista: 1 if g in lista else 0)

dataset_filmes_usado = filmes_usados.copy()

filmes = dataset_filmes_usado.copy()
filmes['date_x'] = pd.to_datetime(filmes['date_x'])

filmes['mes'] = filmes['date_x'].dt.month

filmes['mes_sin'] = np.sin(2 * np.pi * filmes['mes'] / 12)
filmes['mes_cos'] = np.cos(2 * np.pi * filmes['mes'] / 12)

q1 = filmes['lucro'].quantile(0.33) 
q2 = filmes['lucro'].quantile(0.66)

bins = [-float('inf'), 0, q2, float('inf')]
labels = ['Prejuízo', 'Lucro baixo', 'Lucro alto']

filmes['categoria_lucro'] = pd.cut(filmes['lucro'],bins=bins,labels=labels)

dataset_filmes_usado = filmes.copy()
dataset_filmes_usado.to_csv("filmes_final_completo.csv", index=False)

dataset_final_c = dataset_filmes_usado[['Year of Release','mes_sin','mes_cos','Run Time in minutes','budget_x','media_elenco','media_direcao','mediana_elenco','mediana_direcao','Drama','Mystery','Action','Family','Documentary','Crime','Romance','War','Comedy','Fantasy','Adventure','Thriller','Science Fiction','Western','History','Horror','Animation','Music','TV Movie','categoria_lucro']]

dataset_final_c = dataset_final_c.rename(columns={
                                              'Year of Release': 'ano_lancamento',
                                              'mes_sin': 'mes_seno',
                                              'Run Time in minutes': 'duracao',
                                              'budget_x': 'orcamento'
                                              })




''' ---Tratamento de Dados--- '''

def tratar_dados(dataset):
    dataset_tratado = dataset.copy()

    # Colunas que estar zerada significa ausencia ou inconsistencia de dados.
    colunas = ['duracao','orcamento','media_elenco','media_direcao','mediana_elenco','mediana_direcao']

    for coluna in colunas:
        dataset_tratado = dataset_tratado[dataset_tratado[coluna] != 0]
    
    # Retirar Tuplas com Valores Nulos
    dataset_tratado = dataset_tratado.dropna()
    return dataset_tratado


df = dataset_final_c.copy()
# Chama a função para tratamento de Dados
df_tratado = tratar_dados(df)

print(len(df_tratado), df_tratado)
df_tratado.to_csv("dataset_filmes_class.csv", index=False)