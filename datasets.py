import pandas as pd, numpy as np, re 

# Leitura dos datasets
data = pd.read_csv('data.csv')
movies = pd.read_csv("imdb_movies.csv")
data['names'] = data['Movie Name']

filmes = pd.merge(data, movies, on='names')
# Merge dos dois dataset, para pegar os filmes que estão nos dois.

filmes.to_csv("filmes.csv")
# colocar o dataset filmes em um arquivo csv e salvar

dataset_filmes_usado = pd.read_csv("filmes.csv")

dataset_filmes_usado = dataset_filmes_usado[dataset_filmes_usado['Year of Release'] >= 2010]
#Pegar filmes lançados a partir de 2010

dataset_filmes = pd.read_csv("/content/filmes.csv")
dataset_filmes_usado = dataset_filmes.copy()

fatores_ipca = {2010: 2.03, 2011: 1.92,2012: 1.81,2013: 1.71,2014: 1.61,2015: 1.45,2016: 1.31, 2017: 1.23,2018: 1.19,2019: 1.14,2020: 1.09,2021: 1.00,2022: 0.94,2023: 0.90,2024: 0.86,2025: 1.00}
# Fatores de Correção da inflação

dataset_filmes_usado["fator_ipca"] = dataset_filmes_usado["Year of Release"].map(fatores_ipca)
dataset_filmes_usado["receita_corrigida"] = (dataset_filmes_usado["revenue"] - dataset_filmes_usado["budget_x"]) / dataset_filmes_usado["fator_ipca"]
# Alterar a receita calculando com o fator ipca para que os dados não sejam prejudicados pela inflação

dataset_filmes_usado = dataset_filmes_usado.dropna(subset=["revenue"])
# Remove as tuplas que a coluna Revenue - (Receita) é nulo

dados = dataset_filmes_usado.copy()

dados['crew'] = dados['crew'].fillna("").apply(lambda x: [ator.strip() for ator in x.split(',') if ator.strip()])
dados['diretor'] = dados['diretor'].fillna("").apply(lambda x: [d.strip() for d in x.split(',') if d.strip()])

lista_de_atores = sorted(set(a for sublist in dados['crew'] for a in sublist))
lista_de_diretores = sorted(set(d for sublist in dados['diretor'] for d in sublist))


def nota_rapida(pessoa: str, tipo: str) -> float:
    if tipo == 'ator':
        col = 'crew'
    elif tipo == 'diretor':
        col = 'diretor'
    else:
        raise ValueError("Tipo deve ser 'ator' ou 'diretor'.")

    df = dados[[col, 'Movie Rating']].copy()
    df[col] = df[col].astype(str).str.split(',')
    df = df.explode(col)
    df[col] = df[col].str.strip()
    pessoa_filmes = df[df[col] == pessoa]

    if len(pessoa_filmes) == 0:
        return 0

    return pessoa_filmes['Movie Rating'].mean()

atores = {}
diretores = {}

for ator in lista_de_atores:
  atores[ator] = nota_rapida(ator,"ator")

for dir in lista_de_diretores:
  diretores[dir] = nota_rapida(dir, "diretor")


def calcular_nota_elenco(lista_atores, notas_dict):
    notas = [notas_dict[ator] for ator in lista_atores if ator in notas_dict]
    return np.mean(notas) if notas else 0

dados['nota_elenco'] = dados['crew'].apply(lambda lista: calcular_nota_elenco(lista, atores))
dados['nota_direcao'] = dados['diretor'].apply(lambda lista: calcular_nota_elenco(lista, diretores))

dataset_filmes_usado = dados.copy()
filmes_usados = dataset_filmes_usado.copy()

filmes_usados['genre'] = filmes_usados['genre'].fillna("").astype(str)
filmes_usados['genre'] = filmes_usados['genre'].apply(lambda x: [g.strip() for g in x.split(',') if g.strip()])

lista_de_generos = sorted(set(g for sublist in filmes_usados['genre'] for g in sublist))

for g in lista_de_generos:
    filmes_usados[g] = filmes_usados['genre'].apply(lambda lista: 1 if g in lista else 0)

dataset_filmes_usado = filmes_usados.copy()
