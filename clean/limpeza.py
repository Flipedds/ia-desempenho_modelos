import random
import pandas as pd
from sklearn.model_selection import train_test_split

def limpar_dados_banana(dados_banana: pd.DataFrame) -> pd.DataFrame:
    replace = {
        "Size": "Tamanho",
        "Weight": "Peso",
        "Sweetness" : "Doçura",
        "HarvestTime" : "Tempo de colheita",
        "Softness" : "Suavidade",
        "Ripeness" : "Maturação",
        "Acidity" : "Acidez",
        "Quality" : "Qualidade"
    }
    dados_banana = dados_banana.fillna(0)
    dados_banana = dados_banana.rename(columns=replace)
    dados_banana['Qualidade'] = dados_banana['Qualidade'].apply(lambda x: 1 if x == 'Good' else 0)

    mediana_tamanho = dados_banana['Tamanho'].median()
    dados_banana.loc[dados_banana['Tamanho'] < 0, 'Tamanho'] = mediana_tamanho

    mediana_peso = dados_banana['Peso'].median()
    dados_banana.loc[dados_banana['Peso'] < 0, 'Peso'] = mediana_peso

    mediana_docura = dados_banana['Doçura'].median()
    dados_banana.loc[dados_banana['Doçura'] < 0, 'Doçura'] = mediana_docura

    mediana_suavidade = dados_banana['Suavidade'].median()
    dados_banana.loc[dados_banana['Suavidade'] < 0, 'Suavidade'] = mediana_suavidade

    mediana_tempo = dados_banana['Tempo de colheita'].median()
    dados_banana.loc[dados_banana['Tempo de colheita'] < 0, 'Tempo de colheita'] = mediana_tempo

    mediana_maturacao = dados_banana['Maturação'].median()
    dados_banana.loc[dados_banana['Maturação'] < 0, 'Maturação'] = mediana_maturacao

    mediana_acidez = dados_banana['Acidez'].median()
    dados_banana.loc[dados_banana['Acidez'] < 0, 'Acidez'] = mediana_acidez 
    
    return dados_banana

def preparar_para_treinamento(dados_banana: pd.DataFrame) -> tuple:
    X = dados_banana.drop(columns=['Qualidade'])
    y = dados_banana['Qualidade']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=random.uniform(0.10, 0.40))

    return X_train, X_test, y_train, y_test