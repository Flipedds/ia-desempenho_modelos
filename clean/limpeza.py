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
    
    return dados_banana

def preparar_para_treinamento(dados_banana: pd.DataFrame) -> tuple:
    X = dados_banana.drop(columns=['Qualidade'])
    y = dados_banana['Qualidade']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=random.uniform(0.10, 0.40))

    return X_train, X_test, y_train, y_test