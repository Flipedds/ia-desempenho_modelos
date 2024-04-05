import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from clean.limpeza import preparar_para_treinamento

class TreinamentoModelo:
    @staticmethod
    def treinar_modelo_knn(dados_banana: pd.DataFrame) -> float:
        X_train, X_test, y_train, y_test = preparar_para_treinamento(dados_banana)

        knn = KNeighborsClassifier(n_neighbors=random.randint(1, 10))

        knn.fit(X_train, y_train)

        pred = knn.predict(X_test)

        accuracy = accuracy_score(y_test, pred)
        return accuracy
    
    @staticmethod
    def treinar_modelo_arvore_decisao(dados_banana: pd.DataFrame) -> float:
        X_train, X_test, y_train, y_test = preparar_para_treinamento(dados_banana)

        dcst = DecisionTreeClassifier()
        dcst.fit(X_train, y_train)    

        pred = dcst.predict(X_test)

        accuracy = accuracy_score(y_test, pred)
        return accuracy
    
    @staticmethod
    def treinar_modelo_svm(dados_banana: pd.DataFrame) -> float:
        X_train, X_test, y_train, y_test = preparar_para_treinamento(dados_banana)

        svm = SVC()
        svm.fit(X_train, y_train)    

        pred = svm.predict(X_test)

        accuracy = accuracy_score(y_test, pred)
        return accuracy
    
    @staticmethod
    def treinar_modelo_regressao_logistica(dados_banana: pd.DataFrame) -> float:
        X_train, X_test, y_train, y_test = preparar_para_treinamento(dados_banana)

        rgrlog = LogisticRegression()
        rgrlog.fit(X_train, y_train) 

        pred = rgrlog.predict(X_test)

        accuracy = accuracy_score(y_test, pred)
        return accuracy
    
    @staticmethod
    def treinar_modelo_random_forest(dados_banana: pd.DataFrame)-> float:
        X_train, X_test, y_train, y_test = preparar_para_treinamento(dados_banana)

        rdmfr = RandomForestClassifier()
        rdmfr.fit(X_train, y_train)

        pred = rdmfr.predict(X_test)

        accuracy = accuracy_score(y_test, pred)
        return accuracy
    
    @staticmethod
    def treinar_modelo_gaussian_nb(dados_banana: pd.DataFrame) -> float:
        X_train, X_test, y_train, y_test = preparar_para_treinamento(dados_banana)

        gau_nb = GaussianNB()
        gau_nb.fit(X_train, y_train)
        pred = gau_nb.predict(X_test)

        accuracy = accuracy_score(y_test, pred)
        return accuracy