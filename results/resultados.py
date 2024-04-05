import numpy as np

class ListaResultados:
    def __init__(self):
        self.resultados: list = []

    def add_resultado(self, resultado: float):
        self.resultados.append(resultado)

class ListaResultadosKnn(ListaResultados):
    def __init__(self):
        super().__init__()

    def get_resultados(self):
        print("KNN - Resultados")
        print(f" Média Knn: {np.mean(self.resultados)}")
        print(f" Desvio padrão Knn: {np.std(self.resultados)}")
        print("------------------------------------------------------------")

class ListaResultadosArvoreDecisao(ListaResultados):
    def __init__(self):
        super().__init__()

    def get_resultados(self):
        print("Árvore de Decisão - Resultados")
        print(f" Média Árvore de Decisão: {np.mean(self.resultados)}")
        print(f" Desvio padrão Árvore de Decisão: {np.std(self.resultados)}")
        print("------------------------------------------------------------")

class ListaResultadosSvm(ListaResultados):
    def __init__(self):
        super().__init__()

    def get_resultados(self):
        print("SVM - Resultados")
        print(f" Média SVM: {np.mean(self.resultados)}")
        print(f" Desvio padrão SVM: {np.std(self.resultados)}")
        print("------------------------------------------------------------")

class ListaResultadosRegLog(ListaResultados):
    def __init__(self):
        super().__init__()

    def get_resultados(self):
        print("Regressão Logística - Resultados")
        print(f" Média Regressão Logística: {np.mean(self.resultados)}")
        print(f" Desvio padrão Regressão Logística: {np.std(self.resultados)}")
        print("------------------------------------------------------------")

class ListaResultadosRandomForest(ListaResultados):
    def __init__(self):
        super().__init__()

    def get_resultados(self):
        print("Random Forest - Resultados")
        print(f" Média Random Forest: {np.mean(self.resultados)}")
        print(f" Desvio Random Forest: {np.std(self.resultados)}")
        print("------------------------------------------------------------")

class ListaResultadosGausianNb(ListaResultados):
    def __init__(self):
        super().__init__()
        
    def get_resultados(self):
        print("Gausian NB - Resultados")
        print(f" Média Gausian NB: {np.mean(self.resultados)}")
        print(f" Desvio Gausian NB: {np.std(self.resultados)}")
        print("------------------------------------------------------------")