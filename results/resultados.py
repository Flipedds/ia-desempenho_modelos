import numpy as np
class ListaResultados:
    def __init__(self):
        self.resultados: list = []

    def add_resultado(self, resultado: float):
        self.resultados.append(resultado)

    def count_resultados(self) -> int:
        return len(self.resultados)

class ListaResultadosKnn(ListaResultados):
    def __init__(self):
        super().__init__()

    def get_resultados(self):
        print("KNN - Resultados:", end=" ")
        for i in range(self.count_resultados()):
            print(f"{ round(self.resultados[i], 2) }", end=" ")
        print(f"\n Média Knn: {np.mean(self.resultados)}")
        print(f" Desvio padrão Knn: {np.std(self.resultados)}")
        print(f" Quantidade de resultados: {self.count_resultados()}")
        print("------------------------------------------------------------")

class ListaResultadosArvoreDecisao(ListaResultados):
    def __init__(self):
        super().__init__()

    def get_resultados(self):
        print("Árvore de Decisão - Resultados:", end="")
        for i in range(self.count_resultados()):
            print(f"{ round(self.resultados[i], 2) }", end=" ")
        print(f"\n Média Árvore de Decisão: {np.mean(self.resultados)}")
        print(f" Desvio padrão Árvore de Decisão: {np.std(self.resultados)}")
        print(f" Quantidade de resultados: {self.count_resultados()}")
        print("------------------------------------------------------------")

class ListaResultadosSvm(ListaResultados):
    def __init__(self):
        super().__init__()

    def get_resultados(self):
        print("SVM - Resultados:", end=" ")
        for i in range(self.count_resultados()):
            print(f"{ round(self.resultados[i], 2) }", end=" ")
        print(f"\n Média SVM: {np.mean(self.resultados)}")
        print(f" Desvio padrão SVM: {np.std(self.resultados)}")
        print(f" Quantidade de resultados: {self.count_resultados()}")
        print("------------------------------------------------------------")

class ListaResultadosRegLog(ListaResultados):
    def __init__(self):
        super().__init__()

    def get_resultados(self):
        print("Regressão Logística - Resultados:", end=" ")
        for i in range(self.count_resultados()):
            print(f"{ round(self.resultados[i], 2) }", end=" ")
        print(f"\n Média Regressão Logística: {np.mean(self.resultados)}")
        print(f" Desvio padrão Regressão Logística: {np.std(self.resultados)}")
        print(f" Quantidade de resultados: {self.count_resultados()}")
        print("------------------------------------------------------------")

class ListaResultadosRandomForest(ListaResultados):
    def __init__(self):
        super().__init__()

    def get_resultados(self):
        print("Random Forest - Resultados:", end=" ")
        for i in range(self.count_resultados()):
            print(f"{ round(self.resultados[i], 2) }", end=" ")
        print(f"\n Média Random Forest: {np.mean(self.resultados)}")
        print(f" Desvio padrão Random Forest: {np.std(self.resultados)}")
        print(f" Quantidade de resultados: {self.count_resultados()}")
        print("------------------------------------------------------------")

class ListaResultadosGausianNb(ListaResultados):
    def __init__(self):
        super().__init__()
        
    def get_resultados(self):
        print("Gausian NB - Resultados:", end=" ")
        for i in range(self.count_resultados()):
            print(f"{ round(self.resultados[i], 2) }", end=" ")
        print(f"\n Média Gausian NB: {np.mean(self.resultados)}")
        print(f" Desvio padrão Gausian NB: {np.std(self.resultados)}")
        print(f" Quantidade de resultados: {self.count_resultados()}")
        print("------------------------------------------------------------")

class ResultadoFactory:
    def __init__(self) -> None:
        self.lista_knn: ListaResultadosKnn = ListaResultadosKnn()
        self.lista_arvore_decisao: ListaResultadosArvoreDecisao = ListaResultadosArvoreDecisao()
        self.lista_svm: ListaResultadosSvm = ListaResultadosSvm()
        self.lista_reg_log: ListaResultadosRegLog = ListaResultadosRegLog()
        self.lista_rad_for: ListaResultadosRandomForest = ListaResultadosRandomForest()
        self.lista_gau_nb: ListaResultadosGausianNb = ListaResultadosGausianNb()

    def get_lista_knn(self) -> ListaResultadosKnn:
        return self.lista_knn
    
    def get_lista_arvore_decisao(self) -> ListaResultadosArvoreDecisao:
        return self.lista_arvore_decisao
    
    def get_lista_svm(self) -> ListaResultadosSvm:
        return self.lista_svm
    
    def get_lista_reg_log(self) -> ListaResultadosRegLog:
        return self.lista_reg_log
    
    def get_lista_rad_for(self) -> ListaResultadosRandomForest:
        return self.lista_rad_for
    
    def get_lista_gau_nb(self) -> ListaResultadosGausianNb:
        return self.lista_gau_nb