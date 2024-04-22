from typing import override
import numpy as np
class ListaResultados:
    def __init__(self):
        self.resultados: list = []

    def add_resultado(self, resultados: list):
        self.resultados.append(resultados)

    def count_resultados(self) -> int:
        return len(self.resultados)
    
    def get_resultados(self):
        pass

class ListaResultadosKnn(ListaResultados):
    def __init__(self):
        super().__init__()

    @override
    def get_resultados(self):
        print("KNN - Resultados:", end=" ")
        print(f"\nMédia Knn acurrácia: {np.mean([self.resultados[i][0] for i in range(self.count_resultados())])}")
        print(f"Std Knn acurrácia: {np.std([self.resultados[i][0] for i in range(self.count_resultados())])}")

        print(f"Média Knn recall: {np.mean([self.resultados[i][1] for i in range(self.count_resultados())])}")
        print(f"Std Knn recall: {np.std([self.resultados[i][1] for i in range(self.count_resultados())])}")

        print(f"Média Knn especify: {np.mean([self.resultados[i][2] for i in range(self.count_resultados())])}")
        print(f"Std Knn especify: {np.std([self.resultados[i][2] for i in range(self.count_resultados())])}")

        print(f"Média Knn precision: {np.mean([self.resultados[i][3] for i in range(self.count_resultados())])}")
        print(f"Std Knn precision: {np.std([self.resultados[i][3] for i in range(self.count_resultados())])}")

        print(f"Média Knn f1: {np.mean([self.resultados[i][4] for i in range(self.count_resultados())])}")
        print(f"Std Knn f1: {np.std([self.resultados[i][4] for i in range(self.count_resultados())])}")
        print(f" Quantidade de resultados: {self.count_resultados()}")
        print("------------------------------------------------------------")

class ListaResultadosArvoreDecisao(ListaResultados):
    def __init__(self):
        super().__init__()
    
    @override
    def get_resultados(self):
        print("Árvore de Decisão - Resultados:", end="")
        print(f"\nMédia Decision Tree acurrácia: {np.mean([self.resultados[i][0] for i in range(self.count_resultados())])}")
        print(f"Std Decision Tree acurrácia: {np.std([self.resultados[i][0] for i in range(self.count_resultados())])}")

        print(f"Média Decision Tree recall: {np.mean([self.resultados[i][1] for i in range(self.count_resultados())])}")
        print(f"Std Decision Tree recall: {np.std([self.resultados[i][1] for i in range(self.count_resultados())])}")

        print(f"Média Decision Tree especify: {np.mean([self.resultados[i][2] for i in range(self.count_resultados())])}")
        print(f"Std Decision Tree especify: {np.std([self.resultados[i][2] for i in range(self.count_resultados())])}")

        print(f"Média Decision Tree precision: {np.mean([self.resultados[i][3] for i in range(self.count_resultados())])}")
        print(f"Std Decision Tree precision: {np.std([self.resultados[i][3] for i in range(self.count_resultados())])}")

        print(f"Média Decision Tree f1: {np.mean([self.resultados[i][4] for i in range(self.count_resultados())])}")
        print(f"Std Decision Tree f1: {np.std([self.resultados[i][4] for i in range(self.count_resultados())])}")

        print(f" Quantidade de resultados: {self.count_resultados()}")
        print("------------------------------------------------------------")

class ListaResultadosSvm(ListaResultados):
    def __init__(self):
        super().__init__()
    
    @override
    def get_resultados(self):
        print("SVM - Resultados:", end=" ")
        print(f"\nMédia SVM acurrácia: {np.mean([self.resultados[i][0] for i in range(self.count_resultados())])}")
        print(f"Std SVM acurrácia: {np.std([self.resultados[i][0] for i in range(self.count_resultados())])}")

        print(f"Média SVM recall: {np.mean([self.resultados[i][1] for i in range(self.count_resultados())])}")
        print(f"Std SVM recall: {np.std([self.resultados[i][1] for i in range(self.count_resultados())])}")

        print(f"Média SVM especify: {np.mean([self.resultados[i][2] for i in range(self.count_resultados())])}")
        print(f"Std SVM especify: {np.std([self.resultados[i][2] for i in range(self.count_resultados())])}")

        print(f"Média SVM precision: {np.mean([self.resultados[i][3] for i in range(self.count_resultados())])}")
        print(f"Std SVM precision: {np.std([self.resultados[i][3] for i in range(self.count_resultados())])}")

        print(f"Média SVM f1: {np.mean([self.resultados[i][4] for i in range(self.count_resultados())])}")
        print(f"Std SVM f1: {np.std([self.resultados[i][4] for i in range(self.count_resultados())])}")

        print(f" Quantidade de resultados: {self.count_resultados()}")
        print("------------------------------------------------------------")

class ListaResultadosRegLog(ListaResultados):
    def __init__(self):
        super().__init__()

    @override
    def get_resultados(self):
        print("Regressão Logística - Resultados:", end=" ")
        print(f"\nMédia Regressão Logística acurrácia: {np.mean([self.resultados[i][0] for i in range(self.count_resultados())])}")
        print(f"Std Regressão Logística acurrácia: {np.std([self.resultados[i][0] for i in range(self.count_resultados())])}")

        print(f"Média Regressão Logística recall: {np.mean([self.resultados[i][1] for i in range(self.count_resultados())])}")
        print(f"Std Regressão Logística recall: {np.std([self.resultados[i][1] for i in range(self.count_resultados())])}")

        print(f"Média Regressão Logística especify: {np.mean([self.resultados[i][2] for i in range(self.count_resultados())])}")
        print(f"Std Regressão Logística especify: {np.std([self.resultados[i][2] for i in range(self.count_resultados())])}")

        print(f"Média Regressão Logística precision: {np.mean([self.resultados[i][3] for i in range(self.count_resultados())])}")
        print(f"Std Regressão Logística precision: {np.std([self.resultados[i][3] for i in range(self.count_resultados())])}")

        print(f"Média Regressão Logística f1: {np.mean([self.resultados[i][4] for i in range(self.count_resultados())])}")
        print(f"Std Regressão Logística f1: {np.std([self.resultados[i][4] for i in range(self.count_resultados())])}")

        print(f" Quantidade de resultados: {self.count_resultados()}")
        print("------------------------------------------------------------")

class ListaResultadosRandomForest(ListaResultados):
    def __init__(self):
        super().__init__()

    @override
    def get_resultados(self):
        print("Random Forest - Resultados:", end=" ")
        print(f"\nMédia Random Forest acurrácia: {np.mean([self.resultados[i][0] for i in range(self.count_resultados())])}")
        print(f"Std Random Forest acurrácia: {np.std([self.resultados[i][0] for i in range(self.count_resultados())])}")

        print(f"Média Random Forest recall: {np.mean([self.resultados[i][1] for i in range(self.count_resultados())])}")
        print(f"Std Random Forest recall: {np.std([self.resultados[i][1] for i in range(self.count_resultados())])}")

        print(f"Média Random Forest especify: {np.mean([self.resultados[i][2] for i in range(self.count_resultados())])}")
        print(f"Std Random Forest especify: {np.std([self.resultados[i][2] for i in range(self.count_resultados())])}")

        print(f"Média Random Forest precision: {np.mean([self.resultados[i][3] for i in range(self.count_resultados())])}")
        print(f"Std Random Forest precision: {np.std([self.resultados[i][3] for i in range(self.count_resultados())])}")

        print(f"Média Random Forest f1: {np.mean([self.resultados[i][4] for i in range(self.count_resultados())])}")
        print(f"Std Random Forest f1: {np.std([self.resultados[i][4] for i in range(self.count_resultados())])}")

        print(f" Quantidade de resultados: {self.count_resultados()}")
        print("------------------------------------------------------------")

class ListaResultadosGausianNb(ListaResultados):
    def __init__(self):
        super().__init__()

    @override    
    def get_resultados(self):
        print("Gausian NB - Resultados:", end=" ")
        print(f"\nMédia Gausian NB acurrácia: {np.mean([self.resultados[i][0] for i in range(self.count_resultados())])}")
        print(f"Std Gausian NB acurrácia: {np.std([self.resultados[i][0] for i in range(self.count_resultados())])}")

        print(f"Média Gausian NB recall: {np.mean([self.resultados[i][1] for i in range(self.count_resultados())])}")
        print(f"Std Gausian NB recall: {np.std([self.resultados[i][1] for i in range(self.count_resultados())])}")

        print(f"Média Gausian NB especify: {np.mean([self.resultados[i][2] for i in range(self.count_resultados())])}")
        print(f"Std Gausian NB especify: {np.std([self.resultados[i][2] for i in range(self.count_resultados())])}")

        print(f"Média Gausian NB precision: {np.mean([self.resultados[i][3] for i in range(self.count_resultados())])}")
        print(f"Std Gausian NB precision: {np.std([self.resultados[i][3] for i in range(self.count_resultados())])}")

        print(f"Média Gausian NB f1: {np.mean([self.resultados[i][4] for i in range(self.count_resultados())])}")
        print(f"Std Gausian NB f1: {np.std([self.resultados[i][4] for i in range(self.count_resultados())])}")
        
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