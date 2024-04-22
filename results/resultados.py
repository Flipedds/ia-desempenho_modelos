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
        print(f"\nMédia acurácia: {round(np.mean([self.resultados[i][0] for i in range(self.count_resultados())]), 3) * 100}%")
        print(f"Média sensibilidade: {round(np.mean([self.resultados[i][1] for i in range(self.count_resultados())]), 3)* 100}%")
        print(f"Média especificidade: {round(np.mean([self.resultados[i][2] for i in range(self.count_resultados())]), 3)* 100}%")
        print(f"Média precisão: {round(np.mean([self.resultados[i][3] for i in range(self.count_resultados())]), 3)* 100}%")
        print(f"Média f1 score: {round(np.mean([self.resultados[i][4] for i in range(self.count_resultados())]),3)* 100}%")
        print("------------------------------------------------------------")
        print(f"Std acurácia: {round(np.std([self.resultados[i][0] for i in range(self.count_resultados())]),3) * 100}%")
        print(f"Std sensibilidade: {round(np.std([self.resultados[i][1] for i in range(self.count_resultados())]),3) * 100}%")
        print(f"Std especificidade: {round(np.std([self.resultados[i][2] for i in range(self.count_resultados())]),3) * 100}%")
        print(f"Std precisão: {round(np.std([self.resultados[i][3] for i in range(self.count_resultados())]),3)* 100 }%")
        print(f"Std f1 score: {round(np.std([self.resultados[i][4] for i in range(self.count_resultados())]),3)* 100 }%")
        print("------------------------------------------------------------")
        print(f" Quantidade de resultados: {self.count_resultados()}")
        print("------------------------------------------------------------")

class ListaResultadosArvoreDecisao(ListaResultados):
    def __init__(self):
        super().__init__()
    
    @override
    def get_resultados(self):
        print("Árvore de Decisão - Resultados:", end="")
        print(f"\nMédia acurácia: {round(np.mean([self.resultados[i][0] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Média sensibilidade: {round(np.mean([self.resultados[i][1] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Média especificidade: {round(np.mean([self.resultados[i][2] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Média precisão: {round(np.mean([self.resultados[i][3] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Média f1 score: {round(np.mean([self.resultados[i][4] for i in range(self.count_resultados())]),3)* 100}%")
        print("------------------------------------------------------------")
        print(f"Std acurácia: {round(np.std([self.resultados[i][0] for i in range(self.count_resultados())]),3) * 100}%")
        print(f"Std sensibilidade: {round(np.std([self.resultados[i][1] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Std especificidade: {round(np.std([self.resultados[i][2] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Std precisão: {round(np.std([self.resultados[i][3] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Std f1 score: {round(np.std([self.resultados[i][4] for i in range(self.count_resultados())]),3)* 100}%")

        print("------------------------------------------------------------")
        print(f" Quantidade de resultados: {self.count_resultados()}")
        print("------------------------------------------------------------")

class ListaResultadosSvm(ListaResultados):
    def __init__(self):
        super().__init__()
    
    @override
    def get_resultados(self):
        print("SVM - Resultados:", end=" ")
        print(f"\nMédia acurácia: {round(np.mean([self.resultados[i][0] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Média sensibilidade: {round(np.mean([self.resultados[i][1] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Média especificidade: {round(np.mean([self.resultados[i][2] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Média precisão: {round(np.mean([self.resultados[i][3] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Média f1 score: {round(np.mean([self.resultados[i][4] for i in range(self.count_resultados())]),3)* 100}%")
        print("------------------------------------------------------------")
        print(f"Std acurácia: {round(np.std([self.resultados[i][0] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Std sensibilidade: {round(np.std([self.resultados[i][1] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Std especificidade: {round(np.std([self.resultados[i][2] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Std precisão: {round(np.std([self.resultados[i][3] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Std f1 score: {round(np.std([self.resultados[i][4] for i in range(self.count_resultados())]),3)* 100}%")
        print("------------------------------------------------------------")
        print(f" Quantidade de resultados: {self.count_resultados()}")
        print("------------------------------------------------------------")

class ListaResultadosRegLog(ListaResultados):
    def __init__(self):
        super().__init__()

    @override
    def get_resultados(self):
        print("Regressão Logística - Resultados:", end=" ")
        print(f"\nMédia acurácia: {round(np.mean([self.resultados[i][0] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Média sensibilidade: {round(np.mean([self.resultados[i][1] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Média especificidade: {round(np.mean([self.resultados[i][2] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Média precisão: {round(np.mean([self.resultados[i][3] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Média f1 score: {round(np.mean([self.resultados[i][4] for i in range(self.count_resultados())]),3)* 100}%")
        print("------------------------------------------------------------")
        print(f"Std acurácia: {round(np.std([self.resultados[i][0] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Std sensibilidade: {round(np.std([self.resultados[i][1] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Std especificidade: {round(np.std([self.resultados[i][2] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Std precisão: {round(np.std([self.resultados[i][3] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Std f1 score: {round(np.std([self.resultados[i][4] for i in range(self.count_resultados())]),3)* 100}%")
        print("------------------------------------------------------------")
        print(f" Quantidade de resultados: {self.count_resultados()}")
        print("------------------------------------------------------------")

class ListaResultadosRandomForest(ListaResultados):
    def __init__(self):
        super().__init__()

    @override
    def get_resultados(self):
        print("Random Forest - Resultados:", end=" ")
        print(f"\nMédia acurácia: {round(np.mean([self.resultados[i][0] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Média sensibilidade: {round(np.mean([self.resultados[i][1] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Média especificidade: {round(np.mean([self.resultados[i][2] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Média precisão: {round(np.mean([self.resultados[i][3] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Média f1 score: {round(np.mean([self.resultados[i][4] for i in range(self.count_resultados())]),3)* 100}%")
        print("------------------------------------------------------------")
        print(f"Std acurácia: {round(np.std([self.resultados[i][0] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Std sensibilidade: {round(np.std([self.resultados[i][1] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Std especificidade: {round(np.std([self.resultados[i][2] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Std precisão: {round(np.std([self.resultados[i][3] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Std f1 score: {round(np.std([self.resultados[i][4] for i in range(self.count_resultados())]),3)* 100}%")
        print("------------------------------------------------------------")
        print(f" Quantidade de resultados: {self.count_resultados()}")
        print("------------------------------------------------------------")

class ListaResultadosGausianNb(ListaResultados):
    def __init__(self):
        super().__init__()

    @override    
    def get_resultados(self):
        print("Gausian NB - Resultados:", end=" ")
        print(f"\nMédia acurácia: {round(np.mean([self.resultados[i][0] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Média sensibilidade: {round(np.mean([self.resultados[i][1] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Média especificidade: {round(np.mean([self.resultados[i][2] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Média precisão: {round(np.mean([self.resultados[i][3] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Média f1 score: {round(np.mean([self.resultados[i][4] for i in range(self.count_resultados())]),3)* 100}%")
        print("------------------------------------------------------------")
        print(f"Std acurácia: {round(np.std([self.resultados[i][0] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Std sensibilidade: {round(np.std([self.resultados[i][1] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Std especificidade: {round(np.std([self.resultados[i][2] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Std precisão: {round(np.std([self.resultados[i][3] for i in range(self.count_resultados())]),3)* 100}%")
        print(f"Std f1 score: {round(np.std([self.resultados[i][4] for i in range(self.count_resultados())]),3)* 100}%")
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