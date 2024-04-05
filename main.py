import datetime
import pandas as pd
import pandas as pd
import pandas as pd
from results.resultados import *
from train.treinamento import TreinamentoModelo
from clean.limpeza import limpar_dados_banana

def main():
    horario_atual = datetime.datetime.now().time()

    print("------------------------------------------------------------")
    print("Avaliação de desempenho de modelos Árvore de decisão, SVM, Regressão Logística, Random Forest, Naive Bayes e KNN.")
    print("após processo iterativo")
    print("------------------------------------------------------------")
    print(f"Horário de início: {horario_atual}")
    print("------------------------------------------------------------")

    dados_banana: pd.DataFrame = pd.read_csv('datasets/banana_quality.csv', sep=",")
    dados_banana = limpar_dados_banana(dados_banana)

    lista_knn: ListaResultadosKnn = ListaResultadosKnn();
    lista_arvore_decisao: ListaResultadosArvoreDecisao = ListaResultadosArvoreDecisao();
    lista_svm: ListaResultadosSvm = ListaResultadosSvm();
    lista_reg_log: ListaResultadosRegLog = ListaResultadosRegLog();
    lista_rad_for: ListaResultadosRandomForest = ListaResultadosRandomForest();
    lista_gau_nb: ListaResultadosGausianNb = ListaResultadosGausianNb();

    try:
        for i in range(30):
            lista_knn.add_resultado(TreinamentoModelo.treinar_modelo_knn(dados_banana))
            lista_arvore_decisao.add_resultado(TreinamentoModelo.treinar_modelo_arvore_decisao(dados_banana))
            lista_svm.add_resultado(TreinamentoModelo.treinar_modelo_svm(dados_banana))
            lista_reg_log.add_resultado(TreinamentoModelo.treinar_modelo_regressao_logistica(dados_banana))
            lista_rad_for.add_resultado(TreinamentoModelo.treinar_modelo_random_forest(dados_banana))
            lista_gau_nb.add_resultado(TreinamentoModelo.treinar_modelo_gaussian_nb(dados_banana))

        lista_knn.get_resultados()
        lista_arvore_decisao.get_resultados()
        lista_svm.get_resultados()
        lista_reg_log.get_resultados()
        lista_rad_for.get_resultados()
        lista_gau_nb.get_resultados()

        horario_atual = datetime.datetime.now().time()
        print(f"Horário Fim: {horario_atual}")
        print("------------------------------------------------------------")
    except Exception as e:
        print(f"Erro: {e}")
        print("Ocorreu um erro durante a execução do programa !!!!!!")

if __name__ == "__main__":
    main()