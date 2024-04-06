import datetime
import pandas as pd
import pandas as pd
import pandas as pd
from results.resultados import *
from train.treinamento import TreinamentoModelo
from clean.limpeza import limpar_dados_banana

def main():
    horario_atual: datetime.time = datetime.datetime.now().time()

    print("------------------------------------------------------------")
    print("Avaliação de desempenho de modelos: Árvore de decisão, SVM, Regressão Logística, Random Forest, Naive Bayes e KNN.")
    print("Após processo iterativo")
    print("------------------------------------------------------------")
    print(f"Horário de início: {horario_atual}")
    print("------------------------------------------------------------")

    try:
        dados_banana: pd.DataFrame = limpar_dados_banana(pd.read_csv('datasets/banana_quality.csv', sep=","))

        factory: ResultadoFactory = ResultadoFactory()

        for i in range(30):
            factory.get_lista_knn().add_resultado(TreinamentoModelo.treinar_modelo_knn(dados_banana))
            factory.get_lista_arvore_decisao().add_resultado(TreinamentoModelo.treinar_modelo_arvore_decisao(dados_banana))
            factory.get_lista_svm().add_resultado(TreinamentoModelo.treinar_modelo_svm(dados_banana))
            factory.get_lista_reg_log().add_resultado(TreinamentoModelo.treinar_modelo_regressao_logistica(dados_banana))
            factory.get_lista_rad_for().add_resultado(TreinamentoModelo.treinar_modelo_random_forest(dados_banana))
            factory.get_lista_gau_nb().add_resultado(TreinamentoModelo.treinar_modelo_gaussian_nb(dados_banana))

        factory.get_lista_knn().get_resultados()
        factory.get_lista_arvore_decisao().get_resultados()
        factory.get_lista_svm().get_resultados()
        factory.get_lista_reg_log().get_resultados()
        factory.get_lista_rad_for().get_resultados()
        factory.get_lista_gau_nb().get_resultados()

        horario_atual: datetime.time = datetime.datetime.now().time()
        print(f"Horário Fim: {horario_atual}")
        print("------------------------------------------------------------")
    except Exception as e:
        print(f"Erro: {e}")
        print("Ocorreu um erro durante a execução do programa !!!!!!")

if __name__ == "__main__":
    main()