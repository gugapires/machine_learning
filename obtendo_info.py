import pandas as pd
############################# trabalhando com Engenharia ###############################

#comeso = df.loc[(df["NO_OCDE_AREA_GERAL"]=="Engenharia, produção e construção")]
#eng = comeso.to_csv('eng.csv')

leitura_eng = pd.read_csv("train_eng.csv")

data_eng = pd.DataFrame(leitura_eng)

dicionario_eng = ["CO_CATEGORIA_ADMINISTRATIVA","CO_ORGANIZACAO_ACADEMICA","CO_TURNO_ALUNO","CO_GRAU_ACADEMICO","CO_MODALIDADE_ENSINO",
"CO_COR_RACA_ALUNO","NU_ANO_ALUNO_NASC","NU_MES_ALUNO_NASC","NU_DIA_ALUNO_NASC","CO_UF_NASCIMENTO","DT_INGRESSO_CURSO","IN_FINANC_ESTUDANTIL",
"IN_ING_CONVENIO_PECG","IN_SEXO_ALUNO","IN_MOBILIDADE_ACADEMICA"]

k = 0
while k < len(dicionario_eng):
    want_eng = data_eng.loc[: , dicionario_eng[k]]
    k = k + 1

    agrupando_cada_eng = want_eng.value_counts()

    print("---------inicio--------------")
    print(agrupando_cada_eng,"\n\n")

    print(agrupando_cada_eng.describe())
    print("----------fim--------------- \n\n\n")

#####em sexos o numero 1 representa mulher e 0 homem


#CO_GRAU_ACADEMICO 4.0    17729
#CO_MODALIDADE_ENSINO 1.0  19216
#CO_COR_RACA_ALUNO 1.0  14919
#SEXO_ALUNO 0.0    13361
