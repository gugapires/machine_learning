import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

treinar = pd.read_csv("mesclado_train.csv")

df = treinar.fillna(0)
data = df.astype(str)

carac_colunas = ['CO_GRAU_ACADEMICO',
'IN_MOBILIDADE_ACADEMICA',
'CO_COR_RACA_ALUNO',
'CO_MODALIDADE_ENSINO',
'CO_CATEGORIA_ADMINISTRATIVA',
'CO_ORGANIZACAO_ACADEMICA',
'CO_TURNO_ALUNO',
'NU_ANO_ALUNO_NASC',
'NU_MES_ALUNO_NASC',
'NU_DIA_ALUNO_NASC',
'CO_UF_NASCIMENTO',
'IN_FINANC_ESTUDANTIL',
'IN_ING_CONVENIO_PECG']

'''
CO_CATEGORIA_ADMINISTRATIVA,    usado
CO_ORGANIZACAO_ACADEMICA,       usado
CO_TURNO_ALUNO,                 usado
CO_GRAU_ACADEMICO,              usado
CO_MODALIDADE_ENSINO,           usado
CO_COR_RACA_ALUNO,              usado
NU_ANO_ALUNO_NASC,              usado
NU_MES_ALUNO_NASC,              usado
NU_DIA_ALUNO_NASC,              usado
CO_UF_NASCIMENTO,               usado
DT_INGRESSO_CURSO,              defeito
IN_FINANC_ESTUDANTIL,           usado
IN_ING_CONVENIO_PECG,           usado
IN_SEXO_ALUNO,                  defeito
IN_MOBILIDADE_ACADEMICA,        usado
NO_OCDE_AREA_GERAL              defeito
'''


daditos = data.loc[:, carac_colunas]
dadix = daditos.fillna(0)
x = dadix.dropna()

y = treinar.NO_OCDE_AREA_GERAL.astype(str)

logreg = LogisticRegression()
calibrar_dados = logreg.fit(x, y)

testar = pd.read_csv("mesclado_teste.csv")
df_testar = testar.fillna(0)
data_testar = df_testar.astype(str)

novo_x = data_testar.loc[:, carac_colunas]
nova_predicao_de_classe = logreg.predict(novo_x)
teste_de_passageirosId = data_testar.index

final = pd.DataFrame({'index':data_testar.index, 'NO_OCDE_AREA_GERAL':nova_predicao_de_classe})
acabou = final.set_index('index').to_csv('resposta.csv')
