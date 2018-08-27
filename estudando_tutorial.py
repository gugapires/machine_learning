import pandas as pd
import numpy as np
#from sklearn.utils import shuffle
#import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

treinar = pd.read_csv("train.csv")
'''
tabela_head = treinar.head()
print(tabela_head.shape)
'''

#with pd.option_context('mode.use_inf_as_null', True):
#    df = treinar.dropna(subset=['CO_GRAU_ACADEMICO', 'IN_MOBILIDADE_ACADEMICA'])

df = treinar.fillna(0)
data = df.astype(str)

# estou detectando alguem de serviços, logo preciso da caracteristicas
#dessa pessoa

# com esse funciona que belezal
#carac_colunas = ['CO_GRAU_ACADEMICO', 'IN_MOBILIDADE_ACADEMICA']


carac_colunas = ['CO_GRAU_ACADEMICO',
'IN_MOBILIDADE_ACADEMICA',
'CO_COR_RACA_ALUNO',
'CO_MODALIDADE_ENSINO',
'CO_COR_RACA_ALUNO']
#'SEXO_ALUNO']



daditos = data.loc[:, carac_colunas]
dadix = daditos.fillna(0)
x = dadix.dropna()

#x_clean = x.replace([np.inf, -np.inf], np.nan)
#limpo = x.dropna()
#limpo.reset_index()
#df[cat] = le.fit_transform(df[cat].astype(str))

y = treinar.NO_OCDE_AREA_GERAL.astype(str)

logreg = LogisticRegression()
calibrar_dados = logreg.fit(x, y)
#print(calibrar_dados)


testar = pd.read_csv("teste.csv")

df_testar = testar.fillna(0)
data_testar = df_testar.astype(str)


novo_x = data_testar.loc[:, carac_colunas]

nova_predicao_de_classe = logreg.predict(novo_x)

teste_de_passageirosId = data_testar.CO_CATEGORIA_ADMINISTRATIVA

final = pd.DataFrame({'CO_CATEGORIA_ADMINISTRATIVA':data_testar.CO_CATEGORIA_ADMINISTRATIVA, 'NO_OCDE_AREA_GERAL':nova_predicao_de_classe})
acabou = final.set_index('CO_CATEGORIA_ADMINISTRATIVA').to_csv('resposta_dada_pelo_algoritmo_sobre_mesclado.csv')

#x.hist()
#plt.show()



# No pessoal de engenharia tem mais homens que mulheres
# grau academico é maior
# modelidade de ensino só pode ser presencial
# provavelmente todo mundo é branco

# CO_GRAU_ACADEMICO 4.0    17729 --> ok
# CO_MODALIDADE_ENSINO 1.0  19216
# CO_COR_RACA_ALUNO 1.0  14919 --> ok
# SEXO_ALUNO 0.0    13361 --> nan value
# IN_MOBILIDADE_ACADEMICA 0.0    17679
#'CO_COR_RACA_ALUNO',
