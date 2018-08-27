import pandas as pd
from sklearn.linear_model import LogisticRegression

treinar = pd.read_csv("http://bit.ly/kaggletrain")

tabela_head = treinar.head()
# pega a cabeça da tabela

print(tabela_head)

# parch deve ser queimar, tostar
# pclass deve ser primeira classe

caracteristica_das_colunas = ['Pclass', 'Parch']
# aqui eu pego o nome das tabela

x = treinar.loc[:, caracteristica_das_colunas]
# aqui eu quero treinar todas as linhas que pertencem as estas caracteristicas
# (nome das tabelas)

tamanho = x.shape
print("tamanho (coluna x linha)", tamanho)

# agora vamos criar um fator de resposta

y = treinar.Survived
# pq ele pegou survived (caracterisca da tabela)

tamanho_y = y.shape
print(tamanho_y)

# um modelo de classificação

logreg = LogisticRegression()
ajuste = logreg.fit(x, y)
# acima --> ajustar o modelo aos meus dados de treinamento

# Agora precisamos ler nosso dado de testes, é onde faremos
# as predicções. Ou seja, é outro arquivo. A maquina aprendeu
# com aquele passado acima, agora vai "resolver" este para nós.

testar = pd.read_csv('http://bit.ly/kaggletest')

testar.head()
# você percebera que ele não terá a caracteristica, a coluna
# survived, pois é essa caracteristica que treinamos
# para fazer as predicções

# agora vamos criar um novo x, a partir do teste dos dados

novo_x = testar.loc[:, caracteristica_das_colunas]

tamanho_do_novo_x = novo_x.shape

print(tamanho_do_novo_x)

# melhorar nossa predicção em 418 vamos fazer mais uma linha de skicit-learn

nova_predicao_de_classe = logreg.predict(novo_x)

teste_de_passageirosId = testar.PassengerId

print(nova_predicao_de_classe)

pd.DataFrame({'PassengerId': testar.PassengerId, 'Survived': nova_predicao_de_classe}).set_index('PassengerId').to_csv('sub.csv')

# acima estou dizendo que tenho duas colunas, passenderid e survived
# e o passenderid será testado na nova_predicao_de_classe,
# dai o panda vai fazer as relações

# atenção
# tenho que atentar que passenderid será a primeira coluna, pois
# as colunas vem desordenadas, logo preciso ter noção de uma
# caracteristica unica que o passenderid tenha para facilitar
# sua identificação, ou simplesmente adicionando ".set_index('PassenderId')",
# como esta ali em cima

# salvar um objeto python como um dataframe para o disco
# com dataframes usaremos algo chamado pickle
# pois os objetos são chamados de pickle objetos

treinar.to_pickle('train.pkl')
# estamos salvando o dataframe, para que possa ser carregado
# para usar em outro computador
# e para ler com o panda

pd.read_pickle('train.pkl')
