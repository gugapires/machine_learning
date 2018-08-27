import pandas as pd

data_train = {
'PassageiroId': [1, 2, 3, 4, 5, 6],
'Sobreviveu': [0, 1, 1, 1, 0, 0],
'Pclasse': [1, 0, 0, 0, 1, 1],
'Embarcou': [1, 1, 1, 1, 1, 1],
'Estatus': ['rico', 'pobre', 'pobre', 'pobre', 'rico', 'rico']
}

#Criando o DataFrame train

df_train = pd.DataFrame(data_train, columns=['PassageiroId','Sobreviveu','Pclasse','Embarcou','Estatus'])

df_train.to_csv('train.csv')


data_teste = {
'PassageiroId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
'Sobreviveu': [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
'Pclasse': [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0],
'Embarcou': [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1]
}


#Criando o DataFrame teste

df_teste = pd.DataFrame(data_teste, columns=['PassageiroId','Sobreviveu','Pclasse','Embarcou'])

df_teste.to_csv('teste.csv')
