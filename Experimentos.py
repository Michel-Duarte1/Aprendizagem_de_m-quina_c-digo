import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score
import zipfile
import io

#Número de experimentos
num_experimentos = 10

#Caminho para o arquivo zip
caminho_zip_dataset = 'C:/Users/Douglas/Desktop/Michel/adult.zip'

#Extração do conteúdo do arquivo zip para um DataFrame
with zipfile.ZipFile(caminho_zip_dataset, 'r') as zip_ref:
    zip_contents = zip_ref.namelist()

    for file_name in zip_contents:
        if "adult.data" in file_name:
            with zip_ref.open(file_name) as file:
                dados = pd.read_csv(io.StringIO(file.read().decode('utf-8')), header=None)
                break


nomes_colunas = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                 'hours-per-week', 'native-country', 'income']

dados.columns = nomes_colunas

#Converter as variáveis categóricas usando codificação one-hot (dummy encoding)
dados = pd.get_dummies(dados, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'])


print('Nomes das Colunas:', dados.columns)


acuracias_arvore = []
acuracias_rede = []

for i in range(num_experimentos):
    #Divisão estratificada 70%-30%
    X_train, X_test, y_train, y_test = train_test_split(dados.drop('income', axis=1), dados['income'], test_size=0.3, stratify=dados['income'], random_state=i)

    #Treinar e avaliar Árvore de Decisão
    arvore_decisao = DecisionTreeClassifier(random_state=i)
    arvore_decisao.fit(X_train, y_train)
    predicoes_arvore = arvore_decisao.predict(X_test)
    acuracia_total_arvore = accuracy_score(y_test, predicoes_arvore)
    acuracias_arvore.append(acuracia_total_arvore)

    print(f'Experimento {i+1} - Árvore de Decisão:')
    print('Matriz de Confusão:')
    print(confusion_matrix(y_test, predicoes_arvore))
    print(f'Recall: {recall_score(y_test, predicoes_arvore, pos_label=" >50K"):.4f}')
    print(f'Precisão: {precision_score(y_test, predicoes_arvore, pos_label=" >50K"):.4f}')
    print(f'Acurácia Total: {acuracia_total_arvore:.4f}')
    print('-' * 40)

    #Treinar e avaliar Rede Neural
    rede_neural = MLPClassifier(random_state=i)
    rede_neural.fit(X_train, y_train)
    predicoes_rede = rede_neural.predict(X_test)
    acuracia_total_rede = accuracy_score(y_test, predicoes_rede)
    acuracias_rede.append(acuracia_total_rede)

    print(f'Experimento {i+1} - Rede Neural:')
    print('Matriz de Confusão:')
    print(confusion_matrix(y_test, predicoes_rede))
    print(f'Recall: {recall_score(y_test, predicoes_rede, pos_label=" >50K"):.4f}')
    print(f'Precisão: {precision_score(y_test, predicoes_rede, pos_label=" >50K", zero_division=1):.4f}')

    print(f'Acurácia Total: {acuracia_total_rede:.4f}')
    print('-' * 40)

#Cálculo da média e desvio padrão da acurácia total para cada algoritmo
media_acuracia_arvore = np.mean(acuracias_arvore)
desvio_padrao_acuracia_arvore = np.std(acuracias_arvore)

media_acuracia_rede = np.mean(acuracias_rede)
desvio_padrao_acuracia_rede = np.std(acuracias_rede)

print('Resumo das Métricas:')
print(f'Média Acurácia Árvore de Decisão: {media_acuracia_arvore:.4f}')
print(f'Desvio Padrão Acurácia Árvore de Decisão: {desvio_padrao_acuracia_arvore:.4f}')
print(f'Média Acurácia Rede Neural: {media_acuracia_rede:.4f}')
print(f'Desvio Padrão Acurácia Rede Neural: {desvio_padrao_acuracia_rede:.4f}')
