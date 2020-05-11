from sklearn.feature_selection import mutual_info_regression, mutual_info_classif, SelectKBest
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import normalized_mutual_info_score
from info_gain import info_gain
import pandas as pd
import numpy as np
import operator


def calc_MI_scikit(x, y):
    mi = normalized_mutual_info_score(y, x)
    return mi


def calcularMi_Manual(variable, df):
    y = df[variable].values
    x = df.loc[:, df.columns != variable]
    resultado = []
    for (columnName, columnData) in x.iteritems():
        #print('Colunm Name : ', columnName)
        aux = calc_MI_scikit(y, columnData)
        resultado.append(aux)
    resultado = pd.Series(resultado)
    resultado.index = x.columns
    resultado = resultado.sort_values(ascending=False)
    return resultado


def recodeDataframe(dataframe):
    resultado = dataframe
    categorical_feature_mask = resultado.dtypes == object
    categorical_cols = resultado.columns[categorical_feature_mask].tolist()
    labelencoder = LabelEncoder()
    resultado[categorical_cols] = resultado[categorical_cols].apply(
        lambda col: labelencoder.fit_transform(col))
    #X_en= dataframe['Primary Programming Language'].values
    #X_en = labelencoder.fit_transform(X_en)
    #resultado['Primary Programming Language'] = X_en
    return resultado

def calcular_mRMRV2(variable, df):
    seleccionadas = []
    mrmr = {}
    info_gain = calcularMi_Manual(variable, df)
    ordenadas = info_gain.keys().values.tolist()
    seleccionadas.append(ordenadas[0])
    mrmr[ordenadas[0]] = info_gain.iloc[0]
    ordenadas.pop(0)
    iteraciones = len(ordenadas)
    for i in range(iteraciones):
        coefs = {}
        for prueba in ordenadas:
            info_parcial = {}
            Info_prueba = calc_MI_scikit(df[variable], df[prueba])
            for seleccionada in seleccionadas:
                I_parcial = calc_MI_scikit(df[prueba], df[seleccionada])
                info_parcial[seleccionada] = I_parcial 
            coef_parcial = Info_prueba - np.mean(list(info_parcial.values()))
            coefs[prueba] = coef_parcial
        coefs_ordenados = sorted(coefs.items(), key=operator.itemgetter(1), reverse=True)
        print(coefs)
        mrmr[coefs_ordenados[0][0]] = coefs[coefs_ordenados[0][0]]
        ordenadas.remove(coefs_ordenados[0][0])
        seleccionadas.append(coefs_ordenados[0][0])
    resultado = pd.Series(mrmr)
    return resultado