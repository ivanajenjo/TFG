from sklearn.feature_selection import mutual_info_regression, mutual_info_classif, SelectKBest
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import normalized_mutual_info_score
from info_gain import info_gain
import pandas as pd
import numpy as np


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


def calcularMi_ManualInfo_gain(variable, df):
    y = df[variable].values
    x = df.loc[:, df.columns != variable]
    resultado = []
    for (columnName, columnData) in x.iteritems():
        #print('Colunm Name : ', columnName)
        aux = info_gain.intrinsic_value(y, columnData)
        resultado.append(aux)
    resultado = pd.Series(resultado)
    resultado.index = x.columns
    resultado = resultado.sort_values(ascending=False)
    return resultado


def calcularMI(variable, df):
    variables = ['Industry Sector', 'Application Group', 'Development Type', 'Development Platform', 'Language Type', 'Primary Programming Language',
                 'Functional Size', 'Adjusted Function Points', 'Project Elapsed Time', '1st Data Base System', 'Used Methodology']
    X = df.loc[:, variables]
    y = df.loc[:, variable].values
    mi = mutual_info_regression(X, y, discrete_features=[
                                True, True, True, True, True, True, False, False, False, True, True], n_neighbors=1)
    mi = pd.Series(mi)
    mi.index = X.columns
    mi = mi.sort_values(ascending=False)
    return mi


def selectKBestMi(df):
    fs = SelectKBest(score_func=mutual_info_regression, k='all')
    variables = ['Industry Sector', 'Application Group', 'Development Type', 'Development Platform', 'Language Type', 'Primary Programming Language',
                 'Functional Size', 'Adjusted Function Points', 'Project Elapsed Time', '1st Data Base System', 'Used Methodology']
    X = df.loc[:, variables]
    y = df.loc[:, 'Normalised Work Effort Level 1'].values
    fs.fit(X, y)
    fs.index = X.columns
    return fs


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


def calcular_mRMR(variable, df):
    infogain = calcularMi_ManualInfo_gain(variable, df)
    ordenadas = infogain.keys().values.tolist()
    mrmr = []
    seleccionadas = []
    iteraciones = len(ordenadas)
    for i in range(iteraciones):
        seleccionadas.append(ordenadas[0])
        ordenadas.pop(0)
        coefs = []
        for prueba in ordenadas:
            I = calc_MI_scikit(df[variable], df[prueba])
            info_parcial = []
            for seleccionada in seleccionadas:
                I_parcial = calc_MI_scikit(df[seleccionada], df[prueba])
                info_parcial.append(I_parcial)
            coef = I - np.mean(info_parcial)
            coefs.append(coef)
        indices_ordenados = coefs.sort(reverse=True)
        mrmr.append(coefs)
    print(mrmr)
    



""" def calcular_mRMR(variable, df):
    ganancias = calcularMi_ManualInfo_gain(variable, df)
    resultado = {}
    i = 0
    for index, value in ganancias.items():
        if i == 0:
            resultado[index] = value
        i += 1
        #print(f"Index : {index}, Value : {value}")
    resultado = pd.Series(resultado)
    return resultado """
