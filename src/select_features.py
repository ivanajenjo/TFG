import operator

import numpy as np
import pandas as pd
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from info_gain import info_gain
from rpy2.robjects import Formula, pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
from sklearn.feature_selection import (SelectKBest, mutual_info_classif,
                                       mutual_info_regression)
from sklearn.metrics import (adjusted_mutual_info_score, mutual_info_score,
                             normalized_mutual_info_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from fancyimpute import KNN

utils = None
information_gain = None


def setupR_enviroment():
    global utils
    global information_gain
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1)
    packages = ('FSelector')
    utils.install_packages(StrVector(packages))
    FSelector = importr("FSelector")
    information_gain = FSelector.information_gain


def calcularMi_R(variable, df):
    if utils == None:
        setupR_enviroment()
    pandas2ri.activate()
    r_df = ro.conversion.py2rpy(df)
    #fmla = Formula('Normalised_Work_Effort_Level_1~.')
    fmla = Formula(str(variable + '~.'))
    resultado = information_gain(fmla, r_df)
    resultado = resultado.sort_values('attr_importance', ascending=False)
    return resultado


def calcularMi_R_2V(variable1, variable2, df):
    if utils == None:
        setupR_enviroment()
    pandas2ri.activate()
    r_df = ro.conversion.py2rpy(df)
    fmla = Formula(str(variable1 + '~' + variable2))
    resultado = information_gain(fmla, r_df)
    resultado = resultado.iloc[0].values[0]
    return resultado


def calcular_mRMR_R(variable, df):
    seleccionadas = []
    mrmr = {}
    info_gain = calcularMi_R(variable, df)
    ordenadas = info_gain.index.tolist()
    seleccionadas.append(ordenadas[0])
    mrmr[ordenadas[0]] = info_gain.iloc[0].values[0]
    ordenadas.pop(0)
    iteraciones = len(ordenadas)
    for i in range(iteraciones):
        coefs = {}
        for prueba in ordenadas:
            info_parcial = {}
            Info_prueba = calcularMi_R_2V(variable, prueba, df)
            for seleccionada in seleccionadas:
                I_parcial = calcularMi_R_2V(seleccionada, prueba, df)
                info_parcial[seleccionada] = I_parcial
                #print('Prueba ' + str(prueba) + 'Seleccionada ' + str(seleccionada) + 'Info ' + str(I_parcial))
            coef_parcial = Info_prueba - np.mean(list(info_parcial.values()))
            coefs[prueba] = coef_parcial
        coefs_ordenados = sorted(
            coefs.items(), key=operator.itemgetter(1), reverse=True)
        print(coefs)
        mrmr[coefs_ordenados[0][0]] = coefs[coefs_ordenados[0][0]]
        ordenadas.remove(coefs_ordenados[0][0])
        seleccionadas.append(coefs_ordenados[0][0])
    resultado = pd.Series(mrmr)
    return resultado


def recodeDataframe_R(df):
    df.columns = ['Industry_Sector', 'Application_Group', 'Development_Type',
                  'Development_Platform', 'Language_Type', 'Primary_Programming_Language',
                  'Functional_Size', 'Adjusted_Function_Points',
                  'Normalised_Work_Effort_Level_1', 'Project_Elapsed_Time',
                  'First_Data_Base_System', 'Used_Methodology']
    return df


def calc_MI_scikit(x, y):
    mi = normalized_mutual_info_score(y, x)
    return mi


def calcularMi_Manual(variable, df):
    y = df[variable].values
    x = df.loc[:, df.columns != variable]
    #x = df
    resultado = []
    for (columnName, columnData) in x.iteritems():
        #print('Colunm Name : ', columnName)
        aux = calc_MI_scikit(y, columnData)
        resultado.append(aux)
    resultado = pd.Series(resultado)
    resultado.index = x.columns
    resultado = resultado.sort_values(ascending=False)
    return resultado


def calcular_mi(variable, df):
    variables = ['Industry Sector', 'Application Group', 'Development Type', 'Development Platform', 'Language Type', 'Primary Programming Language',
                 'Functional Size', 'Adjusted Function Points', 'Project Elapsed Time', '1st Data Base System', 'Used Methodology']
    X = df.loc[:, variables]
    y = df.loc[:, variable].values
    mi = mutual_info_regression(X, y, n_neighbors=1)
    mi = pd.Series(mi)
    mi.index = X.columns
    mi = mi.sort_values(ascending=False)
    return mi


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
        coefs_ordenados = sorted(
            coefs.items(), key=operator.itemgetter(1), reverse=True)
        print(coefs)
        mrmr[coefs_ordenados[0][0]] = coefs[coefs_ordenados[0][0]]
        ordenadas.remove(coefs_ordenados[0][0])
        seleccionadas.append(coefs_ordenados[0][0])
    resultado = pd.Series(mrmr)
    return resultado


def calcular_mmre(variable, df, k=5):
    total = len(df)
    resultado = pd.DataFrame(columns=['Valor Original', 'Valor Imputado'])
    for i in range(total):
        df_test = df.copy(deep=True)
        dato_original = df_test[variable].iloc[i]
        df_test[variable].iloc[i] = np.nan
        imputer = KNNImputer(n_neighbors=k)
        df_test = imputer.fit_transform(df_test)
        dato_imputado = df_test[i ,8]
        #print(dato_original)
        #print(dato_imputado)
        resultado = resultado.append({'Valor Original':dato_original, 'Valor Imputado':dato_imputado}, ignore_index=True)
        mmre = (1/total)*sum(abs(resultado['Valor Original'] - resultado['Valor Imputado'])/resultado['Valor Original'])
    return mmre


def calcular_mmre_v2(variable, df, k=5):
    total = len(df)
    resultado = pd.DataFrame(columns=['Valor Original', 'Valor Imputado'])
    for i in range(total):  
        df_test = df.copy(deep=True)
        dato_original = df_test[variable].iloc[i]
        df_test[variable].iloc[i] = np.nan
        imputer = KNN(k=k)
        df_test = imputer.fit_transform(df_test)
        dato_imutado = df_test[i ,8]
        resultado = resultado.append({'Valor Original':dato_original, 'Valor Imputado':dato_imutado}, ignore_index=True)
    mmre = (1/total)*sum(abs(resultado['Valor Original'] - resultado['Valor Imputado'])/resultado['Valor Original'])
    return mmre