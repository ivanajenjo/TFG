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
    """Prepara el entorno necesario de R para utilizar FSelector"""
    global utils
    global information_gain
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1)
    packages = ('FSelector')
    utils.install_packages(StrVector(packages))
    FSelector = importr("FSelector")
    information_gain = FSelector.information_gain


def calcular_mi_R(variable, df):
    """Calcula Mutual Information para las variables independientes utilizando FSelector
    
    Parameters:
        variable (String): Variable sobre la cual se quiere calcular MI
        df (pandas.DataFrame): DataFrame

    Returns:
        pandas.DataFrame: DataFrame ordenado con los valores de MI de cada una de las variables del df
    """
    if utils == None:
        setupR_enviroment()
    pandas2ri.activate()
    r_df = ro.conversion.py2rpy(df)
    #fmla = Formula('Normalised_Work_Effort_Level_1~.')
    fmla = Formula(str(variable + '~.'))
    resultado = information_gain(fmla, r_df)
    resultado = resultado.sort_values('attr_importance', ascending=False)
    return resultado


def calcular_mi_R_2v(variable1, variable2, df):
    """Calcula MI entre 2 variables de un DataFrame utilizando FSelector
    
    Parameters:
        variable1 (String): Primera variable utilizada para calcular MI
        variable2 (String): Segunda variable utilizada para calcular MI
        df (pandas.DataFrame): DataFrame

    Returns:
        float: Valor de MI
    """
    if utils == None:
        setupR_enviroment()
    pandas2ri.activate()
    r_df = ro.conversion.py2rpy(df)
    fmla = Formula(str(variable1 + '~' + variable2))
    resultado = information_gain(fmla, r_df)
    resultado = resultado.iloc[0].values[0]
    return resultado


def calcular_mrmr_R(variable, df):
    """Minimum redundancy – Maximum relevance (MRMR) utilizando FSelector para calcular los MI necesarios
    
    Parameters:
        variable (String): Variable sobre la cual se calcula mrmr
        df (pandas.DataFrame): DataFrame

    Returns:
        pandas.Series: Serie con los resultados de mrmr
    """
    seleccionadas = []
    mrmr = {}
    info_gain = calcular_mi_R(variable, df)
    ordenadas = info_gain.index.tolist()
    seleccionadas.append(ordenadas[0])
    mrmr[ordenadas[0]] = info_gain.iloc[0].values[0]
    ordenadas.pop(0)
    iteraciones = len(ordenadas)
    for i in range(iteraciones):
        coefs = {}
        for prueba in ordenadas:
            info_parcial = {}
            Info_prueba = calcular_mi_R_2v(variable, prueba, df)
            for seleccionada in seleccionadas:
                I_parcial = calcular_mi_R_2v(seleccionada, prueba, df)
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


def recode_dataframe_R(df):
    """Cambia el nombre de las columnas para que puedan ser utilizadas correctamente por FSelector
    MÉTODO SOLO VALIDO PARA EL DATAFRAME DEL TRABAJO ORIGINAL

    Parameters:
        df (pandas.DataFrame): Dataframe

    Returns:
        Dataframe modificado
    """
    df.columns = ['Industry_Sector', 'Application_Group', 'Development_Type',
                  'Development_Platform', 'Language_Type', 'Primary_Programming_Language',
                  'Functional_Size', 'Adjusted_Function_Points',
                  'Normalised_Work_Effort_Level_1', 'Project_Elapsed_Time',
                  'First_Data_Base_System', 'Used_Methodology']
    return df


def calc_mi_scikit(x, y):
    mi = normalized_mutual_info_score(y, x)
    return mi


def calcular_mi_manual(variable, df):
    """Calcula Mutual Information para las variables independientes utilizando sklearn.metrics.normalized_mutual_info_score para un DataFrame
    
    Parameters:
        variable (String): Variable sobre la cual se calcula mi
        df (pandas.DataFrame): Dataframe

    Returns:
        pandas.Series: Serie con los resultados de mi
    """
    y = df[variable].values
    x = df.loc[:, df.columns != variable]
    #x = df
    resultado = []
    for (columnName, columnData) in x.iteritems():
        #print('Colunm Name : ', columnName)
        aux = calc_mi_scikit(y, columnData)
        resultado.append(aux)
    resultado = pd.Series(resultado)
    resultado.index = x.columns
    resultado = resultado.sort_values(ascending=False)
    return resultado


def calcular_mi(variable, df):
    """Calcula Mutual Information para las variables independientes utilizando sklearn.feature_selection.mutual_info_regression para un DataFrame
    MÉTODO SOLO VALIDO PARA EL DATAFRAME DEL TRABAJO ORIGINAL
    
    Parameters:
        variable (String): Variable sobre la cual se calcula mi
        df (pandas.DataFrame): Dataframe

    Returns:
        pandas.Series: Serie con los resultados de mi
    """
    variables = ['Industry Sector', 'Application Group', 'Development Type', 'Development Platform', 'Language Type', 'Primary Programming Language',
                 'Functional Size', 'Adjusted Function Points', 'Project Elapsed Time', '1st Data Base System', 'Used Methodology']
    X = df.loc[:, variables]
    y = df.loc[:, variable].values
    mi = mutual_info_regression(X, y, n_neighbors=1)
    mi = pd.Series(mi)
    mi.index = X.columns
    mi = mi.sort_values(ascending=False)
    return mi


def recode_dataframe(dataframe):
    """Recodifica el DataFrame utilizando sklearn.preprocessing.LabelEncoder para las variables categóricas
    
    Parameters:
        dataframe (pandas.DataFrame): Dataframe

    Returns:
        pandas.DataFrame con las variables categóricas codificadas
    """
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


def calcular_mrmr_v2(variable, df):
    """Minimum redundancy – Maximum relevance (MRMR) utilizando sklearn.metrics.normalized_mutual_info_score para calcular los MI necesarios
    
    Parameters:
        variable (String): Variable sobre la cual se calcula mrmr
        df (pandas.DataFrame): DataFrame

    Returns:
        pandas.Series: Serie con los resultados de mrmr
    """
    seleccionadas = []
    mrmr = {}
    info_gain = calcular_mi_manual(variable, df)
    ordenadas = info_gain.keys().values.tolist()
    seleccionadas.append(ordenadas[0])
    mrmr[ordenadas[0]] = info_gain.iloc[0]
    ordenadas.pop(0)
    iteraciones = len(ordenadas)
    for i in range(iteraciones):
        coefs = {}
        for prueba in ordenadas:
            info_parcial = {}
            Info_prueba = calc_mi_scikit(df[variable], df[prueba])
            for seleccionada in seleccionadas:
                I_parcial = calc_mi_scikit(df[prueba], df[seleccionada])
                info_parcial[seleccionada] = I_parcial
            coef_parcial = Info_prueba - np.mean(list(info_parcial.values()))
            coefs[prueba] = coef_parcial
        coefs_ordenados = sorted(
            coefs.items(), key=operator.itemgetter(1), reverse=True)
        #print(coefs)
        mrmr[coefs_ordenados[0][0]] = coefs[coefs_ordenados[0][0]]
        ordenadas.remove(coefs_ordenados[0][0])
        seleccionadas.append(coefs_ordenados[0][0])
    resultado = pd.Series(mrmr)
    return resultado


def calcular_mmre(variable, df, k=5):
    """Calcula mmre imputando los valores con la funcion sklearn.impute.KNNImputer
    EL MÉTODO TODAVIA NO ESTÁ TERMINADO YA QUE SOLO FUNCIONA CON EL DATAFRAME DEL TRABAJO A DIA 28-5

    Parameters:
        variable (String): Variable sobre la cual se va a calcular mmre
        df (pandas.DataFrame): DataFrame
        k (int): Valor utilizado en n_neighbors de KNNImputer
    
    Returns:
        float con el valor de mmre
    """
    total = len(df)
    resultado = pd.DataFrame(columns=['Valor Original', 'Valor Imputado'])
    for i in range(total):
        df_test = df.copy(deep=True)
        dato_original = df_test[variable].iloc[i]
        df_test[variable].iloc[i] = np.nan
        imputer = KNNImputer(n_neighbors=k)
        df_test = imputer.fit_transform(df_test)
        dato_imputado = df_test[i ,8]
        resultado = resultado.append({'Valor Original':dato_original, 'Valor Imputado':dato_imputado}, ignore_index=True)
        mmre = (1/total)*sum(abs(resultado['Valor Original'] - resultado['Valor Imputado'])/resultado['Valor Original'])
    return mmre


def calcular_mmre_v2(variable, df, k=5):
    """Calcula mmre imputando los valores con la funcion fancyimpute.KNN
    EL MÉTODO TODAVIA NO ESTÁ TERMINADO YA QUE SOLO FUNCIONA CON EL DATAFRAME DEL TRABAJO A DIA 28-5

    Parameters:
        variable (String): Variable sobre la cual se va a calcular mmre
        df (pandas.DataFrame): DataFrame
        k (int): Valor utilizado en n_neighbors de fancyimpute.KNN
    
    Returns:
        float con el valor de mmre
    """
    total = len(df)
    resultado = pd.DataFrame(columns=['Valor Original', 'Valor Imputado'])
    for i in range(total):  
        df_test = df.copy(deep=True)
        dato_original = df_test[variable].iloc[i]
        df_test[variable].iloc[i] = np.nan
        imputer = KNN(k=k)
        df_test = imputer.fit_transform(df_test)
        dato_imputado = df_test[i ,8]
        resultado = resultado.append({'Valor Original':dato_original, 'Valor Imputado':dato_imputado}, ignore_index=True)
    mmre = (1/total)*sum(abs(resultado['Valor Original'] - resultado['Valor Imputado'])/resultado['Valor Original'])
    return mmre