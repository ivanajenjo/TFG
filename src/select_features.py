import operator

import numpy as np
import pandas as pd
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from rpy2.robjects import Formula, pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
from sklearn.feature_selection import (SelectKBest, mutual_info_classif,
                                       mutual_info_regression)
from sklearn.metrics import (adjusted_mutual_info_score, mutual_info_score,
                             normalized_mutual_info_score)
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import KNNImputer

utils = None
information_gain = None
knn_r = None
media = None


def setupR_enviroment():
    """Prepara el entorno necesario de R para utilizar FSelector"""
    global utils
    global information_gain
    global knn_r
    global media
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1)
    packages = ('FSelector', 'VIM')
    utils.install_packages(StrVector(packages))
    FSelector = importr("FSelector")
    information_gain = FSelector.information_gain
    VIM = importr("VIM")
    knn_r = VIM.kNN
    base = importr("base")
    media = base.mean


def setupR_enviroment_knn():
    "Prepara el entorno necesario para utilizar knn de vim"
    global utils
    global knn_r
    if utils == None:
        utils = rpackages.importr('utils')
        utils.chooseCRANmirror(ind=1)
    packages = importr("VIM")
    utils.install_packages(StrVector(packages))
    VIM = importr("VIM")
    knn_r = VIM.kNN


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
    FUNCIÓN SOLO VALIDA PARA EL DATAFRAME DEL TRABAJO ORIGINAL

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
    FUNCIÓN SOLO VALIDA PARA EL DATAFRAME DEL TRABAJO ORIGINAL

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

def recode_dataframe_v2(dataframe):
    ordinalencoder = OrdinalEncoder()
    resultado = dataframe[:]
    categorical_feature_mask = resultado.dtypes == object
    categorical_cols = resultado.columns[categorical_feature_mask].tolist()
    for col in categorical_cols:
        ordinalencoder.fit_transform(dataframe[col])

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
        # print(coefs)
        mrmr[coefs_ordenados[0][0]] = coefs[coefs_ordenados[0][0]]
        ordenadas.remove(coefs_ordenados[0][0])
        seleccionadas.append(coefs_ordenados[0][0])
    resultado = pd.Series(mrmr)
    return resultado


def calcular_mmre(variable, df, k=5):
    """Calcula mmre imputando los valores con la funcion sklearn.impute.KNNImputer

    Parameters:
        variable (String): Variable sobre la cual se va a calcular mmre
        df (pandas.DataFrame): DataFrame
        k (int): Valor utilizado en n_neighbors de KNNImputer

    Returns:
        float con el valor de mmre
    """
    total = len(df)
    resultado = pd.DataFrame(columns=['Valor Original', 'Valor Imputado'])
    numero_columna = df.columns.get_loc(variable)
    for i in range(total):
        df_test = df.copy(deep=True)
        dato_original = df_test[variable].iloc[i]
        df_test[variable].iloc[i] = np.nan
        imputer = KNNImputer(n_neighbors=k)
        df_test = imputer.fit_transform(df_test)
        dato_imputado = df_test[i, numero_columna]
        resultado = resultado.append(
            {'Valor Original': dato_original, 'Valor Imputado': dato_imputado}, ignore_index=True)
        mmre = (1/total)*sum(abs(resultado['Valor Original'] -
                                 resultado['Valor Imputado'])/resultado['Valor Original'])
    return mmre


#def calcular_mmre_v2(variable, df, k=5):
#    """Calcula mmre imputando los valores con la funcion fancyimpute.KNN
#
#    Parameters:
#        variable (String): Variable sobre la cual se va a calcular mmre
#        df (pandas.DataFrame): DataFrame
#        k (int): Valor utilizado en n_neighbors de fancyimpute.KNN
#
#    Returns:
#        float con el valor de mmre
#    """
#    total = len(df)
#    resultado = pd.DataFrame(columns=['Valor Original', 'Valor Imputado'])
#    numero_columna = df.columns.get_loc(variable)
#    for i in range(total):
#        df_test = df.copy(deep=True)
#        dato_original = df_test[variable].iloc[i]
#        df_test[variable].iloc[i] = np.nan
#        imputer = KNN(k=k, verbose=False)
#        df_test = imputer.fit_transform(df_test)
#        dato_imputado = df_test[i, numero_columna]
#        resultado = resultado.append(
#            {'Valor Original': dato_original, 'Valor Imputado': dato_imputado}, ignore_index=True)
#    mmre = (1/total)*sum(abs(resultado['Valor Original'] -
#                             resultado['Valor Imputado'])/resultado['Valor Original'])
#    return mmre, resultado


def calcular_mmre_R(variable_a_imputar, df, k=5):
    if utils == None:
        setupR_enviroment()
    total = len(df)
    resultado = pd.DataFrame(columns=['Valor Original', 'Valor Imputado'])
    numero_columna = df.columns.get_loc(variable_a_imputar)
    pandas2ri.activate()
    for i in range(total):
        df_test = df.copy(deep=True)
        dato_original = df_test[variable_a_imputar].iloc[i]
        df_test[variable_a_imputar].iloc[i] = np.nan
        r_df_test = ro.conversion.py2rpy(df_test)
        r_df_test_imputed = knn_r(
            r_df_test, variable=variable_a_imputar, numFun=media, k=k)
        dato_imputado = r_df_test_imputed[variable_a_imputar].iloc[i]
        resultado = resultado.append(
            {'Valor Original': dato_original, 'Valor Imputado': dato_imputado}, ignore_index=True)
    mmre = sum(abs(resultado['Valor Original'] - resultado['Valor Imputado'])/resultado['Valor Original'])/total
    return mmre, resultado

def determinar_numero_variables(variable, variables_numericas, variables_nominales, df, k=2, umbral_mmre=0, verbose=False):
    """Calcula el numero de variables a elegir para la imputacion utilizando KNN y MMRE

    Parameters:
        variable (String): Variable sobre la cual se va a calcular mmre
        variables_numericas (list:String): Lista de Variables numericas del DF
        variables_nominales (list:String): Lista de Variables nominales del DF
        df (pandas.DataFrame): DataFrame
        k (int): Valor utilizado en n_neighbors de fancyimpute.KNN

    Returns:
        Por Terminar...
    """
    umbral = 1 + umbral_mmre/100
    total_iteraciones = len(variables_nominales) + len(variables_numericas)
    hay_numericas = True
    hay_nominales = True
    variables_elegidas = []
    variables_eliminadas = []
    mmres = []
    mmre_min = float('Inf')
    iteracion = 1
    while (hay_numericas or hay_nominales):
        mmre_num = float('Inf')
        mmre_nom = float('Inf')

        if len(variables_numericas) > 0:
            # En la primera iteracion las variables elegidas son [] porque no hay ninguna por lo tanto no se deberia agregar
            if len(variables_elegidas) <= 0:
                campos = [variable, variables_numericas[0]]
            else:
                # Al añadir variables_elegidas, como es una lista se crea una lista de listas por tanto no funciona
                campos = [variable] + variables_elegidas + \
                    [variables_numericas[0]]
            # print(campos)
            mmre_num, results = calcular_mmre(variable, df[campos], k)
        else:
            hay_numericas = False

        if len(variables_nominales) > 0:
            # En la primera iteracion las variables elegidas son [] porque no hay ninguna por lo tanto no se deberia agregar
            if len(variables_elegidas) <= 0:
                campos = [variable, variables_nominales[0]]
            else:
                # Al añadir variables_elegidas, como es una lista se crea una lista de listas por tanto no funciona
                campos = [variable] + variables_elegidas + \
                    [variables_nominales[0]]
            # print(campos)
            mmre_nom, results = calcular_mmre(variable, df[campos], k)
        else:
            hay_nominales = False

        if mmre_num <= mmre_nom:
            if (umbral*mmre_min) >= mmre_num:
                variables_elegidas.append(variables_numericas[0])
                mmres.append(mmre_num)
                if mmre_min > mmre_num:
                    mmre_min = mmre_num
            else:
                variables_eliminadas.append(variables_numericas[0])
            variables_numericas.pop(0)
        else:
            if (umbral*mmre_min) >= mmre_nom:
                variables_elegidas.append(variables_nominales[0])
                mmres.append(mmre_nom)
                if mmre_min > mmre_nom:
                    mmre_min = mmre_nom
            else:
                variables_eliminadas.append(variables_nominales[0])
            variables_nominales.pop(0)

        if verbose and (hay_nominales or hay_numericas):
            print('Iteracion:', iteracion, 'de', total_iteraciones)
            print('Variables elegidas:', variables_elegidas)
            print('Variables eliminadas', variables_eliminadas)
        iteracion += 1

        # Comprobacion para terminar el algoritmo
        if len(variables_nominales) < 1:
            hay_nominales = False

        if len(variables_numericas) < 1:
            hay_numericas = False

    resultado = [variable, variables_elegidas,
                 variables_eliminadas, mmres, umbral_mmre]
    #resultado = {'Variable':[variable], 'Variables Elegidas':variables_elegidas, 'Variables Eliminadas':variables_eliminadas, 'MMREs':mmres, 'Umbral MMRE':[umbral_mmre]}
    #result_df = pd.DataFrame(resultado)

    print('Ejecucion Completa')
    return resultado

def evaluator_r(nfolds, kNN, df, variable):
    kf = KFold(n_splits=nfolds, shuffle=True)
    kf.split(df)
    mmres = []
    for train_index, test_index in kf.split(df):
        #print('Train Index', train_index)
        #print('Test Index', test_index)
        df_train, df_test = df.iloc[train_index], df.iloc[test_index]
        mmre = calcular_mmre_R(variable, df_test, kNN)
        mmres.append(mmre)
    resultado = np.mean(mmres)
    return resultado

def evaluator(nfolds, kNN, df, variable):
    kf = KFold(n_splits=nfolds, shuffle=True)
    kf.split(df)
    mmres = []
    for train_index, test_index in kf.split(df):
        #print('Train Index', train_index)
        #print('Test Index', test_index)
        df_train, df_test = df.iloc[train_index], df.iloc[test_index]
        mmre = calcular_mmre(variable, df_test, kNN)
        mmres.append(mmre)
    resultado = np.mean(mmres)
    return resultado

def greedy_forward_selection(valor_knn, variable, var_ordenadas, df, umbral_mmre=0, verbose=False):
    valor_de_nfolds = 3
    total_iteraciones = len(var_ordenadas)
    umbral = 1 + umbral_mmre/100
    variables_elegidas = []
    variables_eliminadas = []
    mmres = []
    mmre_min = float('Inf')
    iteracion = 1
    while iteracion <= total_iteraciones:
        campos = [variable] + variables_elegidas + [var_ordenadas[0]]
        mmre_calc = evaluator(valor_de_nfolds, valor_knn, df[campos], variable)
        if ((umbral*mmre_min) >= mmre_calc):
            variables_elegidas.append(var_ordenadas[0])
            mmres.append(mmre_calc)
            if mmre_min > mmre_calc:
                mmre_min = mmre_calc
        else:
            variables_eliminadas.append(var_ordenadas[0])
        var_ordenadas.pop(0)

        if verbose:
            print('Iteracion', iteracion, 'de', total_iteraciones)
            print('Variables elegidas', variables_elegidas)
            print('Variables eliminadas', variables_eliminadas)
        iteracion += 1
    resultado = [variable, variables_elegidas,
                 variables_eliminadas, mmres, umbral_mmre]
    return resultado

def greedy_forward_selection_r(valor_knn, variable, var_ordenadas, df, umbral_mmre=0, verbose=False):
    valor_de_nfolds = 3
    total_iteraciones = len(var_ordenadas)
    umbral = 1 + umbral_mmre/100
    variables_elegidas = []
    variables_eliminadas = []
    mmres = []
    mmre_min = float('Inf')
    iteracion = 1
    while iteracion <= total_iteraciones:
        campos = [variable] + variables_elegidas + [var_ordenadas[0]]
        mmre_calc = evaluator_r(valor_de_nfolds, valor_knn, df[campos], variable)
        if ((umbral*mmre_min) >= mmre_calc):
            variables_elegidas.append(var_ordenadas[0])
            mmres.append(mmre_calc)
            if mmre_min > mmre_calc:
                mmre_min = mmre_calc
        else:
            variables_eliminadas.append(var_ordenadas[0])
        var_ordenadas.pop(0)

        if verbose:
            print('Iteracion', iteracion, 'de', total_iteraciones)
            print('Variables elegidas', variables_elegidas)
            print('Variables eliminadas', variables_eliminadas)
        iteracion += 1
    resultado = [variable, variables_elegidas,
                 variables_eliminadas, mmres, umbral_mmre]
    return resultado

def doquire_forward_selection(valor_knn, variable, var_numericas, var_nominales, df, umbral_mmre=0, verbose=False):
    valor_de_nfolds = 3
    total_iteraciones = len(var_nominales) + len(var_numericas)
    umbral = 1 + umbral_mmre/100
    hay_numericas = True
    hay_nominales = True
    variables_elegidas = []
    variables_eliminadas = []
    mmres = []
    mmre_min = float('Inf')
    iteracion = 1
    while hay_nominales or hay_numericas:
        mmre_num = float('Inf')
        mmre_nom = float('Inf')
        if len(var_numericas) > 0:
            campos = [variable] + variables_elegidas + [var_numericas[0]]
            mmre_num = evaluator(
                valor_de_nfolds, valor_knn, df[campos], variable)

        if len(var_nominales) > 0:
            campos = [variable] + variables_elegidas + [var_nominales[0]]
            mmre_nom = evaluator(
                valor_de_nfolds, valor_knn, df[campos], variable)

        if mmre_num <= mmre_nom:
            if umbral*mmre_min >= mmre_num:
                variables_elegidas.append(var_numericas[0])
                mmres.append(mmre_num)
                if mmre_min > mmre_num:
                    mmre_min = mmre_num
            else:
                variables_eliminadas.append(var_numericas[0])
            var_numericas.pop(0)
        else:
            if umbral*mmre_min >= mmre_nom:
                variables_elegidas.append(var_nominales[0])
                mmres.append(mmre_nom)
                if mmre_min > mmre_nom:
                    mmre_min = mmre_nom
            else:
                variables_eliminadas.append(var_nominales[0])
            var_nominales.pop(0)

        if verbose:
            print('Iteracion', iteracion, 'de', total_iteraciones)
            print('Variables Elegidas', variables_elegidas)
            print('Variables Eliminadas', variables_eliminadas)

        if len(var_nominales) < 1:
            hay_nominales = False

        if len(var_numericas) < 1:
            hay_numericas = False

        iteracion += 1
    resultado = [variable, variables_elegidas,
                 variables_eliminadas, mmres, umbral_mmre]
    return resultado

def doquire_forward_selection_r(valor_knn, variable, var_numericas, var_nominales, df, umbral_mmre=0, verbose=False):
    valor_de_nfolds = 3
    total_iteraciones = len(var_nominales) + len(var_numericas)
    umbral = 1 + umbral_mmre/100
    hay_numericas = True
    hay_nominales = True
    variables_elegidas = []
    variables_eliminadas = []
    mmres = []
    mmre_min = float('Inf')
    iteracion = 1
    while hay_nominales or hay_numericas:
        mmre_num = float('Inf')
        mmre_nom = float('Inf')
        if len(var_numericas) > 0:
            campos = [variable] + variables_elegidas + [var_numericas[0]]
            mmre_num = evaluator_r(
                valor_de_nfolds, valor_knn, df[campos], variable)

        if len(var_nominales) > 0:
            campos = [variable] + variables_elegidas + [var_nominales[0]]
            mmre_nom = evaluator_r(
                valor_de_nfolds, valor_knn, df[campos], variable)

        if mmre_num <= mmre_nom:
            if umbral*mmre_min >= mmre_num:
                variables_elegidas.append(var_numericas[0])
                mmres.append(mmre_num)
                if mmre_min > mmre_num:
                    mmre_min = mmre_num
            else:
                variables_eliminadas.append(var_numericas[0])
            var_numericas.pop(0)
        else:
            if umbral*mmre_min >= mmre_nom:
                variables_elegidas.append(var_nominales[0])
                mmres.append(mmre_nom)
                if mmre_min > mmre_nom:
                    mmre_min = mmre_nom
            else:
                variables_eliminadas.append(var_nominales[0])
            var_nominales.pop(0)

        if verbose:
            print('Iteracion', iteracion, 'de', total_iteraciones)
            print('Variables Elegidas', variables_elegidas)
            print('Variables Eliminadas', variables_eliminadas)

        if len(var_nominales) < 1:
            hay_nominales = False

        if len(var_numericas) < 1:
            hay_numericas = False

        iteracion += 1
    resultado = [variable, variables_elegidas,
                 variables_eliminadas, mmres, umbral_mmre]
    return resultado