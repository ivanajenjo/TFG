from sklearn.feature_selection import mutual_info_regression, mutual_info_classif, SelectKBest
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score, adjusted_mutual_info_score
from scipy.stats import chi2_contingency
from info_gain import info_gain
import pandas as pd
import numpy as np

def calc_MI(x, y):
    c_xy = np.histogram2d(x, y)[0]
    g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood")
    mi = 0.5 * g / c_xy.sum()
    return mi

def calc_MI_scikit(x, y):
    mi = mutual_info_score(y, x)
    return mi

def calcularMiManual(variable, df):
    y = df[variable].values
    x = df.loc[:, df.columns != variable]
    column = []
    resultado = []
    for (columnName, columnData) in x.iteritems():
        #print('Colunm Name : ', columnName)
        aux = calc_MI_scikit(y, columnData)
        resultado.append(aux)
    resultado = pd.Series(resultado)
    resultado.index = x.columns
    return resultado

def calcularMiManualInfo_gain(variable, df):
    y = df[variable].values
    x = df.loc[:, df.columns != variable]
    column = []
    resultado = []
    for (columnName, columnData) in x.iteritems():
        #print('Colunm Name : ', columnName)
        aux = info_gain.intrinsic_value(y, columnData)
        resultado.append(aux)
    resultado = pd.Series(resultado)
    resultado.index = x.columns
    return resultado

def calcularMI(variable, df):
    variables = ['Industry Sector', 'Application Group', 'Development Type', 'Development Platform', 'Language Type', 'Primary Programming Language',
                 'Functional Size', 'Adjusted Function Points', 'Project Elapsed Time', '1st Data Base System', 'Used Methodology']
    X = df.loc[:, variables]
    y = df.loc[:, variable].values
    mi = mutual_info_regression(X, y, discrete_features=[True, True, True, True, True, True, False, False, False, True, True])
    mi = pd.Series(mi)
    mi.index = X.columns
    mi.sort_values(ascending=False)
    return mi

def selectKBestMi(df):
    fs = SelectKBest(score_func=mutual_info_regression, k='all')
    variables = ['Industry Sector', 'Application Group', 'Development Type', 'Development Platform', 'Language Type', 'Primary Programming Language',
                 'Functional Size', 'Adjusted Function Points', 'Project Elapsed Time', '1st Data Base System', 'Used Methodology']
    X = df.loc[:, variables]
    y = df.loc[:,'Normalised Work Effort Level 1'].values
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