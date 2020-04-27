from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def ordenar_porMI(variable, df):
    df.columns
    variables = ['Industry Sector', 'Application Group', 'Development Type', 'Development Platform', 'Language Type', 'Primary Programming Language',
                 'Functional Size', 'Adjusted Function Points', 'Project Elapsed Time', '1st Data Base System', 'Used Methodology']
    X = df.loc[:, variables]
    y = df.loc[:, variable].values
    mi = mutual_info_regression(X, y)
    mi = pd.Series(mi)
    mi.index = X.columns
    mi.sort_values(ascending=False)
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


def recodeDataframeV2(dataframe):
    resultado = dataframe
    variables = ['Industry Sector', 'Application Group', 'Development Type', 'Development Platform',
                 'Language Type', 'Primary Programming Language', '1st Data Base System', 'Used Methodology']
    resultado[variables] = resultado[variables].astype('category')
    resultado[variables] = resultado[variables].cat.codes
    return resultado