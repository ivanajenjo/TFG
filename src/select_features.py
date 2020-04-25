from sklearn.feature_selection import mutual_info_classif
import pandas as pd

def ordenar_porMI(var, df):
    resultado = mutual_info_classif(df, var)