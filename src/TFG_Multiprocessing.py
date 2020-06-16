import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import select_features
from multiprocessing import Pool, cpu_count

#Not Working

resultados = []

def main():
    pd.options.mode.chained_assignment = None
    #archivo = '..\data\ISBSG - Release May 2017 R1.csv'
    archivo ='D:\\Users\\ivana\\Documents\\TFG\\data\\ISBSG DATA Release 12.csv'
    df = pd.read_csv(archivo, sep = ';', low_memory = False)
    #variables = ['Data Quality Rating', 'UFP rating', 'Industry Sector','Application Group', 'Development Type', 'Development Platform', 'Language Type', 'Primary Programming Language', 'Count Approach', 'Functional Size', 'Adjusted Function Points', 'Normalised Work Effort Level 1', 'Summary Work Effort', 'Project Elapsed Time', 'Business Area Type', '1st Data Base System', 'Used Methodology', 'Resource Level', 'Max Team Size', 'Average Team Size', 'Input count', 'Output count', 'Enquiry count', 'File count', 'Interface count', 'Agile Method Used']
    variables = ['Data Quality Rating', 'UFP rating', 'Industry Sector','Application Group', 'Development Type', 'Development Platform', 'Language Type', 'Primary Programming Language', 'Count Approach', 'Functional Size', 'Adjusted Function Points', 'Normalised Work Effort Level 1', 'Summary Work Effort', 'Project Elapsed Time', 'Business Area Type', '1st Data Base System', 'Used Methodology', 'Resource Level', 'Max Team Size', 'Average Team Size', 'Input count', 'Output count', 'Enquiry count', 'File count', 'Interface count']
    df = df.loc[:, variables]
    filtro = ((df['Data Quality Rating'] == 'A') | (df['Data Quality Rating'] == 'B')) & ((df['UFP rating'] == 'A') | (df['UFP rating'] == 'B')) 
    df = df.loc[filtro, :]
    filtro = (df['Normalised Work Effort Level 1'].notnull()) & (df['Normalised Work Effort Level 1'] == df['Summary Work Effort'])
    df = df.loc[filtro, :]
    filtro = df['Count Approach'] == 'IFPUG 4+'
    df = df.loc[filtro, :]
    #variables = ['Industry Sector','Application Group', 'Development Type', 'Development Platform', 'Language Type', 'Primary Programming Language', 'Functional Size', 'Adjusted Function Points', 'Normalised Work Effort Level 1', 'Project Elapsed Time', 'Business Area Type', '1st Data Base System', 'Used Methodology', 'Max Team Size', 'Average Team Size', 'Input count', 'Output count', 'Enquiry count', 'File count', 'Interface count', 'Agile Method Used']
    variables = ['Industry Sector','Application Group', 'Development Type', 'Development Platform', 'Language Type', 'Primary Programming Language', 'Functional Size', 'Adjusted Function Points', 'Normalised Work Effort Level 1', 'Project Elapsed Time', 'Business Area Type', '1st Data Base System', 'Used Methodology', 'Max Team Size', 'Average Team Size', 'Input count', 'Output count', 'Enquiry count', 'File count', 'Interface count']
    df = df.loc[:, variables]
    df = df.dropna(axis=1, thresh=int(0.5*len(df)))
    df = df.dropna()
    print(len(df))
    df['Project Elapsed Time'] = df['Project Elapsed Time'].str.replace(',', '.').astype(float)
    programmingLenguaje = {'A:G':'Unspecified', 'ASP.Net':'ASP', 'BASIC':'Visual Basic', 'CSP':'Unspecified', 'Visual C':'C'}
    df['Primary Programming Language'].replace( programmingLenguaje, inplace = True)
    database = {'[;].*':';','ACCESS[; ].*':'ACCESS', 'MS Access':'ACCESS', 'ACCESS;':'ACCESS', 'ADABAS;':'ADABAS', 'Micosoft.*':'Attain', 'DB2[; /].*':'DB2', 'IBM DB2':'DB2', 'UDB2':'DB2', 'Domino[ ].*':'Domino', 'LOTUS.*':'Domino', 'Notes.*':'Domino', 'Exchange.*':'Exchange', 'FOXPRO;':'Foxpro', 'HIRDB;':'HIRDB', 'DB[/].*':'IMS', 'DEDB;':'IMS', 'IDMS[; -].*':'IMS', 'IMS.*':'IMS', 'MS[- ]SQL[; ].*':'MS SQL', 'MSDE.*':'MS SQL', 'SQL Server[; ].*':'MS SQL', 'SQL;':'MS SQL', 'VSE/.*':'MS SQL', 'NCR;':'NCR', 'Oracle.*':'ORACLE', 'Personal O.*':'ORACLE', 'RDB[; ].*':'ORACLE', 'CICS;':'ORACLE', 'SAS;':'SAS', 'Solid;':'Solid', 'SYBASE.*':'SYBASE', 'YES':'Unspecified', 'ISAM;':'Unspecified', 'multiple;':'Unspecified', 'VSAM[; ].*':'Unspecified', 'WATCOM[; ].*':'Watcom', 'WGRES;':'WGRES'}
    df['1st Data Base System'].replace( database, inplace = True, regex = True)
    df['1st Data Base System'].replace( {'ACCESS;':'ACCESS'}, inplace = True, regex = True)
    print('Empieza el Calculo')
    mi = select_features.calcular_mi_manual('Normalised Work Effort Level 1', df)
    variables_por_mi = list(mi.index.values)
    df = select_features.recode_dataframe(df)
    pool = Pool(processes=(cpu_count() - 1))
    for i in range(10):
        var = variables_por_mi[:]
        pool.apply_async(thread_process, args=(i, var, df))
        print('Iteracion', i)
    pool.close()
    pool.join()
    final_value = np.mean(resultados)
    print(final_value)

def thread_process(i, var, df):
    mmre = select_features.greedy_forward_selection(1, 'Normalised Work Effort Level 1', var, df, seed=i)
    resultados.append((mmre[3])[-1])

if __name__ == "__main__":
    main()