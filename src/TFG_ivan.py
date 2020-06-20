import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import select_features
import time

df = None

def importBD():
    global df
    archivo ='D:\\Users\\ivana\\Documents\\TFG\\data\\ISBSG DATA Release 12.csv'
    df = pd.read_csv(archivo, sep = ';', low_memory = False)
    variables = ['Data Quality Rating', 'UFP rating', 'Industry Sector','Application Group', 'Development Type', 'Development Platform', 'Language Type', 'Primary Programming Language', 'Count Approach', 'Functional Size', 'Adjusted Function Points', 'Normalised Work Effort Level 1', 'Summary Work Effort', 'Project Elapsed Time', 'Business Area Type', '1st Data Base System', 'Used Methodology', 'Resource Level', 'Max Team Size', 'Average Team Size', 'Input count', 'Output count', 'Enquiry count', 'File count', 'Interface count']
    df = df.loc[:, variables]
    filtro = ((df['Data Quality Rating'] == 'A') | (df['Data Quality Rating'] == 'B')) & ((df['UFP rating'] == 'A') | (df['UFP rating'] == 'B')) 
    df = df.loc[filtro, :]
    filtro = (df['Normalised Work Effort Level 1'].notnull()) & (df['Normalised Work Effort Level 1'] == df['Summary Work Effort'])
    df = df.loc[filtro, :]
    filtro = df['Count Approach'] == 'IFPUG 4+'
    df = df.loc[filtro, :]
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

def main():
    global df
    pd.options.mode.chained_assignment = None
    importBD()
    print('Calculo MI')
    var_categoricas = df.select_dtypes(exclude=["number"]).columns.values
    var_numericas = df.select_dtypes(include=["number"]).columns.values
    mi = select_features.calcular_mi_manual('Normalised Work Effort Level 1', df)
    variables_por_mi = list(mi.index.values)
    mrmr = select_features.calcular_mrmr_v2('Normalised Work Effort Level 1', df)
    variables_por_mrmr = list(mrmr.index.values)
    var_c_mi = list(set(variables_por_mi) & set(var_categoricas))
    var_n_mi = list(set(variables_por_mi) & set(var_numericas))
    var_c_mrmr = list(set(variables_por_mrmr) & set(var_categoricas))
    var_n_mrmr = list(set(variables_por_mrmr) & set(var_numericas))
    mmres = list()
    ks = list()
    var_elegidas = list()
    resultados = list()
    iteraciones = 500
    print('Recode DF')
    df = select_features.recode_dataframe(df)
    print('Empieza el Bucle')
    start = time.time()

    #k in range(1, 5) para que k sea de 1 a 4
    for k in range(2, 3):
        print ('K =', k)
        for i in range(iteraciones):
            print('Iteracion', i, 'Método 1 k', k)
            iteration_start = time.time()
            var = variables_por_mi[:]
            mmre = select_features.greedy_forward_selection(k, 'Normalised Work Effort Level 1', var, df)
            mmres.append((mmre[3])[-1])
            var_elegidas.append(mmre[1])
            ks.append(k)
            iteration_time = time.time() - iteration_start
            print(iteration_time, 'Segundos')
            resultados.append(str((mmre[3])[-1]) + ';' + str(k) + ';' + str(mmre[1]) + ';' + str(1) + ';' + str(iteration_time) + ';' + str(i))

        for i in range(iteraciones):
            print('Iteracion', i, 'Método 2 k', k)
            iteration_start = time.time()
            var = variables_por_mrmr[:]
            mmre = select_features.greedy_forward_selection(k, 'Normalised Work Effort Level 1', var, df)
            mmres.append((mmre[3])[-1])
            var_elegidas.append(mmre[1])
            ks.append(k)
            iteration_time = time.time() - iteration_start
            print(iteration_time, 'Segundos')
            resultados.append(str((mmre[3])[-1]) + ';' + str(k) + ';' + str(mmre[1]) + ';' + str(2) + ';' + str(iteration_time) + ';' + str(i))

        for i in range(iteraciones):
            print('Iteracion', i, 'Método 3 k', k)
            iteration_start = time.time()
            mmre = select_features.doquire_forward_selection(k, 'Normalised Work Effort Level 1', var_n_mi[:], var_c_mi[:], df)
            mmres.append((mmre[3])[-1])
            var_elegidas.append(mmre[1])
            ks.append(k)
            iteration_time = time.time() - iteration_start
            print(iteration_time, 'Segundos')
            resultados.append(str((mmre[3])[-1]) + ';' + str(k) + ';' + str(mmre[1]) + ';' + str(3) + ';' + str(iteration_time) + ';' + str(i))

        for i in range(iteraciones):
            print('Iteracion', i, 'Método 4 k', k)
            iteration_start = time.time()
            mmre = select_features.doquire_forward_selection(k, 'Normalised Work Effort Level 1', var_n_mrmr[:], var_c_mrmr[:], df)
            mmres.append((mmre[3])[-1])
            var_elegidas.append(mmre[1])
            ks.append(k)
            iteration_time = time.time() - iteration_start
            print(iteration_time, 'Segundos')
            resultados.append(str((mmre[3])[-1]) + ';' + str(k) + ';' + str(mmre[1]) + ';' + str(4) + ';' + str(iteration_time) + ';' + str(i))
             
    end = time.time()

    print('Tiempo de ejecucion', end-start)
    convertirACsv(resultados)


def convertirACsv(lista):
    import csv

    with open("csv.csv", 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_MINIMAL, delimiter= ";")
        row = 'MMRE' + ';' + 'k' + ';' + 'Variables Elegidas' + ';' + 'Metodo' + ';' + 'Tiempo' + ';' + 'Iteracion'
        wr.writerow(row.split(';'))
        for i in lista:
            #print(i)
            wr.writerow(i.split(';'))
    print("escrito en csv")

if __name__ == "__main__":
    main()

