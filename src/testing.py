import pandas as pd

archivo = 'data\ISBSG - Release May 2017 R1.csv'
df = pd.read_csv(archivo, sep = ';', low_memory = False)
variables = ['ISBSG Project ID', 'Data Quality Rating', 'UFP rating', 'Application Group', 'Development Type', 'Development Platform', 'Language Type', 'Primary Programming Language', 'Count Approach', 'Functional Size', 'Adjusted Function Points', 'Normalised Work Effort Level 1', 'Summary Work Effort', 'Project Elapsed Time', 'Business Area Type', 'Used Methodology', 'Resource Level', 'Max Team Size', 'Average Team Size', 'Input count', 'Output count', 'Enquiry count', 'File count', 'Interface count', 'Agile Method Used']
df = df.loc[:, variables]
df