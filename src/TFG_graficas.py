import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    data = pd.read_csv("src\csv.csv", sep=';')
    print(data)
    filtro = data['Metodo'] == 1
    data1 = data.loc[filtro,:]
    data1.plot(kind='line', x='Iteracion', y='MMRE')
    plt.show()

if __name__ == "__main__":
    main()