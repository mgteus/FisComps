import os
from unicodedata import name 
import numpy
import csv
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Tuple


PARTICULAS = List[Tuple[float,float]]
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def salva_valores(**data)-> None:
    """
    Funcao que cria um csv para guardar as infos da simulacao

    retorna None
    """
    if 'name' not in data.keys():
        raise TypeError('Voce deve especificar um nome para o arquivo a ser criado.')
    
    filename = data['name'] + '.csv'

    del data['name']


    cwd = os.getcwd()

    full_filename = os.path.join(cwd, filename)


    if os.path.exists(full_filename):
        if data['new']:
                del  data['new']
                with open(full_filename, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([col for col in data.keys()])
                    writer.writerow([value for value in data.values()])
            
        else:
            del data['new']
            with open(full_filename, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([value for value in data.values()]) 
    else:
        del data['new']
        with open(full_filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([col for col in data.keys()])
                writer.writerow([value for value in data.values()])   

    return full_filename

def salva_velocidades_particulas(particulas: PARTICULAS = [], new:bool = True) -> None:
    """
    Funcao que recebe a lista de particulas e salva suas informaçoes no arquivo
    velocs_particulas.csv

    """
    path = os.getcwd()
    path_completo = os.path.join(path, 'velocs_particulas.csv')

    if len(particulas) > 1:
        nmr_particulas = len(particulas)
        colunas = ["P"+str(i) for i in range(nmr_particulas)]
        velocidades = [part[1] for part in particulas]

        if new:
            with open(path_completo, 'w') as file:
                writer = csv.writer(file)
                writer.writerow([col for col in colunas])
                writer.writerow([velo for velo in velocidades])
            return
        else:
            with open(path_completo, 'a') as file:
                writer = csv.writer(file)
                writer.writerow([velo for velo in velocidades])
            return
            
    else:
        return

def read_csv(path: str = ''):
    """
    Funcao que le o csv requistado e devolve como dataframe
    """

    if not path:   
        raise TypeError('Voce deve especificar um arquivo a ser lido')

    df = pd.read_csv(path)

    return df

def grafico_energias(**data) -> plt.figure:
    """
    Funcao que plota a serie dos dados enviados
    """
    fig, ax = plt.subplots(figsize=(16,9), dpi=60)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)

    timeline = [i for i in range(len([j for j in data.values()][0]))]

    if 'KBT' in data.keys():
        kbt_list = [data['KBT'] for i in timeline]
        data['KBT'] = kbt_list
    
    if 'title' in data.keys():
        plt.title(data['title'], fontsize=30)
        
        del data['title']

    

    for item, values in data.items():
        plt.plot(timeline, values, label=item, lw=3)
    
    plt.ylabel('Energia', fontsize=25)
    plt.xlabel('Tempo', fontsize=25)
    plt.legend(fontsize=25)
    
    return fig

    
#TODO fazer funcao que le, funcao que plota os graficos pedido (hist e st)
if __name__ == '__main__':
    # salva_valores(name='ahhg', g=0, f=0)
    # for i in range(100):
    #     path = salva_valores(name='ahhg', g=i-1, f=i+4)
    
    # print(read_csv(path))

    # data = {'a':[1,2,3,4,5],
    #         'b':[1,2,3,4]}

    # x = grafico_energias(a=[1,2,3,4,5], b=[2,4,6,8,10], KBT=2.5)

    # plt.show()
    t = 0
    for i in range(100):    
        print(t, t == 0)
        t += i/100
