# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 20:01:34 2021

@author: mgteus

"""
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 10:17:21 2021

@author: mgteus

"""

import numba
import numpy as np
import random as rd 
import matplotlib.pyplot as plt
import time


from numba import jit, config, threading_layer, set_num_threads
from numpy.random import random as npr
from numpy.random import default_rng
from chartmodules import plot_from_csv, restar_csv

# seeds
rd.seed(42)
np.random.seed(42)
rng = default_rng(seed=42)

L = 16 # L = tamanho da rede
L2 = L**2
S = np.array(rd.choices([-1,1], k=L2), dtype=np.float32) # rede  para t=0

# # TEMP = array com as temperaturas 
TEMP = 2.269 #np.array([2.269], dtype=np.float32)


# T = tempo de simulação
Teq = 10**5

T = 10**7 - Teq

pm = 60 # tempo entre medidas

viz = np.zeros((L2,4),dtype=np.int64)
# ISING EM SERIE COM NUMBA



def escreve(t, mag, E):
    with open("C:/Users/mateu/workspace/MonteCarlo/Ising/data.csv", "a") as file:
        file.write("{}, {}, {}\n".format(t, mag, E)) 
    return

def init_viz(L2):
    viz = np.zeros((L2,4),dtype=np.int64)
    for sitio in range(L2):
        n1 = ((sitio//L -1 +L2)%L)*L + sitio%L
        n2 = ((sitio//L)%L)*L+(sitio+1+L)%L
        n3 = ((sitio//L +1 +L2)%L)*L +sitio%L
        n4 = ((sitio//L)%L)*L+(sitio-1+L)%L
    
        viz[sitio][0] = n1
        viz[sitio][1] = n2
        viz[sitio][2] = n3
        viz[sitio][3] = n4
    return viz

def init_medidas(s, viz):
    E = 0 
    mag = np.sum(s)                         
    viz = init_viz(L2)                      
    for i in range(L2):
      for j in range(4):
        E = E + s[i]*(s[viz[i][j]])
    E = E*(-1/2)
    
    return E, mag
    
    

# funcao responsavel pelos MCS
@jit(nopython=True)
def passo(s, mag, E, viz):
    for i in range(L2):
             sitio = np.random.randint(L2) # vou escolher um sitio aleatorio
             deltae = 2*s[sitio]*(s[viz[sitio,0]] + s[viz[sitio,1]] +s[viz[sitio,2]]+s[viz[sitio,3]])
             prob = np.exp(-deltae/TEMP)
             rfloat1 = npr(1)[0]  # num aleatorio [0,1) | npr() = numpy.random.random()
             if rfloat1 < prob: # if para flipar o sitio
                 s[sitio] = s[sitio]*(-1)
                 mag = mag + 2*s[sitio] # ajustamos a mag
                 E = E + deltae
    return mag, E
        

def dinamica(s):
        restar_csv() # reiniciando medidas
        viz = init_viz(L2) # iniciando vizinhos
        E, mag = init_medidas(s, viz) # iniciando energia e mag
        
    
        for t in range(Teq):              # loop para equilibrar
            mag, E = passo(s, mag, E, viz)
        
        for t in range(T):                # loop com medidas
            mag, E = passo(s, mag, E, viz)
            if t%(50)==0:
                escreve(t, mag, E)
                
        print("Resultados para T={}".format(T+Teq))

            
            
     
      




start_time = time.time()


s = S.copy()  
dinamica(s)

print("--- %s segundos ---" % (time.time() - start_time))
    













#from chartmodules import plot_from_csv, restar_csv



#plot_from_csv()
#
 



     
   
"""
CRIAR FUNCAO COM O NUMBA APENAS PARA O PASSOS DE MONTE CARLO
FUNCAO PASSO(s, E, mag)
"""


#print("--- %s seconds ---" % (time.time() - start_time))



# mag_list_abs = [abs(i) for i in medidas_mag[10**5:]]

# np.mean(mag_list_abs)

# en_list_abs =  [i for i in medidas_en[10**5:]]

# np.mean(en_list_abs)




# from chartmodules import *

# hist_en(medidas_en)

# len(medidas_en[10**5:])

# plt.plot(medidas_mag[10**5:])

# hist_mag(medidas_mag)



# plt.hist(medidas_mag[10**5:], bins=280)


# np.min(medidas_en[10**5:])


# plt.hist(medidas_en[10**5:], bins=)



