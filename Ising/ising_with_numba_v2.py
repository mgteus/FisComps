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

# seeds
rd.seed(42)
np.random.seed(42)
rng = default_rng(seed=42)

L = 16 # L = tamanho da rede
L2 = L**2
S = np.array(rd.choices([-1,1], k=L2), dtype=np.float32) # rede  para t=0

# # TEMP = array com as temperaturas 
TEMP = np.array([2.269], dtype=np.float32)


# TMAX = tempo de simulação
TMAX = 10**6




# ISING EM SERIE COM NUMBA



def escreve(t, mag, E):
    with open("data.csv", "a") as file:
        file.write("{}, {}, {}\n".format(t, mag, E)) 
    return




# viz = matriz de vizinhos
@jit(nopython=True)
def init_viz(L2):
    viz = np.zeros((L2,4),dtype=numba.int64)
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



@jit(nopython=True)
def dinamica(s, medidas_mag, medidas_en):
    for temp in TEMP:
        E = 0 
        mag = np.sum(s) # definimos a mag dentro do loop com numba
        viz = init_viz(L2)  # iniciamos os vizinhos
        for i in range(L2):
          for j in range(4):
            E = E + s[i]*(s[viz[i][j]])
        E = E*(-1/2)      
        for t in range(TMAX):
            # rotina da dinamica
            if t < 10**5:
               for i in range(L2):
                 sitio = np.random.randint(L2) # vou escolher um sitio aleatorio
                 deltae = 2*s[sitio]*(s[viz[sitio,0]] + s[viz[sitio,1]] +s[viz[sitio,2]]+s[viz[sitio,3]])
                 prob = np.exp(-deltae/temp)
                 rfloat1 = npr(1)[0]  # num aleatorio [0,1) | npr() = numpy.random.random()
                 if rfloat1 < prob: # if para flipar o sitio
                     s[sitio] = s[sitio]*(-1)
                     mag = mag + 2*s[sitio] # ajustamos a mag
                     E = E + deltae  
            
            else:
              for i in range(L2):
                sitio = np.random.randint(L2) # vou escolher um sitio aleatorio
                deltae = 2*s[sitio]*(s[viz[sitio,0]] + s[viz[sitio,1]] +s[viz[sitio,2]]+s[viz[sitio,3]])
                prob = np.exp(-deltae/temp)
                rfloat1 = npr(1)[0]  # num aleatorio [0,1) | npr() = numpy.random.random()
                if rfloat1 < prob: # if para flipar o sitio
                    s[sitio] = s[sitio]*(-1)
                    mag = mag + 2*s[sitio] # ajustamos a mag
                    E = E + deltae  
                    
                    #escreve(t, mag, E)
                if t%(50)==0:
                    medidas_mag[t] = mag # salvando o valor da mag do passo t 
                    medidas_en[t] = E    # salvando o valor da energia no passo t
                
                
            
"""
CRIAR FUNCAO COM O NUMBA APENAS PARA O PASSOS DE MONTE CARLO
FUNCAO PASSO(s, E, mag)
"""



start_time = time.time()



s = S.copy() #np.array(rd.choices([-1,1], k=L2), dtype=np.float32)


medidas_mag = np.zeros(TMAX, dtype=np.float32) # vetor com as medidas da magnitude
medidas_en = np.zeros(TMAX, dtype=np.float32)  # vetor com as medidas da energia


dinamica(s, medidas_mag, medidas_en)

print("--- %s seconds ---" % (time.time() - start_time))



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



