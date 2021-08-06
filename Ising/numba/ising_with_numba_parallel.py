# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 19:08:41 2021

@author: mgteus

"""
import numba
import numpy as np
import random as rd 
import matplotlib.pyplot as plt
import time


from numba import jit, config, threading_layer, set_num_threads, prange
from numpy.random import random as npr
from numpy.random import default_rng
from chartmodules import *

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




config.THREADING_LAYER = 'threadsafe'
# função da dinâmica de MC
@jit(nopython=True, parallel=True)
def dinamica_p(s, medidas_mag, medidas_en):
    for temp in TEMP:
        E = 0 
        mag = np.sum(s)/L2 # definimos a mag dentro do loop com numba
        viz = init_viz(L2)  # iniciamos os vizinhos
        for i in range(L2):
          for j in range(4):
            E = E + s[i]*(s[viz[i][j]])
        E = E*(-1/2)       
        for t in range(TMAX):
            # rotina da dinamica
            if t > 10**5:
                for i in prange(L2):
                    sitio = np.random.randint(L2) # vou escolher um sitio aleatorio
                    deltae = 2*s[sitio]*(s[viz[sitio,0]] + s[viz[sitio,1]] +s[viz[sitio,2]]+s[viz[sitio,3]])
                    prob = np.exp(-deltae/temp)
                    rfloat1 = npr(1)[0]  # num aleatorio [0,1) | npr() = numpy.random.random()
                    if rfloat1 < prob: # if para flipar o sitio
                        s[sitio] = s[sitio]*(-1)
                        mag = mag + 2*s[sitio] # ajustamo a mag
                        E = E + deltae
            else:
                for i in prange(L2):
                    sitio = np.random.randint(L2) # vou escolher um sitio aleatorio
                    deltae = 2*s[sitio]*(s[viz[sitio,0]] + s[viz[sitio,1]] +s[viz[sitio,2]]+s[viz[sitio,3]])
                    prob = np.exp(-deltae/temp)
                    rfloat1 = npr(1)[0]  # num aleatorio [0,1) | npr() = numpy.random.random()
                    if rfloat1 < prob: # if para flipar o sitio
                        s[sitio] = s[sitio]*(-1)
                        mag = mag + 2*s[sitio] # ajustamo a mag
                        E = E + deltae
                    
                    medidas_mag[t] = mag # salvando o valor da mag do passo t 
                    medidas_en[t] = E    # salvando o valor da energia no passo t
                
            


c = time.time()


s = S.copy() #np.array(rd.choices([-1,1], k=L2), dtype=np.float32)


medidas_mag = np.zeros(TMAX, dtype=np.float32) # vetor com as medidas da magnitude
medidas_en = np.zeros(TMAX, dtype=np.float32)  # vetor com as medidas da energia


dinamica_p(s, medidas_mag, medidas_en)

print(time.time() - c,"segundos")


#plt.plot(medidas_mag)

#hist_mag(medidas_mag)



