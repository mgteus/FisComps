# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 20:01:34 2021

@author: mgteus

"""
# -*- coding: utf-8 -*-
import itertools

import numba
import numpy as np
import random as rd 
import matplotlib.pyplot as plt
import time
import matplotlib as mpl


from matplotlib import cm
from numba import jit, config, threading_layer, set_num_threads
from numpy.random import random as npr
from numpy.random import default_rng
from chartmodules import plot_from_csv, restart_csv

# seeds
rd.seed(42)
np.random.seed(42)
rng = default_rng(seed=42)

for TEMP in [3.0]:
    T = 2.2
    #for L in [16,32,64]:
    L = 4 # L = tamanho da rede
    L2 = L**2
    S = np.array(rd.choices([1], k=L2), dtype=np.float32) # rede  para t=0
    
    # # TEMP = array com as temperaturas 
    #TEMP = 3 #np.array([2.269], dtype=np.float32)
    
    
    # Teq = tempo de equilibrio
    Teq = 10**5
    # T = tempo de simulação
    T = 10**7 - Teq
    
    pm = 50 # tempo entre medidas
    
    viz = np.zeros((L2,4),dtype=np.int64)
    # ISING EM SERIE COM NUMBA
    
    E    = 0
    peso = 0
    Z    = 0
    mag  = 0
    m2   = 0
    E2   = 0
    e    = 0
    
    
    for i in range(S.shape[0]):
        for j in range(-1,2,2):  
            print(f'primeiro valor = {S[i]*j}, i={i}')
            S[i] = j
            for i in range(L2):
              for j in range(4):
                E = E + s[i]*(s[viz[i][j]])
            E = E*(-1/2)
            
            peso = np.exp(-E/T)
            
            Z = Z + peso
            
            e = e + E/L2*peso
            
            mag = mag + abs(np.sum(S))/L2*peso
        



    
    
    # def make_plot(s, t):
    #     splot = np.zeros(shape=(L,L), dtype=int)
    #     for j in range(L):
    #         for i in range(L):
    #             sitio = i+j*L
    #             splot[i][j] = s[sitio]
                
    #     fig, ax = plt.subplots(figsize=(16,9), dpi=60)
    #     ax.tick_params(axis="x", labelsize=20)
    #     ax.tick_params(axis="y", labelsize=20)
    #     ax.spines['left'].set_linewidth(2)
    #     ax.spines['bottom'].set_linewidth(2)
    #     ax.spines['right'].set_linewidth(2)
    #     ax.spines['top'].set_linewidth(2)
        
    #     plot = ax.imshow(splot, cmap='winter', label='teste')
    #     #Colorbar confgis
    #     cmap = plt.cm.cool
    #     bounds = np.linspace(-1, 1, 3)
    #     norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    #     m = cm.ScalarMappable(cmap='winter')
    #     m.set_array([-1,1])
    #     plt.colorbar(m, ax=ax,norm=norm ,boundaries=bounds,
    #     spacing='proportional', ticks=[-1,1], format='%1i')
        
        
    #     plt.title("MCS={}".format(t), fontsize=20)
    #     plt.savefig('C:/Users/mateu/workspace/MonteCarlo/Ising/images/teste{:03}.png'.format(t), format='png')
        
        
    
    def escreve(t, mag, E, TEMP):
        if TEMP == 2.2:
            path = "C:/Users/mateu/workspace/MonteCarlo/Ising/data22.csv"
        elif TEMP == 2.4:
            path = "C:/Users/mateu/workspace/MonteCarlo/Ising/data24.csv"
        else:
            path = "C:/Users/mateu/workspace/MonteCarlo/Ising/data30.csv"
            
        with open(path, "a") as file:
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
            
    
    def dinamica(s, TEMP):
            restart_csv(TEMP)             # reiniciando medidas
            viz = init_viz(L2)            # iniciando vizinhos
            E, mag = init_medidas(s, viz) # iniciando energia e mag
            
        
            for t in range(Teq):              # loop para equilibrar
                mag, E = passo(s, mag, E, viz)
                
            for t in range(T):                # loop com medidas
                mag, E = passo(s, mag, E, viz)
                if t%(50)==0:
                    escreve(t, mag, E, TEMP)
                
                    
            print("Resultados para T={}".format(T+Teq))
    
                
                
         
           
    
    
    
    
    start_time = time.time()
    
    
    s = S.copy()  
    dinamica(s, TEMP)
    
    print("--- {} segundos para L = {} ---".format(time.time() - start_time, TEMP))














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



