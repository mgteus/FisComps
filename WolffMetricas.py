"""
Created on Fri May  7 08:38:43 2021

@author: mgteus

"""
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import random as rd
import time






#---------------------PARAMETROS----------------------------------
rd.seed(42)
np.random.seed(42)
#temperatura
TEMP = np.linspace(0.25,5,25)
#numero de spins
N = 64
N2 = N**2
#numero de trocas
TMAX = 500
rng = default_rng(seed=42)

#lista para as metricas
metricas=[]
metricas2=[]



#---------------------PARAMETROS----------------------------------

#--------------VIZINHOS------------------
viz=np.zeros(shape=(N2,4),dtype=int)
for sitio in range(N2):
    n1 = ((sitio//N -1 +N2)%N)*N + sitio%N
    n2 = ((sitio//N)%N)*N+(sitio+1+N)%N
    n3 = ((sitio//N +1 +N2)%N)*N +sitio%N
    n4 = ((sitio//N)%N)*N+(sitio-1+N)%N
    viz[sitio][0] = n1
    viz[sitio][1] = n2
    viz[sitio][2] = n3
    viz[sitio][3] = n4
#------------VIZINHOS---------------------

#---------------- "BUFFER" --------------
def cluster_din(sitio):
    stack = []
    oldspin = s[sitio]
    newspin = (-1)*s[sitio]
    s[sitio] = newspin
    sp=1
    stack.append(sitio)
    
    while (sp):
        sp = sp-1
        atual = stack[sp]
        stack.pop()        
    
        for j in range(4):
            nn=viz[atual][j]
            if s[nn] == oldspin:   
             rfloat1 = rng.random()
             if (rfloat1<prob) : #IF da inclusão no buffer
                 
                 stack.append(nn)
                 sp = sp+1
                 s[nn] = newspin
#---------------- "BUFFER" -----------------
 
            
     
 # ----------- CLUSTER SIZE ----------------   
def cluster_n(d1,d2):
    cluster_size = (d1[1][1]-d2[1][1])/N2
    #if cluster_size
    lista_cluster_size.append(abs(cluster_size))

 # ----------- CLUSTER SIZE ----------------  
    
 

#=============== DINAMICA ======================
for temp in TEMP:
    #iniciando a rede a cada troca de temperatura
    s = rd.choices([-1, 1], k=N2)
    #prob do flip
    prob = 1 - np.exp(-2/temp)
    #matriz para o plot
    splot = np.zeros(shape=(N,N), dtype=int)
    
    #lista para contagem do tamanho dos clusters
    lista_cluster_size=[]
    
    for t in range(TMAX):
        for j in range(N):
            for i in range(N):
                sitio = i+j*N
                splot[i][j] = s[sitio]    
        #ROTINA:
        #d1 = cluster size inicial 
        d1 = np.unique(s,  return_counts=True)
        
        # Sorteio do spin 
        sitio = np.random.randint(N2)
        
        #passo o sitio para o buffer
        cluster_din(sitio)
        
        #d2 = cluster size após o passo 
        d2 = np.unique(s,  return_counts=True)
        
        
        #calculando o tamanho do cluster
        cluster_size = (d1[1][0]-d2[1][0])/N2
        lista_cluster_size.append(abs(cluster_size))
        
    media_c = np.mean(lista_cluster_size)
    cluster_sorted = np.sort(lista_cluster_size)
    metricas.append(cluster_sorted[-1])
    metricas2.append(media_c)
    
    print("TERMINEI T={}".format(temp))
    
#=============== DINAMICA ======================    

  



fig, ax = plt.subplots(figsize=(16,9), dpi=90)

ax.tick_params(axis="x", labelsize=20)
ax.tick_params(axis="y", labelsize=20)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

    
    
    
plt.plot(TEMP, metricas, label='max', lw=3)
plt.scatter(TEMP, metricas, s=3)
#plt.plot(TEMP, metricas2, label='media')
plt.legend(loc='best', fontsize=20)
plt.show()
    







