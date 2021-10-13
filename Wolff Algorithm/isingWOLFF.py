"""
Created on Fri May  7 08:38:43 2021

@author: mgteus

"""
import os 
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import random as rd
import time
from matplotlib.animation import   FuncAnimation
import matplotlib as mpl
from matplotlib import cm
mpl.rc('figure', max_open_warning = 0)


#---------------------PARAMETROS----------------------------------
#seeds
rd.seed(42)
np.random.seed(42)
#temperatura
TEMP=1.8
#numero de spins
N = 64
N2 = N**2
#numero de trocas
TMAX = 300


#vetor da rede
s = rd.choices([1,-1], k=N2)


#gerador de numero aleatorio
rng = default_rng(seed=42)

#prob do flip
prob = 1 - np.exp(-2/TEMP)
    
#matriz para o plot
splot = np.zeros(shape=(N,N), dtype=int)

#lista para contagem do tamanho dos clusters
lista_cluster_size=[]



#---------------------PARAMETROS----------------------------------

#--------------VIZINHOS------------------
#def init_vizinos():
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
            if s[nn] == oldspin: #IF da orientação do vizinho
             rfloat1 = rng.random()
             if (rfloat1<prob) : #IF da inclusão no cluster
                 
                 stack.append(nn)
                 sp = sp+1
                 s[nn] = newspin
                 
                 
                 
#---------------- "BUFFER" -----------------




def cluster_din0(sitio):
    """Função que gera os clusters na cor roxa para visualização."""
    stack = []
    oldspin = s[sitio]
    newspin = 0
    s[sitio] = 0
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
 
            
     
 # ----------- CLUSTER SIZE ----------------   
def cluster_n(d1,d2):
    cluster_size = (d1[1][1]-d2[1][1])/N2
    #if cluster_size
    lista_cluster_size.append(abs(cluster_size))

 # ----------- CLUSTER SIZE ----------------  
    
 

#=============== DINAMICA ======================
print(os.getcwd())

for t in range(TMAX):
    for j in range(N):
        for i in range(N):
            sitio = i+j*N
            splot[i][j] = s[sitio]
            
    fig, ax = plt.subplots(figsize=(16,9), dpi=60)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    
    plot = ax.imshow(splot, cmap='cool', label='teste')
    #Colorbar confgis
    cmap = plt.cm.cool
    bounds = np.linspace(-1, 1, 3)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    m = cm.ScalarMappable(cmap='cool')
    m.set_array([-1,1])
    plt.colorbar(m, ax=ax,norm=norm ,boundaries=bounds,
    spacing='proportional', ticks=[-1,1], format='%1i')
    
    if t in [256,257,258]:    
        plt.title("T={} e MCsteps={}".format(TEMP, t), fontsize=20)
        plt.savefig(r"Wolff Algorithm\images\teste{:03}.png".format(t), format='png')
        plt.show()
    

        

    """ROTINA"""    
    # Sorteio do spin 
    sitio = np.random.randint(N2)
    
    #passo o sitio para o buffer
    cluster_din(sitio)
    

    
#=============== DINAMICA ======================    






    
    
    







