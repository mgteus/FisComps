# -*- coding: utf-8 -*-
"""
Created on Fri May  7 08:38:43 2021

@author: mgteus

"""
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import random as rd
import time
from matplotlib.animation import   FuncAnimation
import matplotlib as mpl
mpl.rc('figure', max_open_warning = 0)

#---------------------PARAMETROS----------------------------------
# temperatura
TEMP = 2.8
# numero de spins
N = 64
N2 = N**2
# numero de trocas
TMAX = 200
#vetor da rede
s = rd.choices([-1,1], k=N2)
# gerador de numero aleatorio
rng = default_rng()
#prob do flip
prob = 1 - np.exp(-2/TEMP)
#matriz para o plot
splot = np.zeros(shape=(N,N), dtype=int)
rd.seed(42)


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
                 
        
    return 
#---------------- "BUFFER" -----------------

#----------------- PLOT FUNCTION --------------------
def plot_din(s, t):
    # transformando s em 2D para o plot
    for j in range(N):
        for i in range(N):
          sitio = i+j*N
          splot[i][j] = s[sitio]
          
    plt.title("TEMPO {}".format(t))
    return plt.matshow(splot, cmap='cool', fignum=None)

#----------------- PLOT FUNCTION --------------------



#-------DINÂMICA-----------------
plot_din(s, t=0)
for t in range(TMAX):
    # rotina da dinamica
    # vou escolher um sitio aleatorio
    sitio = np.random.randint(N2)

    
    cluster_din(sitio)
    plot_din(s, t)
    """PLOT"""
    
    

#--------DINÂMICA--------------
    








