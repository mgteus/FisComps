import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import random as rd
import time
from numba import jit

## PARAMETROS
# temperatura
TEMP = np.array(2.269, dtype=np.float32)
rd.seed(42)
np .random.seed(42)
#H = ? para campo diferente de zero
#J = ? para J diferente de 1
# numero de spins
N = 16
N2 = N**2
# tempo de SIM
TMAX = 1000
a = np.array(rd.choices([-1,1], k=N2), dtype=np.float32)
rng = default_rng(seed=42)
#np.seed(42)

s = np.random.randint(2, size=N2)
#print(s)
s = 2*s-1
#print(s)
mag = np.sum(a)/N2
mag2 = np.sum(s)/N2
#print(mag)
#fator = 1.0/N
#fator2 = 2.0/N
medidas1 = np.zeros(TMAX, dtype=np.float32)
medidas2 = np.zeros(TMAX, dtype=np.float32)


# defino uma matriz de vizinhos para nao ter que calcular
# os vizinhos a cada passo
@jit(nopython=True)
def init_viz(N2):
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
    return viz
    
    
    
    


@jit(nopython=True)
def dinamica():
    for temp in TEMP:
        mag2 = np.sum(s)/N2
        viz = init_viz(N2)
        for t in range(TMAX):
            # rotina da dinamica
            for i in range(N2):
                # vou escolher um sitio aleatorio
                sitio = np.random.randint(N2)
                deltae = 2*s[sitio]*(s[viz[sitio,0]] + s[viz[sitio,1]] +s[viz[sitio,2]]+s[viz[sitio,3]])
                prob = np.exp(-deltae/temp)
                rfloat1 = rng.random()
                if(rfloat1<prob) :
                    # flipo o sitio
                    s[sitio]*=-1;
                    mag2 = mag2 + 2*s[sitio]
            medidas1[t] = mag2
        



plt.plot(medidas1, label="mag")
#print(sum(medidas6)/TMAX)
#plt.xlim(0,1500)
#print(time.process_time()-c)
# plt.plot(medidas6, label="T={}".format(temp))
# plt.xlim(0,1000)
plt.legend(loc='best')
plt.show()




#TMAX = 1000    2.234s
#TMAX = 10000   22.843s
#TMAX = 100000  224.573s
