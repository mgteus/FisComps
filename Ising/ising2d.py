import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import random as rd
import time
from numba import njit

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
@njit(nopython=True)
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
    
    
    
    


@njit(nopython=True)
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
        



from concurrent.futures import ThreadPoolExecutor
import numpy as np
import numba
from numba import prange
import threading


#@njit(nogil=True, parallel=True)
def faz_muita_conta(N, seed):
    np.random.seed(seed)
    print("Eu sou a thread", threading.current_thread())
    x = np.random.randint(N)
    print(x)
    y = np.random.randint(N)
    print(y)
    g = 0 
    for i in range(N*60):
        g+=np.exp(i)
    
    print(g)
    
    for i in range(10**4):
        for i in range(40):
            
            if x > y:
                print("AH")
                x = np.random.randint(N)
                y = np.random.randint(N)
            else:
                print("OH")
                x = np.random.randint(N)
                y = np.random.randint(N)
        
    vec = np.array((x,y), dtype=numba.float64)
    
    return vec*vec
    



with ThreadPoolExecutor(12) as Executor:
    task1 = Executor.submit(faz_muita_conta, 6, 42)
    task2 = Executor.submit(faz_muita_conta, 5, 12)
    task3 = Executor.submit(faz_muita_conta, 8, 52)
    task4 = Executor.submit(faz_muita_conta, 6, 40)
    task5 = Executor.submit(faz_muita_conta, 5, 10)
    task6 = Executor.submit(faz_muita_conta, 8, 32)
    task7 = Executor.submit(faz_muita_conta, 6, 35)
    task8 = Executor.submit(faz_muita_conta, 5, 1)
    task9 = Executor.submit(faz_muita_conta, 8, 69)
    task10 = Executor.submit(faz_muita_conta, 6, 5)
    task11 = Executor.submit(faz_muita_conta, 5, 3)
    task12 = Executor.submit(faz_muita_conta, 8, 9)
    
    









