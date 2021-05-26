import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import random as rd
import time

## PARAMETROS
# temperatura
TEMP = [2.3]
#H = ? para campo diferente de zero
#J = ? para J diferente de 1
# numero de spins
N = 16
N2 = N**2
# tempo de SIM
TMAX = 10000
a = rd.choices([-1,1], k=N)
a
rng = default_rng()

s = np.random.randint(2, size=N2)
#print(s)
s = 2*s-1
#print(s)
mag = np.sum(a)/N
mag2 = np.sum(s)/N
#print(mag)
#fator = 1.0/N
#fator2 = 2.0/N
medidas6 = np.zeros(TMAX, dtype=np.float32)

#print(s)

# defino uma matriz de vizinhos para nao ter que calcular
# os vizinhos a cada passo
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
c = time.process_time()
for temp in TEMP:
    for t in range(TMAX):
        # rotina da dinamica
        for i in range(N2):
            # vou escolher um sitio aleatorio
            sitio = np.random.randint(N2)
            #print(sitio,s[sitio])
            # E = - J (si. si-1 + si. si+1)
            #ei = - (s[sitio]*s[sitio-1] + s[sitio]*s[sitio+1])
            #ei = - s[sitio]*(s[sitio-1] + s[sitio+1])
            # si -> -si
            #ef =  s[sitio]*(s[sitio-1] + s[sitio+1])
            #deltae = ef - ei = 2*ef
            # ex: si = +1 e soma viz = +2
            #deltae = 2*s[sitio]*(s[sitio-1] + s[sitio+1])
            #deltae = 2*s[sitio]*(s[viz[sitio][[0]]] + s[viz[sitio][[1]]])
            deltae = 2*s[sitio]*(s[viz[sitio,0]] + s[viz[sitio,1]] +s[viz[sitio,2]]+s[viz[sitio,3]])
            prob = np.exp(-deltae/temp)
            rfloat1 = rng.random()
            if(rfloat1<prob) :
                # flipo o sitio
                s[sitio]*=-1;
                mag = mag + 2*s[sitio]
    
        medidas6[t] = mag
    plt.plot(medidas6, label="T={}".format(temp))
    
    #print(s)



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
