import os
import numpy as np
from numpy.lib.function_base import rot90
from numpy.random import random as npr
from numpy.random import default_rng
import matplotlib.pyplot as plt
import random as rd
import time
import matplotlib
from matplotlib.animation import   FuncAnimation
import matplotlib as mpl
from matplotlib import cm
from modules import energy_ts, mag_ts
from modules import plot_snapshot
from modules import write_list_to_file
from modules import get_list_from_file
from numba import int32, float32
from numba.experimental import jitclass

from concurrent.futures import ThreadPoolExecutor, thread


mpl.rc('figure', max_open_warning = 0)

def snapshot(potts, save):
    """
    Funcao que plota uma snapshot do MCS atual
    """
    splot = np.zeros(shape=(potts.N, potts.N), dtype=int)

    for j in range(potts.N):
        for i in range(potts.N):
            sitio = i+j*potts.N
            splot[i][j] = potts.s[sitio]
    plot_snapshot(splot, f"MCS = {potts.MCS} com Q={potts.Q} e T = {potts.temp}",
         potts.Q, save, potts.t)

    return


def create_rede(temp, N, Q, TMAX, alg ) -> list:

    x = Potts(temp=temp, N=N, Q=Q, TMAX=TMAX, alg=alg)
    x.run(TMAX)

    return [x.en_list, x.lista_sitios]

class Potts():

    def init_viz(N):
        N2 = N**2

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

    def  __init__(self, temp, N, Q, TMAX, alg):
        rd.seed(42)
        np.random.seed(42)
        self.temp = temp
        self.N = N
        self.N2 = N**2
        self.Q = Q
        self.TMAX = TMAX
        self.alg = alg
        self.krone = np.zeros(shape=(Q,Q), dtype=int)
        self.en_list = []
        self.mag_list = []
        self.MCS = 0
        np.fill_diagonal(self.krone, 1)
        # iniciando a rede

        # if Q == 2:
        #     self.q_list = [0,1] # lista de q's
        #     self.s = rd.choices(self.q_list, k=self.N2)
        #     self.magne = np.sum(self.s)
        # else:
        self.q_list = [i for i in range(1, self.Q+1)] # lista de q's
        self.s = rd.choices(self.q_list, k=self.N2)

        self.splot = np.zeros(shape=(self.N, self.N), dtype=int)
        self.t = 0

        #gerador de numero aleatorio
        self.rng = default_rng(seed=42)
        #prob do flip
        self.prob = 1 - np.exp(-1./self.temp)
        #criando matriz de vizinhos
        self.viz = Potts.init_viz(self.N)
        #energia
        self.E = 0
        self.E_ = 0
        #magnetizacao
        self.magne = 0
        #peso
        self.peso = np.exp(-self.E/self.temp)


        self.lista_sitios = [np.random.randint(self.N2) for i in range(5*10**5+30)]
        self.lista_rands = [self.rng.random() for _ in range(10*4+30)]
        
        #TMAX*self.N2+1
        time.sleep(3)

        return

    def show_status(self):
        """
        Funcao pra plotar o status atual da rede
        """

        print(self.temp)
        print(self.N, self.N2)
        print(self.Q)
        print(self.krone[5][3])

        return

    def cluster_din(self, sitio):
        stack = []
        oldspin = self.s[sitio]
        possibles_newspin = [ x for x in self.q_list if x != oldspin ]
        newspin = rd.choice(possibles_newspin)
        self.s[sitio] = newspin
        sp = 1
        stack.append(sitio)

        while (sp):
            sp = sp - 1
            atual = stack[sp]
            stack.pop()

            for j in range(4):
                nn = self.viz[atual][j]
                if self.s[nn] == oldspin: #IF da orientação do vizinho
                    #print(self.t)
                    rfloat1 = self.rng.random()
                    if (rfloat1 < self.prob) : #IF da inclusão no cluster
                        stack.append(nn)
                        sp = sp + 1
                        self.s[nn] = newspin
        return

    def metropolis_din(self, sitio):
        possibles_q = [q for q in self.q_list if q != self.s[sitio]]
        new_state  = rd.choices(possibles_q)[0]

        E1 = 0
        E2 = 0

        for j in range(4):
            if self.s[sitio] == self.s[self.viz[sitio][j]]:
                E1 = E1 - 1
        for j in range(4):
            if new_state == self.s[self.viz[sitio][j]]:
                E2 = E2 -1

        deltae = E2 - E1

        if deltae <= 0:
            self.s[sitio] = new_state
        else:
            rfloat1 = self.rng.random()
            if rfloat1 < self.prob: # if para flipar o sitio
                self.s[sitio] = new_state

    def energia_e_mag(self):

        self.E = 0
        q_dict = {}
        for sitio in range(self.N2):  # passando por todos os sitios da rede
            #mag
            if self.s[sitio] not in q_dict:
                q_dict[self.s[sitio]] = 1
            else:
                q_dict[self.s[sitio]]+=1

            #energia
            for j in range(1, 3):     # j = {1,2}   # loop nos vizinhos
                if self.s[sitio] == self.s[self.viz[sitio][j]]: # if da orientacao sitio-vizinho
                    self.E = self.E - 1

        N_max = max(q_dict.values())
        self.magne = (self.Q*(N_max/self.N2) -1)/(self.Q - 1)

        # ajustando E_0
        if self.E_ == 0:
            self.E_ = self.E

        return self.E, self.magne

    def mag(self):

        return

    def step(self):
        """
        Funcao da dinamica de Monte Carlo do modelo
        """
        for i in range(1):
            if self.t%50:
                self.E, self.magne = Potts.energia_e_mag(self)
                self.en_list.append(self.E)
                self.mag_list.append(self.magne)


            sitio = self.lista_sitios[self.t]


            if self.alg == 0: # wolff
                Potts.cluster_din(self, sitio)
            elif self.alg == 1: # metropolis
                Potts.metropolis_din(self, sitio)
            else:
                pass # fazer banho termico

            self.t = self.t + 1
        self.MCS = self.MCS + 1

        return

    def run(self, n_passos: int=0):
        if n_passos != 0:
            for _ in range(n_passos):
                Potts.step(self)
        else:
            for _ in range(self.TMAX):
                Potts.step(self)

        return


if __name__ == '__main__':

    threads = 4
    with ThreadPoolExecutor(threads) as Exec:
       #                                 temp, N, Q, TMAX, alg
        task1 = Exec.submit(create_rede, 0.91, 32, 4, 10, 0)
        task2 = Exec.submit(create_rede, 1.21, 32, 4, 10, 0)
        task3 = Exec.submit(create_rede, 0.61, 32, 4, 10, 0)
        
    t1 = task1.result()
    t2 = task2.result()
    t3 = task3.result()
    

    temps = [0.91, 1.51, 0.31]#
    alg = [0,0,0]
    
    mag_list = [t1[1][:50], t2[1][:50], t3[1][:50]]

    mag_ts(mag_list, temps, alg, 1000)
    en_list = [t1[0][:50], t2[0][:50], t3[0][:50]]

        






    
    # y = Potts(temp=0.72, N=64, Q=9, TMAX=10**5, alg=0)
    # y.run(2*10**4)

    # snapshot(y, True)
    # for i in range(20):
    #     y.run(1)
    #     snapshot(y, True)
    
    # mag_ts(y.mag_list, y.temp, y.alg, y.TMAX)

    # energy_ts(y.en_list, y.temp, y.alg, y.TMAX)
    






    #
    #y.run(10**6)

    #write_list_to_file(y.en_list, 'wolff_q4_t1')
    #mag_list.append(y.en_list)

    #x = Potts(temp=2.269, N=32, Q=10, TMAX=1, alg=0)


    #plt.plot()
    # x.run(10**5)
    # write_list_to_file(x.en_list, 'wolff_q10')

    #mag_list.append(x.en_list)
    # snapshot(y, True)
    # for i in range(20):
    #     y.run(1)
    #     snapshot(y, True)
    # # # write_list_to_file(y.mag_list, 'TESTE_M-1')
    # # # write_list_to_file(y.en_list, 'TESTE_E-1')


    #en = get_list_from_file('wolff_q4.txt')
    #energy_ts(time_series=en, temps=0.702, algs=0, MCS=10**5)

    #y.run(10)
    #snapshot(y, False)
    # for i in range(10):
    #     y.run(1)
        #snapshot(y, False)
    #snapshot(y, False)
    # rng = default_rng(seed=42)
    # mag_ts([rng.random() for i in range(1000)], 10, [1], 100)



    # for i in range(1):
    #     y.run()
    #     snapshot(y, False)



    # en_list = []
    # mg_list = []
    # files   = []
    # temps_list = [0.7, 0.7, 10, 10]
    # algs = [0,1,0,1]
    # for temp, alg in zip(temps_list, algs):
    #     x = Potts(temp=temp, N=64, Q=10, TMAX=100, alg=alg)
    #     x.run()

    #     en_list.append(x.en_list)
    #     mg_list.append(x.mag_list)
    #     filename_e = f"{alg}_en_{str(temp).replace('.', '-')}"
    #     filename_m = f"{alg}_mag_{str(temp).replace('.', '-')}"
    #     files.append(filename_e)
    #     files.append(filename_m)
    #     write_list_to_file(x.en_list, filename_e)
    #     write_list_to_file(x.mag_list, filename_m)
    #     print(f'T = {temp}, alg={alg} Finalizado')

    # y = Potts(temp=0.7, N=64, Q=10, TMAX=100, alg=1)
    # y.run()


    # mag_ = y.mag_list
    # mag__ = get_list_from_file('0_mag_0-7.txt')


    # #energy_ts(mag_, 0.7, [1], 100)
    # plt.hist(mag__, label='W', bins=100)
    # plt.hist(mag_, label='M', bins=100)

    # plt.legend()
    # plt.show()


    """
    TESTAR COM O NUMBA CLASSES E AJEITAR MAGNETIZACAO PARA CALCULAR A DO ISING
    NA FUNCAO MAG, CONFERIR SE Q == 2, CASO SEJA IGUAL A 2 CALCULAR A MAG IGUAL
    AO ISING PADRAO
    """



