import os 
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import random as rd
import time
import matplotlib
from matplotlib.animation import   FuncAnimation
import matplotlib as mpl
from matplotlib import cm
from modules import energy_ts
from modules import plot_snapshot
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
    plot_snapshot(splot, f"MCS = {potts.MCS} com Q={potts.Q} e T = {potts.temp}", potts.Q, save)

    return


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

    def  __init__(self, temp, N, Q, TMAX):
        rd.seed(42)
        np.random.seed(42)
        self.temp = temp
        self.N = N
        self.N2 = N**2
        self.Q = Q
        self.TMAX = TMAX
        self.krone = np.zeros(shape=(Q,Q), dtype=int)
        self.en_list = []
        self.MCS = 0
        np.fill_diagonal(self.krone, 1)
        # iniciando a rede

        if Q == 2:
            self.q_list = [-1,1] # lista de q's
            self.s = rd.choices(self.q_list, k=self.N2)
            self.magne = np.sum(self.s)
        else:
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
        #peso
        self.peso = np.exp(-self.E/self.temp)

        # Z
        #self.Z = self.peso

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
        possibles_newspin = [ x for x in range(1, self.Q+1) if x != oldspin ]
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
                    rfloat1 = self.rng.random()
                    if (rfloat1 < self.prob) : #IF da inclusão no cluster
                    
                        stack.append(nn)
                        sp = sp + 1
                        self.s[nn] = newspin

    def energia(self):

        self.E = 0 

        for sitio in range(self.N2):  # passando por todos os sitios da rede
            for j in range(1, 3):     # j = {1,2}   # loop nos vizinhos 
                if self.s[sitio] == self.s[self.viz[sitio][j]]: # if da orientacao sitio-vizinho
                    self.E = self.E + 1
        return self.E

    def mag(self):

        return
        
        # q_dict = {} # np.zeros(shape=(self.N, self.N), dtype=int)

        # for q in self.s:
        #     if q not in q_dict:
        #         q_dict[q] = 1
        #     else:
        #         q_dict[q] += 1 

        
        # # Plot histogram.
        # x_span = self.q_list[0] - self.q_list[-1]
        
        # norm = matplotlib.colors.Normalize(vmin=.5, vmax=self.Q+.5)
        # cm = matplotlib.cm.get_cmap('rainbow')

        # y = plt.get_cmap('rainbow', self.Q)

        


        # C = [y(x/self.Q) for x in self.q_list]
        # plt.bar(q_dict.keys(), q_dict.values(), color=C)
        # plt.show()
        # C2 = [cm(x)  for x in self.q_list]
        # plt.bar(q_dict.keys(), q_dict.values(), color=C2)
        # plt.show()

    def step(self):
        """
        Funcao da dinamica de Monte Carlo do modelo
        """
        for i in range(self.N):
            if self.t%50:
                self.E = Potts.energia(self)
                self.en_list.append(self.E)

            sitio = np.random.randint(self.N2)
            
            Potts.cluster_din(self, sitio)

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
    en_list = []
    temps_list = [0.8, 2., 2.2, 2.5, 3]
    for temp in temps_list:
        x = Potts(temp=temp, N=32, Q=5, TMAX=10**4)
        x.run(100)

        en_list.append(x.en_list)
    
    energy_ts(en_list, temps_list)
        

     
    

    """ 
    TESTAR COM O JAX E AJEITAR MAGNETIZACAO PARA CALCULAR A DO ISING
    NA FUNCAO MAG, CONFERIR SE Q == 2, CASO SEJA IGUAL A 2 CALCULAR A MAG IGUAL
    AO ISING PADRAO
    """



