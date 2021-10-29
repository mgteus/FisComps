import os 
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import random as rd
import time
from matplotlib.animation import   FuncAnimation
import matplotlib as mpl
from matplotlib import cm
from modules import plot_snapshot
mpl.rc('figure', max_open_warning = 0)




class Potts():

    def init_viz():

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
        np.fill_diagonal(self.krone, 1)
        # iniciando a rede

        self.s = rd.choices(range(1, self.Q+1), k=self.N2)
        self.splot = np.zeros(shape=(self.N, self.N), dtype=int)
        self.t = 0

        #gerador de numero aleatorio
        rng = default_rng(seed=42)
        #prob do flip
        prob = 0


        return
    
    def show_status(self):
        """
        Funcao pra plotar o status atual da rede
        """

        print(self.temp)
        print(self.N, self.N2)
        print(self.Q)
        print(self.krone[5][3])

    def snapshot(self):
        """
        Funcao que plota uma snapshot do T atual
        """
        for j in range(self.N):
            for i in range(self.N):
                sitio = i+j*self.N
                self.splot[i][j] = self.s[sitio]
        plot_snapshot(self.splot, f"TEMPO = {self.t}", self.Q)

        return

    def run(self, plot: bool=False):
        """
        Funcao da dinamica de Monte Carlo do modelo
        """
        sitio = np.random.randint(self.N2)

        """
        Da pra usar a mesma dinamica que o WOLFF (email Heitor), cuidar com as medidas 
        dos observáveis e usar q=2 como espelho pra saber se está tudo certo.


        Da pra paralelilzar usando o checkerboard (mestrado Luis)
        """

        return 
    



if __name__ == '__main__':
    x = Potts(temp=2.1, N=64, Q=10, TMAX=10)
    x.show_status()

    #x.snapshot()
