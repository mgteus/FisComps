from cProfile import label
import numpy as np

import matplotlib.pyplot as plt
from typing import List, Tuple

from aux_funcs import salva_valores, read_csv, grafico_energias, salva_velocidades_particulas
# DEFINICOES DE ESTRUTURAS
PARTICULAS = List[Tuple[float,float]]

# DEFINICOES DAS CONSTANTES
GAMMA = 2.0
TEMP  = 0.8
KB    = 1.0
K     = 3.0
MASSA = 1.0
TAU   = 0.01
GKBT  = 2*(GAMMA*KB*TEMP)/MASSA
SEED  = 16

# seed para o pGNA
np.random.seed(SEED)


def inicia_lista_particulas(N: int = 1
                     ,x0: float = 0
                     ,v0: float = 0,
                     ) -> PARTICULAS: 
    """
    Funcao que inicia a lista de N particulas com velocidade v0 na posiçao x0.


    Retorna uma lista no formato PARTICULAS.
    """
    particulas = []
    for _ in range(N):
        particulas.append((x0,v0))

    return particulas


def forca_externa(x: float = 0, k: float = 0) -> float:
    """
    Funcao da forca externa que devolve -x*k
    """
    return -x*k


def passo(particula: Tuple[float,float])-> Tuple[float,float]:
    """
    Funcao de passo para uma particula

    retorna uma tupla com os novos valores de x e v
    """
    x = particula[0]
    v = particula[1]
    forca = forca_externa(x = x, k = K) # K = CONSTANTE

    # atualizando valores de x e v
    gauss = np.random.normal()
    v_aux = (1-GAMMA*TAU)*v + (forca/MASSA + np.sqrt(GKBT/TAU)*gauss)*TAU
    v = v_aux 

    x = x + v*TAU

    return (x, v)


def calc_energias(particulas: PARTICULAS)-> Tuple[float, float]:
    """
    Funcao que calcula a energia cinetica media e energia potencial media.

    retorna uma tupla com os valores de cada energia (K, U)
    """
    K = 0
    U = 0
    nmr_de_particulas = len(particulas)
    for i in range(nmr_de_particulas):
        part = particulas[i]
        x = part[0]
        v = part[1]

        # atualizando valores das energias
        K  += (1/2)*MASSA*(v*v)

        U  +=  (1/2)*K*(x*x)


    return K, U


if  __name__ == '__main__':
    # variaveis iniciais
    # N  = 1000   # numero de particulas
    # x0 = 0      # posicao inicial 
    # v0 = 0      # velocidade inicial 
    # TMAX = 100  # tempo maximo de simulacao
    # DT = TAU    # passo temporal 
    # T = 0       # tempo inicial 
    # K = 0       # energia cinetica  
    # U = 0       # energia potencial 
    # # dinamica
    # particulas = inicia_lista_particulas(N=N, x0=x0, v0=v0)
    # pos = []
    # # loop temporal
    # while T < TMAX:
    #     path_ = salva_valores(name = 'teste1',
    #                              T = T,
    #                              K = K/N,
    #                              U = 2*U/N,
    #                            new = T <= 0)
    #     salva_velocidades_particulas(particulas = particulas,
    #                                         new = T <= 0)
    #     # loop nas particulas
    #     for i in range(N):
    #         #recebe a particula
    #         part = particulas[i]
            
                

    #         #atualiza a particula
    #         part = passo(part)

    #         #salva a particula novamente na lista
    #         particulas[i] = part

    #     # calculando energia 
    #     K, U = calc_energias(particulas=particulas)
    #     velos = [i[0] for i in particulas]
    #     pos.append(sum(velos))
    #     # atualizando o tempo
    #     T = T + DT
    

    # df = read_csv(path_)

    # grafico_energias(K   = df['K'],
    #                  U   = df['U'],
    #                  KBT = KB*TEMP/2,
    #                  )
    # plt.show()
    # grafico_energias(x=pos)
    # plt.show()

    # grafico_energias(U=df['U'])
    # plt.show()

    # df = read_csv(r'C:\Users\mateu\Desktop\UFRGS\2021(2)\Processos Estocásticos\teste1.csv')

    # fig, ax = plt.subplots(figsize=(16,9), dpi=120)
 
    
    # ax.tick_params(axis="x", labelsize=20)
    # ax.tick_params(axis="y", labelsize=20)
    # ax.spines['left'].set_linewidth(2)
    # ax.spines['bottom'].set_linewidth(2)
    # ax.spines['right'].set_linewidth(2)
    # ax.spines['top'].set_linewidth(2)

    # plt.plot([KB*TEMP/2 for i in range(2000)], label='KBT', lw=3, ls= '--', c='k')
    # plt.plot(df['K'].iloc[0:2000], label='K', lw=2, c='r')
    # plt.plot(df['U'].iloc[0:2000], label='U', lw=2, c='b')
    # plt.title('Gráfico de K e U', fontsize=30)
    # plt.xlabel(r'Passos ($dt$)', fontsize=25)
    # plt.ylabel('Energia', fontsize=25)
    # plt.legend()

    # plt.show()



    dfv = read_csv(r'C:\Users\mateu\Desktop\UFRGS\2021(2)\Processos Estocásticos\velocs_particulas.csv')

    dfv['P1'].plot()
    plt.show()





        
    

