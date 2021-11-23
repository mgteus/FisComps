import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
mpl.rc('figure', max_open_warning = 0)
import numpy as np
import collections
import os




def plot_snapshot(rede: np.array = np.array([0,0]), title: str = "",
                 Q: int=0, save: bool = False, MCS: int = 0) -> plt.figure:
    """
    Funcao que plota o estado atual da rede, recebendo um array
    """

    if Q == 0:
        print('Q = 0')
        return


    fig, (ax0, ax1) = plt.subplots(ncols=2, nrows=1,figsize=(16,9), dpi=120, gridspec_kw={'width_ratios': [3, 1]})
    fig.suptitle(title, fontsize=35)
    ax0.tick_params(axis="x", labelsize=20)
    ax0.tick_params(axis="y", labelsize=20)
    ax0.spines['left'].set_linewidth(2)
    ax0.spines['bottom'].set_linewidth(2)
    ax0.spines['right'].set_linewidth(2)
    ax0.spines['top'].set_linewidth(2)
    
    #Colorbar confgis
    cmap = plt.get_cmap('rainbow', Q)


    mat = ax0.imshow(rede,cmap=cmap, vmin = 1-.5, vmax = Q+.5)
    #tell the colorbar to tick at integers
    cbar = plt.colorbar(mat, ticks=range(1,Q+1))
    cbar.ax.tick_params(labelsize=20) 

    
    ax0.set_title(f'Rede {rede.shape[0]}x{rede.shape[1]}', fontsize=30)
    ax0.grid(color='k', linewidth=4, which='minor')
    ax0.set_frame_on(False)


    ax1.tick_params(axis="x", labelsize=20)
    ax1.tick_params(axis="y", labelsize=20)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(2)
    ax1.spines['top'].set_linewidth(2)

    ax1.set_title('Histograma', fontsize=30)

    q_dict = {} # np.zeros(shape=(self.N, self.N), dtype=int)

    for q in rede.reshape(-1):
        if q not in q_dict:
            q_dict[q] = 1
        else:
            q_dict[q] += 1 
    q_dict = collections.OrderedDict(sorted(q_dict.items()))

    norm = mpl.colors.Normalize(vmin=.5, vmax=Q+.5)
    ax1.barh(list(q_dict.keys()), list(q_dict.values()), color = [cmap(norm(x)) for x in q_dict.keys()])
    ax1.set_yticks(list(q_dict.keys()))

    fig.tight_layout()
    if save:
        path = r'C:\Users\mateu\workspace\MonteCarlo\Potts\img'
        filename = 'fig{:03d}.png'.format(MCS)
        plt.savefig(os.path.join(path, filename), format='png')
    else:
        plt.show()


    return





def energy_ts(time_series: list = [0], temps: list = [0]) -> plt.figure:
    """
    Funcao que devolve um gráfico da série temporal dos valores de energia
    """

    """
    TENTAR FAZER UM GRAFICO DA SERIE TEMPORAL EM CIMA (RETANGULAR) E ABAIXO 
    HISTOGRAMAS DAS DISTRIBUIÇOES DOS VALORES DE ENERGIA
    """
    if isinstance(time_series[0], list): # if para saber se veio uma ou mais series temporais          
            n_energias = len(time_series)          
    else:
        n_energias = 1

    fig = plt.figure(constrained_layout=True, figsize=(16, 9))
    subfigs = fig.subfigures(2, 1, wspace=0.07) 
    plot_en_ts = subfigs[0]
    plot_en_hist = subfigs[1]


    """ INICIO PRIMEIRO PLOT """
    # plot pricipal da serie temporal
    ax = plot_en_ts.gca()
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)
    #ax.set_xticks(range(len(time_series[0])))
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)

    if n_energias != 1: # plot de várias temperaturas
        cmap = plt.get_cmap('jet', len(temps))
        norm = mpl.colors.Normalize(vmin=.5, vmax=max(temps)+0.5)
        for ts, temp in zip(time_series, temps):         
            ax.plot(ts, lw=2, c=cmap(norm(temp)), label=f"T = {temp}")

        ax.legend()
        plot_en_ts.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ticks=temps)

    else:
        cmap = plt.get_cmap('jet', 1)
        norm = mpl.colors.Normalize(vmin=.5, vmax= temps + 0.5)
        ax.plot(time_series, lw=3, c=cmap(norm(temps)), label=f"T = {temps}")
        ax.legend()
        
    """ FIM PRIMEIRO PLOT """
        
    # plot secundario, histogramas das energias
    plot_en_hist_sub = plot_en_hist.subfigures(1, n_energias)


    if isinstance(plot_en_hist_sub, np.ndarray):   # se for lista eh pq existe mais de uma energia
        for en_plot, temp, ts in zip(plot_en_hist_sub, temps, time_series):
            ax_hist = en_plot.subplots(1,1)
            ax_hist.hist(ts, label=temp, color=cmap(norm(temp)))
            en_plot.suptitle(f'Histograma para T = {temp}')
            ax_hist.legend()
         
    else:
        plot_en_hist_sub.suptitle(f'Histograma para T = {temps}')
        ax_hist = plot_en_hist_sub.subplots(1,1)                                  
        ax_hist.hist(time_series, label=temps, color=cmap(norm(temps)))


        

    plt.show()

    








if __name__ == '__main__':
    # energy_ts(time_series=[1,2,3,4,5,1,1,1,1],temps=5)
    # import random
    # x = 100
    #energy_ts(time_series=[[1*random.random() for i in range(x)],
                            # [20*random.random() for i in range(x)],
                            # [30*random.random() for i in range(x)],
                            # [40*random.random() for i in range(x)],
                            # [50*random.random() for i in range(x)]],temps=[1, 2, 3, 4, 5])

    for i in range(10):
        print('ahhh{:03d}'.format(i))