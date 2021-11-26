import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
mpl.rc('figure', max_open_warning = 0)
import numpy as np
import collections
import os




def plot_snapshot(rede: np.array = np.array([0,0]), title: str = "",
                 Q: int=0, save: bool = False, t: int = 0) -> plt.figure:
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
    for k, v in q_dict.items():
        ax1.annotate(f"s = {str(v)}", (0.56*v, k), fontsize=20)

    fig.tight_layout()
    if save:
        path = r'C:\Users\mateu\workspace\MonteCarlo\Potts\img'
        filename = 'fig{:07d}.png'.format(t)
        plt.savefig(os.path.join(path, filename), format='png')
    else:
        plt.show()


    return





def energy_ts(time_series: list = [0], temps: list = [0],algs:list = [0], MCS: int=0) -> plt.figure:
    """
    Funcao que devolve um gráfico da série temporal dos valores de energia
    """
    

    if isinstance(time_series[0], list): # if para saber se veio uma ou mais series temporais          
            n_energias = len(time_series) 
            ls = ['-' if i==0 else '--' for i in algs]
            algs = ['W' if alg==0 else 'M' for alg in algs] 
          
    else:
        n_energias = 1
        if algs == 0:
            algs = 'W'
            ls = '-'
        else:
            algs = 'M'
            ls   = '--'

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

    ax.set_title(f'Série Temporal de E para {MCS+1}MCS')

    if n_energias != 1: # plot de várias temperaturas
        cmap = plt.get_cmap('jet', len(set(temps)))
        norm = mpl.colors.Normalize(vmin=.5, vmax=max(temps)+0.5)
        for ts, temp, alg, ls_ in zip(time_series, temps, algs, ls):         
            ax.plot(ts, lw=2, c=cmap(norm(temp)), label=f"T = {temp} - {alg}", ls=ls_)

        ax.legend()
        plot_en_ts.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ticks=temps)

    else:
        cmap = plt.get_cmap('jet', 1)
        norm = mpl.colors.Normalize(vmin=.5, vmax= temps + 0.5)
        ax.plot(time_series, lw=3, c=cmap(norm(temps)), label=f"T = {temps} - {algs}", ls=ls)
        ax.legend()
        
    """ FIM PRIMEIRO PLOT """
        
    # plot secundario, histogramas das energias
    plot_en_hist_sub = plot_en_hist.subfigures(1, n_energias)


    if isinstance(plot_en_hist_sub, np.ndarray):   # se for lista eh pq existe mais de uma energia
        for en_plot, temp, ts, alg in zip(plot_en_hist_sub, temps, time_series, algs):
            ax_hist = en_plot.subplots(1,1)
            n_bins = int(abs(max(ts)-min(ts))/np.sqrt(len(ts)))
            n_bins = int(np.sqrt(len(ts)/3))
            n_bins = int(1 + 3.22*np.log(len(ts)))
            ax_hist.hist(ts, label=temp, color=cmap(norm(temp)), bins=n_bins)
            en_plot.suptitle(f'Histograma de E para T = {temp} ({alg})')
            ax_hist.legend()
         
    else:
        plot_en_hist_sub.suptitle(f'Histograma de E para T = {temps} ({algs[0]})')
        ax_hist = plot_en_hist_sub.subplots(1,1)      
        n_bins = 1 + 3.22*np.log(len(time_series))
        ax_hist.hist(time_series, label=temps, color=cmap(norm(temps)), 
                    bins = int(n_bins))

    plt.show()
    return

def mag_ts(time_series: list = [0], temps: list = [0], algs: list = [0], MCS: int = 0) -> plt.figure:
    """
    Funcao que devolve um gráfico da série temporal dos valores de magnetizacao
    """


    if isinstance(time_series[0], list): # if para saber se veio uma ou mais series temporais          
            n_energias = len(time_series)
            ls = ['-' if i==0 else '--' for i in algs]
            algs = ['W' if alg==0 else 'M' for alg in algs]      
    else:
        n_energias = 1
        if algs == 0:
            algs = 'W'
            ls = '-'
        else:
            algs = 'M'
            ls = '--'
        

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

    ax.set_title(f'Série Temporal de M para {MCS+1}MCS')

    if n_energias != 1: # plot de várias temperaturas
        cmap = plt.get_cmap('jet', len(set(temps)))
        norm = mpl.colors.Normalize(vmin=.5, vmax=max(temps)+0.5)
        for ts, temp, alg, ls_ in zip(time_series, temps, algs, ls):      
            ax.plot(ts, lw=2, c=cmap(norm(temp)), label=f"T = {temp} - {alg}", ls =ls_)

        ax.legend()
        plot_en_ts.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ticks=temps)

    else:
        cmap = plt.get_cmap('jet', 1)
        norm = mpl.colors.Normalize(vmin=.5, vmax= temps + 0.5)
        ax.plot(time_series, lw=3, c=cmap(norm(temps)), label=f"T = {temps} - {algs}", ls=ls)
        ax.legend()
        
    """ FIM PRIMEIRO PLOT """
        
    # plot secundario, histogramas das energias
    plot_en_hist_sub = plot_en_hist.subfigures(1, n_energias)


    if isinstance(plot_en_hist_sub, np.ndarray):   # se for lista eh pq existe mais de uma energia
        for en_plot, temp, ts, alg in zip(plot_en_hist_sub, temps, time_series, algs):
            ax_hist = en_plot.subplots(1,1)
            n_bins = int(abs(max(ts)-min(ts))/np.sqrt(len(ts)))
            n_bins = int(np.sqrt(len(ts)/3))
            n_bins = int(1 + 3.22*np.log(len(ts)))
            ax_hist.hist(ts, label=temp, color=cmap(norm(temp)), bins=n_bins)
            en_plot.suptitle(f'Histograma de M para T = {temp} ({alg})')
            ax_hist.legend()
         
    else:
        plot_en_hist_sub.suptitle(f'Histograma de M para T = {temps} ({algs})')
        ax_hist = plot_en_hist_sub.subplots(1,1)      
        n_bins = int(abs(max(time_series)-min(time_series))/np.sqrt(len(time_series)))
        n_bins = int(np.sqrt(len(time_series)/3))
        n_bins = 1 + 3.22*np.log(len(time_series))
        ax_hist.hist(time_series, label=temps, color=cmap(norm(temps)), 
                    bins = int(n_bins))


        

    plt.show()
    





def write_list_to_file(list_of: list=[], filename: str='') -> None:
    """
    Funcao que escreve os valores de uma lista em um arquivo 
    """
    path = r'Potts\data'
    path_c = os.path.join(path, filename)

    if os.path.exists(path_c):
        filename = filename + '_new'
    else:
        with open(path_c+'.txt', 'w') as file:
            for val in list_of:
                file.write(f'{val}\n')
    return 
    

def get_list_from_file(filename: str='') -> list:
    """
    Funcao que retorna a lista de valores apos ler o arquivo _filename_
    """

    path = os.path.join('Potts\data', filename)
    values = []
    if os.path.exists(path):
        with open(path, 'r') as file:
            lines = file.readlines()
            lines = [float(line.rstrip()) for line in lines]
            
        return lines


    else: 
        return None

def plota_tcs():
    x = np.linspace(1, 10, 10000)
    x1 = np.linspace(1,10,10)
    
    def f(x):
        return 1/(np.log(1+np.sqrt(x)))

    

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(16,9), dpi=120)
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)
    #ax.set_xticks(range(len(time_series[0])))
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)

    plt.scatter(x1, f(x1), c='r', label=r'$T_c$')
    plt.plot(x, f(x), ls='--', c='k')
    for val in x1:
        plt.annotate(r'$T_c$'+'= {}'.format(round(f(val), 3)), (val, round(f(val), 5)), fontsize=15)
    
    plt.title('Temperatura Crítica x Q', fontsize=25)
    plt.ylabel(r'$J$', fontsize=20, rotation=0)
    plt.xlabel(r'$Q$', fontsize=20)
    form_tex = r'$T_c = [ln(1+\sqrt{Q})]^{-1}$'
    plt.annotate(form_tex, (x1[5], f(x1[1])), fontsize=40)
    plt.xlim(0, 12)
    plt.xticks(range(11))
    plt.yticks(np.linspace(0.6, 1.5, 15))
    plt.legend(fontsize=15)

    plt.show()
    return

if __name__ == '__main__':
    # energy_ts(time_series=[1,2,3,4,5,1,1,1,1],temps=5)
    # import random
    # x = 100
    #energy_ts(time_series=[[1*random.random() for i in range(x)],
                            # [20*random.random() for i in range(x)],
                            # [30*random.random() for i in range(x)],
                            # [40*random.random() for i in range(x)],
                            # [50*random.random() for i in range(x)]],temps=[1, 2, 3, 4, 5])

    a = [1,2,3,4,5,6]
    b = [1,2,3,4,5]

    for i, j in zip(a, b):
        print(i, j)