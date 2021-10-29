import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
mpl.rc('figure', max_open_warning = 0)
import numpy as np




def plot_snapshot(rede: np.array = np.array([0,0]), title: str = "", Q: int=0) -> plt.figure:
    """
    Funcao que plota o estado atual da rede, recebendo um array
    """
    if Q == 0:
        print('Q = 0')
        return
    _, ax = plt.subplots(figsize=(16,9), dpi=60)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    
    #Colorbar confgis
    cmap = plt.get_cmap('rainbow', Q)
    # set limits .5 outside true range
    mat = ax.imshow(rede,cmap=cmap, vmin = 1-.5, vmax = Q+.5)
    #tell the colorbar to tick at integers
    cbar = plt.colorbar(mat, ticks=range(1,Q+1))
    cbar.ax.tick_params(labelsize=20)

    plt.title(title, fontsize=20)
    ax.grid(color='k', linewidth=4, which='minor')
    ax.set_frame_on(False)

    
    plt.show()


    return


if __name__ == '__main__':
    pass