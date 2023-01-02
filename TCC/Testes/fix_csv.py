from cProfile import label
import re
from turtle import color
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import datetime
import time
import collections


def main():
    folder_path = r'C:\Users\Mateus\Workspace\FisComps\TCC\Testes\testes20220907'
    #path = r'G:\.shortcut-targets-by-id\16w29Ho_PIrwdrQLLixlmi06nYdZVEN3d\Bolsa\TCC\POTTS'

    path = r'TCC\Testes\testes20220907\q3-df80.gzip'

    L = 80
    df = pd.read_parquet(path)

    #df.to_parquet(os.path.join(folder_path, 'q3-df80.gzip'))

    #new_df = pd.DataFrame()
    #df.drop(['Q', 'S'], aixs=1, inplace=True)
    

    print('df', df.memory_usage(deep=True).sum()/10e5, 'MB', df.shape)
    print(df.info())


    Q = df['Q'].iat[0]

    DFS = []

    for T in df['T'].unique():
        df_ = df.loc[df['T'] == T].copy()
        for S in df_['S'].unique():
            df__ = df_.loc[df_['S'] == S ].copy()
            df__.reset_index(drop=True, inplace=True)
            #print('df__', df__.memory_usage(deep=True).sum()/10e6, 'MB')
            L = df__.shape[1] - 3

            #print(L)

            q = np.array(df__.iloc[0][:-3])
            aux = {'Q':[int(Q) for _ in range(L)]
                ,'s':[i+1 for i in range(L)]
                ,'q': q
                ,'S':[S for _ in range(L)]
                ,'T':[T for _ in range(L)]}    

            dff = pd.DataFrame(aux)   
            DFS.append(dff) 

    DF = pd.concat(DFS, ignore_index=True)

    # print(df__['S'].values)
    # print(dff.loc[dff['s'] == 46]['S'].values)

    print('DF', DF.memory_usage(deep=True).sum()/10e5, 'MB', DF.shape)
    print(DF.info())


    #print(np.array(df__.iloc[df__.index[0]][:-3]))
    #print(aux)
    # print(df.loc[0, 'S'])
    # print(df.loc[5678, 'S'])
    # print(len(df.columns[:-3]))
    # print(len(df.columns))
    # print(df.columns)
    # print(df.shape)
    #print(len(df['T'].unique()))
    #print(df['T'].unique())
    #print(df['S'].unique())
    # for t in df['T'].unique():
    #     seeds.append(
    #         len(
    #             df.loc[df['T'] == t]['S'].unique()
    #             )
    #                 )

    # fig, ax = plt.subplots(figsize=(16,9))

    # df_ = np.array(df[df.columns[:-3]].iloc[0])

    # df_ = df_.reshape((L,L))
    
    # ax.imshow(df_)
    # ax.grid(color='k', linewidth=4, which='minor')
    # #ax.set_frame_on(False)



    # plt.show()


    return


def test_gpu():

    print('torch.cuda.is_available()',torch.cuda.is_available())


    print('torch.cuda.device_count()', torch.cuda.device_count())


    print('torch.cuda.current_device()', torch.cuda.current_device())

    print("torch.cuda.device(0)", torch.cuda.device(0))


    print(torch.cuda.get_device_name(0))



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    print('Mem Infos')
    global_free, total_gpu = torch.cuda.mem_get_info()


    print(f"global_free = {round(global_free/1024**3,1)}")
    print(f"total_gpu = {round(total_gpu/1024**3,1)}")


    
    return 

def create_pkl(path:str = '', L:int = 80):

    new_path = path.split('.')[0] + '.pkl'

   



    df = pd.read_parquet(path)
    cols = list(df.columns[:-3])


    final_dict = {'rede':[],'temp':[]}

    for i in range(len(df)):
        # rede
        array = np.array(df.iloc[i][cols])
        array = np.reshape(array, (L,L))
        # temp
        temp = df.iloc[i]['T']


        final_dict['rede'].append(array)
        final_dict['temp'].append(temp)


    dff = pd.DataFrame(final_dict)
    dff = dff.loc[dff['temp'] > 0.2].reset_index(drop=True)

    #dff.to_csv(r'TCC\Testes\testes20220907\test.csv', index=False)
    dff.to_pickle(new_path)

    
    print('dff', dff.memory_usage(deep=True).sum()/10e5, 'MB', dff.shape)
    print(new_path, 'finalizado')

    return

def plot_dataset_temperature(temp):
    temps = []
    for data in temp:
        temps.append(data)

    bins = np.linspace(0.494973, 1.484973, 10)

    plt.hist(temps, bins=bins)
    plt.show()
    return

def ajusta_dsfs(path:str = '', L:int = 0):
    files = [fil for _, _, fil in os.walk(path)][0]
    DFS = []

    for file in files:
        #print(file)
        _path = os.path.join(path, file)

        TEMP = file.split('Q')[0].split('T')[-1]
        #print(TEMP)

        SEED = file.split('.dsf')[0].split('S')[-1]
        #print(SEED)

        Q = file.split('Q')[-1].split('L')[0]
        #print(Q)


        if str(L) != file.split('S')[0].split('L')[-1]:
            print('erro no L')
        #print(L)

        
        with open(_path, 'r', encoding='latin-1') as fil:
            LINE = []
            lines = fil.readlines()
            lines = [lin.split(' ')[:-1] for lin in lines]

            for lin in lines:
                LINE.extend(lin)


            df_ = pd.DataFrame({str(i):float(l) for i,l in zip(range(L**2), LINE)}, index=[0])
            df_['T'] = float(TEMP)
            df_['Q'] = float(Q)
            df_['S'] = float(SEED)
            DFS.append(df_)
            
        #print(LINES, len(LINES[0]))
    
    DFS = pd.concat(DFS, ignore_index=True)
    print(DFS.shape)

    print(DFS.head())

        # falta juntar todos os dataframes, um em cima do outro 
        # LINES tem as 6400 colunas certinhas ja
    final_path = r'TCC\Testes\testes20221010'
    filename = 'q' + str(Q) + '-df' + str(L) + '.gzip'

    filename = os.path.join(final_path, filename)
    
    DFS.to_parquet(filename)
    


    return

def plot_results(TIN, TOUT, TEMP_MIN, TEMP_MAX, TC):
    import matplotlib as mpl
    
    fig, ax = plt.subplots(figsize = (16,9), dpi=120)
    my_cmap = plt.get_cmap("rainbow")
    rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
    #plt.title(r'$T$ x $T_{r}$   | Q=4', fontsize=30)

    line = np.linspace(TEMP_MIN*0.83, TEMP_MAX*1.15, 200)


    #print('TEMPS_IN', len(TEMPS_IN))
    #print('TEMPS_OUT', len(TEMPS_OUT))
    # data points
    plot = ax.scatter(TIN, TOUT, c=TOUT
                ,edgecolor='black', s=50, label=r'$T_{nn}$', cmap=my_cmap)
    fig.colorbar(plot, label='T')
    # Tc line
    plt.vlines(TC, ymin=TEMP_MIN*0.55, ymax=TEMP_MAX*1.15
                ,color='black', label=r'$T_{C} = $'+str(round(TC,4)), lw=3, ls='--')
    # T = T line
    plt.plot(line, line, color='black', label=r'$T_{R}=T_{nn}$', lw=3)

    plt.xlabel(r'$T$', fontsize=15)
    plt.ylabel(r'$T_{nn}$', fontsize=15)

    plt.xlim(TEMP_MIN*0.65,TEMP_MAX*1.01)
    plt.ylim(TEMP_MIN*0.65,TEMP_MAX*1.01)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # fig.colorbar(mpl.cm.ScalarMappable(cmap=my_cmap)
    #             ,ax=ax
    #             ,norm=mpl.colors.Normalize(vmin=TEMP_MIN, vmax=TEMP_MAX))
                #values=np.linspace(TEMP_MIN, TEMP_MAX, 10))
                # ,values=np.linspace(TEMP_MIN, TEMP_MAX, 10)
                #,ticks=np.linspace(TEMP_MIN, TEMP_MAX, 10))
                # ,drawedges=True)
    
    #print(colors) 
    plt.legend(fontsize=15)

    plt.savefig('resultados_q4.png', format='png')
    #plt.show()

    return

def U_simples(TIN, TOUT, TC,title:str='', paper: bool = False):
    df = pd.DataFrame({'TIN':TIN, 'TOUT':TOUT})
    UT = {}
    temps = df['TIN'].unique()
    for t in temps:
        df_ = df.loc[df['TIN'] == t].copy()
        u_mean = np.mean(df_['TOUT'])
        if paper:
            df_['U'] = np.sqrt(np.power(df_['TOUT'] - u_mean, 2))
        else:
            df_['U'] = np.sqrt(np.power(df_['TIN'] - df_['TOUT'], 2))
        UT[t] = np.mean(df_['U'])
    ordered_dict = {k: v for k, v in sorted(UT.items(), key=lambda item: item[0])}

    return ordered_dict

def U_full(TIN, TOUT, TC, run, title:str=''
            ,paper: bool = False
            ,save: bool = False
            ,Q:int = 2):
    
    df = pd.DataFrame({'TIN':TIN, 'TOUT':TOUT, 'RUN':run})
    fig, ax = plt.subplots(figsize=(12,8), dpi=120)
    #print(len(df['RUN'].unique()))
    #print(df['RUN'].unique()[:-1])
    gray_colors = ['dimgray', 'gray', 'darkgray'] #, 'silver', 'lightgrey']
    for r, color in zip(df['RUN'].unique()[:3], gray_colors):
        df_aux = df.loc[df['RUN'] == r].copy()
        UT = {}
        temps = df_aux['TIN'].unique()
        for t in temps:
            df_ = df_aux.loc[df_aux['TIN'] == t].copy()
            u_mean = np.mean(df_['TOUT'])
            if paper:
                df_['U'] = np.sqrt(np.power(df_['TOUT'] - u_mean, 2))
            else:
                df_['U'] = np.sqrt(np.power(df_['TIN'] - df_['TOUT'], 2))
            UT[t] = np.mean(df_['U'])

        ordered_dict = {k: v for k, v in sorted(UT.items(), key=lambda item: item[0])}
        if r == 2:
            plt.plot(ordered_dict.keys(), ordered_dict.values(), c=color,alpha=0.4, label='Séries')
        else:
            plt.plot(ordered_dict.keys(), ordered_dict.values(), c=color, alpha=0.4)
    # # ultima run 
    # df_aux = df.loc[df['RUN'] == df['RUN'].unique()[-1]].copy()
    # UT = {}
    # temps = df_aux['TIN'].unique()
    # for t in temps:
    #     df_ = df_aux.loc[df_aux['TIN'] == t].copy()
    #     u_mean = np.mean(df_['TOUT'])
    #     if paper:
    #         df_['U'] = np.sqrt(np.power(df_['TOUT'] - u_mean, 2))
    #     else:
    #         df_['U'] = np.sqrt(np.power(df_['TIN'] - df_['TOUT'], 2))
    #     UT[t] = np.mean(df_['U'])

    # ordered_dict = {k: v for k, v in sorted(UT.items(), key=lambda item: item[0])}
    # plt.plot(ordered_dict.keys(), ordered_dict.values(), c='k', alpha=0.1, label='Séries')


    UT = {}
    temps = df['TIN'].unique()
    for t in temps:
        df_ = df.loc[df['TIN'] == t].copy()
        u_mean = np.mean(df_['TOUT'])
        if paper:
            df_['U'] = np.sqrt(np.power(df_['TOUT'] - u_mean, 2))
        else:
            df_['U'] = np.sqrt(np.power(df_['TIN'] - df_['TOUT'], 2))
        UT[t] = np.mean(df_['U'])
    ordered_dict = {k: v for k, v in sorted(UT.items(), key=lambda item: item[0])}


    plt.vlines(x=TC, ls='--', color='k', label=r'$T_C$ = '+str(round(TC,3)), ymax=1, ymin=0)
    plt.plot(ordered_dict.keys(), ordered_dict.values(), c='r')
    plt.scatter(UT.keys(), UT.values(), label=r'$U(T)$', c='r')

    #$plt.title(title, fontsize=25)
    plt.ylabel(r'$U(T)$',fontsize=25)
    plt.xlabel(r'T', fontsize=25)
    #plt.scatter(min(UT, key=UT.get), min(UT.values()), c='r', label='Min')
    plt.legend(loc='upper left', fontsize=15)
    #plt.xlim(0.5, 1)
    plt.ylim(0, 0.1) 
    plt.xticks(fontsize=25)   
    plt.yticks(fontsize=25)


    if save:
        path_ = r'C:\Users\Mateus\Workspace\FisComps\TCC\Testes\testes20221020\graficos\20221113'
        filename = f'Q{Q}_{max(run)}_v2.png'
        print('salvando', filename)
        plt.savefig(os.path.join(path_, filename))


    #plt.show()
    return UT

def ajusta_dsfs_v2(path:str = ''
                  ,L:int = 0
                  ,batch:int = 100
                  ,path_to_save:str=''):
    # TERMINAR CODIGO PARA LER OS ARQUIVOS EM BATCHES
    #path_ = r'TCC\Testes\testes20221010\auxliar'
    files = [fil for _, _, fil in os.walk(path)][0]

    DFS = []
    total_files = len(files)


    sum_ = 0
    RATIO = total_files//batch


    for i in range(batch):
        sum_ += len(files[i*RATIO:i*RATIO+RATIO])

    if sum_ != total_files:
        print(sum_, total_files)
        print('ERRO NO TAMANHO DO BATCH')
        return 

    DF1 = []
    DF2 = []
    for i in range(batch):
        dfs = []
        for file in files[i*RATIO:i*RATIO+RATIO]:
            _path = os.path.join(path, file)

            TEMP = file.split('Q')[0].split('T')[-1]
            #print(TEMP)

            SEED = file.split('.dsf')[0].split('S')[-1]
            #print(SEED)

            Q = file.split('Q')[-1].split('L')[0]
            #print(Q)

            
            l = int(file.split('S')[0].split('L')[-1])

            if str(L) != file.split('S')[0].split('L')[-1]:
                print('erro no L')
                return

            with open(_path, 'r', encoding='latin-1') as fil:
                LINE = []
                lines = fil.readlines()
                lines = [lin.split(' ')[:-1] for lin in lines]

                for lin in lines:
                    LINE.extend(lin)
                L2 = l*l
                # aux_dict = {
                #     'p':[i for i in range(1, L2+1)]
                #     ,'q':[int(i) for i in LINE]
                #     ,'T':[float(TEMP)]*L2
                #     ,'S':[int(SEED)]*L2
                #     ,'Q':[int(Q)]*L2
                #                 }

                # print(len(aux_dict['T']))
                # print(len(aux_dict['S']))
                # print(len(aux_dict['q']))
                # print(len(aux_dict['Q']))
                # print(len(aux_dict['p']))

                #df_ = pd.DataFrame(aux_dict)
                #print(df_.info(verbose=True, memory_usage='deep'))
                #data_types = ['int8','int8', 'float16', 'int32', 'int8']

                # for col, typ in zip(df_.columns, data_types):
                #     df_[col] = df_[col].astype(typ)
                #print(df_.info(verbose=True, memory_usage='deep'))
                #DF1.append(df_)
                #print('df', df_.memory_usage(deep=True).sum()/10e5, 'MB', df_.shape)

                df_ = pd.DataFrame({str(i):np.int8(l) for i,l in zip(range(L**2), LINE)}, index=[0])
                df_['T'] = np.float32(TEMP)
                df_['Q'] = np.int8(Q)
                df_['S'] = np.int32(SEED)
                #print('df', df_.memory_usage(deep=True).sum()/10e5, 'MB', df_.shape)
                dfs.append(df_)

        dfp = pd.concat(dfs, ignore_index=True)
        
        filename = 'df' + str(i) + '.gzip'
        file_path = os.path.join(path_to_save, filename)
        dfp.to_parquet(file_path)
        print('arquivo', i+1, 'salvo')

    # df1 = pd.concat(DF1, ignore_index=True)
    # print('df1', df1.memory_usage(deep=True).sum()/10e5, 'MB', df1.shape)

    
    print('finalizado')
                
    return 

def junta_dfs(path:str=''):

    dfs_f = [fil for _, _, fil in os.walk(path)][0]
    DFS = []
    for fil in dfs_f:
        pat = os.path.join(path, fil)
        df_aux = pd.read_parquet(pat)
        DFS.append(df_aux)

    df2 = pd.concat(DFS, ignore_index=True)
    print('df', df2.memory_usage(deep=True).sum()/10e5, 'MB', df2.shape)

    return df2

def ajsuta_temps(path:str=''):
    return

def junta_results(Q:int = 8,
                N: int =18,
                E: int=50,
                L: int=120,
                date_lim: str = '',
                  path: str=''):

    files = [fil for _, _, fil in os.walk(path)][0]
    selected_files = []
    hash_ = f'results_N{N}_E{E}_Q{Q}_L{L}'
    for file in files:
        if file.startswith(hash_):
            selectec_path = os.path.join(path, file)
            selected_files.append(selectec_path)

    return selected_files

def plot_snapshot_com_hist(rede: np.array = np.array([0,0]), title: str = "",
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

def plot_snapshot(rede: np.array = np.array([0,0]), title: str = "",
                 Q: int=0, save: bool = False, t: int = 0, temp:float=0.0) -> plt.figure:
    """
    Funcao que plota o estado atual da rede, recebendo um array
    """

    if Q == 0:
        print('Q = 0')
        return


    fig, ax0 = plt.subplots(ncols=1, nrows=1,figsize=(12,12), dpi=120)
    #fig.suptitle(title, fontsize=35)
    ax0.tick_params(axis="x", labelsize=10)
    ax0.tick_params(axis="y", labelsize=10)
    ax0.spines['left'].set_linewidth(2)
    ax0.spines['bottom'].set_linewidth(2)
    ax0.spines['right'].set_linewidth(2)
    ax0.spines['top'].set_linewidth(2)
    
    #Colorbar confgis
    cmap = plt.get_cmap('rainbow', Q)


    mat = ax0.imshow(rede,cmap=cmap, vmin = 1-.5, vmax = Q+.5)
    #tell the colorbar to tick at integers
    #cbar = plt.colorbar(mat, ticks=range(1,Q+1))
    #cbar.ax.tick_params(labelsize=20) 

    
    ax0.set_title(title+f'T = {temp}', fontsize=20)
    ax0.grid(color='k', linewidth=4, which='minor')
    ax0.set_frame_on(False)

    fig.tight_layout()
    if save:
        path = r'C:\Users\mateu\workspace\MonteCarlo\Potts\img'
        filename = 'fig{:07d}.png'.format(t)
        plt.savefig(os.path.join(path, filename), format='png')
    else:
        plt.show()


    return

def calcula_min_U(Q:int,E=100):
    path=r'C:\Users\Mateus\Workspace\FisComps\TCC\Testes\testes20221020\data'
    list_ = junta_results(Q=Q, E=E, path=path)
    TC =  (1/(np.log(1+np.sqrt(Q))))
    dfs = []
    for i in range(len(list_)):
        fil = list_[i]
        df_ = pd.read_csv(fil)
        df_['run'] = i+1
        df_ = df_.loc[df_['TR'] > 0.4].copy()
        dfs.append(df_)

    DF = pd.concat(dfs)
    max_temp = DF['Tnn'].max()
    DFS = []
    temps_finais = []
    for run in DF['run'].unique():
        df_aux = DF.loc[DF['run'] == run].copy()

        dfU = U_simples(df_aux['TR'], df_aux['Tnn'], TC, paper=True)

        dfU = pd.DataFrame({'T':dfU.keys(), 'U':dfU.values()})
        U_min = min(dfU['U'])

        t = dfU.loc[dfU['U'] == U_min]['T'].values[0]
        temps_finais.append(t)

    Tmean = np.mean(temps_finais)
    Tstd  = np.std(temps_finais)

    T_menos = Tmean - Tstd
    T_mais = Tmean + Tstd


    if TC > T_menos and TC < T_mais:
        Ok = True
    else:
        Ok = False
    print(Q,'&',round(TC, 3),'&',  round(Tmean, 2),r'$\pm$', round(Tstd, 4), r'\\ \hline', Ok)

    return [t/TC for t in temps_finais]

def calcula_max_preTC(Q:int,E=100):
    path=r'C:\Users\Mateus\Workspace\FisComps\TCC\Testes\testes20221020\data'
    list_ = junta_results(Q=Q, E=E, path=path)
    TC =  (1/(np.log(1+np.sqrt(Q))))
    dfs = []
    for i in range(len(list_)):
        fil = list_[i]
        df_ = pd.read_csv(fil)
        df_['run'] = i+1
        df_ = df_.loc[df_['TR'] > 0.4].copy()
        dfs.append(df_)

    DF = pd.concat(dfs)

    DFS = []
    temps_finais = []
    for run in DF['run'].unique():
        df_aux = DF.loc[DF['run'] == run].copy()

        dfU = U_simples(df_aux['TR'], df_aux['Tnn'], TC, paper=True)

        dfU = pd.DataFrame({'T':dfU.keys(), 'U':dfU.values()})
        U_min = min(dfU['U'])

        t = dfU.loc[dfU['U'] == U_min]['T'].values[0]
        temps_finais.append(t)

    Tmean = np.mean(temps_finais)
    Tstd  = np.std(temps_finais)


    dfs = []
    for i in range(len(list_)):
        fil = list_[i]
        df_ = pd.read_csv(fil)
        df_['run'] = i+1
        df_ = df_.loc[df_['TR'] < Tmean+Tstd].copy()
        dfs.append(df_)

    DF = pd.concat(dfs)

    temps_preTC_max = []
    for run in DF['run'].unique():
        df_aux = DF.loc[DF['run'] == run].copy()

        dfU = U_simples(df_aux['TR'], df_aux['Tnn'], TC, paper=True)

        dfU = pd.DataFrame({'T':dfU.keys(), 'U':dfU.values()})
        U_max = max(dfU['U'])

        temps_preTC_max.append(U_max)

    preTC_max_mean = round(np.mean(temps_preTC_max), 4)
    preTC_max_std  = round(np.std(temps_preTC_max), 4)

    print('Q', Q, 'Umax', preTC_max_mean, round(1.96*preTC_max_std, 6)
            , 'lb', round(preTC_max_mean-1.96*preTC_max_std, 6), 'ub',round(preTC_max_mean+1.96*preTC_max_std, 6) )
    return

if __name__ == '__main__':



    # files = [
    #     r'TCC\Testes\testes20221020\q2-df120.gzip',
    #     r'TCC\Testes\testes20221020\q3-df120.gzip',
    #     r'TCC\Testes\testes20221020\q4-df120.gzip',
    #     r'TCC\Testes\testes20221020\q6-df120.gzip',
    #     r'TCC\Testes\testes20221020\q8-df120.gzip',       
    # ]
    # for file in files:
    #     create_pkl(file, L=120)

    new_path = r'C:\Users\Mateus\Workspace\FisComps\TCC\Testes\testes20221020\q6_df120_df50.pkl'
    df = pd.read_pickle(new_path)
    #print(df['temp'].unique())
    # L=120
    index = 200
    rede = df['rede'].iloc[index]
    temp = round(df['temp'].iloc[index], 3)
    plot_snapshot(rede, Q=6, temp=temp)


    exit()


    # values = []
    # for q in [2,3,4,6,7,8]:
    #     x = calcula_min_U(q, 100)
    #     #x = [k/max(x) for k in x]
    #     values.extend(x)    
    # fig, ax = plt.subplots(figsize=(15,15), dpi=120)

    
    # #plt.title(f'Distribuição das Temperaturas Mínimas  ({len(values)} séries)')
    # #vals, bins = np.histogram(values, bins = bins)
    # plt.hist(values,   color='grey', alpha=0.7, align='mid', label='Dados')
    # plt.vlines(x=np.mean(values), ymin=0, ymax=50
    #             , ls='--', label=f'Media = {round(np.mean(values), 3)}'
    #             ,lw=3, colors='k')
    # plt.ylabel('Frequência', fontsize=25)
    # plt.ylim(0, 42)
    # plt.xlabel(r'$T_{CR}/T_{C}$', fontsize=25)
    # plt.xticks(fontsize=30)
    # plt.yticks(fontsize=25)
    # plt.legend(fontsize=25)
    # plt.savefig(r'hist_teste.png')
    # plt.show()



    #for q in [2,3,4,6,7,8]:
       # calcula_max_preTC(q, 100)


    # df = pd.read_pickle(r'C:\Users\Mateus\Workspace\FisComps\TCC\Testes\testes20221020\q2-df120.pkl')
    # df['temp'].plot()
    # plt.show()

    # # loss plot tcc


    # Q=4
    #list_  = junta_results(Q=Q,E=100,path=r'C:\Users\Mateus\Workspace\FisComps\TCC\Testes\testes20221020\data')
    path_ = r'Testes\testes20221020\data\loss_N18_E100_Q2_L120_20221115_18h37mB64_0.5.csv'
    df = pd.read_csv(path_)

    # dfs = []
    # for i in range(len(list_[:1])):
    #     fil = list_[i]
    #     df_ = pd.read_csv(fil)
    #     df_['run'] = i+1
    #     #df_ = df_.loc[df_['TR'] > 0.4].copy()
    #     dfs.append(df_)

    # DF = pd.concat(dfs)
    # TC =  (1/(np.log(1+np.sqrt(Q))))

    # # plot_results(DF['TR'], DF['Tnn'], min(DF['Tnn']), max(DF['Tnn']), TC)
    # fig, ax0 = plt.subplots(figsize=(16,5))
    # ax0.tick_params(axis="x", labelsize=10)
    # ax0.tick_params(axis="y", labelsize=10)
    # ax0.spines['left'].set_linewidth(2)
    # ax0.spines['bottom'].set_linewidth(2)
    # ax0.spines['right'].set_linewidth(2)
    # ax0.spines['top'].set_linewidth(2)
    # plt.title('Loss x Épocas', fontsize=25)
    # plt.plot(df['epochs']+1, df['loss'], lw=3, c='r', label='Loss')
    # plt.scatter(df['epochs']+1, df['loss'], c='r', s=25)
    # plt.xlabel('log(Época)', fontsize=25)
    # plt.ylabel('Loss', fontsize=25)
    # plt.xscale('log')
    # plt.legend(fontsize=15)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)

    # plt.show()


    # print(df['loss'].iloc[-1])











    UTODOS = {}
    for Q in [2,3,4,6,7,8]:
        save = True
        list_  = junta_results(Q=Q,E=100,path=r'C:\Users\Mateus\Workspace\FisComps\TCC\Testes\testes20221020\data')

        dfs = []
        for i in range(len(list_)):
            fil = list_[i]
            df_ = pd.read_csv(fil)
            df_['run'] = i+1
            #df_ = df_.loc[df_['TR'] > 0.4].copy()
            dfs.append(df_)

        DF = pd.concat(dfs)
        print(f'{len(dfs)} arquivos encontrados para Q{Q}')
        print(DF.shape)
        TC =  (1/(np.log(1+np.sqrt(Q))))
        

        #title = r'$U(T)$'+ f' para Q{Q} com {len(dfs)} séries'
        UTODOS[Q] = U_full(DF['TR'], DF['Tnn'], TC,DF['run'], paper=True, save=save, Q=Q)

    # print(UTODOS.keys())
    # fig, ax = plt.subplots(figsize=(9,9), dpi=120)
    # for q in UTODOS.keys():
    #     data = UTODOS[q]
    #     TC =  (1/(np.log(1+np.sqrt(q))))
    #     ordered_dict = {k: v for k, v in sorted(data.items(), key=lambda item: item[0])}
    #     max_ = max(ordered_dict.keys())
    #     plt.plot([k/TC for k in ordered_dict.keys()], ordered_dict.values()
    #         , alpha=1, lw=3, label=f'Q={q}')
    #     #plt.scatter([k/TC for k in data.keys()], [k/TC for k in data.values()], label=f'Q={q}')

    # # #plt.title('U(T) x T')
    # plt.ylabel(r'$U(T)$', fontsize=25)
    # plt.xlabel(r'$T/T_{C}$', fontsize=25)
    # plt.xticks(fontsize=25)
    # plt.yticks(fontsize=25)
    # # #plt.scatter(min(UT, key=UT.get), min(UT.values()), c='r', label='Min')
    # #plt.xlim(0.4, 0.6)
    # #plt.ylim(0, 0.02)
    # plt.legend(fontsize=25)
    # plt.show()
    #test_gpu()
    #main()
    #path_ = r'C:\Users\Mateus\results_Q2_20221017'
    #path__ = r'TCC\Testes\testes20221010\auxliar'

    # base_path = r'C:\Users\Mateus\Workspace\FisComps\TCC\Testes\testes20221020'
    # base_content_path = r'C:\Users\Mateus\results_20221020'




    # Q=6
    # TC =  (1/(np.log(1+np.sqrt(Q))))

    # # print(TC)
    # L=120

    # df = pd.read_parquet(r'D:\Backup\TCC\auxiliar\q6\df30.gzip')
    # cols = list(df.columns[:-3])


    # final_dict = {'rede':[],'temp':[]}

    # for i in range(len(df)):
    #     # rede
    #     array = np.array(df.iloc[i][cols])
    #     array = np.reshape(array, (L,L))
    #     # temp
    #     temp = df.iloc[i]['T']


    #     final_dict['rede'].append(array)
    #     final_dict['temp'].append(temp)


    # dff = pd.DataFrame(final_dict)
    # dff = dff.loc[dff['temp'] > 0.2].reset_index(drop=True)

    # #dff.to_csv(r'TCC\Testes\testes20220907\test.csv', index=False)
    # dff.to_pickle(new_path)

    
    # print('dff', dff.memory_usage(deep=True).sum()/10e5, 'MB', dff.shape)
    # print(new_path, 'finalizado')





    # Qs = [2,3,4,6,8]
    # paths_to_save = [
    #     os.path.join(base_path, f'Q{q}') for q in Qs
    # ]

    
    # contents_path = [
    #     os.path.join(base_content_path, f'results_Q{q}_20221020') for q in Qs
    # ]

    # for base, path in zip(contents_path, paths_to_save):
    #     ajusta_dsfs_v2(base, 120, 100, path)
    # path = r'C:\Users\Mateus\results_20221020\results_Q7_20221020'
    # path2 = r'C:\Users\Mateus\Workspace\FisComps\TCC\Testes\testes20221020'
    # ajusta_dsfs_v2(path, 120, 100, path2)

    # for base, q in zip(paths_to_save, Qs):
    #     df_ = junta_dfs(base)
    #     filename = f'q{q}-df120.gzip'
    #     new_base = r'TCC\Testes\testes20221020'
    #     path_ = os.path.join(new_base, filename)

    #     df_.to_parquet(path_)


    #df = junta_dfs(path__)
    #new_p = r'TCC\Testes\testes20221010\q2-df120.pkl'
    # #df = pd.read_pickle(new_p)
    # new_p = r'TCC\Testes\testes20221010\q2-df120.gzip'
    
    # #create_pkl(new_p, 120)
    # for i in range(3):
    #     print(i)
    #print(df.head())
    # df['T'].plot()
    # plt.show()
    #ajusta_dsfs(r'C:\Users\Mateus\results_Q7_20221012', 120)
    #df2 = ajusta_dsfs_v2(r'C:\Users\Mateus\results_Q7_20221012', 120)
    # ARRUMAR NO FORMATO CHAVE-VALOR, TEM 15K DE COLUNAS CADA REDE
    # 15k de colunas x 10k de linhas
    #df = junta_dfs(path_)
    # #path_ = r'TCC\Testes\testes20221010\teste.gzip'
    # df = pd.read_pickle(r'TCC\Testes\testes20221010\teste.gzip')
    # df.to_parquet(r'TCC\Testes\testes20221010\teste.gzip')
    # #print(df.head())
    # create_csv(path=r'TCC\Testes\testes20221010\teste.gzip', L=120)

    # path = r'TCC\Testes\testes20221010\q8-df120.pkl'
    # df = pd.read_pickle(path)

    # print(df.head())
    # df_ = df.loc[df['temp'] > 0.2].reset_index(drop=True).copy()
    #path = r'TCC\Testes\testes20221010\q8-df120_v2.pkl'
    #df = pd.read_pickle(path)
    #print(df.head())
    #$create_csv(path, 120)






    #path = r'TCC\Testes\testes20220907\q8-df80.gzip'

    #create_csv(path=path)
    # df = pd.read_csv(r'TCC\Testes\testes20221010\data\results_N18_E100_Q8_L120_20221012_14h17mB32_0.8.csv')

    # 
    # df_ = U(df['TR'], df['Tnn'], TC)
    #df = pd.read_pickle(r'TCC\Testes\testes20221003\q8-df80.pkl')

    #print(df.head())
    #print(df.shape)

    #print(min(df['TR']), max(df['TR']))
    #plot_results(df['TR'], df['Tnn'], min(df['TR']), max(df['TR']))

   

    # print(df2.shape)
    # print(df2.head())

    # print(len(df2['TIN'].unique()))

    # for item in df['Tnn']:
    #     print(type(item))
    #fmt = r'%Y%m%d_%Hh%Mm'
    #print(datetime.datetime.today().strftime(fmt))
    #ajusta_dsfs(path, 80)





    #df = pd.read_pickle(r'TCC\Testes\testes20220907\test.pkl')
    #df = pd.read_parquet(r'TCC\Testes\testes20220907\q3-df80.gzip')
    #print(df.shape)
    #print(df.head())
    #print(df.head())

    # plot_dataset_temperature(df['temp'])



    # for Q in [3, 4, 7, 8]:
    #     print(Q, (1/(np.log(1+np.sqrt(Q)))))

    # paths = [
    #      r'TCC\Testes\testes20220907\q3-df80.gzip'
    #     ,r'TCC\Testes\testes20220907\q4-df80.gzip'
    #     ,r'TCC\Testes\testes20220907\q7-df80.gzip'
    # ]
    
    # for path in paths:
    #     create_csv(path=path)

    # df = pd.read_pickle(r'TCC\Testes\testes20220907\test.pkl')

    # print(df.head())
    # print(max(df['temp']), min(df['temp']))
    # print(df.shape)
    # print(len(df['temp'].unique()))
    # #test_gpu()
    

