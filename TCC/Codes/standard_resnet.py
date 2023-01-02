import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import datetime

#machine learning libraries
import torch
import torchvision.models as models
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torch.nn.modules import conv
from torch.nn.modules.utils import _pair
from torchvision.models import resnet18, resnet34, resnet152, resnet50, resnet101

#ranger optimizer #https://github.com/mpariente/Ranger-Deep-Learning-Optimizer
from pytorch_ranger import Ranger  
from pytorch_ranger import RangerVA
from pytorch_ranger import RangerQH

#fastai
# from fastai.data.core import Datasets
# from fastai.basics import *
# from fastai.vision.all import *
# from fastai.callback.all import *

#removing warnings
import warnings
warnings.filterwarnings("ignore")








class ising_dataset(Dataset):
  def __init__(self, dataname):
    self.data   = dataname['rede']
    self.target = dataname['temp']

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    current_sample = np.array(self.data[idx])
    current_sample = np.expand_dims(current_sample, axis=0)
    current_sample = current_sample.astype(np.float32)
    current_sample = torch.from_numpy(current_sample)
    current_target = self.target[idx]
    
    current_target = current_target.astype(np.float32)

    return torch.tensor(current_sample), torch.tensor(current_target)

"""###Functions"""

class Mish_layer(nn.Module):
  '''
  The class represents Mish activation function.
  '''
  def __init__(self):
    super(Mish_layer,self).__init__()

  def forward(self,x):
    return x*torch.tanh(F.softplus(x))

#randon split dataframe
def split_indexes(dataframe, train_size):
  num_train = len(dataframe)
  indices = list(range(num_train))

  split = int(np.floor(train_size * num_train))
  split2 = int(np.floor((train_size+(1-train_size)/2) * num_train))

  np.random.shuffle(indices)

  return indices[:split], indices[split2:], indices[split:split2]

def mse_loss_wgtd(pred, true, wgt=1.):
  loss = wgt*(pred-true).pow(2)
  return loss.mean()

def root_mean_squared_error(p, y): 
    return torch.sqrt(mse_loss_wgtd(p.view(-1), y.view(-1)))

def mae_loss_wgtd(pred, true, wgt=1.):
    loss = wgt*(pred-true).abs()
    return loss.mean()



def pipe(N          : int   = 18
        ,epochs     : int   = 10
        ,L          : int   = 80
        ,Q          : int   = 3
        ,train_size : float = 0.8
        ,batch_size : int   = 32
        ,root       : str   = ''
        ,save_nn    : bool  = False
        ,save_data  : bool  = False
        ,plots      : int   = 0 
        ,run_text   : str   = ''):

    if plots == 0:
        loss_plot = False
        results_plot = False
    elif plots == 1:
        loss_plot = True
        results_plot = False
    elif plots == 2:
        loss_plot = True
        results_plot = True
    else:
        loss_plot = False
        results_plot = False



    ROOT = r'TCC\Testes\testes20221020'
    DATE_HASH = datetime.datetime.today().strftime(r'%Y%m%d_%Hh%Mm')
    EXEC_HASH = f'N{N}_E{epochs}_Q{Q}_L{L}_' + DATE_HASH + f'B{batch_size}_{train_size}'

    #Data Loading
    pickle_path = ROOT + f"\q{Q}-df{L}.pkl"
    pickle_data = pd.read_pickle(pickle_path)

    #definitions
    L2 = L*L
    TEMP_MAX = max(pickle_data['temp'])
    TEMP_MIN = min(pickle_data['temp'])
    TC =  (1/(np.log(1+np.sqrt(Q))))
    



    # Train, Test, Validation SPLIT
    train_idx, test_idx, val_idx = split_indexes(pickle_data, train_size)

    data_train = ising_dataset(dataname=pickle_data.loc[train_idx].reset_index(drop=True))
    data_test = ising_dataset(dataname=pickle_data.loc[test_idx].reset_index(drop=True))
    data_val = ising_dataset(dataname=pickle_data.loc[val_idx].reset_index(drop=True))


    # print(f'data_train = {len(data_train)}')
    # print(f'data_test = {len(data_test)}')
    # print(f'data_val = {len(data_val)}')




    # device settings
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')



    # dataset loader
    train_dataloader = DataLoader(data_train, shuffle=True, batch_size=batch_size, num_workers=2)
    test_dataloader = DataLoader(data_test, shuffle=True, batch_size=batch_size, num_workers=2)
    val_dataloader = DataLoader(data_val, shuffle=True, batch_size=batch_size, num_workers=2)


    # clear cache
    torch.cuda.empty_cache

    # plot_dataset_temperature(train_dataloader)
    # plot_dataset_temperature(test_dataloader)
    # plot_dataset_temperature(val_dataloader)

    model_dict = {
         18  : resnet18(pretrained=True)
        ,34  : resnet34(pretrained=True)
        ,50  : resnet50(pretrained=True)
        ,101 : resnet101(pretrained=True)
        ,152 : resnet152(pretrained=True)
                }

    model = model_dict[N]
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(in_features=1, out_features=1, bias=True) # Change final layer to predict one value


    mynet = model.to(device)
    loss_function = root_mean_squared_error
    optimizer = Ranger(mynet.parameters(), lr=1e-3) # colocando o mesmo lr do paper

    train_history = []

    mynet.train() 

    print(f'INICANDO {EXEC_HASH}')

    for epoch in range(0, epochs): # 5 epochs at maximum
        print(f'Starting epoch {epoch+1}', run_text)
        print(f'-----------------')
            
        train_loss = []

        for data in train_dataloader:
            inputs, targets = data[0].float(), data[1].float()
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = mynet(inputs)
            
            loss = loss_function(outputs, targets)
            
            loss.backward()
            optimizer.step() 
            optimizer.zero_grad()

            train_loss.append(loss.item())

        train_loss = np.array(train_loss)
        train_history.append(train_loss.mean())

    print('Testing process has finished.')


    #save NN

    #load_nn = 0

    if save_nn == 1:
        net_path = ROOT + f'nn\weights_{EXEC_HASH}.pt'
        torch.save({'weights': mynet.state_dict()}, net_path)

    # if load_nn == 1:
    #     s = torch.load(net_path)
    #     mynet = mynet()
    #     mynet.load_state_dict(s['weights'])
    #     # mynet.state_dict()

    



    # LOSS PLOT
    fig, ax = plt.subplots(figsize=(16,9))
    x_axis = np.arange(0,epochs,1)
    plt.title(f'Loss x Epoch - {EXEC_HASH}', fontsize=30)
    plt.plot(x_axis, train_history, lw=3, label='Loss')
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.legend(fontsize=15)
    if loss_plot:
        root_ = os.path.join(ROOT, 'plots')
        loss_plot_path = os.path.join(root_, f'loss_{EXEC_HASH}.jpeg')
        plt.savefig(loss_plot_path)


    # SAVE LOSS DATA
    if save_data:
        df_ = pd.DataFrame({'epochs':x_axis, 'loss':train_history})
        root_ = os.path.join(ROOT, 'data')
        df_path = os.path.join(root_, f'loss_{EXEC_HASH}.csv')
        df_.to_csv(df_path, index=False)



    mynet.eval()

    val_loss = []
    t_in = []
    t_out = []

    for data in val_dataloader:
        inputs, targets = data[0].float(), data[1].float()
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = mynet(inputs)
    
        t_in.append(targets.cpu().detach().numpy())
        t_out.append(outputs.cpu().detach().numpy())
    

    loss = loss_function(outputs, targets)  
    
    val_loss.append(loss.item())

    val_loss = np.array(val_loss)
    val_loss = val_loss.mean()

    TEMPS_IN  = []
    TEMPS_OUT = []

    for item in range(len(t_in)):
        for i in t_in[item]:
            TEMPS_IN.append(i)

    for item in range(len(t_out)):
        for i in t_out[item]:
            TEMPS_OUT.append(i[0])

    print()
    print('Validation process has finished.')
    print(f'loss = ',val_loss)


    fig, ax = plt.subplots(figsize = (16,9))
    my_cmap = plt.get_cmap("rainbow")
    rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
    plt.title(f'Results - {EXEC_HASH}', fontsize=30)

    line = np.linspace(TEMP_MIN*0.83, TEMP_MAX*1.15, 200)


    #print('TEMPS_IN', len(TEMPS_IN))
    #print('TEMPS_OUT', len(TEMPS_OUT))
    # data points
    plot = ax.scatter(TEMPS_IN, TEMPS_OUT, c=TEMPS_OUT
                ,edgecolor='black', s=50, label=r'$T_{nn}$', cmap=my_cmap)
    fig.colorbar(plot)
    # Tc line
    plt.vlines(TC, ymin=TEMP_MIN*0.85, ymax=TEMP_MAX*1.15
                ,color='black', label=r'$T_{C} = $'+str(round(TC,4)), lw=3, ls='--')
    # T = T line
    plt.plot(line, line, color='black', label=r'$T_{R}=T_{nn}$', lw=3)

    plt.xlabel(r'$T_{R}$', fontsize=15)
    plt.ylabel(r'$T_{nn}$', fontsize=15)

    plt.xlim(TEMP_MIN*0.85,TEMP_MAX*1.15)
    plt.ylim(TEMP_MIN*0.85,TEMP_MAX*1.15)
    # fig.colorbar(cm.ScalarMappable(cmap=my_cmap)
    #             ,ax=ax
    #             ,values=np.linspace(TEMP_MIN, TEMP_MAX, 10)
    #             ,ticks=np.linspace(TEMP_MIN, TEMP_MAX, 10)
    #             ,drawedges=True)
    plt.legend(fontsize=15)
    if results_plot:  
        root_ = os.path.join(ROOT, 'plots')
        results_plot_path = os.path.join(root_, f'results_{EXEC_HASH}.jpeg')    
        plt.savefig(results_plot_path)
        

    #print(f'TEMPS_OUT', TEMPS_OUT)
    # SAVE TEMPS PLOT DATA
    if save_data:
        df_ = pd.DataFrame({'TR':[float(t) for t in TEMPS_IN]
                            ,'Tnn':[float(t) for t in TEMPS_OUT]})
        root_ = os.path.join(ROOT, 'data')
        df_path = os.path.join(root_, f'results_{EXEC_HASH}.csv')
        df_.to_csv(df_path, index=False)


    # NAO SEI OQ ACONTECE DAQUI PRA BAIXO
    for data in test_dataloader:
        inputs, targets = data[0].float(), data[1].float()
        inputs = inputs.to(device)
        targets = targets.to(device) 
        outputs = mynet(inputs)
        loss = loss_function(outputs, targets)
        
    #print(outputs.shape)
    # print(targets.shape)
            
    # plt.imshow(inputs[0][0].cpu())
    # plt.title(r'$T_p$ = ' + str(outputs[0][0].item()) + ' ; $T_r$ = ' + str(targets[0]))
    # plt.show()

    mydict = mynet.state_dict()
    cnn_w = mynet.state_dict().keys()
    check = []
    for item in cnn_w:
        b = np.array(mydict[item].cpu())
        if b.ndim == 4:
            check.append(item)

    kk = []
    for item in check:
        array = np.array(mydict[item].cpu())
        array = array[0,0,:,:]
        if array.shape == (1,1):
            pass#print(array.shape)
        else:
            kk.append(array)

    #print(len(kk))

    #three plots
    def multi_plot_snapshot(TEMPS_IN, TEMPS_OUT, cc, filename):

        #plot
        fig, axs = plt.subplots(1, 3, figsize=(15,14))
        
        axs[0].imshow(TEMPS_IN, cmap='Greys')
        axs[0].tick_params(left = False, right = False,
                            labelleft=False, labelbottom = False,
                            bottom = False)

        axs[1].imshow(cc, cmap='Greys')
        axs[1].tick_params(left = False, right = False,
                            labelleft=False, labelbottom = False,
                            bottom = False)

        axs[2].imshow(TEMPS_OUT, cmap='Greys')
        axs[2].tick_params(left = False, right = False,
                            labelleft=False, labelbottom = False,
                            bottom = False)
        plt.show()

        return

    return



if __name__ == '__main__':

    pipe(epochs=1, L=120, N=18)








