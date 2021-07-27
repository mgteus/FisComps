# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 12:10:40 2021

@author: mgteus

"""


""" criar as funcoes para plotar os graficos de serie temporal e histogramas"""

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd


    

def hist_en(medidas_en):
    # PLOT MAG HIST##
    #configs
    #-------------------------- PLT CONFIGS ----------------------------------------
    fig, ax = plt.subplots(figsize=(16,9), dpi=120)
    plt.title(' Histograma da Energia', fontsize=30)
    plt.xlabel('mag', fontsize=15)
    plt.ylabel('H(mag)', fontsize=15)
    
    #---------------------- CHART CONFIGURATION ------------------------------------
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    
    if len(medidas_en)>10**5:
    
        # histogram configs
        n_bins = int(1 + 3.22*np.log(len(medidas_en[10**5+1:]))) # sturge's rule
        freqs_en, vals_en = np.histogram(medidas_en[10**5+1:], n_bins)
        
        
        plt.scatter(vals_en[:-1], freqs_en, label='E')
        plt.legend(loc='best', fontsize=20)
    
        return plt.show()
    
    else:
        
        n_bins = int(1 + 3.22*np.log(len(medidas_en))) # sturge's rule
        freqs_en, vals_en = np.histogram(medidas_en, 28)
        
        
        plt.scatter(vals_en[:-1], freqs_en, label='E')
        plt.legend(loc='best', fontsize=20)
    
        return plt.show()
        
        




def hist_mag(medidas_mag):
    ## PLOT MAG HIST##
    #configs
    #-------------------------- PLT CONFIGS ----------------------------------------
    fig, ax = plt.subplots(figsize=(16,9), dpi=120)
    plt.title(' Histograma da Magnitude', fontsize=30)
    plt.xlabel('mag', fontsize=15)
    plt.ylabel('H(mag)', fontsize=15)
    
    #---------------------- CHART CONFIGURATION ------------------------------------
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    
    if len(medidas_mag)>10**5:
        # histgram configs
        n_bins = int(1 + 3.22*np.log(len(medidas_mag[10**5:]))) # sturge's rule
        freqs, vals = np.histogram(medidas_mag[10**5:], n_bins)
        
        
        plt.scatter(vals[:-1], freqs, label='mag')
        plt.legend(loc='upper center', fontsize=20)
        
        return plt.show()
    
    
    else:
        n_bins = int(1 + 3.22*np.log(len(medidas_mag))) # sturge's rule
        freqs, vals = np.histogram(medidas_mag, n_bins)
        
        
        plt.scatter(vals[:-1], freqs, label='mag')
        plt.legend(loc='upper center', fontsize=20)
        
        return plt.show()
        
    
    





def plot_from_csv(parallel=False):
    if parallel:
        path_to_csv = 'C:/Users/mateu/workspace/MonteCarlo/Ising/data_paralelo.csv'
    else:   
        path_to_csv = 'C:/Users/mateu/workspace/MonteCarlo/Ising/data22.csv'
        path_to_csv2 = 'C:/Users/mateu/workspace/MonteCarlo/Ising/data24.csv'
        path_to_csv3 = 'C:/Users/mateu/workspace/MonteCarlo/Ising/data30.csv'
    
    
    path_to_figs = 'C:/Users/mateu/workspace/MonteCarlo/Ising/charts/'
    df = pd.read_csv(path_to_csv) 
    df2 = pd.read_csv(path_to_csv2) 
    df3 = pd.read_csv(path_to_csv3) 
    
    #time_array = df['tempo']
    mag_array  = df['mag']
    en_array   = df['energia']
    
    mag_array2  = df2['mag']
    en_array2   = df2['energia']
    
    
    mag_array3  = df3['mag']
    en_array3   = df3['energia']
    
    mag_width = 2
    mag_bins = np.arange(min(mag_array), max(mag_array) + mag_width, mag_width)
    mag_bins2 = np.arange(min(mag_array2), max(mag_array2) + mag_width, mag_width)
    mag_bins3 = np.arange(min(mag_array3), max(mag_array3) + mag_width, mag_width)
    
    en_width = 4
    en_bins = np.arange(min(en_array), max(en_array) + en_width, en_width)
    en_bins2 = np.arange(min(en_array2), max(en_array2) + en_width, en_width)
    en_bins3 = np.arange(min(en_array3), max(en_array3) + en_width, en_width)
    

    
    fig, ax = plt.subplots(figsize=(16,9), dpi=120)
    #-------------------- CHART CONFIGURATION ---------------------#
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    
    
    plt.title('Histograma da Magnetização', fontsize=30)
    plt.xlabel('mag', fontsize=20)
    plt.ylabel('H(mag)', fontsize=20)
    
    
    
    
    plt.hist(mag_array, bins=mag_bins, color='r', label='T = 2.2')
    plt.hist(mag_array2, bins=mag_bins2, color='navy', label='T = 2.4')
    plt.hist(mag_array3, bins=mag_bins3, color='green', label='T = 3.0')
    
    plt.legend(loc='best', fontsize=20)
    
    if parallel:
        figname1 = path_to_figs+'mag_hist_p.jpeg'
    else:
        figname1 = path_to_figs+'mag_hist.jpeg'
        
    plt.savefig(figname1)
    
    plt.show()
    
    
   
    
    
    fig, ax = plt.subplots(figsize=(16,9), dpi=120)
    #-------------------- CHART CONFIGURATION ---------------------#
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    plt.title('Histograma da Energia', fontsize=30)
    plt.xlabel('Energia', fontsize=20)
    plt.ylabel('H(E)', fontsize=20)
    
    
    plt.hist(en_array, bins=en_bins, color='red', label='T = 2.2')
    plt.hist(en_array2, bins=en_bins2, color='navy', label='T = 2.4')
    plt.hist(en_array3, bins=en_bins3, color='green', label='T = 30')
    plt.legend(loc='best', fontsize=20)
    
    if parallel:
        figname2 = path_to_figs+'en_hist_p.jpeg'
    else:
        figname2 = path_to_figs+'en_hist.jpeg'
        
    plt.savefig(figname2)
    
    plt.show()
    
    
    
    return
    
    
    
    
    

def restart_csv(TEMP, parallel=False):
    if parallel:
        path_to_csv = 'C:/Users/mateu/workspace/MonteCarlo/Ising/data_paralelo.csv'
    else:
        path_to_csv = 'C:/Users/mateu/workspace/MonteCarlo/Ising/data.csv'  
        
    if TEMP == 2.2:
       path_to_csv = "C:/Users/mateu/workspace/MonteCarlo/Ising/data22.csv"
    elif TEMP == 2.4:
       path_to_csv = "C:/Users/mateu/workspace/MonteCarlo/Ising/data24.csv"
    else:
       path_to_csv = "C:/Users/mateu/workspace/MonteCarlo/Ising/data30.csv"
    
    with open(path_to_csv, 'w') as file:
        file.write('tempo,mag,energia\n')
        pass
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

