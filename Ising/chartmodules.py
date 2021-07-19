# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 12:10:40 2021

@author: mgteus

"""


""" criar as funcoes para plotar os graficos de serie temporal e histogramas"""

import matplotlib.pyplot as plt
import numpy as np 





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
        freqs_en, vals_en = np.histogram(medidas_en, n_bins)
        
        
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
        

