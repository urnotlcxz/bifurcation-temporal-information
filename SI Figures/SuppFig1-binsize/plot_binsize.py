# -*- coding: utf-8 -*-
"""
Script reproducing figures in the manuscript titled

Bifurcation enhances temporal information encoding in the olfactory periphery

This script plots data testing different binsizes 

Copyright 2024 Nirag Kadakia, Kiri Choi
"""

import matplotlib.pyplot as plt
import pickle
import os, sys

sys.path.append(os.path.abspath('../../utils'))

from paper_utils import save_fig

SAVEFIG = False

#%%

file = './data/Na_K_mu_bintest_4.pkl'
with open(file, 'rb') as f:
	data_dir = pickle.load(f)
    
MI_tot = data_dir['MI_tot']


#%% Supp Fig 1

labels = ['50', '75', '100', '150', '200', '500', '1000']

fig,ax = plt.subplots(7,7,figsize=(10,10), sharex=True)

for i in range(len(MI_tot)):
    for j in range(len(MI_tot[0])):
        ax[i][j].plot(MI_tot[i][j], lw=5)
        if i == 0:
            ax[i][j].set_xlabel(labels[j], fontsize=25, labelpad=10)
            ax[i][j].xaxis.set_label_position('top')
            ax[i][j].set_xticklabels([])
        elif i != 6:
            ax[i][j].set_xticklabels([])
        else:
            ax[i][j].set_xticks([0, 3.6, 7.2])
            ax[i][j].set_xticklabels([0.01, 0.1, 1], rotation=45, fontsize=20)
        if j == 6:
            ax[i][j].set_ylabel(labels[i], rotation=270, labelpad=25, fontsize=25)
            ax[i][j].yaxis.set_label_position('right')
            ax[i][j].set_yticklabels([])
        elif j != 0:
            ax[i][j].set_yticklabels([])
        else:
            ax[i][j].set_yticks([1, 2])
            ax[i][j].set_yticklabels([1, 2], fontsize=20)
        
        ax[i][j].set_ylim(0, 2.75)
        ax[i][j].tick_params(width=2, length=5)
        ax[i][j].spines['top'].set_linewidth(2)
        ax[i][j].spines['left'].set_linewidth(2)
        ax[i][j].spines['bottom'].set_linewidth(2)
        ax[i][j].spines['right'].set_linewidth(2)

plt.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()

