# -*- coding: utf-8 -*-
"""
Script reproducing figures in the manuscript titled

Bifurcation enhances temporal information encoding in the olfactory periphery

This script plots MI- plot over mu using ORN model

Copyright 2024 Nirag Kadakia, Kiri Choi
"""


import numpy as np
import pickle
import matplotlib.pyplot as plt
import os, sys

sys.path.append(os.path.abspath('../../models'))
sys.path.append(os.path.abspath('../../utils'))

from models import *
from paper_utils import save_fig, gen_plot

SAVEFIG = False

#%% Supp Fig 8B
    
file = './data/NaP_K_adapting_MI_mu_sweep.pkl'

with open(file, 'rb') as f:
    data_dir = pickle.load(f)

params = data_dir['params']
mus = params['mus']
sigmas = params['sigmas']
MI_neg = data_dir['MI_neg']

fig, ax = gen_plot(1.2, 1.2)
colors = plt.cm.Greens(np.linspace(0.4, 1, len(mus)))
mus = [4, 6, 10, 15]
sigmas = np.logspace(-2, 0.5, 12)
for im in range(len(mus)):
    plt.plot(sigmas, MI_neg[im], color=colors[im], lw=1)
plt.yticks([0, 0.5, 1], fontsize=7)
plt.xscale('log')
plt.xticks(fontsize=7)
plt.xlim(0.01, 1)
if SAVEFIG:
    save_fig('NaP_K_neg_MI')
plt.show()

fig, ax = gen_plot(1.2, 1.2)
colors = plt.cm.Greens(np.linspace(0.5, 1, len(mus)))
mus = [4, 6, 10, 15]
sigmas = np.logspace(-2, 0.5, 12)
for im in range(len(mus)):
    plt.plot(sigmas/mus[im], MI_neg[im], color=colors[im], lw=1)
plt.yticks([0, 0.5, 1], fontsize=7)
plt.xscale('log')
plt.xticks(fontsize=7)
plt.xlim(0.001, 0.1)
if SAVEFIG:
    save_fig('ORN_MI_mean_adaptation_mu_scaled')
plt.show()

