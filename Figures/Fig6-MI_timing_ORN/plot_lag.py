# -*- coding: utf-8 -*-
"""
Script reproducing figures in the manuscript titled

Bifurcation enhances temporal information encoding in the olfactory periphery

This script plots lag traces from cross-correlation

Copyright 2024 Nirag Kadakia, Kiri Choi
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import os, sys

sys.path.append(os.path.abspath('../../models'))
sys.path.append(os.path.abspath('../../utils'))

from models import *
from paper_utils import save_fig

SAVEFIG = False

#%% Fig 6A

dt1 = 0.1
dt2 = 0.05
mus = np.logspace(0.65, 1.2, 10)
sigma = 0.4

lags_ORN_Ca = np.load('./data/NaKCa_lag_2.npy')
lags_NaK = np.load('./data/NaK_lag_3.npy')

lags_ORN_Ca_m = np.mean(lags_ORN_Ca, axis=1)
lags_ORN_Ca_err = scipy.stats.sem(lags_ORN_Ca, axis=1)
lags_NaK_m = np.mean(lags_NaK, axis=1)
lags_NaK_err = scipy.stats.sem(lags_NaK, axis=1)

Ca_spearman = scipy.stats.spearmanr(mus, np.array(lags_ORN_Ca_m))
NaK_spearman = scipy.stats.spearmanr(mus, np.array(lags_NaK_m))

fig = plt.figure(figsize=(3.25,3))
plt.plot(mus, (lags_NaK_m-lags_NaK_m[0])*dt2, lw=3, color='tab:green')
plt.plot(mus, (lags_ORN_Ca_m-lags_ORN_Ca_m[0])*dt1, lw=3, color='tab:red')
plt.xticks([5, 10, 15], fontsize=15)
plt.yticks([-2, -1, 0, 1, 2], fontsize=15)
plt.text(8.3,-1.45,r'$\rho$={0}, p={1}'.format(round(NaK_spearman[0],3), round(NaK_spearman[1],3)),
         fontsize=13, color='tab:green')
plt.text(8.3,-1.8,r'$\rho$={0}, p={1}'.format(round(Ca_spearman[0],3), round(Ca_spearman[1],3)),
         fontsize=13, color='tab:red')
plt.xlabel('$\mu$', fontsize=15)
plt.ylabel('$\Delta$Lag ($ms$)', fontsize=15)
plt.tight_layout()
if SAVEFIG:
    save_fig('model_firing_lag', tight_layout=False)
plt.show()