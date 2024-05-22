# -*- coding: utf-8 -*-
"""
Script reproducing figures in the manuscript titled

Bifurcation enhances temporal information encoding in the olfactory periphery

This script generates adaptation schematics using Na+K+Ca model

Copyright 2024 Nirag Kadakia, Kiri Choi
"""

import numpy as np
import os, sys

sys.path.append(os.path.abspath('../../models'))

data = np.loadtxt('./data/NaP_K_adapting_timetrace_same_sigma.npy')

N = 500000

SAVE = False

#%%

from models import *

a = ORN_model(dt=0.05, N=N, odor_tau=0.5, odor_mu=0, odor_sigma=0, 
                    rate_sigma=0, k_rate=1/10)

a.stim = data[:,1]
a.integrate([-63, 0, 0, 0])
a.calc_rate()

if SAVE:
    data_new = np.vstack((a.stim, a.rate))
    np.save('./data/NaP_K_adapting_timetrace_same_sigma_new.npy', data_new.T, allow_pickle=True)
