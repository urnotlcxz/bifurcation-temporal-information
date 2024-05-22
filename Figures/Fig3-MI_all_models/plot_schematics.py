# -*- coding: utf-8 -*-
"""
Script reproducing figures in the manuscript titled

Bifurcation enhances temporal information encoding in the olfactory periphery

This script plots insets explaining MI-

Copyright 2024 Nirag Kadakia, Kiri Choi
"""

import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append(os.path.abspath('../../models'))
sys.path.append(os.path.abspath('../../utils'))

from models import *
from paper_utils import save_fig, gen_plot

SAVEFIG = False

#%% Fig 3A

mu = 4.54
N = 200000
dt = 0.05
a = NaP_K_model(dt=dt, N=N, odor_tau=500, odor_mu=mu, odor_sigma=0.1, 
				rate_sigma=0.01	, k_rate=1/50)
a.gen_sig_trace_OU(seed=2, smooth_T=20)
a.integrate()
a.calc_rate()

mu = 4.54
N = 200000
dt = 0.05
a = NaP_K_model(dt=dt, N=N, odor_tau=500, odor_mu=mu, odor_sigma=0.1, 
				rate_sigma=0.01	, k_rate=1/50)
a.gen_sig_trace_OU(seed=2, smooth_T=20)
a.integrate()
a.calc_rate()

fig, ax = gen_plot(1.2, 0.8)
plt.xticks([])
plt.yticks([])
bin_stim = a.stim > mu
splits = np.where(np.diff(bin_stim) != 0)[0]
split_stim = [a.stim[:splits[0]]]
split_Tt = [a.Tt[:splits[0]]]
colors = [True if a.stim[0] > 4.54 else False]
for iS in range(1, len(splits)):
	colors.append(not colors[-1])
	split_stim.append(a.stim[splits[iS - 1]:splits[iS]])
	split_Tt.append(a.Tt[splits[iS - 1]:splits[iS]])
colors.append(not colors[-1])
split_Tt.append(a.Tt[splits[iS]:])
split_stim.append(a.stim[splits[iS]:])
for iS in range(0, len(split_stim)):
	if colors[iS] == True:
		color = 'k'
	else:
		color = 'green'
	plt.plot(split_Tt[iS], split_stim[iS], color=color, lw=1)
plt.axhline(4.54, color='b', ls='--', lw=1)
if SAVEFIG:
    save_fig('ex_odor_colored')
plt.show()


