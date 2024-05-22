# -*- coding: utf-8 -*-
"""
Script reproducing figures in the manuscript titled

Bifurcation enhances temporal information encoding in the olfactory periphery

This script plots introductary schematics

Copyright 2024 Nirag Kadakia, Kiri Choi
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size
import os, sys

sys.path.append(os.path.abspath('../../models'))
sys.path.append(os.path.abspath('../../utils'))

from models import *
from paper_utils import save_fig

def dose_resp(cur):
    return np.interp(cur, dose_resp_x, dose_resp_y)
def inv_dose_resp(rate):
    return np.interp(rate, dose_resp_y, dose_resp_x)

SAVEFIG = False

min_x = 300
max_x = 500
nums = 2
fig_w = 1.1
fig_h = 0.5

def gen_fig(_w, _h):
    fig = plt.figure(figsize=(6, 6))
    h = [Size.Fixed(0.3), Size.Fixed(_w)]
    v = [Size.Fixed(0.3), Size.Fixed(_h)]
    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    ax = fig.add_axes(divider.get_position(),
                  axes_locator=divider.new_locator(nx=1, ny=1))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return fig, ax


#%% Fig 1B

dose_resp_x = []
dose_resp_y = []
for i in np.arange(0, 20, 0.1):
    N = 10000
    a = NaP_K_model(dt=0.05, N=N, odor_tau=0.5, odor_mu=0, odor_sigma=0, 
                        rate_sigma=0, k_rate=1/10)
    a.stim = np.ones(N)*i
    a.integrate()
    a.calc_rate()
    dose_resp_x.append(i)
    dose_resp_y.append(a.rate[N//3])

fig, ax = gen_fig(0.7, 0.6)
plt.plot(dose_resp_x, dose_resp_y, color='k')
plt.xticks(range(0, 21, 5), fontsize=7)
plt.yticks(range(0, 201, 50), fontsize=7)
if SAVEFIG:
    save_fig('dose_response', tight_layout=False)
plt.show()

colors = ['0.1', '0.6']
for idx, i in enumerate([3, 10]):
    N = 3000
    a = NaP_K_model(dt=0.05, N=N, odor_tau=0.5, odor_mu=0, odor_sigma=0, 
                        rate_sigma=0, k_rate=1/10)
    a.stim = np.ones(N)*i
    a.stim[:2*N//6] = 0
    a.stim[int(4*N//6):] = 0
    a.integrate()
    a.calc_rate()
    
    fig, ax = gen_fig(0.7, 0.2)
    plt.plot(a.Tt - a.Tt[-1]/6, a.stim, color=colors[idx], lw=1)
    plt.ylim(-2, 11)
    plt.xticks([], fontsize=7)
    plt.yticks(fontsize=7)
    plt.xlim(0, a.Tt[-1]*4/6)
    if SAVEFIG:
        save_fig('stim_%d' % i, tight_layout=False)
    plt.show()
    
    fig, ax = gen_fig(0.7, 0.4)
    plt.plot(a.Tt - a.Tt[-1]/6, a.x[:, 0], color=colors[idx], lw=1)
    plt.xticks([], fontsize=7)
    plt.yticks(fontsize=7)
    plt.xlim(0, a.Tt[-1]*4/6)
    plt.ylim(-85, 20)
    if SAVEFIG:
        save_fig('V_%d' % i, tight_layout=False)
    plt.show()
    

#%% 
mns = [13, 15]
amps = [11, 12]
xs = []
rates = []
stims = []
Tts = []
bin_arrs = []

for i in range(len(amps)):
    N = 30000
    a = NaP_K_model(dt=0.05, N=N, odor_tau=0.5, odor_mu=mns[i], odor_sigma=0, 
                        rate_sigma=0, k_rate=1/5)
    a.stim = [0.]
    np.random.seed(10)
    rands = np.random.uniform(0, 1, N)
    for j in range(1, N):
        if np.exp(-1/200) < rands[j]:
            if a.stim[j - 1] == 0:
                a.stim.append(1.)
            else:
                a.stim.append(0.)
        else:
            a.stim.append(a.stim[j - 1])
    a.stim = np.array(a.stim)
    a.stim -= 1
    a.stim *= amps[i]
    a.stim += mns[i]
    a.integrate()
    a.calc_rate()
    
    xs.append(a.x[:, 0])
    rates.append(a.rate)
    Tts.append(a.Tt)
    stims.append(a.stim)
    bin_arrs.append(a.bin_arr)
    
    if i == 0:
        a.stim = inv_dose_resp(a.rate)
        a.integrate()
        a.calc_rate()

        xs.append(a.x[:, 0])
        rates.append(a.rate)
        Tts.append(a.Tt)
        stims.append(a.stim)
        bin_arrs.append(a.bin_arr)


#%% Fig 1C

mn = 8
amp = 6

N = 3000
a = NaP_K_model(dt=0.05, N=N, odor_tau=0.5, odor_mu=mns[i], odor_sigma=0, 
                    rate_sigma=0, k_rate=1/5)
a.stim = [0.]
np.random.seed(3)
rands = np.random.uniform(0, 1, N)
for j in range(1, N):
    if np.exp(-1/200) < rands[j]:
        if a.stim[j - 1] == 0:
            a.stim.append(1.)
        else:
            a.stim.append(0.)
    else:
        a.stim.append(a.stim[j - 1])
a.stim = np.array(a.stim)
a.stim -= 1
a.stim *= amp
a.stim += mn

# Fluctuating
a.integrate()
a.stim_fluct = a.stim.copy()
a.x_fluct = a.x.copy()

# Remove blip
a.stim[int(58/a.dt):int(85/a.dt)] = mn
a.stim = np.array(a.stim)
a.integrate()
a.stim_const = a.stim.copy()
a.x_const = a.x.copy()

beg = 40
end = 90
fig_w = 0.8
fig_h = 0.5
fig, ax = gen_fig(fig_w, fig_h)
ax.plot(a.Tt - beg, a.stim_const, color='k', lw=1)
ax.plot(a.Tt - beg, a.stim_fluct, color='r', lw=1)
ax.set_xlim(0, end - beg)
plt.xticks(np.arange(0, 100, 25), fontsize=7)
plt.yticks(fontsize=7)
ax.set_ylim(0, 10)
ax.axhline(4.54, color='k', ls='--', lw=1)
if SAVEFIG:
    save_fig('rate_drop_spike_delay_stim', tight_layout=False)
plt.show()

fig, ax = gen_fig(fig_w, fig_h)
ax.set_xlim(0, end - beg)
plt.xticks(np.arange(0, 100, 25), fontsize=7)
plt.yticks(fontsize=7)
ax.plot(a.Tt - beg, a.x_fluct[:, 0], color='r', lw=1)
if SAVEFIG:
    save_fig('rate_drop_spike_delay_V', tight_layout=False)
plt.show()

fig, ax = gen_fig(fig_w, fig_h)
ax.set_xlim(0, end - beg)
plt.xticks(np.arange(0, 100, 25), fontsize=7)
plt.yticks(fontsize=7)
ax.plot(a.Tt - beg, a.x_const[:, 0], color='k', lw=1)
if SAVEFIG:
    save_fig('rate_drop_spike_delay_V_const_curr', tight_layout=False)
plt.show()


#%% Fig 1D

colors = ['dodgerblue', 'k', 'orangered']
lss =['-', '-', '-']
fig, ax = gen_fig(fig_w, fig_h)
for i in range(nums):
    ax.plot(Tts[i], stims[i], color=colors[i], ls='-', lw=1)
    plt.xticks(fontsize=7)
    plt.yticks([0, 5, 10], fontsize=7)
    plt.xlim(min_x, max_x)
    plt.ylim(0, 15)
    plt.axhline(4.54, color='k', ls='--', lw=0.5)
if SAVEFIG:
    save_fig('compare_stims', tight_layout=False)
plt.show()
    
fig, ax = gen_fig(fig_w, fig_h)
for i in range(nums):
    plt.plot(Tts[i], xs[i], color=colors[i], ls='-', lw=1)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.xlim(min_x, max_x)
if SAVEFIG:
    save_fig('compare_vs', tight_layout=False)
plt.show()
    
fig, ax = gen_fig(fig_w, fig_h)
for i in range(nums):
    plt.plot(Tts[i], rates[i], color=colors[i], ls=lss[i], lw=1)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.xlim(min_x, max_x)
if SAVEFIG:
    save_fig('compare_rates', tight_layout=False)
plt.show()




