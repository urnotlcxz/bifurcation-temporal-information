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

# SAVEFIG = False
SAVEFIG = True

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


#%% Fig 1C

dose_resp_x = []
dose_resp_y = []
for i in np.arange(14, 17, 0.01):
    N = 50000
    a = SimplifiedLIFModel(dt=0.05, N=N, odor_tau=0.5, odor_mu=0, odor_sigma=0, 
                        rate_sigma=0, k_rate=1/20, adaptation_strength=0.0, V_th=-50, V_reset=-70, V_rest=-65)   # NaP_K_model
    a.stim = np.ones(N)*i   # 恒定电流输入
    a.integrate()
    a.calc_rate()
    dose_resp_x.append(i)
    # start = int(0.2 * N)
    start = N // 4
    end = 3 * N // 4
    avg_rate = np.mean(a.rate[start: end])  # 去掉起始/结束的瞬态
    dose_resp_y.append(avg_rate)
    # dose_resp_y.append(a.rate[N//3])  # 取中间时间点的发放率
    
    print("input current:", i, "rate:", a.rate[N-1])
dose_resp_x = np.array(dose_resp_x)
dose_resp_y = np.array(dose_resp_y)
np.savez('dose_response.npz', x=dose_resp_x, y=dose_resp_y)

fig, ax = gen_fig(0.7, 0.6)
# plt.plot(a.rate)   # 发放率震荡说明了滤波器太“窄”或模拟时间太短:
# 计算 firing rate 时用了 gaussian_filter()、如果 sigma = 1/self.k_rate/self.dt 太小，会造成 rate 不够平滑
# 或者如果 spikes 本身周期性强（比如周期性发放），rate 就会出现震荡。
plt.plot(dose_resp_x, dose_resp_y, color='k')
plt.xticks(range(14, 17, 1), fontsize=7)

plt.yticks(range(0, 100, 20), fontsize=7)
threshold_idx = np.argmax(np.array(dose_resp_y) > 0)
if threshold_idx > 0:
    plt.axvline(x=dose_resp_x[threshold_idx], color='r', linestyle='--', 
                label=f'Threshold = {dose_resp_x[threshold_idx]:.1f} pA')
    plt.legend()
plt.xlabel("Input Current (pA)")
plt.ylabel("Firing Rate (Hz)")
plt.title("Dose-Response")

if SAVEFIG:
    save_fig('dose_response', tight_layout=False)
plt.show()

colors = ['0.1', '0.6']
for idx, i in enumerate([5, 20]):
    N = 3000
    a = SimplifiedLIFModel(dt=0.05, N=N, odor_tau=0.5, odor_mu=0, odor_sigma=0, 
                        rate_sigma=0, k_rate=1/20)  # NaP_K_model
    a.stim = np.ones(N)*i
    a.stim[:2*N//6] = 0
    a.stim[int(4*N//6):] = 0
    a.integrate()
    a.calc_rate()
    
    fig, ax = gen_fig(0.7, 0.2)
    plt.plot(a.Tt - a.Tt[-1]/6, a.stim, color=colors[idx], lw=1)
    plt.ylim(-2, 20)
    plt.xticks([], fontsize=7)
    plt.yticks(fontsize=7)
    plt.xlim(0, a.Tt[-1]*4/6)
    if SAVEFIG:
        save_fig('stim_%d' % i, tight_layout=False)
    plt.show()
    
    fig, ax = gen_fig(0.7, 0.4)
    # plt.plot(a.Tt - a.Tt[-1]/6, a.x[:, 0], color=colors[idx], lw=1)
    plt.plot(a.Tt - a.Tt[-1]/6, a.V, color=colors[idx], lw=1)

    plt.xticks([], fontsize=7)
    plt.yticks(fontsize=7)
    plt.xlim(0, a.Tt[-1]*4/6)
    plt.ylim(-85, 20)
    if SAVEFIG:
        save_fig('V_%d' % i, tight_layout=False)
    plt.show()
    

# %% Fig 1D
mns = [13, 15]
amps = [11, 12]
xs = []
rates = []
stims = []
Tts = []
bin_arrs = []

for i in range(len(amps)):
    N = 30000
    a = SimplifiedLIFModel(dt=0.05, N=N, odor_tau=0.5, odor_mu=mns[i], odor_sigma=0, 
                        rate_sigma=0, k_rate=1/10)  #k_rate=1/5
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
    
    # xs.append(a.x[:, 0])
    xs.append(a.V)
    rates.append(a.rate)
    Tts.append(a.Tt)
    stims.append(a.stim)
    # bin_arrs.append(a.bin_arr)
    bin_arrs.append(a.spikes)
    
    if i == 0:
        a.stim = inv_dose_resp(a.rate)
        a.integrate()
        a.calc_rate()

        # xs.append(a.x[:, 0])
        xs.append(a.V)
        rates.append(a.rate)
        Tts.append(a.Tt)
        stims.append(a.stim)
        # bin_arrs.append(a.bin_arr)
        bin_arrs.append(a.spikes)


# %% Fig 1D

mn = 8
amp = 6

N = 3000
a = SimplifiedLIFModel(dt=0.05, N=N, odor_tau=0.5, odor_mu=mns[i], odor_sigma=0, 
                    rate_sigma=0, k_rate=1/10)  # k_rate=1/5

# 构造 binary 状态输入（类似突变信号）
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

# 模拟 fluctuating 情况（有跳变）
a.integrate()
a.stim_fluct = a.stim.copy()
# a.x_fluct = a.x.copy()
a.V_fluct = a.V.copy()

# Remove blip  去除某段跳变，构造 constant 情况
a.stim[int(58/a.dt):int(85/a.dt)] = mn
a.stim = np.array(a.stim)
a.integrate()
a.stim_const = a.stim.copy()
# a.x_const = a.x.copy()
a.V_const = a.V.copy()

beg = 40
end = 90
fig_w = 0.8
fig_h = 0.5
# ---- 图 1：刺激轨迹（突变 vs 平滑）
fig, ax = gen_fig(fig_w, fig_h)
ax.plot(a.Tt - beg, a.stim_const, color='k', lw=1)
ax.plot(a.Tt - beg, a.stim_fluct, color='r', lw=1)
ax.set_xlim(0, end - beg)
plt.xticks(np.arange(0, 100, 25), fontsize=7)
plt.yticks(fontsize=7)
ax.set_ylim(0, 20)
ax.axhline(15.0, color='k', ls='--', lw=1)
if SAVEFIG:
    save_fig('rate_drop_spike_delay_stim', tight_layout=False)
plt.show()

# ---- 图 2：膜电位轨迹（fluctuating）
fig, ax = gen_fig(fig_w, fig_h)
ax.set_xlim(0, end - beg)
plt.xticks(np.arange(0, 100, 25), fontsize=7)
plt.yticks(fontsize=7)
# ax.plot(a.Tt - beg, a.x_fluct[:, 0], color='r', lw=1)
ax.plot(a.Tt - beg, a.V_fluct, color='r', lw=1)
if SAVEFIG:
    save_fig('rate_drop_spike_delay_V', tight_layout=False)
plt.show()

# ---- 图 3：膜电位轨迹（constant）
fig, ax = gen_fig(fig_w, fig_h)
ax.set_xlim(0, end - beg)
plt.xticks(np.arange(0, 100, 25), fontsize=7)
plt.yticks(fontsize=7)
# ax.plot(a.Tt - beg, a.x_const[:, 0], color='k', lw=1)
ax.plot(a.Tt - beg, a.V_const, color='k', lw=1)
if SAVEFIG:
    save_fig('rate_drop_spike_delay_V_const_curr', tight_layout=False)
plt.show()






