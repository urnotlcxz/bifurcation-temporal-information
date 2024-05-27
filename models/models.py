"""
Biophysical models for neurons.

Copyright 2024 Nirag Kadakia, Kiri Choi

This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate, convolve
from scipy.interpolate import splrep, splev
from utils import calc_MI_xy


class NaP_K_model():
    
    def __init__(self, dt=0.1, N=1000000, odor_tau=200, odor_mu=4.53, odor_sigma=0.3, 
                 k_rate=1./200, T_last_k_rate=1/100, rate_sigma=0.5):
        self.N = N 
        self.dt = dt
        self.Tt = np.linspace(0, self.N*self.dt, self.N)
        
        self.C = 1.
        self.EL = -80.
        self.gL = 8.
        self.gNa = 20.
        self.gK = 10.
        self.Vm = -20.
        self.km = 15.
        self.Vn = -25.
        self.kn = 5.
        self.tau = 1.
        self.ENa = 60.
        self.EK = -90.

        self.I_mu = odor_mu
        self.I_sigma = odor_sigma
        self.I_tau = odor_tau
        
        self.k_rate = k_rate
        self.T_last_k_rate = T_last_k_rate
        self.rate_sigma = rate_sigma
        
    def df(self, x, iT):
        V, n = x
        m_inf = (1 + np.exp((self.Vm - V)/self.km))**-1
        n_inf = (1 + np.exp((self.Vn - V)/self.kn))**-1

        dV = (self.stim[iT] - self.gL*(V - self.EL) - self.gNa*m_inf*(V - self.ENa)\
              - self.gK*n*(V - self.EK))/self.C
        dn = (n_inf - n)/self.tau

        return np.array([dV, dn])

    def gen_sig_trace_OU(self, seed=0, smooth_T=None):
        rng = np.random.default_rng(seed)
        rands = rng.normal(0, 1, self.N)
        st = np.zeros(self.N)
        D = self.I_sigma**2*2/self.I_tau
        for iT in range(1, self.N):
            st[iT] = st[iT - 1]*np.exp(-1/self.I_tau*self.dt) + \
                (D/2*self.I_tau*(1 - np.exp(-2/self.I_tau*self.dt)))**0.5*rands[iT]
        if smooth_T is not None:
            st = gaussian_filter(st, sigma=int(smooth_T/self.dt))
        self.stim = st + self.I_mu
    
    def gen_sig_trace_OU_binary(self, seed=0, smooth_T=None, partial='none'):
        rng = np.random.default_rng(seed)
        rands = rng.normal(0, 1, self.N)
        st = np.zeros(self.N)
        D = self.I_sigma**2*2/self.I_tau
        for iT in range(1, self.N):
            st[iT] = st[iT - 1]*np.exp(-1/self.I_tau*self.dt) + \
                (D/2*self.I_tau*(1 - np.exp(-2/self.I_tau*self.dt)))**0.5*rands[iT]
        if smooth_T is not None:
            st = gaussian_filter(st, sigma=int(smooth_T/self.dt))
        if partial == 'up':
            st[st>0] = self.I_sigma
        elif partial == 'down':
            st[st<0] = -self.I_sigma
        else:
            st[st>0] = self.I_sigma
            st[st<0] = -self.I_sigma
        self.stim = st + self.I_mu
    
    def gen_sig_trace_Gaussian(self, seed=0):
        rng = np.random.default_rng(seed)
        I_vec = rng.normal(self.I_mu, self.I_sigma, int(self.N/200*self.dt))
        I_vec_Tt = np.linspace(0, self.N, self.N)
        tck = splrep(np.linspace(0, self.N, int(self.N/200*self.dt)), I_vec)
        self.stim = splev(I_vec_Tt, tck)
        
    def gen_sig_trace_sinusoidal(self, amplitude=1, theta=0.5, thres=0):
        I_vec_Tt = np.linspace(0, self.N, self.N)
        I_vec = amplitude*np.power(1 + np.sin(np.linspace(0,1*np.pi,len(I_vec_Tt))*np.pi*2/theta),3) - thres
        I_vec[I_vec<0] = 0
        self.stim = I_vec
        
    def calc_rate(self, filt_len=30000, filt_type='Gaussian', offset=None):
        V = self.x[:, 0]
        peak_idxs = find_peaks(V, 0)[0]
        self.bin_arr = np.zeros(self.N)
        self.bin_arr[peak_idxs] = 1

        if filt_type == 'exp':
            Tt = np.linspace(0, self.dt*filt_len, filt_len)
            filt = self.dt*self.k_rate*np.exp(-1.*Tt*self.k_rate)
            self.rate = convolve(filt, self.bin_arr, mode='full')[:self.N]
        elif filt_type == 'Gaussian':
            self.rate = gaussian_filter(self.bin_arr, sigma=1/self.k_rate/self.dt)
        elif filt_type == 'delayed_exp':
            Tt = np.linspace(0, self.dt*filt_len, filt_len)
            filt = self.dt*self.k_rate**3/2*(Tt**2.0*np.exp(-Tt*self.k_rate))
            if offset is None:
                offset = 2/self.k_rate
            self.rate = convolve(filt, self.bin_arr, mode='full')[:self.N]
            self.rate = np.roll(self.rate, -int(offset/self.dt))
        elif filt_type == 'Cauchy':
            Tt = np.linspace(0, self.dt*filt_len, filt_len)
            if offset is None:
                offset = 2/self.k_rate
            filt = self.dt*self.k_rate/np.pi*(1 + ((Tt - offset)*self.k_rate)**2.)**-1
            self.rate = convolve(filt, self.bin_arr, mode='full')[:self.N]
            self.rate = np.roll(self.rate, -int(offset/self.dt))
        
        self.rate *= 1000/self.dt
        self.rate += np.random.normal(0, self.rate_sigma, self.N)
        
    def calc_T_to_last_hit(self):
        V = self.x[:, 0]
        peak_idxs = find_peaks(V, 0)[0]
        
        self.T_to_last_hit = np.zeros(self.N)
        self.T_to_last_hit[:peak_idxs[0]] = range(peak_idxs[0])
        for iD in range(len(peak_idxs) - 1):
            beg = peak_idxs[iD]
            end = peak_idxs[iD + 1]
            self.T_to_last_hit[beg:end] = range(end - beg)
        self.T_to_last_hit *= self.dt
        self.T_to_last_hit_smth = gaussian_filter(self.T_to_last_hit, 
                                    sigma=1/self.T_last_k_rate/self.dt)
        
    def integrate(self):
        x = np.zeros((self.N, 2))
        x[0,0] = -63
        for iT in range(self.N - 1):
            x[iT + 1] = x[iT] + self.dt*self.df(x[iT], iT)
        self.x = x
        
    def calc_MI(self, stim_lims=[0, 10], num_stim_bins=200, rate_lims=[0, 200],
                num_rate_bins=300, split_stim=None, cutout=0):
        stim_bins = np.linspace(stim_lims[0], stim_lims[1], num_stim_bins)
        rate_bins = np.linspace(rate_lims[0], rate_lims[1], num_rate_bins)

        rate = self.rate
        stim = self.stim
        rate = rate[int(cutout/self.dt):]
        stim = stim[int(cutout/self.dt):]
        
        if split_stim is None:
            pass
        elif split_stim == 'neg':
            rate = rate[stim < self.I_mu]
            stim = stim[stim < self.I_mu]
        elif split_stim == 'pos':
            rate = rate[stim > self.I_mu]
            stim = stim[stim > self.I_mu]
        
        self.MI = calc_MI_xy(stim, stim_bins, rate, rate_bins)
        
    def calc_avg_dose_response(self, num_stim_bins=100):
        stim_bins = np.linspace(min(self.stim), max(self.stim), num_stim_bins)
        stim_binned = np.digitize(self.stim, bins=stim_bins, right=True) - 1
        rates_binned = [[] for i in range(len(stim_bins))]
        
        dose_response_avg = []
        dose_response_std = []
        for iS in range(len(self.stim)):
            rates_binned[stim_binned[iS]].append(self.rate[iS])
        for iS in range(num_stim_bins - 1):
            if len(rates_binned[iS]) > 100:
                dose_response_avg.append(np.mean(rates_binned[iS]))
                dose_response_std.append(np.std(rates_binned[iS]))
            else:
                dose_response_avg.append(np.nan)
                dose_response_std.append(np.nan)
        self.dose_response_bins = (stim_bins[1:] + stim_bins[:-1])/2.
        self.dose_response_avg = np.array(dose_response_avg)
        self.dose_response_std = np.array(dose_response_std)


class ORN_model_Ca(NaP_K_model):

    def __init__(self, N=1000000, dt=0.1, odor_tau=300, odor_mu=5, odor_sigma=1, 
                 k_rate=1/30., rate_sigma=0.5, class_2=False):
        self.N = N
        self.dt = dt
        self.Tt = np.linspace(0, self.N*self.dt, self.N)
        
        if class_2 == True:
            self.beta_w = -15
        else:
            self.beta_w = 0
        self.g_fast = 20
        self.g_slow = 20
        self.g_leak = 2
        self.E_Na = 50 
        self.E_K = -100
        self.E_leak = -70
        self.phi_w = 0.15*2/2.5
        self.C = 1.0*2.5
        self.beta_m = -1.2
        self.gamma_m = 18
        self.gamma_w = 10
        self.kA = 1
        self.A0 = 0.075
        self.odor_beg_time = 100
        self.stim_noise_sigma = 1e-3
        self.stim_gain = 500
        self.tau_Ca = 250
        
        self.filt_sigma_ms = 50
        self.peak_height = 30
        
        self.I_mu = odor_mu
        self.I_sigma = odor_sigma
        self.I_tau = odor_tau
        
        # For filtering to get firing rate; timescle and noise
        self.k_rate = k_rate
        self.rate_sigma = rate_sigma
        
    def dVdt(self, V, w, _Aa):
        val = 1./self.C*(self.I_olf(_Aa) - self.g_fast*self.m_inf(V)*(V - self.E_Na)\
                         - self.g_slow*w*(V - self.E_K) - self.g_leak*(V - self.E_leak))
        return val

    def dwdt(self, w, V):
        return self.phi_w*(self.w_inf(V) - w)/self.tau_w(V)

    def m_inf(self, V):
        return 0.5*(1 + np.tanh((V - self.beta_m)/self.gamma_m))

    def w_inf(self, V):
        return 0.5*(1 + np.tanh((V - self.beta_w)/self.gamma_w))

    def tau_w(self, V):
        return 1./np.cosh((V - self.beta_w)/(2*self.gamma_w))

    def I_olf(self, _Aa):
        return self.stim_gain*_Aa*(np.random.normal(1, self.stim_noise_sigma))
    
    def Aa(self, Ca, iT):
        return self.stim[iT]/(0.1 + self.stim[iT] + Ca/1)

    def dCadt(self, _Aa, iT, Ca):
        return (0.76875*self.stim[iT] - 0.0625*Ca)/self.tau_Ca

    def euler(self, V, w, _Aa, Ca, iT):    
        V_new = V + self.dt*self.dVdt(V, w, _Aa)
        w_new = w + self.dt*self.dwdt(w, V)
        Ca_new = max(Ca + self.dt*self.dCadt(_Aa, iT, Ca), 0)
        return (V_new, w_new, Ca_new)

    def integrate(self, X_init=np.zeros(3)):
        V_arr = [-70]
        w_arr = [X_init[0]]
        Ca_arr = [X_init[1]]
        Aa_arr = [X_init[2]]
        
        for iT in range(1, self.N):
            _Aa = self.Aa(Ca_arr[-1], iT)
            V_new, w_new, Ca_new = self.euler(V_arr[-1], w_arr[-1], _Aa, 
                                              Ca_arr[-1], iT)
            V_arr.append(V_new)
            w_arr.append(w_new)
            Ca_arr.append(Ca_new)
            Aa_arr.append(_Aa)
        
        self.x = np.vstack((np.array(V_arr), np.array(w_arr))).T
        self.x = np.vstack((self.x.T, Ca_arr, Aa_arr)).T


class LN_model():
    
    def __init__(self, N=1000000, odor_tau=200, odor_mu=4.53, 
                 odor_sigma=0.3, k_rate=0.1, rate_sigma=0.5, NF_type='relu', 
                 SNIC_response=None):
        self.N = N
        self.NF_type = NF_type

        self.I_mu = odor_mu
        self.I_sigma = odor_sigma
        self.I_tau = odor_tau
        
        self.k_rate = k_rate
        self.rate_sigma = rate_sigma
        
        if SNIC_response != None:
            self.SNIC_response = np.load(SNIC_response)
        
    def dose_resp(self, cur):
        return np.interp(cur, self.SNIC_response[0], self.SNIC_response[1]/100)

    def dose_resp_approx(self, cur):
        return np.nan_to_num(10/(1*np.pi/(np.sqrt(0.3*(cur-4.51))) + 4.6))
    
    def gen_sig_trace_OU(self, seed=0, smooth_T=None):
        rng = np.random.default_rng(seed)
        rands = rng.normal(0, 1, self.N)
        stim = np.zeros(self.N)
        D = self.I_sigma**2*2/self.I_tau
        for iT in range(1, self.N):
            stim[iT] = stim[iT - 1]*np.exp(-1/self.I_tau) + \
                (D/2*self.I_tau*(1 - np.exp(-2/self.I_tau)))**0.5*rands[iT]
        if smooth_T is not None:
            stim = gaussian_filter(stim, sigma=smooth_T)
            
        self.stim = stim + self.I_mu
        self.thresh_stim = gaussian_filter(self.stim, sigma=1/self.k_rate)
    
    def gen_firing_rate(self):
        if self.NF_type == 'relu':
            self.rate = (self.thresh_stim - self.I_mu)*(self.thresh_stim > self.I_mu)
        elif self.NF_type == 'snic':
            self.rate = self.dose_resp(self.thresh_stim)
        elif self.NF_type == 'quad':
            self.rate = self.dose_resp_approx(self.thresh_stim)
        
        self.rate += np.random.normal(0, self.rate_sigma, self.N)

    def get_opt_corr_lag(self, no_lag=False):
        if no_lag == False:
            corr = correlate(self.rate, self.stim, mode='full')
            opt_lag = np.argmax(corr[len(corr)//2:])
            self.lagged_stim = np.roll(self.stim, opt_lag)
        else:
            self.lagged_stim = np.roll(self.stim, 0)
    
    def calc_MI(self, stim_lims=[0, 10], num_stim_bins=200, rate_lims=[0, 200],
                num_rate_bins=300, split_stim=None, cutout=0):
        stim_bins = np.linspace(stim_lims[0], stim_lims[1], num_stim_bins)
        rate_bins = np.linspace(rate_lims[0], rate_lims[1], num_rate_bins)

        rate = self.rate/np.max(self.rate)
        stim = self.lagged_stim
        rate = rate[int(cutout):]
        stim = stim[int(cutout):]
        
        if split_stim is None:
            pass
        elif split_stim == 'neg':
            rate = rate[stim < self.I_mu]
            stim = stim[stim < self.I_mu]
        elif split_stim == 'pos':
            rate = rate[stim > self.I_mu]
            stim = stim[stim > self.I_mu]
        
        self.MI = calc_MI_xy(stim, stim_bins, rate, rate_bins)
        
        
class NL_model():
    
    def __init__(self, N=1000000, odor_tau=200, odor_mu=4.53, 
                 odor_sigma=0.3, k_rate=0.1, rate_sigma=0.5, NF_type='relu', 
                 SNIC_response=None):
        self.N = N
        self.NF_type = NF_type

        self.I_mu = odor_mu
        self.I_sigma = odor_sigma
        self.I_tau = odor_tau
        
        self.k_rate = k_rate
        self.rate_sigma = rate_sigma
        
        if SNIC_response != None:
            self.SNIC_response = np.load(SNIC_response)
        
    def dose_resp(self, cur):
        return np.interp(cur, self.SNIC_response[0], self.SNIC_response[1]/100)
    
    def dose_resp_approx(self, cur):
        return np.nan_to_num(10/(1*np.pi/(np.sqrt(0.3*(cur-self.I_mu))) + 4.6))
    
    def gen_sig_trace_OU(self, seed=0, smooth_T=None):
        rng = np.random.default_rng(seed)
        rands = rng.normal(0, 1, self.N)
        stim = np.zeros(self.N)
        D = self.I_sigma**2*2/self.I_tau
        for iT in range(1, self.N):
            stim[iT] = stim[iT - 1]*np.exp(-1/self.I_tau) + \
              (D/2*self.I_tau*(1 - np.exp(-2/self.I_tau)))**0.5*rands[iT]
        if smooth_T is not None:
            stim = gaussian_filter(stim, sigma=smooth_T)
            
        self.stim = stim + self.I_mu
        
    def gen_firing_rate(self):
        if self.NF_type == 'relu':
            self.thresh_stim = (self.stim - self.I_mu)*(self.stim > self.I_mu)
        elif self.NF_type == 'snic':
            self.thresh_stim = self.dose_resp(self.stim)
        elif self.NF_type == 'quad':
            self.thresh_stim = self.dose_resp_approx(self.stim)
            
        self.rate = gaussian_filter(self.thresh_stim, sigma=1/self.k_rate)
        self.rate += np.random.normal(0, self.rate_sigma, self.N)

    def get_opt_corr_lag(self, no_lag=False):
        if no_lag == False:
            corr = correlate(self.rate, self.stim, mode='full')
            opt_lag = np.argmax(corr[len(corr)//2:])
            self.lagged_stim = np.roll(self.stim, opt_lag)
        else:
            self.lagged_stim = np.roll(self.stim, 0)
    
    def calc_MI(self, stim_lims=[0, 10], num_stim_bins=200, rate_lims=[0, 200],
                num_rate_bins=300, split_stim=None, cutout=0):
        stim_bins = np.linspace(stim_lims[0], stim_lims[1], num_stim_bins)
        rate_bins = np.linspace(rate_lims[0], rate_lims[1], num_rate_bins)

        rate = self.rate/np.max(self.rate)
        stim = self.lagged_stim
        rate = rate[int(cutout):]
        stim = stim[int(cutout):]
        
        if split_stim is None:
            pass
        elif split_stim == 'neg':
            rate = rate[stim < self.I_mu]
            stim = stim[stim < self.I_mu]
        elif split_stim == 'pos':
            rate = rate[stim > self.I_mu]
            stim = stim[stim > self.I_mu]
        
        self.MI = calc_MI_xy(stim, stim_bins, rate, rate_bins)
        

class Hopf_Subcritical_model():

    def __init__(self, dt=0.1, N=1000000, I_tau=200, I_mu=4.53,
                 I_sigma=0.3, k_rate=1./200, T_last_k_rate=1/100,
                 rate_sigma=0.5):
        self.N = N
        self.dt = dt
        self.Tt = np.linspace(0, self.N*self.dt, self.N)
        
        self.EK = -84
        self.EL = -60
        self.ECa = 120
        self.gK = 8
        self.gL = 2
        self.gCa = 4
        self.C = 20
        self.v1 = -1.2
        self.v2 = 18
        self.v3 = 2
        self.v4 = 30
        self.phi = 0.04
        
        self.V0 = -60
        self.w0 = 0.1

        self.I_mu = I_mu
        self.I_sigma = I_sigma
        self.I_tau = I_tau
        
        self.k_rate = k_rate
        self.T_last_k_rate = T_last_k_rate
        self.rate_sigma = rate_sigma
        
    def df(self, x, iT):
        V, w = x
        
        m_inf = 0.5*(1 + np.tanh((V - self.v1)/self.v2))
        w_inf = 0.5*(1 + np.tanh((V - self.v3)/self.v4))
        lam_w = self.phi*np.cosh((V - self.v3)/(2*self.v4))

        dV = (self.stim[iT] + self.gL*(self.EL - V) + self.gK*w*(self.EK - V) +\
        self.gCa*m_inf*(self.ECa - V))/self.C

        dw = lam_w*(w_inf-w)
        
        return np.array([dV, dw])

    def gen_sig_trace_OU(self, seed=0, smooth_T=None):
        np.random.seed(seed)
        rands = np.random.normal(0, 1, self.N)
        stim = np.zeros(self.N)
        D = self.I_sigma**2*2/self.I_tau
        for iT in range(1, self.N):
            stim[iT] = stim[iT - 1]*np.exp(-1/self.I_tau*self.dt) + \
                (D/2*self.I_tau*(1 - np.exp(-2/self.I_tau*self.dt)))**0.5*rands[iT]
        if smooth_T is not None:
            stim = gaussian_filter(stim, sigma=int(smooth_T/self.dt))
        self.stim = stim + self.I_mu
        
    def gen_sig_trace_Gaussian(self, seed=0):
        np.random.seed(seed)
        I_vec = np.random.normal(self.I_mu, self.I_sigma,
                                 int(self.N/self.I_tau*self.dt))
        I_vec_Tt = np.linspace(0, self.N, len(I_vec))
        self.stim = np.interp(range(self.N), I_vec_Tt, I_vec)
        
    def calc_rate(self, filt_len=30000, filt_type='Gaussian', offset=None):
        V = self.x[:, 0]
        peak_idxs = find_peaks(V, 0)[0]
        
        bin_arr = np.zeros(self.N)
        bin_arr[peak_idxs] = 1

        if filt_type == 'exp':
            Tt = np.linspace(0, self.dt*filt_len, filt_len)
            filt = self.dt*self.k_rate*np.exp(-1.*Tt*self.k_rate)
            self.rate = convolve(filt, bin_arr, mode='full')[:self.N]
        elif filt_type == 'Gaussian':
            self.rate = gaussian_filter(bin_arr, sigma=1/self.k_rate/self.dt)
        elif filt_type == 'delayed_exp':
            Tt = np.linspace(0, self.dt*filt_len, filt_len)
            filt = self.dt*self.k_rate**3/2*(Tt**2.0*np.exp(-Tt*self.k_rate))
            if offset is None:
                offset = 2/self.k_rate
            self.rate = convolve(filt, bin_arr, mode='full')[:self.N]
            self.rate = np.roll(self.rate, -int(offset/self.dt))
        elif filt_type == 'Cauchy':
            Tt = np.linspace(0, self.dt*filt_len, filt_len)
            if offset is None:
                offset = 2/self.k_rate
            filt = self.dt*self.k_rate/np.pi*(1 + ((Tt - offset)*self.k_rate)**2.)**-1
            self.rate = convolve(filt, bin_arr, mode='full')[:self.N]
            self.rate = np.roll(self.rate, -int(offset/self.dt))
        
        self.rate *= 1000/self.dt
        self.rate += np.random.normal(0, self.rate_sigma, self.N)
        
    def calc_T_to_last_hit(self):
        V = self.x[:, 0]
        peak_idxs = find_peaks(V, 0)[0]
        
        self.T_to_last_hit = np.zeros(self.N)
        self.T_to_last_hit[:peak_idxs[0]] = range(peak_idxs[0])
        for iD in range(len(peak_idxs) - 1):
            beg = peak_idxs[iD]
            end = peak_idxs[iD + 1]
            self.T_to_last_hit[beg:end] = range(end - beg)
        self.T_to_last_hit *= self.dt
        self.T_to_last_hit_smth = gaussian_filter(self.T_to_last_hit,
                                    sigma=1/self.T_last_k_rate/self.dt)
        
    def integrate(self):
        x = np.zeros((self.N, 2))
        x[0] = [self.V0,self.w0]
        for iT in range(self.N - 1):
            x[iT + 1] = x[iT] + self.dt*self.df(x[iT], iT)
        self.x = x

    def calc_MI(self, num_stim_bins=800, num_rate_bins=400, rate_bounds=[0, 25.], 
                split_stim=None, cutout=0):
        stim_bins = np.linspace((np.mean(self.stim)-3*np.std(self.stim)), 
                                (np.mean(self.stim)+3*np.std(self.stim)), 
                                num_stim_bins)
        rate_bins = np.linspace(rate_bounds[0], rate_bounds[1], num_rate_bins)

        rate = self.rate
        stim = self.stim
        rate = rate[int(cutout/self.dt):]
        stim = stim[int(cutout/self.dt):]
        
        if split_stim is None:
            pass
        elif split_stim == 'neg':
            rate = rate[stim < self.I_mu]
            stim = stim[stim < self.I_mu]
        elif split_stim == 'pos':
            rate = rate[stim > self.I_mu]
            stim = stim[stim > self.I_mu]
        
        self.MI = calc_MI_xy(stim, stim_bins, rate, rate_bins)
        
    def calc_avg_dose_response(self, num_stim_bins=100):
        stim_bins = np.linspace(min(self.stim), max(self.stim), num_stim_bins)
        stim_binned = np.digitize(self.stim, bins=stim_bins, right=True) - 1
        rates_binned = [[] for i in range(len(stim_bins))]
        
        dose_response_avg = []
        dose_response_std = []
        for iS in range(len(self.stim)):
            rates_binned[stim_binned[iS]].append(self.rate[iS])
        for iS in range(num_stim_bins - 1):
            if len(rates_binned[iS]) > 100:
                dose_response_avg.append(np.mean(rates_binned[iS]))
                dose_response_std.append(np.std(rates_binned[iS]))
            else:
                dose_response_avg.append(np.nan)
                dose_response_std.append(np.nan)
        self.dose_response_bins = (stim_bins[1:] + stim_bins[:-1])/2.
        self.dose_response_avg = np.array(dose_response_avg)
        self.dose_response_std = np.array(dose_response_std)


        
        