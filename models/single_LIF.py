from models import *
import matplotlib.pyplot as plt
import numpy as np
import os, sys


# # 初始化模型

model = SimplifiedLIFModel(N=10000, odor_mu=0, odor_sigma=0)

# # 1. 生成 OU 输入信号
model.gen_sig_trace_OU(seed=0)
model.stim = np.ones(10000)*14   # 恒定电流输入
# # 2. 模拟神经元响应
model.integrate()

# # 3. 计算发放率
model.calc_rate()

# # 4. 可选：计算互信息
# model.calc_MI()


plt.scatter(x=14,y=model.rate[10000//3])
plt.ylabel("Firing Rate (Hz)")
# # plt.title("LIF Model Dose-Response")
# # plt.savefig('dose_response.png')
plt.show()

# # 5. 画图（如下）
# plt.figure(figsize=(12,5))
# plt.subplot(2,1,1)
# plt.plot(model.Tt, model.stim, label='OU Stimulus')
# plt.ylabel("Stimulus (pA)")
# plt.legend()

# plt.subplot(2,1,2)
# plt.plot(model.Tt, model.rate, label='Firing Rate')
# plt.ylabel("Rate (Hz)")
# plt.xlabel("Time (ms)")
# plt.legend()
# plt.tight_layout()
# # plt.savefig('model_response.png')
# plt.show()

# sys.path.append(os.path.abspath('../utils'))
# from paper_utils import gen_plot

# sigmas = np.logspace(0, 1.5, 10)  # 从小到大变化
# avgs = []
# bins = []

# for sigma in sigmas:
#     print("Simulating for sigma =", sigma)
#     model = SimplifiedLIFModel(
#         dt=0.05, N=5_000_000, odor_mu=4.54, odor_sigma=sigma, k_rate=1/50, rate_sigma=0.2
#     )
#     model.gen_sig_trace_OU(seed=0, smooth_T=10)
#     model.integrate()
#     model.calc_rate()
#     model.calc_avg_dose_response(num_stim_bins=100)
    
#     avgs.append(model.dose_response_avg)
#     bins.append(model.dose_response_bins)

# # 画图

# colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(sigmas)))
# fig = gen_plot(1.3, 1.1)

# for i, sigma in enumerate(sigmas):
#     plt.plot(bins[i] - 4.54, avgs[i], color=colors[i], lw=0.7, label=f"{sigma:.2f}")

# plt.axvline(0, color='k', lw=1, ls='--')
# plt.xlim(-2, 2)
# plt.ylim(0, 80)
# plt.xlabel("Input (centered at μ=4.54)")
# plt.ylabel("Firing rate (Hz)")
# plt.legend(fontsize=6)
# plt.tight_layout()
# plt.show()
