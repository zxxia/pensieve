import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.utils import compute_cdf

# ROOT = '../results/noise_exp'
# ROOT = '../results/bug_fix'
# ROOT = '../results/entropy_weight_exp'
ROOT = '../results/exponential_traces'

NOISES = [0, ]#1, 2, 3,]

NAMES = ['timestamp', 'bit rate', 'buffer size',
         'rebuf', 'video chunk size', 'delay', 'reward']

ls_list = ['-', '--', ':']
SCHEMES = ['bb', 'mpc', 'sim_rl_pretrain', 'sim_rl_train_noise0']  # , 'sim_rl_train_noise1', 'sim_rl_train_noise2', 'sim_rl_train_noise3', ]


# fig, axes = plt.subplots(2, 3, figsize=(12, 8))
# for noise, ax in zip(NOISES, axes.flatten()):
#     rewards = {}
#     for scheme in SCHEMES:
#         rewards[scheme] = []
#         log_files = sorted(glob.glob(os.path.join(
#             ROOT, f'noise_{noise}', scheme, 'log*')))
#         for i, log_file in enumerate(log_files):
#             log = pd.read_csv(log_file, sep='\t', names=NAMES)
#             rewards[scheme].append(np.sum(log['reward'][1:]))
#
#     for i, scheme in enumerate(SCHEMES):
#         sorted_reward, cdf = compute_cdf(rewards[scheme])
#         ax.plot(sorted_reward, cdf, label=scheme)
#     ax.set_title(f"Test on +{noise} in throughput")
#     ax.set_ylabel('CDF')
#     ax.set_xlabel('QoE')
#     ax.set_ylim(0, 1.0)
#     ax.legend()
# fig.tight_layout()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for plot_idx, (noise, ax) in enumerate(zip(NOISES, axes.flatten())):
    rewards = {}
    for scheme in SCHEMES:
        rewards[scheme] = []
        log_files = sorted(glob.glob(os.path.join(
            ROOT, f'noise_{noise}', scheme, 'log*')))
        for i, log_file in enumerate(log_files):
            log = pd.read_csv(log_file, sep='\t', names=NAMES)
            rewards[scheme].append(np.sum(log['reward'][1:]))

    mean_rewards = [np.mean(rewards[scheme]) for scheme in SCHEMES]
    mean_rewards_err = [np.std(rewards[scheme])/np.sqrt(len(rewards[scheme])) for scheme in SCHEMES]
    bars = ax.bar(np.arange(len(SCHEMES)), mean_rewards)
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(round(height)),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    ax.errorbar(np.arange(len(SCHEMES)), mean_rewards, xerr=0,
                yerr=mean_rewards_err, fmt='o', color='Black', elinewidth=1,
                capthick=1, errorevery=1, alpha=1, ms=2, capsize=3)
    ax.set_title(f"Test on +{noise}noise in throughput of test set")
    ax.set_ylabel('Mean Reward')
    ax.set_xlabel('Schemes')
    ax.set_xticks(np.arange(len(SCHEMES)))
    ax.set_xticklabels(SCHEMES, rotation=90)
    # bars[plot_idx+3].set_color('#ff7f0e')
    # if noise == 0:
    #     ax.set_ylim(0, )
    ax.legend()
fig.tight_layout()


fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for plot_idx, (noise, ax) in enumerate(zip(NOISES, axes.flatten())):
    rewards = {}
    for scheme in SCHEMES:
        rewards[scheme] = []
        log_files = sorted(glob.glob(os.path.join(
            ROOT, f'noise_{noise}_train', scheme, 'log*')))
        for i, log_file in enumerate(log_files):
            log = pd.read_csv(log_file, sep='\t', names=NAMES)
            rewards[scheme].append(np.sum(log['reward'][1:]))

    mean_rewards = [np.mean(rewards[scheme]) for scheme in SCHEMES]
    mean_rewards_err = [np.std(rewards[scheme])/np.sqrt(len(rewards[scheme])) for scheme in SCHEMES]
    bars = ax.bar(np.arange(len(SCHEMES)), mean_rewards)
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(round(height)),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    ax.errorbar(np.arange(len(SCHEMES)), mean_rewards, xerr=0,
                yerr=mean_rewards_err, fmt='o', color='Black', elinewidth=1,
                capthick=1, errorevery=1, alpha=1, ms=2, capsize=3)
    ax.set_title(f"Test on +{noise}noise in throughput of train set")
    ax.set_ylabel('Mean Reward')
    ax.set_xlabel('Schemes')
    ax.set_xticks(np.arange(len(SCHEMES)))
    ax.set_xticklabels(SCHEMES, rotation=90)
    # bars[plot_idx+3].set_color('#ff7f0e')
    # if noise == 0:
    #     ax.set_ylim(0, )
    ax.legend()
fig.tight_layout()
# SCHEMES = ['bb', 'mpc', 'sim_rl_pretrain',  ]
# fig, axes = plt.subplots(2, 2, figsize=(12, 8))
# for plot_idx, (noise, ax) in enumerate(zip(NOISES, axes.flatten())):
#     rewards = {}
#     for scheme in SCHEMES:
#         rewards[scheme] = []
#         log_files = sorted(glob.glob(os.path.join(
#             ROOT, f'noise_{noise}_train', scheme, 'log*')))
#         for i, log_file in enumerate(log_files):
#             log = pd.read_csv(log_file, sep='\t', names=NAMES)
#             rewards[scheme].append(np.sum(log['reward'][1:]))
#
#     mean_rewards = [np.mean(rewards[scheme]) for scheme in SCHEMES]
#     mean_rewards_err = [np.std(rewards[scheme])/np.sqrt(len(rewards[scheme])) for scheme in SCHEMES]
#     bars = ax.bar(np.arange(len(SCHEMES)), mean_rewards)
#     for bar in bars:
#         height = bar.get_height()
#         ax.annotate('{}'.format(round(height)),
#                     xy=(bar.get_x() + bar.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')
#
#     ax.errorbar(np.arange(len(SCHEMES)), mean_rewards, xerr=0,
#                 yerr=mean_rewards_err, fmt='o', color='Black', elinewidth=1,
#                 capthick=1, errorevery=1, alpha=1, ms=2, capsize=3)
#     ax.set_title(f"Test on +{noise}noise in throughput")
#     ax.set_ylabel('Mean Reward')
#     ax.set_xlabel('Schemes')
#     ax.set_xticks(np.arange(len(SCHEMES)))
#     ax.set_xticklabels(SCHEMES, rotation=90)
#     bars[plot_idx+3].set_color('#ff7f0e')
#     # if noise == 0:
#     #     ax.set_ylim(0, )
#     ax.legend()
# fig.tight_layout()
# fig.savefig("./scheme_perf_vs_bw_noise_on_train_trace_bug_fix.pdf")
plt.show()
