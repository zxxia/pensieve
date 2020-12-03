import os

import matplotlib.pyplot as plt
import pandas as pd


def compute_cdf(data):
    length = len(data)
    sorted_data = sorted(data)
    cdf = [i / length for i, val in enumerate(sorted_data)]
    return sorted_data, cdf


def training_plot(root):
    log_test_filename = os.path.join(root, 'results_noise_0', 'log_test')
    if not os.path.exists(log_test_filename):
        log_test_filename = os.path.join(root, 'results_noise0', 'log_test')
    log_val_filename = os.path.join(root, 'results_noise_0', 'log_val')
    if not os.path.exists(log_val_filename):
        log_val_filename = os.path.join(root, 'results_noise0', 'log_val')

    log_train_filename = os.path.join(root, 'results_noise_0', 'log_train_e2e')
    if not os.path.exists(log_train_filename):
        log_train_filename = os.path.join(root, 'results_noise0',
                                          'log_train_e2e')
    log_test = pd.read_csv(log_test_filename, sep='\t')
    log_val = pd.read_csv(log_val_filename, sep='\t')
    log_train_e2e = pd.read_csv(log_train_filename, sep='\t')

    fig, axes = plt.subplots(3, 1, figsize=(15, 11))
    axes[0].plot(log_train_e2e['epoch'],
                 log_train_e2e['rewards_mean'], label='Train')
    axes[0].set_title('trained with original data')
    axes[0].set_xlabel('iter')
    axes[0].set_ylabel('reward')
    # axes[0].set_ylim(0, )
    axes[0].legend(loc='lower right')
    axes[1].plot(log_val['epoch'], log_val['rewards_mean'], label='Val')
    axes[1].set_xlabel('iter')
    axes[1].set_ylabel('reward')
    axes[1].legend(loc='lower right')
    # axes[1].set_ylim(0, )
    axes[2].plot(log_test['epoch'], log_test['rewards_mean'], label='Test')
    axes[2].set_xlabel('iter')
    axes[2].set_ylabel('reward')
    axes[2].legend(loc='lower right')
    fig.tight_layout()
