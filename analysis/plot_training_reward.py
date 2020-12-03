import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plotting training curves.")
    parser.add_argument("--root", type=str, required=True,
                        help='dir to log_train_e2e, log_val, log_test files.')
    parser.add_argument("--experiment-name", type=str, default=None,
                        help='experiment name(used as figure name).')
    parser.add_argument("--output-dir", type=str, default=None,
                        help='output path for the image if specified.')
    return parser.parse_args()


args = parse_args()

log_test = pd.read_csv(os.path.join(args.root,  'log_test'), sep='\t')
log_val = pd.read_csv(os.path.join(args.root,  'log_val'), sep='\t')
log_train = pd.read_csv(os.path.join(args.root, 'log_train_e2e'), sep='\t')


fig, axes = plt.subplots(3, 1, figsize=(20, 18))
axes[0].plot(log_train['epoch'], log_train['rewards_mean'], label='Train')
axes[0].set_title('Train')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Reward')
axes[0].legend(loc='lower right')

axes[1].plot(log_val['epoch'], log_val['rewards_mean'], label='Val')
axes[1].set_title('Validation')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Reward')
axes[1].legend(loc='lower right')

axes[2].plot(log_test['epoch'], log_test['rewards_mean'], label='Test')
axes[1].set_title('Test')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Reward')
axes[2].legend(loc='lower right')

fig.tight_layout()

if args.output_dir is None:
    plt.show()
elif args.output_dir is not None and args.experiment_name is None:
    plt.savefig(os.path.join(args.output_dir, 'training_curve.jpg'))
elif args.output_dir is not None and args.experiment_name is not None:
    plt.savefig(os.path.join(
        args.output_dir, '{}_training_curve.jpg'.format(args.experiment_name)))
