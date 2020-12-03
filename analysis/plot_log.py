import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAMES = ['timestamp', 'bit rate', 'buffer size',
         'rebuf', 'video chunk size', 'delay', 'reward']

MPC_NAMES = ['timestamp', 'bit rate', 'buffer size', 'rebuf',
             'video chunk size', 'delay', 'reward', 'future_bandwidth']

SCHEMES = ['RobustMPC',  'RL']
PREFIXS = ['sim_mpc',  'sim_rl']
COLORS = ['C0', 'C1']


def parse_args():
    parser = argparse.ArgumentParser(description="Plotting log details.")
    parser.add_argument("--trace-dir", type=str, required=True,
                        help='dir to all test traces.')
    parser.add_argument("--mpc-log-dir", type=str, default=None,
                        help='dir to all MPC log files.')
    parser.add_argument("--pensieve-log-dir", type=str, default=None,
                        help='dir to all pensieve log files')
    parser.add_argument("--output-dir", type=str, default=None,
                        help='output path.')

    return parser.parse_args()


args = parse_args()

log_dirs = [args.mpc_log_dir, args.pensieve_log_dir]
trace_files = sorted(glob.glob(os.path.join(args.trace_dir, '*')))

for i, trace_file in enumerate(trace_files):
    rewards = {}
    bitrate = {}
    ts = {}
    buf = {}
    rebuf = {}
    future_bandwidth = {}
    trace_name = os.path.basename(trace_file)
    print(i, trace_name)
    for scheme, prefix, log_dir in zip(SCHEMES, PREFIXS, log_dirs):
        if log_dir is None:
            continue
        log_file = os.path.join(log_dir, 'log_'+prefix+'_'+trace_name)
        # print(log_file)
        if scheme == 'RobustMPC':
            log = pd.read_csv(log_file, sep='\t', names=MPC_NAMES)
            future_bandwidth[scheme] = log['future_bandwidth']
        else:
            log = pd.read_csv(log_file, sep='\t', names=NAMES)
        rewards[scheme] = log['reward']
        bitrate[scheme] = log['bit rate']
        ts[scheme] = log['timestamp']
        buf[scheme] = log['buffer size']
        rebuf[scheme] = log['rebuf']

    trace = pd.read_csv(trace_file, sep='\t', header=None)
    fig, axes = plt.subplots(5, 1, figsize=(24, 12))
    for ax_idx, ax in enumerate(axes):
        for scheme, color in zip(SCHEMES, COLORS):
            if scheme not in rewards:
                continue
            if ax_idx == 0:
                avg_reward = np.sum(rewards[scheme][1:])
                ax.plot(ts[scheme]-ts[scheme][0], rewards[scheme], 'o-',
                        c=color,
                        label=scheme+" Reward: {:.2f}".format(avg_reward))
                ax.set_ylabel('chunk reward')
            elif ax_idx == 1:
                ax.plot(ts[scheme]-ts[scheme][0],
                        bitrate[scheme], 'o-', c=color, label=scheme)
                ax.set_ylabel('bitrate(Kbps)')
            elif ax_idx == 2:
                ax.plot(ts[scheme]-ts[scheme][0],
                        buf[scheme], 'o-', c=color, label=scheme)
                ax.set_ylabel('buffer length(s)')
            elif ax_idx == 3:
                ax.plot(ts[scheme]-ts[scheme][0],
                        rebuf[scheme], 'o-', c=color, label=scheme)
                ax.set_ylabel('rebuffer(s)')
            elif ax_idx == 4 and scheme == 'RobustMPC':
                ax.plot(trace[0], trace[1]*1000, 'o-', c='r', label='trace')
                ax.plot(ts[scheme]-ts[scheme][0],
                        future_bandwidth[scheme]*1000*8, 'o-', c=color,
                        label='mpc predict')
                ax.set_ylabel('bw(Kbps)')
        # ax.set_xlim(0, 210)
        ax.set_xlabel('timestamp(second)')
        ax.legend()
    plt.title(trace_name)
    fig.tight_layout()
    if args.output_dir is None:
        plt.show()
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        plt.savefig(os.path.join(args.output_dir, 'trace_{}.jpg'.format(i)))
    plt.close()
