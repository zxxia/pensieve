import argparse
import os

import env
from utils.utils import adjust_traces, load_traces
import numpy as np

# bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and
# time), chunk_til_video_end
S_INFO = 6
S_LEN = 8  # take how many frames in the past
A_DIM = 6
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000
RESEVOIR = 5  # BB
CUSHION = 10  # BB
SUMMARY_DIR = '../results/tmp'
# LOG_FILE = './results/log_sim_bb'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size
# download_time reward

TEST_TRACES = '../data/test'

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pensieve testing script.")
    parser.add_argument("--test_trace_dir", type=str,
                        required=True, help='dir to all test traces.')
    parser.add_argument("--summary_dir", type=str,
                        required=True, help='output path.')
    parser.add_argument("--noise", type=float, default=0,)
    parser.add_argument("--duration", type=float, default=1.0)

    return parser.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.summary_dir, exist_ok=True)
    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_traces(
        args.test_trace_dir)
    all_cooked_time, all_cooked_bw = adjust_traces(
        all_cooked_time, all_cooked_bw, bw_noise=args.noise,
        duration_factor=args.duration)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw, fixed=True)

    log_path = os.path.join(args.summary_dir, 'log_sim_bb_' +
                            all_file_names[net_env.trace_idx])
    log_file = open(log_path, 'w', 1)

    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    r_batch = []

    video_count = 0

    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
            net_env.get_video_chunk(bit_rate)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
            - REBUF_PENALTY * rebuf \
            - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                      VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
        r_batch.append(reward)

        last_bit_rate = bit_rate

        # log time_stamp, bit_rate, buffer_size, reward
        log_file.write(str(time_stamp / M_IN_K) + '\t' +
                       str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                       str(buffer_size) + '\t' +
                       str(rebuf) + '\t' +
                       str(video_chunk_size) + '\t' +
                       str(delay) + '\t' +
                       str(reward) + '\n')

        if buffer_size < RESEVOIR:
            bit_rate = 0
        elif buffer_size >= RESEVOIR + CUSHION:
            bit_rate = A_DIM - 1
        else:
            bit_rate = (A_DIM - 1) * (buffer_size - RESEVOIR) / float(CUSHION)

        bit_rate = int(bit_rate)

        if end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here
            r_batch = []

            print("video count", video_count)
            video_count += 1

            if video_count > len(all_file_names):
                break

            log_path = os.path.join(
                args.summary_dir,
                'log_sim_bb_' + all_file_names[net_env.trace_idx])
            log_file = open(log_path, 'w', 1)


if __name__ == '__main__':
    main()
