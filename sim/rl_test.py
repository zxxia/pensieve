import argparse
import os

import a3c
import env
import numpy as np
import tensorflow as tf
from constants import (A_DIM, BUFFER_NORM_FACTOR, DEFAULT_QUALITY, M_IN_K,
                       RANDOM_SEED, S_INFO, S_LEN, VIDEO_BIT_RATE)

from utils.utils import adjust_traces, linear_reward, load_traces

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


os.environ['CUDA_VISIBLE_DEVICES'] = ''

tf.logging.set_verbosity(tf.logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pensieve testing script.")
    parser.add_argument("--test_trace_dir", type=str,
                        required=True, help='dir to all test traces.')
    parser.add_argument("--summary_dir", type=str,
                        required=True, help='output path.')
    parser.add_argument("--model_path", type=str, required=True,
                        help='model path')
    parser.add_argument("--noise", type=float, default=0,)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--env-random-start", action="store_true",
                        help='environment will not randomly start a new trace'
                        'in training stage if environment is not fixed if '
                        'specified.')

    return parser.parse_args()


def main():
    args = parse_args()
    summary_dir = args.summary_dir
    nn_model = args.model_path
    test_trace_dir = args.test_trace_dir
    os.makedirs(summary_dir, exist_ok=True)

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_traces(
        test_trace_dir)
    print("Loaded {} traces." .format(len(all_file_names)))
    all_cooked_time, all_cooked_bw = adjust_traces(
        all_cooked_time, all_cooked_bw, bw_noise=args.noise,
        duration_factor=args.duration)

    if args.env_random_start:
        net_env = env.Environment(all_cooked_time=all_cooked_time,
                                  all_cooked_bw=all_cooked_bw,
                                  all_file_names=all_file_names, fixed=True)
    else:
        net_env = env.EnvironmentNoRandomStart(
            all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw,
            all_file_names=all_file_names, fixed=True)

    log_path = os.path.join(summary_dir, 'log_sim_rl_' +
                            all_file_names[net_env.trace_idx])
    log_file = open(log_path, 'w')

    with tf.Session() as sess:

        actor = a3c.ActorNetwork(sess, state_dim=[S_INFO, S_LEN],
                                 action_dim=A_DIM)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        if nn_model is not None:  # NN_MODEL is the path to file
            saver.restore(sess, nn_model)
            print("Testing model restored from {}.".format(nn_model))

        time_stamp = 0

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

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
            # print(time_stamp, net_env.trace_time[-1], video_chunk_remain)

            # reward is video quality - rebuffer penalty - smoothness
            reward = linear_reward(bit_rate, last_bit_rate, rebuf)

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
            log_file.flush()

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / \
                float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / \
                float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :A_DIM] = np.array(
                next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            # state[5, -1] = np.minimum(
            #     video_chunk_remain,
            #     CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
            state[5, -1] = video_chunk_remain / net_env.total_video_chunk

            action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
            # action_cumsum = np.cumsum(action_prob)
            # bit_rate = (action_cumsum > np.random.randint(
            #     1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            bit_rate = action_prob.argmax()
            # Note: we need to discretize the probability into 1/RAND_RANGE
            # steps, because there is an intrinsic discrepancy in passing
            # single state and batch states

            s_batch.append(state)

            entropy_record.append(a3c.compute_entropy(action_prob[0]))

            if end_of_video:
                log_file.write('\n')
                log_file.close()

                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here

                s_batch = []
                a_batch = []
                r_batch = []

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)
                entropy_record = []

                video_count += 1
                print("Processed {} traces".format(video_count))

                if video_count >= len(all_file_names):
                    break

                log_path = os.path.join(
                    summary_dir,
                    'log_sim_rl_{}'.format(all_file_names[net_env.trace_idx]))
                log_file = open(log_path, 'w')


if __name__ == '__main__':
    main()
