from warnings import simplefilter
from utils.utils import (adjust_traces, load_traces, linear_reward)
from tensorflow.python.util import deprecation
import csv
import logging
import multiprocessing as mp
import os
import time
from datetime import datetime

import a3c
from constants import (VIDEO_BIT_RATE, DEFAULT_QUALITY, M_IN_K,
                       MODEL_SAVE_INTERVAL)
import env
import numpy as np
import src.config as config
import tensorflow as tf
from typing import List
from numpy.random import RandomState

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


deprecation._PRINT_DEPRECATION_WARNINGS = False

simplefilter(action='ignore', category=FutureWarning)


tf.logging.set_verbosity(tf.logging.INFO)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def entropy_weight_decay_func(epoch):
    # linear decay
    return np.maximum(-0.05/(10**4) * epoch + 0.5, 0.1)
    # return 0.5


# def entropy_weight_decay_func(epoch):
#     # linear decay
#     return np.maximum((1-0.1)/(10**5) * epoch + 1, 0.1)


def learning_rate_decay_func(epoch):
    # linear decay
    return 0.0001
    # if epoch < 30000:
    #     return 0.0001
    # elif epoch < 60000:
    #     return 0.00005
    # else:
    #     return 0.00001


def test(args, test_traces_dir, actor, log_output_dir, noise, duration,
         buffer_thresh):
    np.random.seed(args.RANDOM_SEED)
    assert len(VIDEO_BIT_RATE) == args.A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_traces(
        test_traces_dir)
    # handle the noise and duration variation here
    all_cooked_time, all_cooked_bw = adjust_traces(
        all_cooked_time, all_cooked_bw, bw_noise=noise,
        duration_factor=duration)

    for trace_time, trace_bw, trace_file_name in zip(
            all_cooked_time, all_cooked_bw, all_file_names):

        net_env = env.NetworkEnvironment(
            trace_time=trace_time, trace_bw=trace_bw,
            trace_video_same_duration_flag=True,
            trace_file_name=trace_file_name, fixed=True,
            video_size_file_dir='../data/video_sizes',
            buffer_thresh=buffer_thresh * 1000)

        log_path = os.path.join(log_output_dir, 'log_sim_rl_{}'.format(
                                trace_file_name))
        log_file = open(log_path, 'w', 1)

        time_stamp = 0

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros(args.A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((args.S_INFO, args.S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, rebuf, \
                video_chunk_size, next_video_chunk_sizes, \
                end_of_video, video_chunk_remain = \
                net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # reward is video quality - rebuffer penalty - smoothness
            reward = linear_reward(bit_rate, last_bit_rate, rebuf)
            # reward = opposite_linear_reward(bit_rate, last_bit_rate, rebuf)

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

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((args.S_INFO, args.S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be args.S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / \
                float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / args.BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / \
                float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / \
                args.BUFFER_NORM_FACTOR  # 10 sec
            state[4, :args.A_DIM] = np.array(next_video_chunk_sizes) / \
                M_IN_K / M_IN_K  # mega byte
            state[5, -1] = video_chunk_remain / net_env.total_video_chunk

            action_prob = actor.predict(np.reshape(
                state, (1, args.S_INFO, args.S_LEN)))
            # action_cumsum = np.cumsum(action_prob)
            # bit_rate = (action_cumsum > np.random.randint(
            #     1, args.RAND_RANGE) / float(args.RAND_RANGE)).argmax()
            # TODO: Zhengxu: Why compute bitrate this way?
            bit_rate = action_prob.argmax()
            # Note: we need to discretize the probability into
            # 1/args.RAND_RANGE steps, because there is an intrinsic
            # discrepancy in passing single state and batch states

            s_batch.append(state)

            entropy_record.append(a3c.compute_entropy(action_prob[0]))

            if end_of_video:
                break


def testing(args, epoch, actor, log_file, trace_dir, test_log_folder, noise,
            duration, buffer_thresh):
    # clean up the test results folder
    os.system('rm -r ' + test_log_folder)
    os.makedirs(test_log_folder, exist_ok=True)

    # run test script
    print('test on buffer thresh {}'.format(buffer_thresh))
    # test(args, trace_dir, actor, test_log_folder, noise, duration,
    #      buffer_thresh)

    cmd = "python ../sim/rl_test_opt.py --test_trace_dir {} --summary_dir {} "\
           "--model_path {} --video-size-file-dir {} --buffer-thresh {}".format(
            trace_dir, test_log_folder, actor, '../data/video_sizes', buffer_thresh)
    os.system(cmd)

    # append test performance to the log
    rewards = []
    test_log_files = os.listdir(test_log_folder)
    for test_log_file in test_log_files:
        reward = []
        with open(os.path.join(test_log_folder, test_log_file), 'r') as f:
            for line in f:
                parse = line.split()
                try:
                    reward.append(float(parse[-1]))
                except IndexError:
                    break
        rewards.append(np.sum(reward[1:]))

    rewards = np.array(rewards)

    rewards_min = np.min(rewards)
    rewards_5per = np.percentile(rewards, 5)
    rewards_mean = np.mean(rewards)
    rewards_median = np.percentile(rewards, 50)
    rewards_95per = np.percentile(rewards, 95)
    rewards_max = np.max(rewards)

    log_file.write(str(epoch) + '\t' +
                   str(rewards_min) + '\t' +
                   str(rewards_5per) + '\t' +
                   str(rewards_mean) + '\t' +
                   str(rewards_median) + '\t' +
                   str(rewards_95per) + '\t' +
                   str(rewards_max) + '\t' +
                   str(buffer_thresh) + '\n')
    log_file.flush()
    return rewards_mean


def testing_mpc(epoch, log_file, trace_dir, test_log_folder, buffer_thresh):
    # clean up the test results folder
    os.system('rm -r ' + test_log_folder)
    os.makedirs(test_log_folder, exist_ok=True)
    cmd = 'python ../sim/mpc_opt.py --test_trace_dir {} --video-size-file-dir {} ' \
          '--buffer-thresh {} --summary_dir {}'.format(
                trace_dir, '../data/video_sizes', buffer_thresh,
                test_log_folder)
    print(cmd)
    os.system(cmd)

    # append test performance to the log
    rewards = []
    test_log_files = os.listdir(test_log_folder)
    for test_log_file in test_log_files:
        reward = []
        with open(os.path.join(test_log_folder, test_log_file), 'r') as f:
            for line in f:
                parse = line.split()
                try:
                    reward.append(float(parse[-2]))
                except IndexError:
                    break
        rewards.append(np.sum(reward[1:]))

    rewards = np.array(rewards)

    rewards_min = np.min(rewards)
    rewards_5per = np.percentile(rewards, 5)
    rewards_mean = np.mean(rewards)
    rewards_median = np.percentile(rewards, 50)
    rewards_95per = np.percentile(rewards, 95)
    rewards_max = np.max(rewards)

    log_file.write(str(epoch) + '\t' +
                   str(rewards_min) + '\t' +
                   str(rewards_5per) + '\t' +
                   str(rewards_mean) + '\t' +
                   str(rewards_median) + '\t' +
                   str(rewards_95per) + '\t' +
                   str(rewards_max) + '\t' +
                   str(buffer_thresh) + '\n')
    log_file.flush()
    return rewards_mean


def central_agent(args, net_params_queues, exp_queues):
    # Visdom Logs

    prng = RandomState(42)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.INFO)

    assert len(net_params_queues) == args.NUM_AGENTS
    assert len(exp_queues) == args.NUM_AGENTS

    logging.basicConfig(filename=os.path.join(args.summary_dir, 'log_central'),
                        filemode='w', level=logging.INFO)

    with tf.Session() as sess, \
            open(os.path.join(args.summary_dir, 'log_test'),
                 'w', 1) as test_log_file, \
            open(os.path.join(args.summary_dir, 'log_train'),
                 'w', 1) as log_central_file, \
            open(os.path.join(args.summary_dir, 'log_val'),
                 'w', 1) as val_log_file, \
            open(os.path.join(args.summary_dir, 'log_train_e2e'),
                 'w', 1) as train_e2e_log_file,\
            open(os.path.join(args.summary_dir, 'mpc_log_val'),
                 'w', 1) as mpc_val_log_file:
        log_writer = csv.writer(log_central_file, delimiter='\t')
        log_writer.writerow(['epoch', 'loss', 'avg_reward', 'avg_entropy'])
        test_log_file.write("\t".join(
            ['epoch', 'rewards_min', 'rewards_5per', 'rewards_mean',
             'rewards_median', 'rewards_95per', 'rewards_max\n']))
        val_log_file.write("\t".join(
            ['epoch', 'rewards_min', 'rewards_5per', 'rewards_mean',
             'rewards_median', 'rewards_95per', 'rewards_max', 'buffer_thresh\n']))
        mpc_val_log_file.write("\t".join(
            ['epoch', 'rewards_min', 'rewards_5per', 'rewards_mean',
             'rewards_median', 'rewards_95per', 'rewards_max', 'buffer_thresh\n']))
        train_e2e_log_file.write("\t".join(
            ['epoch', 'rewards_min', 'rewards_5per', 'rewards_mean',
             'rewards_median', 'rewards_95per', 'rewards_max', 'buffer_thresh\n']))

        actor = a3c.ActorNetwork(sess,
                                 state_dim=[args.S_INFO, args.S_LEN],
                                 action_dim=args.A_DIM,)
        # learning_rate=args.ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[args.S_INFO, args.S_LEN],
                                   learning_rate=args.CRITIC_LR_RATE)

        logging.info('actor and critic initialized')
        summary_ops, summary_vars = a3c.build_summaries()

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(
            args.summary_dir, sess.graph)  # training monitor
        saver = tf.train.Saver(max_to_keep=20)  # save neural net parameters

        tmp_saver = tf.train.Saver(max_to_keep=1)  # save neural net parameters

        # restore neural net parameters
        if args.nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, args.nn_model)
            print("Model restored.")

        epoch = 0

        # assemble experiences from agents, compute the gradients
        max_avg_reward = None
        training_buffer_thresh_range = [60, 60]
        t_start = time.time()
        while True:
            # start_t = time.time()
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            critic_net_params = critic.get_network_params()
            for i in range(args.NUM_AGENTS):
                net_params_queues[i].put([actor_net_params, critic_net_params,
                                          training_buffer_thresh_range])
                # Note: this is synchronous version of the parallel training,
                # which is easier to understand and probe. The framework can be
                # fairly easily modified to support asynchronous training.
                # Some practices of asynchronous training (lock-free SGD at
                # its core) are nicely explained in the following two papers:
                # https://arxiv.org/abs/1602.01783
                # https://arxiv.org/abs/1106.5730

            # record average reward and td loss change
            # in the experiences from the agents
            total_batch_len = 0.0
            total_reward = 0.0
            total_td_loss = 0.0
            total_entropy = 0.0
            total_agents = 0.0

            # assemble experiences from the agents
            actor_gradient_batch = []
            critic_gradient_batch = []

            # linear entropy weight decay(paper sec4.4)
            entropy_weight = entropy_weight_decay_func(epoch)
            current_learning_rate = learning_rate_decay_func(epoch)

            for i in range(args.NUM_AGENTS):
                s_batch, a_batch, r_batch, terminal, info = exp_queues[i].get()
                assert len(s_batch) > 0, "{}: {}".format(i, s_batch)

                actor_gradient, critic_gradient, td_batch = \
                    a3c.compute_gradients(
                        s_batch=np.stack(s_batch, axis=0),
                        a_batch=np.vstack(a_batch),
                        r_batch=np.vstack(r_batch),
                        terminal=terminal, actor=actor, critic=critic,
                        entropy_weight=entropy_weight)

                actor_gradient_batch.append(actor_gradient)
                critic_gradient_batch.append(critic_gradient)

                total_reward += np.sum(r_batch)
                total_td_loss += np.sum(td_batch)
                total_batch_len += len(r_batch)
                total_agents += 1.0
                total_entropy += np.sum(info['entropy'])

            # compute aggregated gradient
            assert args.NUM_AGENTS == len(actor_gradient_batch)
            assert len(actor_gradient_batch) == len(critic_gradient_batch)
            for i in range(len(actor_gradient_batch)):
                actor.apply_gradients(
                    actor_gradient_batch[i], current_learning_rate)
                critic.apply_gradients(critic_gradient_batch[i])

            # log training information
            epoch += 1
            avg_reward = total_reward / total_agents
            avg_td_loss = total_td_loss / total_batch_len
            avg_entropy = total_entropy / total_batch_len

            logging.info('Epoch: ' + str(epoch) +
                         ' TD_loss: ' + str(avg_td_loss) +
                         ' Avg_reward: ' + str(avg_reward) +
                         ' Avg_entropy: ' + str(avg_entropy))
            log_writer.writerow([epoch, avg_td_loss, avg_reward, avg_entropy])

            summary_str = sess.run(summary_ops, feed_dict={
                summary_vars[0]: avg_td_loss,
                summary_vars[1]: avg_reward,
                summary_vars[2]: avg_entropy
            })

            writer.add_summary(summary_str, epoch)
            writer.flush()

            tmp_save_path = tmp_saver.save(sess, os.path.join(
                    args.summary_dir, "tmp_nn_model_ep.ckpt"))
            if epoch % MODEL_SAVE_INTERVAL == 0:
                t_used = time.time() - t_start
                print('epoch {} :{}s'.format(epoch, t_used))
                t_start = time.time()
                print('epoch', epoch - 1)
                small_ranges = np.linspace(4, 124, 7)
                all_val_rewards = []
                all_mpc_val_rewards = []
                for buffer_thresh in range(4, 124+1):
                    _ = testing(args, epoch, tmp_save_path, test_log_file,
                                args.test_trace_dir,
                                os.path.join(args.summary_dir, 'test_results'),
                                args.noise, args.duration, buffer_thresh)
                for lower_bound, upper_bound in zip(small_ranges[:-1],
                                                    small_ranges[1:]):
                    # pick a buffer thresh in the small range
                    # buffer_thresh = np.random.randint(lower_bound, upper_bound)
                    buffer_thresh = prng.randint(lower_bound, upper_bound)
                    # _ = testing(args, epoch, actor, train_e2e_log_file,
                    #             args.train_trace_dir,
                    #             os.path.join(args.summary_dir, 'test_results'),
                    #             args.noise, args.duration, buffer_thresh)
                    # print('done evaluating training set')
                    # _ = testing(args, epoch, actor, test_log_file,
                    #             args.test_trace_dir,
                    #             os.path.join(args.summary_dir, 'test_results'),
                    #             args.noise, args.duration, buffer_thresh)
                    print('done evaluating testing set')

                    # TODO: run MPC on the same set of traces
                    mpc_val_reward = testing_mpc(
                        epoch, mpc_val_log_file, args.val_trace_dir,
                        os.path.join(args.summary_dir, 'mpc_test_results'),
                        buffer_thresh)
                    all_mpc_val_rewards.append(mpc_val_reward)
                    test_mean_reward = testing(
                        args, epoch, tmp_save_path, val_log_file, args.val_trace_dir,
                        os.path.join(args.summary_dir, 'test_results'),
                        args.noise, args.duration, buffer_thresh)
                    all_val_rewards.append(test_mean_reward)
                    print('done evaluating val set')
                # update the buffer thresh range
                # all_mpc_val_rewards = [746.4478944073453, 969.4447477621553,
                #         968.6286708424519, 968.6286708424519, 968.6286708424519, 968.6286708424519]
                reward_diffs = np.array(all_mpc_val_rewards) - np.array(all_val_rewards)
                target_idx = np.argsort(reward_diffs)[-1]
                training_buffer_thresh_range = [small_ranges[target_idx],
                                                small_ranges[target_idx+1]]
                # training_buffer_thresh_range = [60, 60]
                print(all_mpc_val_rewards)
                print(all_val_rewards)
                # print(reward_diffs)
                print('train with', training_buffer_thresh_range)

                if max_avg_reward is None or (np.mean(all_val_rewards) >
                                              max_avg_reward):
                    max_avg_reward = np.mean(all_val_rewards)
                    # Save the neural net parameters to disk.
                    save_path = saver.save(
                        sess, os.path.join(
                            args.summary_dir,
                            "nn_model_ep_{}.ckpt".format(epoch)))
                    logging.info("Model saved in file: " + save_path)

            # end_t = time.time()
            # print(f'epoch{epoch-1}: {end_t - start_t}s')


def agent(args, agent_id, all_cooked_time, all_cooked_bw, all_file_names,
          net_params_queue, exp_queue):

    with tf.Session() as sess, open(os.path.join(
            args.summary_dir, 'log_agent_{}'.format(agent_id)),
            'w', 1) as log_file:

        if not args.no_agent_logging:
            log_file.write('\t'.join(['time_stamp', 'bit_rate', 'buffer_size',
                                      'rebuffer', 'video_chunk_size', 'delay',
                                      'reward', 'epoch', 'trace_idx',
                                      'mahimahi_ptr', 'buffer_thresh'])+'\n')
        actor = a3c.ActorNetwork(sess,
                                 state_dim=[args.S_INFO, args.S_LEN],
                                 action_dim=args.A_DIM,)
        # learning_rate=args.ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[args.S_INFO, args.S_LEN],
                                   learning_rate=args.CRITIC_LR_RATE)

        # initial synchronization of the network parameters from the
        # coordinator
        actor_net_params, critic_net_params, training_buffer_thresh_range = \
            net_params_queue.get()
        actor.set_network_params(actor_net_params)
        critic.set_network_params(critic_net_params)

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros(args.A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((args.S_INFO, args.S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        time_stamp = 0
        epoch = 0
        np.random.seed(agent_id)

        # TODO: random start
        # if isinstance(args.buffer_thresh, List):
        #     buffer_thresh = np.random.randint(*args.buffer_thresh)
        # else:
        #     buffer_thresh = args.buffer_thresh
        if training_buffer_thresh_range[0] == training_buffer_thresh_range[1]:
            buffer_thresh = training_buffer_thresh_range[0]
        else:
            buffer_thresh = np.random.randint(*training_buffer_thresh_range)

        # if isinstance(args.link_rtt, List):
        #     link_rtt = np.random.randint(*args.link_rtt)
        # else:
        #     link_rtt = args.link_rtt
        trace_idx = np.random.randint(len(all_cooked_time))
        net_env = env.NetworkEnvironment(
            trace_time=all_cooked_time[trace_idx],
            trace_bw=all_cooked_bw[trace_idx],
            trace_file_name=all_file_names[trace_idx],
            video_size_file_dir='../data/video_sizes',
            buffer_thresh=buffer_thresh * 1000)
        while True:  # experience video streaming forever

            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, rebuf, \
                video_chunk_size, next_video_chunk_sizes, \
                end_of_video, video_chunk_remain = \
                net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # -- linear reward --
            # reward is video quality - rebuffer penalty - smoothness
            reward = linear_reward(bit_rate, last_bit_rate, rebuf)

            r_batch.append(reward)

            last_bit_rate = bit_rate

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((args.S_INFO, args.S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be args.S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / \
                float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / args.BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / \
                float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / \
                args.BUFFER_NORM_FACTOR  # 10 sec
            state[4, :args.A_DIM] = np.array(next_video_chunk_sizes) / \
                M_IN_K / M_IN_K  # mega byte
            state[5, -1] = video_chunk_remain / net_env.total_video_chunk

            # compute action probability vector
            action_prob = actor.predict(np.reshape(
                state, (1, args.S_INFO, args.S_LEN)))
            action_cumsum = np.cumsum(action_prob)
            bit_rate = (action_cumsum > np.random.randint(
                1, args.RAND_RANGE) / float(args.RAND_RANGE)).argmax()
            # Note: we need to discretize the probability into
            # 1/args.RAND_RANGE steps, because there is an intrinsic
            # discrepancy in passing single state and batch states

            entropy_record.append(a3c.compute_entropy(action_prob[0]))

            if not args.no_agent_logging:  # and epoch % MODEL_SAVE_INTERVAL:
                # log time_stamp, bit_rate, buffer_size, reward
                log_file.write(str(time_stamp) + '\t' +
                               str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                               str(buffer_size) + '\t' +
                               str(rebuf) + '\t' +
                               str(video_chunk_size) + '\t' +
                               str(delay) + '\t' +
                               str(reward) + '\t' +
                               str(epoch) + '\t' +
                               str(trace_idx) + '\t' +
                               str(net_env.trace_ptr) + '\t' +
                               str(buffer_thresh) + '\n')

            # report experience to the coordinator
            if len(r_batch) >= args.TRAIN_SEQ_LEN or end_of_video:
                exp_queue.put([s_batch[1:],  # ignore the first chuck
                               a_batch[1:],  # since we don't have the
                               r_batch[1:],  # control over it
                               end_of_video,
                               {'entropy': entropy_record}])

                # synchronize the network parameters from the coordinator
                actor_net_params, critic_net_params, \
                    training_buffer_thresh_range = net_params_queue.get()
                actor.set_network_params(actor_net_params)
                critic.set_network_params(critic_net_params)

                epoch += 1
                s_batch = []
                a_batch = []
                r_batch = []
                entropy_record = []

                # so that in the log we know where video ends
                if not args.no_agent_logging:
                    log_file.write('\n')

            # store the state and action into batches
            if end_of_video:
                # if isinstance(args.buffer_thresh, List):
                #     buffer_thresh = np.random.randint(*args.buffer_thresh)
                # else:
                #     buffer_thresh = args.buffer_thresh
                if training_buffer_thresh_range[0] == training_buffer_thresh_range[1]:
                    buffer_thresh = training_buffer_thresh_range[0]
                else:
                    buffer_thresh = np.random.randint(
                        *training_buffer_thresh_range)
                # print('{} agent {} train with {}, {}'.format(
                #     epoch, agent_id, trace_idx, training_buffer_thresh_range))

                # if isinstance(args.link_rtt, List):
                #     link_rtt = np.random.randint(*args.link_rtt)
                # else:
                #     link_rtt = args.link_rtt
                trace_idx = np.random.randint(len(all_cooked_time))
                net_env = env.NetworkEnvironment(
                    trace_time=all_cooked_time[trace_idx],
                    trace_bw=all_cooked_bw[trace_idx],
                    trace_file_name=all_file_names[trace_idx],
                    video_size_file_dir='../data/video_sizes',
                    buffer_thresh=buffer_thresh*1000)
                # use the default action here
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY
                s_batch.append(np.zeros((args.S_INFO, args.S_LEN)))
            else:
                s_batch.append(state)

            action_vec = np.zeros(args.A_DIM)
            action_vec[bit_rate] = 1
            a_batch.append(action_vec)


def main(args):

    start_time = datetime.now()
    start_time_string = start_time.strftime("%Y%m%d_%H%M%S")
    args.start_time = start_time_string

    np.random.seed(args.RANDOM_SEED)
    assert len(VIDEO_BIT_RATE) == args.A_DIM

    # create result directory
    if not os.path.exists(args.summary_dir):
        os.makedirs(args.summary_dir)

    config.log_config(args)

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in range(args.NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(args, net_params_queues, exp_queues))
    coordinator.start()

    all_cooked_time, all_cooked_bw, all_file_names = load_traces(
        args.train_trace_dir)
    all_cooked_time, all_cooked_bw = adjust_traces(
        all_cooked_time, all_cooked_bw,
        bw_noise=args.noise, duration_factor=args.duration)
    agents = []
    for i in range(args.NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(args, i, all_cooked_time, all_cooked_bw,
                                       all_file_names, net_params_queues[i],
                                       exp_queues[i])))
    for i in range(args.NUM_AGENTS):
        agents[i].start()

    # wait unit training is done
    coordinator.join()


if __name__ == '__main__':
    args = config.parse_args()
    main(args)
