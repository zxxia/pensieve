import csv
import logging
import multiprocessing as mp
import os
import time

import a3c
import env
import numpy as np
import tensorflow as tf

import visdom
import src.config as config

from utils.utils import adjust_traces, load_traces
from datetime import datetime


# Visdom Settings
vis = visdom.Visdom()
assert vis.check_connection()
PLOT_COLOR = 'red'

tf.logging.set_verbosity(tf.logging.INFO)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and
# time), chunk_til_video_end
MODEL_SAVE_INTERVAL = 100
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
HD_REWARD = [1, 2, 3, 12, 15, 20]
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
NOISE = 0
DURATION = 1

# SUMMARY_DIR = f'../results/Lesley_more_traces_for_pensieve/results_noise{NOISE}'
# SUMMARY_DIR = f'../results/entropy_weight_exp/results_noise{NOISE}'
# SUMMARY_DIR = f'../results/lr_exp/results_noise{NOISE}'
# SUMMARY_DIR = '../results/results_duration_quarter'  # .format(DURATION)
# SUMMARY_DIR = '../results/results_duration_quarter'  # .format(DURATION)
# SUMMARY_DIR = './results_noise-1'
# TEST_LOG_FOLDER = os.path.join(SUMMARY_DIR, 'test_results')
# TRAIN_TRACES = './cooked_traces/'  # original trace location
# TRAIN_TRACES = './train_sim_traces'

# TRAIN_TRACES = '../data/train'
# VAL_TRACES = '../data/val/'
# TEST_TRACES = '../data/test'
# NN_MODEL = './results/pretrain_linear_reward.ckpt'
# NN_MODEL = None


# def entropy_weight_decay_func(epoch):
#     # linear decay
#     return np.maximum(-0.05/(10**4) * epoch + 0.5, 0.1)

def entropy_weight_decay_func(epoch):
    # linear decay
    if epoch < 5000:
        entropy = 1000.0
    if epoch < 10000:
        entropy = 2
    if entropy < 20000:
        entropy = 1
    if entropy < 50000:
        entropy = 0.5
    return entropy



def test(args, test_traces_dir, actor, log_output_dir, noise, duration):
    np.random.seed(args.RANDOM_SEED)
    assert len(VIDEO_BIT_RATE) == args.A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_traces(
        test_traces_dir)
    # handle the noise and duration variation here
    # all_cooked_time, all_cooked_bw = adjust_traces(
    #     all_cooked_time, all_cooked_bw,
    #     args.RANDOM_SEED, duration_factor=duration)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    log_path = os.path.join(log_output_dir, 'log_sim_rl_{}'.format(
                            all_file_names[net_env.trace_idx]))
    log_file = open(log_path, 'w')

    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    action_vec = np.zeros(args.A_DIM)
    action_vec[bit_rate] = 1

    s_batch = [np.zeros((args.S_INFO, args.S_LEN))]
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

        # reward is video quality - rebuffer penalty - smoothness
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
        log_file.flush()

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
        state[4, :args.A_DIM] = np.array(
            next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = np.minimum(video_chunk_remain,
                                  args.CHUNK_TIL_VIDEO_END_CAP) / float(args.CHUNK_TIL_VIDEO_END_CAP)

        action_prob = actor.predict(np.reshape(
            state, (1, args.S_INFO, args.S_LEN)))
        action_cumsum = np.cumsum(action_prob)
        bit_rate = (action_cumsum > np.random.randint(
            1, args.RAND_RANGE) / float(args.RAND_RANGE)).argmax()
        # TODO: Zhengxu: Why compute bitrate this way?
        # bit_rate = action_prob.argmax()
        # Note: we need to discretize the probability into 1/args.RAND_RANGE steps,
        # because there is an intrinsic discrepancy in passing single state and batch states

        s_batch.append(state)

        entropy_record.append(a3c.compute_entropy(action_prob[0]))

        if end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]

            action_vec = np.zeros(args.A_DIM)
            action_vec[bit_rate] = 1

            s_batch.append(np.zeros((args.S_INFO, args.S_LEN)))
            a_batch.append(action_vec)
            entropy_record = []

            video_count += 1

            if video_count >= len(all_file_names):
                break

            log_path = os.path.join(
                log_output_dir,
                'log_sim_rl_{}'.format(all_file_names[net_env.trace_idx]))
            log_file = open(log_path, 'w')


def testing(args, epoch, actor, log_file, trace_dir, test_log_folder, noise, duration):
    # clean up the test results folder
    os.system('rm -r ' + test_log_folder)
    os.makedirs(test_log_folder, exist_ok=True)

    # run test script
    test(args, trace_dir, actor, test_log_folder, noise, duration)

    # append test performance to the log
    rewards = []
    test_log_files = os.listdir(test_log_folder)
    print(len(test_log_files))
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
                   str(rewards_max) + '\n')
    log_file.flush()
    return rewards_mean


def central_agent(args, net_params_queues, exp_queues):
    # Visdom Logs
    testing_epochs = []
    training_losses = []
    testing_mean_rewards = []
    average_rewards = []
    average_entropies = []

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.INFO)

    assert len(net_params_queues) == args.NUM_AGENTS
    assert len(exp_queues) == args.NUM_AGENTS

    logging.basicConfig(filename=os.path.join(args.summary_dir, 'log_central'),
                        filemode='w', level=logging.INFO)

    with tf.Session() as sess, \
            open(os.path.join(args.summary_dir, 'log_test'), 'w', 1) as test_log_file, \
            open(os.path.join(args.summary_dir, 'log_train'), 'w', 1) as log_central_file, \
            open(os.path.join(args.summary_dir, 'log_val'), 'w', 1) as val_log_file, \
            open(os.path.join(args.summary_dir, 'log_train_e2e'), 'w', 1) as train_e2e_log_file:
        log_writer = csv.writer(log_central_file, delimiter='\t')
        log_writer.writerow(['epoch', 'loss', 'avg_reward', 'avg_entropy'])
        test_log_file.write("\t".join(
            ['epoch', 'rewards_min', 'rewards_5per', 'rewards_mean',
             'rewards_median', 'rewards_95per', 'rewards_max\n']))
        val_log_file.write("\t".join(
            ['epoch', 'rewards_min', 'rewards_5per', 'rewards_mean',
             'rewards_median', 'rewards_95per', 'rewards_max\n']))
        train_e2e_log_file.write("\t".join(
            ['epoch', 'rewards_min', 'rewards_5per', 'rewards_mean',
             'rewards_median', 'rewards_95per', 'rewards_max\n']))

        actor = a3c.ActorNetwork(sess,
                                 state_dim=[args.S_INFO,
                                            args.S_LEN], action_dim=args.A_DIM,
                                 learning_rate=args.ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[args.S_INFO, args.S_LEN],
                                   learning_rate=args.CRITIC_LR_RATE)

        logging.info('actor and critic initialized')
        summary_ops, summary_vars = a3c.build_summaries()

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(
            args.summary_dir, sess.graph)  # training monitor
        saver = tf.train.Saver(max_to_keep=1000)  # save neural net parameters

        # restore neural net parameters
        if args.nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, args.nn_model)
            print("Model restored.")

        epoch = 0

        # assemble experiences from agents, compute the gradients
        max_avg_reward = None
        max_train_reward = None

        while True:
            start_t = time.time()
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            critic_net_params = critic.get_network_params()
            for i in range(args.NUM_AGENTS):
                net_params_queues[i].put([actor_net_params, critic_net_params])
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

            for i in range(args.NUM_AGENTS):
                s_batch, a_batch, r_batch, terminal, info = exp_queues[i].get()

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
            # assembled_actor_gradient = actor_gradient_batch[0]
            # assembled_critic_gradient = critic_gradient_batch[0]
            # for i in range(len(actor_gradient_batch) - 1):
            #     for j in range(len(assembled_actor_gradient)):
            #             assembled_actor_gradient[j] += actor_gradient_batch[i][j]
            #             assembled_critic_gradient[j] += critic_gradient_batch[i][j]
            # actor.apply_gradients(assembled_actor_gradient)
            # critic.apply_gradients(assembled_critic_gradient)
            for i in range(len(actor_gradient_batch)):
                actor.apply_gradients(actor_gradient_batch[i])
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

            if epoch % MODEL_SAVE_INTERVAL == 0:
                _ = testing(args, epoch, actor, train_e2e_log_file,
                            args.train_trace_dir,
                            os.path.join(args.summary_dir, 'test_results'),
                            args.noise, args.duration)
                _ = testing(args, epoch, actor, test_log_file, args.test_trace_dir,
                            os.path.join(args.summary_dir, 'test_results'),
                            args.noise, args.duration)

                # Visdom log and plot
                test_mean_reward = testing(
                    args, epoch, actor, val_log_file, args.val_trace_dir,
                    os.path.join(args.summary_dir, 'test_results'),
                    args.noise, args.duration)
                testing_epochs.append(epoch)
                testing_mean_rewards.append(test_mean_reward)
                average_rewards.append(np.sum(avg_reward))
                average_entropies.append(avg_entropy)

                suffix = args.start_time
                if args.description is not None:
                    suffix = args.description
                trace = dict(x=testing_epochs, y=testing_mean_rewards, mode="markers+lines", type='custom',
                             marker={'color': PLOT_COLOR,
                                     'symbol': 104, 'size': "5"},
                             text=["one", "two", "three"], name='1st Trace')
                layout = dict(title="Pensieve_Val_Reward " + suffix,
                              xaxis={'title': 'Epoch'},
                              yaxis={'title': 'Mean Reward'})
                vis._send(
                    {'data': [trace], 'layout': layout, 'win': 'Pensieve_testing_mean_reward_' + args.start_time})
                trace = dict(x=testing_epochs, y=average_rewards, mode="markers+lines", type='custom',
                             marker={'color': PLOT_COLOR,
                                     'symbol': 104, 'size': "5"},
                             text=["one", "two", "three"], name='1st Trace')
                layout = dict(title="Pensieve_Training_Reward " + suffix,
                              xaxis={'title': 'Epoch'},
                              yaxis={'title': 'Mean Reward'})
                vis._send(
                    {'data': [trace], 'layout': layout, 'win': 'Pensieve_training_mean_reward_' + args.start_time})
                trace = dict(x=testing_epochs, y=average_entropies, mode="markers+lines", type='custom',
                             marker={'color': PLOT_COLOR,
                                     'symbol': 104, 'size': "5"},
                             text=["one", "two", "three"], name='1st Trace')
                layout = dict(title="Pensieve_Training_Mean Entropy " + suffix,
                              xaxis={'title': 'Epoch'},
                              yaxis={'title': 'Mean Entropy'})
                vis._send(
                    {'data': [trace], 'layout': layout, 'win': 'Pensieve_training_mean_entropy_' + args.start_time})

                avg_val_reward = testing(
                    args, epoch, actor, val_log_file, args.val_trace_dir,
                    os.path.join(args.summary_dir, 'test_results'),
                    args.noise, args.duration)

                if max_avg_reward is None or avg_val_reward > max_avg_reward:
                    max_avg_reward = avg_val_reward
                    # Save the neural net parameters to disk.
                    save_path = saver.save(sess,
                                           os.path.join(args.summary_dir+"/model_saved/", f"nn_model_ep_{epoch}.ckpt"))
                    logging.info("Model saved in file: " + save_path)

                if epoch >= args.TOTAL_EPOCH:
                    exit()

            end_t = time.time()
            print(f'epoch{epoch-1}: {end_t - start_t}s')


def agent(args, agent_id, all_cooked_time, all_cooked_bw, net_params_queue,
          exp_queue):

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw,
                              random_seed=agent_id)

    with tf.Session() as sess, open(os.path.join(
            args.summary_dir, f'log_agent_{agent_id}'), 'w') as log_file:
        actor = a3c.ActorNetwork(sess,
                                 state_dim=[args.S_INFO,
                                            args.S_LEN], action_dim=args.A_DIM,
                                 learning_rate=args.ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[args.S_INFO, args.S_LEN],
                                   learning_rate=args.CRITIC_LR_RATE)

        # initial synchronization of the network parameters from the coordinator
        actor_net_params, critic_net_params = net_params_queue.get()
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
        while True:            # experience video streaming forever

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
            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                - REBUF_PENALTY * rebuf \
                - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                          VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

            # -- log scale reward --
            # log_bit_rate = np.log(VIDEO_BIT_RATE[bit_rate] / float(VIDEO_BIT_RATE[-1]))
            # log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[-1]))

            # reward = log_bit_rate \
            #          - REBUF_PENALTY * rebuf \
            #          - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)

            # -- HD reward --
            # reward = HD_REWARD[bit_rate] \
            #          - REBUF_PENALTY * rebuf \
            #          - SMOOTH_PENALTY * np.abs(HD_REWARD[bit_rate] - HD_REWARD[last_bit_rate])

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
            state[4, :args.A_DIM] = np.array(
                next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[5, -1] = np.minimum(video_chunk_remain,
                                      args.CHUNK_TIL_VIDEO_END_CAP) / float(args.CHUNK_TIL_VIDEO_END_CAP)

            # compute action probability vector
            action_prob = actor.predict(np.reshape(
                state, (1, args.S_INFO, args.S_LEN)))
            action_cumsum = np.cumsum(action_prob)
            bit_rate = (action_cumsum > np.random.randint(
                1, args.RAND_RANGE) / float(args.RAND_RANGE)).argmax()
            # Note: we need to discretize the probability into 1/args.RAND_RANGE steps,
            # because there is an intrinsic discrepancy in passing single state and batch states

            entropy_record.append(a3c.compute_entropy(action_prob[0]))

            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write(str(time_stamp) + '\t' +
                           str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                           str(buffer_size) + '\t' +
                           str(rebuf) + '\t' +
                           str(video_chunk_size) + '\t' +
                           str(delay) + '\t' +
                           str(reward) + '\t' +
                           str(epoch) + '\t' +
                           str(net_env.trace_idx) + '\t' +
                           str(net_env.mahimahi_ptr)+'\n')
            log_file.flush()

            # report experience to the coordinator
            if len(r_batch) >= args.TRAIN_SEQ_LEN or end_of_video:
                exp_queue.put([s_batch[1:],  # ignore the first chuck
                               a_batch[1:],  # since we don't have the
                               r_batch[1:],  # control over it
                               end_of_video,
                               {'entropy': entropy_record}])

                # synchronize the network parameters from the coordinator
                actor_net_params, critic_net_params = net_params_queue.get()
                actor.set_network_params(actor_net_params)
                critic.set_network_params(critic_net_params)

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]
                del entropy_record[:]

                # so that in the log we know where video ends
                log_file.write('\n')

            # store the state and action into batches
            if end_of_video:
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here

                action_vec = np.zeros(args.A_DIM)
                action_vec[bit_rate] = 1

                s_batch.append(np.zeros((args.S_INFO, args.S_LEN)))
                a_batch.append(action_vec)

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

    all_cooked_time, all_cooked_bw, _ = load_traces(args.train_trace_dir)
    # all_cooked_time, all_cooked_bw = adjust_traces(
    #     all_cooked_time, all_cooked_bw,
    #     args.RANDOM_SEED, duration_factor=args.duration)
    agents = []
    for i in range(args.NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(args, i, all_cooked_time, all_cooked_bw,
                                       net_params_queues[i], exp_queues[i])))
    for i in range(args.NUM_AGENTS):
        agents[i].start()

    # wait unit training is done
    coordinator.join()


if __name__ == '__main__':
    args = config.parse_args()

    main(args)