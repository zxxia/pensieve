import math
import os
import numpy as np
import random
import env


NAMES = ['timestamp', 'bandwidth']
SUMMARY_DIR_NOISE = "../noise_traces"

def load_traces(trace_dir):
    trace_files = os.listdir(trace_dir)
    all_ts = []
    all_bw = []
    all_file_names = []
    for trace_file in trace_files:
        file_path = os.path.join(trace_dir, trace_file)
        ts_list = []
        bw_list = []
        with open(file_path, 'r') as f:
            for line in f:
                ts, bw = line.split()
                ts = float(ts)
                bw = float(bw)
                ts_list.append(ts)
                bw_list.append(bw)
        all_ts.append(ts_list)
        all_bw.append(bw_list)
        all_file_names.append(trace_file)

    return all_ts, all_bw, all_file_names

def adjust_traces(all_ts, all_bw, test_traces_dir, random_seed, duration_factor=1):
    new_all_bw = []
    new_all_ts = []
    np.random.seed(random_seed)
    #### add noise to a segment of noise


    #### add noise to each BW
    for trace_ts, trace_bw in zip(all_ts, all_bw):
        duration = trace_ts[-1]
        new_duration = duration_factor * duration
        new_trace_ts = []
        new_trace_bw = []
        for i in range(math.ceil(duration_factor)):
            for t, bw in zip(trace_ts, trace_bw):
                if (t + i * duration) <= new_duration:
                    new_trace_ts.append(t + i * duration)
                    noise = np.random.normal(0, 0.05, 1)
                    if noise < -0.05:
                        noise = 0
                    delta = 1+float(noise)
                    new_trace_bw.append(bw*delta)

        new_all_ts.append(new_trace_ts)
        new_all_bw.append(new_trace_bw)
    assert len(new_all_ts) == len(all_ts)
    assert len(new_all_bw) == len(all_bw)

    # all_cooked_time, all_cooked_bw, all_file_names = load_traces(
    #     test_traces_dir )
    # net_env = env.Environment( all_cooked_time=all_cooked_time,
    #                            all_cooked_bw=all_cooked_bw, fixed=True )
    # # log new traces with noise
    #
    # noise_dir = os.path.join("../noise_traces/", str(random_seed))
    # try:
    #     os.mkdir(noise_dir)
    # except:
    #     pass
    #
    # log_path = os.path.join(noise_dir,
    #                         all_file_names[net_env.trace_idx])
    #
    # log_file = open( log_path, 'w' )
    # for index in range( len( new_all_ts ) ):
    #     log_file.write( str( new_all_ts[index] ) + '\t' + str(new_all_bw[index] ) + "\n" )
    # log_file.close()

    return new_all_ts, new_all_bw

def adjust_traces_one_random(all_ts, all_bw, random_seed, robust_noise, sample_length):
    adjust_n_random_traces(all_ts, all_bw, random_seed, robust_noise, sample_length, number_pick=1)
    # new_all_bw = all_bw.copy()
    # new_all_ts = all_ts.copy()
    # np.random.seed(random_seed)
    #
    # number_of_traces = len(all_ts)
    # random_trace_index = random.randint(0, number_of_traces - 1)
    # trace_bw = new_all_bw[random_trace_index]
    #
    # ########
    # # use your randomization code from the notebook on new_all_bw
    # ########
    # start_index = random.randint( 0, len( trace_bw ) - sample_length )
    # sublist = trace_bw[start_index: start_index + sample_length]
    # trace_bw[start_index:start_index + sample_length] = [i * float(1+robust_noise) for i in sublist]
    #
    # assert len(new_all_ts) == len(all_ts)
    # assert len(new_all_bw) == len(all_bw)
    #
    # return new_all_ts, new_all_bw


def adjust_n_random_traces(all_ts, all_bw, random_seed, robust_noise, sample_length, number_pick):
    new_all_bw = all_bw.copy()
    new_all_ts = all_ts.copy()
    np.random.seed(random_seed)

    number_of_traces = len(all_ts)

    # we need n random index numbers from the set
    # do this n times
    random_trace_indices = random.sample(range(0, number_of_traces - 1), number_pick)

    for ri in random_trace_indices:
        trace_bw = new_all_bw[ri]

        start_index = random.randint( 0, len( trace_bw ) - sample_length )
        sublist = trace_bw[start_index: start_index + sample_length]
        new_sublist = []
        for i in sublist:
            i = i*float(1+robust_noise)
            # if i + robust_noise > 0:
            #     i = i + robust_noise
            # else:
            #     i = i
            new_sublist.append(i)
        trace_bw[start_index:start_index + sample_length] = new_sublist

    assert len(new_all_ts) == len(all_ts)
    assert len(new_all_bw) == len(all_bw)

    return new_all_ts, new_all_bw


def compute_cdf(data):
    """ Return the cdf of input data.

    Args
        data(list): a list of numbers.

    Return
        sorted_data(list): sorted list of numbers.

    """
    length = len(data)
    sorted_data = sorted(data)
    cdf = [i / length for i, val in enumerate(sorted_data)]
    return sorted_data, cdf
