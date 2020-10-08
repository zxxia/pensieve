#! /bin/bash

# immediately exit the bash if an error encountered
set -e

NOISE=0
DURATION=1
# TRACE_PATH="./cooked_traces"
# TRACE_PATH="../sim/train_sim_traces"
# TRACE_PATH="../sim/test_sim_traces"
# TRACE_PATH="../sim/test_sim_traces"
TRACE_PATH="../data/test"
# TRACE_PATH="../data/train"
# SUMMARY_DIR="../results/duration_exp/duration_quarter"
# NN_MODELS=( "../sim/results_noise0.5/nn_model_ep_5300.ckpt" )
SIMULATOR_DIR="../sim"


# LOG_FILES=( 'pretrain' 'duration' 'duration_half' 'duration_quarter' 'duration_double' )
LOG_FILES=( '0' '1' '2' '3' '4' )


NN_MODELS=(
             "../results/results_noise0_new/nn_model_ep_20100.ckpt"
             "../results/results_noise1/nn_model_ep_23200.ckpt"
             "../results/results_noise2/nn_model_ep_22500.ckpt"
             "../results/results_noise3/nn_model_ep_18100.ckpt"
             "../results/results_noise4/nn_model_ep_20500.ckpt" )
for NOISE in 0 1 2 3 4; do
    DURATION=1
    SUMMARY_DIR="../results/noise_exp/noise_${NOISE}"
    # SUMMARY_DIR="../results/noise_${NOISE}"
    # SUMMARY_DIR="../results/noise_exp/noise_${NOISE}_train"
    # python bb.py \
    #     --test_trace_dir ${TRACE_PATH} \
    #     --summary_dir ${SUMMARY_DIR}/bb\
    #     --noise ${NOISE} \
    #     --duration ${DURATION}
    # python mpc.py \
    #     --test_trace_dir ${TRACE_PATH} \
    #     --summary_dir ${SUMMARY_DIR}/mpc\
    #     --noise ${NOISE} \
    #     --duration ${DURATION} &

    # for ((i=0;i<${#NN_MODELS[@]};++i)); do
    #     python rl_test.py \
    #         --test_trace_dir ${TRACE_PATH} \
    #         --summary_dir ${SUMMARY_DIR}/sim_rl_train_noise${LOG_FILES[i]}\
    #         --model_path ${NN_MODELS[i]} \
    #         --noise ${NOISE} \
    #         --duration ${DURATION} &
    # done
    python ${SIMULATOR_DIR}/rl_test.py \
        --test_trace_dir ${TRACE_PATH} \
        --summary_dir ${SUMMARY_DIR}/sim_rl_pretrain \
        --model_path ${SIMULATOR_DIR}/models/pretrain_linear_reward.ckpt \
        --noise ${NOISE} \
        --duration ${DURATION}
done
