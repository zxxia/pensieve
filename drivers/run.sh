#! /bin/bash

# immediately exit the bash if an error encountered
set -e

NOISE=0
DURATION=1
# TRACE_PATH="./cooked_traces"
# TRACE_PATH="../sim/train_sim_traces"
# TRACE_PATH="../sim/test_sim_traces"
# TRACE_PATH="../sim/test_sim_traces"
# TRACE_PATH="../data/test"
# TRACE_PATH="../data/train"
# TRACE_PATH="../data/Lesley_more_traces_for_pensieve/cooked_test_traces"
# TRACE_PATH="../data/Lesley_more_traces_for_pensieve/cooked_traces"
TRACE_PATH="../data/exponential_traces/test"
# SUMMARY_DIR="../results/duration_exp/duration_quarter"
# NN_MODELS=( "../sim/results_noise0.5/nn_model_ep_5300.ckpt" )
SIMULATOR_DIR="../sim"


LOG_FILES=( '0' '1' '2' '3' )

# bug fixed models
# NN_MODELS=(
#              "../results/bug_fix/results_noise0/nn_model_ep_47000.ckpt"
#              "../results/bug_fix/results_noise1/nn_model_ep_28100.ckpt"
#              "../results/bug_fix/results_noise2/nn_model_ep_19100.ckpt"
#              "../results/bug_fix/results_noise3/nn_model_ep_95100.ckpt" )
#              # "../results/bug_fix/results_noise4/nn_model_ep_20500.ckpt" )

# extropy fixed models
NN_MODELS=(
             "../results/entropy_weight_exp/results_noise0/nn_model_ep_83400.ckpt"
             "../results/bug_fix/results_noise1/nn_model_ep_28100.ckpt"
             "../results/entropy_weight_exp/results_noise2/nn_model_ep_64200.ckpt"
             "../results/bug_fix/results_noise3/nn_model_ep_95100.ckpt" )
             # "../results/bug_fix/results_noise4/nn_model_ep_20500.ckpt" )

# exponential_traces
NN_MODELS=( "../results/exponential_traces/results_noise_0_duration_1/nn_model_ep_47500.ckpt" )
             # "../results/bug_fix/results_noise4/nn_model_ep_20500.ckpt" )
for NOISE in 0 ; do
    # SUMMARY_DIR="../results/noise_exp/noise_${NOISE}"
    # SUMMARY_DIR="../results/tmp/noise_${NOISE}"
    # SUMMARY_DIR="../results/noise_${NOISE}"
    # SUMMARY_DIR="../results/noise_exp/noise_${NOISE}_train"
    # SUMMARY_DIR="../results/Lesley_more_traces_for_pensieve/noise_${NOISE}"
    # SUMMARY_DIR="../results/entropy_weight_exp/noise_${NOISE}_train"
    SUMMARY_DIR="../results/exponential_traces/noise_${NOISE}"
    # python ${SIMULATOR_DIR}/bb.py \
    #     --test_trace_dir ${TRACE_PATH} \
    #     --summary_dir ${SUMMARY_DIR}/bb\
    #     --noise ${NOISE} \
    #     --duration ${DURATION}
    # python ${SIMULATOR_DIR}/mpc.py \
    #     --test_trace_dir ${TRACE_PATH} \
    #     --summary_dir ${SUMMARY_DIR}/mpc\
    #     --noise ${NOISE} \
    #     --duration ${DURATION} &
#
    for ((i=0;i<${#NN_MODELS[@]};++i)); do
        python ${SIMULATOR_DIR}/rl_test.py \
            --test_trace_dir ${TRACE_PATH} \
            --summary_dir ${SUMMARY_DIR}/sim_rl_train_noise${LOG_FILES[i]}\
            --model_path ${NN_MODELS[i]} \
            --noise ${NOISE} \
            --duration ${DURATION} &
    done
    # python ${SIMULATOR_DIR}/rl_test.py \
    #     --test_trace_dir ${TRACE_PATH} \
    #     --summary_dir ${SUMMARY_DIR}/sim_rl_pretrain \
    #     --model_path ${SIMULATOR_DIR}/models/pretrain_linear_reward.ckpt \
    #     --noise ${NOISE} \
    #     --duration ${DURATION}
done
