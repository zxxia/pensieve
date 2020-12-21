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
TRACE_PATH="../data/synthetic_traces/trace_to_test_vary_duration_5500"
# TRACE_PATH="../data/train"
# TRACE_PATH="../data/Lesley_more_traces_for_pensieve/cooked_test_traces"
# TRACE_PATH="../data/Lesley_more_traces_for_pensieve/cooked_traces"
# TRACE_PATH="../data/exponential_traces/test"
# TRACE_PATH="../data/step_traces/test"
# TRACE_PATH="../data/step_traces_period20/test"
# TRACE_PATH="../data/step_traces_period50/test"
# TRACE_PATH="../data/step_traces_period40_changing_peak/test"
# TRACE_PATH="../data/201608_train"
# TRACE_PATH="../data/201608_test"
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
NN_MODELS=( "../results/entropy_weight_exp/results_noise0/nn_model_ep_83400.ckpt")
             # "../results/bug_fix/results_noise1/nn_model_ep_28100.ckpt"
             # "../results/entropy_weight_exp/results_noise2/nn_model_ep_64200.ckpt"
             # "../results/bug_fix/results_noise3/nn_model_ep_95100.ckpt" )
             # "../results/bug_fix/results_noise4/nn_model_ep_20500.ckpt" )

# exponential_traces
# NN_MODELS=( "../results/exponential_traces/results_noise_0_duration_1/nn_model_ep_47500.ckpt" )
             # "../results/bug_fix/results_noise4/nn_model_ep_20500.ckpt" )
# step_traces
# NN_MODELS=( "../results/step_traces/results_noise_0_duration_1/nn_model_ep_6200.ckpt" )
# NN_MODELS=( "../results/step_traces_period20/results_noise_0_duration_1/nn_model_ep_25000.ckpt" )
# NN_MODELS=( "../results/step_traces_period40_changing_peak/results_noise_0_duration_1/nn_model_ep_45900.ckpt" )
# NN_MODELS=( "../results/step_traces_period50/results_noise_0_duration_1/nn_model_ep_18600.ckpt" )
# NN_MODELS=( "../results/1000_fcc201608_traces/noise_0/nn_model_ep_82500.ckpt" )
# NN_MODELS=( "../results/results/results_noise_0/nn_model_ep_19500.ckpt" )
# NN_MODELS=( "../results/results_original/results_noise_0/nn_model_ep_17800.ckpt" )
# NN_MODELS=( "../results/results_original_change_pred/results_noise_0/nn_model_ep_108500.ckpt" )
# NN_MODELS=( "../results/repeat_entropy_weight_exp1/results_noise_0/nn_model_ep_164500.ckpt" )
# NN_MODELS=( "../results/remove_bad_traces/results_noise_0/nn_model_ep_58200.ckpt" )
# NN_MODELS=( "../results/synthetic_traces_T_l_1_2_exp/results_noise_0/nn_model_ep_22400.ckpt" )
# NN_MODELS=( "../results/synthetic_traces_T_l_1_6_exp/results_noise_0/nn_model_ep_24300.ckpt" )
# NN_MODELS=( "../results/synthetic_traces_T_l_1_10_exp/results_noise_0/nn_model_ep_21900.ckpt" )
NN_MODELS=( "../results/synthetic_traces_duration_10_30_exp/results_noise_0/nn_model_ep_39000.ckpt" )
NN_MODELS=( "../results/synthetic_traces_duration_10_100_exp/results_noise_0/nn_model_ep_3800.ckpt" )
NN_MODELS=( "../results/synthetic_traces_duration_10_300_exp/results_noise_0/nn_model_ep_4900.ckpt" )
NN_MODELS=( "../results/synthetic_traces_duration_10_500_exp/results_noise_0/nn_model_ep_4800.ckpt" )
for NOISE in 0 ; do
    # SUMMARY_DIR="../results/noise_exp/noise_${NOISE}"
    # SUMMARY_DIR="../results/tmp/noise_${NOISE}"
    # SUMMARY_DIR="../results/noise_${NOISE}"
    # SUMMARY_DIR="../results/noise_exp/noise_${NOISE}_train"
    # SUMMARY_DIR="../results/Lesley_more_traces_for_pensieve/noise_${NOISE}"
    # SUMMARY_DIR="../results/entropy_weight_exp/noise_${NOISE}_train"
    # SUMMARY_DIR="../results/bug_fix/noise_${NOISE}"
    # SUMMARY_DIR="../results/entropy_weight_exp/noise_${NOISE}_train"
    # SUMMARY_DIR="../results/exponential_traces/noise_${NOISE}"
    # SUMMARY_DIR="../results/step_traces/noise_${NOISE}"
    # SUMMARY_DIR="../results/step_traces_period20/noise_${NOISE}"
    # SUMMARY_DIR="../results/step_traces_period40_changing_peak/noise_${NOISE}"
    # SUMMARY_DIR="../results/step_traces_period50/noise_${NOISE}"
    # SUMMARY_DIR="../results/1000_fcc201608_traces/noise_${NOISE}_test_original"
    # SUMMARY_DIR="../results/results/noise_${NOISE}"
    # SUMMARY_DIR="../results/results_original/noise_${NOISE}"
    # SUMMARY_DIR="../results/results_original_change_pred/noise_${NOISE}"
    # SUMMARY_DIR="../results/repeat_entropy_weight_exp1/noise_${NOISE}"
    # SUMMARY_DIR="../results/remove_bad_traces/noise_${NOISE}"
    # SUMMARY_DIR="../results/synthetic_traces_T_l_1_2_exp/noise_${NOISE}"
    # SUMMARY_DIR="../results/synthetic_traces_T_l_1_6_exp/noise_${NOISE}"
    # SUMMARY_DIR="../results/synthetic_traces_T_l_1_10_exp/noise_${NOISE}"
    SUMMARY_DIR="../results/synthetic_traces_duration_10_30_exp/noise_${NOISE}"
    SUMMARY_DIR="../results/synthetic_traces_duration_10_100_exp/noise_${NOISE}"
    SUMMARY_DIR="../results/synthetic_traces_duration_10_300_exp/noise_${NOISE}"
    # SUMMARY_DIR="../results/synthetic_traces_duration_10_500_exp/noise_${NOISE}"
    # python ${SIMULATOR_DIR}/bb.py \
    #     --test_trace_dir ${TRACE_PATH} \
    #     --summary_dir ${SUMMARY_DIR}/bb\
    #     --noise ${NOISE} \
    #     --duration ${DURATION}
    # python ${SIMULATOR_DIR}/mpc.py \
    #     --test_trace_dir ${TRACE_PATH} \
    #     --summary_dir ${SUMMARY_DIR}/mpc\
    #     --noise ${NOISE} \
    #     --duration ${DURATION} # &
    python ${SIMULATOR_DIR}/mpc_optimized.py \
        --test_trace_dir ${TRACE_PATH} \
        --summary_dir ${SUMMARY_DIR}/mpc
    # cp -r ../results/entropy_weight_exp/noise_0/mpc ${SUMMARY_DIR}/

    # for ((i=0;i<${#NN_MODELS[@]};++i)); do
    #     python ${SIMULATOR_DIR}/rl_test.py \
    #         --test_trace_dir ${TRACE_PATH} \
    #         --summary_dir ${SUMMARY_DIR}/sim_rl_train_noise${LOG_FILES[i]}\
    #         --model_path ${NN_MODELS[i]} \
    #         --noise ${NOISE} \
    #         --duration ${DURATION}
    # done
    # python ${SIMULATOR_DIR}/rl_test.py \
    #     --test_trace_dir ${TRACE_PATH} \
    #     --summary_dir ${SUMMARY_DIR}/sim_rl_pretrain \
    #     --model_path ${SIMULATOR_DIR}/models/pretrain_linear_reward.ckpt \
    #     --noise ${NOISE} \
    #     --duration ${DURATION}
done
