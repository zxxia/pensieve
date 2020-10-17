#! /bin/bash

# immediately exit the bash if an error encountered
set -e

DURATION=1
# TRACE_PATH="./cooked_traces"
# TRACE_PATH="../sim/train_sim_traces"
# TRACE_PATH="../sim/test_sim_traces"
# TRACE_PATH="../sim/test_sim_traces"
TRACE_PATH="../data/test"
#TRACE_PATH="../data/train"
# SUMMARY_DIR="../results/duration_exp/duration_quarter"
# NN_MODELS=( "../sim/results_noise0.5/nn_model_ep_5300.ckpt" )
SIMULATOR_DIR="../sim"


# LOG_FILES=( 'pretrain' 'duration' 'duration_half' 'duration_quarter' 'duration_double' )
#LOG_FILES=( '0' '1' '2' '3')


NN_MODELS="../results/robust_exp/noise_3/nn_model_ep_36800.ckpt" #noise=0.03
#RANDOM_SEED=41


for RANDOM_SEED in 11 12 13 14 15 16 17 18 19 20; do
#    DURATION=1
SUMMARY_DIR="../results/robust_test_unbound_add/seed_${RANDOM_SEED}"
#     SUMMARY_DIR="../results/noise_${NOISE}"
#     SUMMARY_DIR="../results/noise_exp/noise_${NOISE}_train"

     python ${SIMULATOR_DIR}/mpc.py \
         --test_trace_dir ${TRACE_PATH} \
         --summary_dir ${SUMMARY_DIR}\
         --random_seed ${RANDOM_SEED}  \
         --duration ${DURATION} &

#     for ((i=0;i<${#NN_MODELS[@]};++i)); do
      python ${SIMULATOR_DIR}/rl_test.py \
             --test_trace_dir ${TRACE_PATH} \
             --summary_dir ${SUMMARY_DIR}\
             --model_path ${NN_MODELS[i]} \
             --random_seed ${RANDOM_SEED} \
             --duration ${DURATION} &
     done


# done
