#! /bin/bash

# immediately exit the bash if an error encountered
set -e

DURATION=1
# TRACE_PATH="./cooked_traces"
# TRACE_PATH="../sim/train_sim_traces"
# TRACE_PATH="../sim/test_sim_traces"
# TRACE_PATH="../sim/test_sim_traces"
#TRACE_PATH="../data/test"
TRACE_PATH="../data/train"
# SUMMARY_DIR="../results/duration_exp/duration_quarter"
# NN_MODELS=( "../sim/results_noise0.5/nn_model_ep_5300.ckpt" )
SIMULATOR_DIR="../sim"


# LOG_FILES=( 'pretrain' 'duration' 'duration_half' 'duration_quarter' 'duration_double' )
#LOG_FILES=( '0' '1' '2' '3')


#NN_MODELS="../data/model_saved/nn_model_ep_36800.ckpt" #noise=0.03
NN_MODELS="../data/model_saved/nn_model_ep_94000.ckpt" # model from Zhengxu
#RANDOM_SEED=41


    for RANDOM_SEED in {1..100}; do
      SUMMARY_DIR="../results/robustness_train/robust_all_mul_-0.1_length20_pick54/seed_${RANDOM_SEED}"
      #     SUMMARY_DIR="../results/noise_${NOISE}"
      #     SUMMARY_DIR="../results/noise_exp/noise_${NOISE}_train"

        #for ((i=0;i<${#NN_MODELS[@]};++i)); do
            python ${SIMULATOR_DIR}/rl_test.py \
                   --test_trace_dir ${TRACE_PATH} \
                   --summary_dir ${SUMMARY_DIR}\
                   --model_path ${NN_MODELS[i]} \
                   --random_seed ${RANDOM_SEED} \
                   --ROBUST_NOISE=-0.1 \
                   --SAMPLE_LENGTH=20 \
                   --NUMBER_PICK=53 \
                   --duration ${DURATION} &

           python ${SIMULATOR_DIR}/bb.py \
                 --test_trace_dir ${TRACE_PATH} \
                 --summary_dir ${SUMMARY_DIR}\
                 --random_seed ${RANDOM_SEED}  \
                 --ROBUST_NOISE=-0.1 \
                 --SAMPLE_LENGTH=20 \
                 --NUMBER_PICK=53 \
                 --duration ${DURATION} &

            python ${SIMULATOR_DIR}/mpc.py \
                 --test_trace_dir ${TRACE_PATH} \
                 --summary_dir ${SUMMARY_DIR}\
                 --random_seed ${RANDOM_SEED}  \
                 --ROBUST_NOISE=-0.1 \
                 --SAMPLE_LENGTH=20 \
                 --NUMBER_PICK=53 \
                 --duration ${DURATION}


    done

