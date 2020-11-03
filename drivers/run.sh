#! /bin/bash

# immediately exit the bash if an error encountered
set -e

DURATION=1
TRACE_PATH="../data/FCC-DR-exp/FCC-test"
# TRACE_PATH="../sim/train_sim_traces"
# TRACE_PATH="../sim/test_sim_traces"
# TRACE_PATH="../sim/test_sim_traces"
# TRACE_PATH="../data/test"
#TRACE_PATH="../data/train"
# SUMMARY_DIR="../results/duration_exp/duration_quarter"
# NN_MODELS=( "../sim/results_noise0.5/nn_model_ep_5300.ckpt" )
SIMULATOR_DIR="../sim"


# LOG_FILES=( 'pretrain' 'duration' 'duration_half' 'duration_quarter' 'duration_double' )
LOG_FILES=( '35000' '34600' '33000' '32100' '29700' '28400' '27700')



NN_MODELS=("../results/Norway-DR-train/multiply-1-2.2-train/model_saved/nn_model_ep_35000.ckpt"
          "../results/Norway-DR-train/multiply-1-2.2-train/model_saved/nn_model_ep_34600.ckpt"
          "../results/Norway-DR-train/multiply-1-2.2-train/model_saved/nn_model_ep_33000.ckpt"
          "../results/Norway-DR-train/multiply-1-2.2-train/model_saved/nn_model_ep_32100.ckpt"
          "../results/Norway-DR-train/multiply-1-2.2-train/model_saved/nn_model_ep_29700.ckpt"
          "../results/Norway-DR-train/multiply-1-2.2-train/model_saved/nn_model_ep_28400.ckpt"
          "../results/Norway-DR-train/multiply-1-2.2-train/model_saved/nn_model_ep_27700.ckpt")
#RANDOM_SEED=41


    for RANDOM_SEED in 1; do
      SUMMARY_DIR="../results/DR-test/Norway-multiply-1-2.2"
      #     SUMMARY_DIR="../results/noise_${NOISE}"
      #     SUMMARY_DIR="../results/noise_exp/noise_${NOISE}_train"

        for ((i=0;i<${#NN_MODELS[@]};++i)); do
            python ${SIMULATOR_DIR}/rl_test.py \
                   --test_trace_dir ${TRACE_PATH} \
                   --summary_dir ${SUMMARY_DIR}/model_${LOG_FILES[i]}/seed_1\
                   --model_path ${NN_MODELS[i]} \
                   --random_seed ${RANDOM_SEED} \
                   --ROBUST_NOISE=0 \
                   --SAMPLE_LENGTH=0 \
                   --NUMBER_PICK=0 \
                   --duration ${DURATION} &

#           python ${SIMULATOR_DIR}/bb.py \
#                 --test_trace_dir ${TRACE_PATH} \
#                 --summary_dir ${SUMMARY_DIR}\
#                 --random_seed ${RANDOM_SEED}  \
#                 --ROBUST_NOISE=-0.3 \
#                 --SAMPLE_LENGTH=20 \
#                 --NUMBER_PICK=253 \
#                 --duration ${DURATION} &

#            python ${SIMULATOR_DIR}/mpc.py \
#                 --test_trace_dir ${TRACE_PATH} \
#                 --summary_dir ${SUMMARY_DIR}\
#                 --random_seed ${RANDOM_SEED}  \
#                 --ROBUST_NOISE=0 \
#                 --SAMPLE_LENGTH=0 \
#                 --NUMBER_PICK=0 \
#                 --duration ${DURATION}
          done

    done

