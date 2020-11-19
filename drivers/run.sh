#! /bin/bash

# immediately exit the bash if an error encountered
set -e

DURATION=1
#TRACE_PATH="../data/Norway-DR-exp/val-norm-0.6-0.5-noise"
# TRACE_PATH="../sim/train_sim_traces"
# TRACE_PATH="../sim/test_sim_traces"
# TRACE_PATH="../sim/test_sim_traces"
# TRACE_PATH="../data/test"
#TRACE_PATH="../data/train"
# SUMMARY_DIR="../results/duration_exp/duration_quarter"
# NN_MODELS=( "../sim/results_noise0.5/nn_model_ep_5300.ckpt" )
SIMULATOR_DIR="../sim"


# LOG_FILES=( 'pretrain' 'duration' 'duration_half' 'duration_quarter' 'duration_double' )
#LOG_FILES=( '1' '2' '3' '4' '5' '6' '7' '8')
LOG_FILES=( '1')



# NN_MODELS="../results/synthetic-train/train-2000-trace-train100/model_saved/nn_model_ep_4400.ckpt"
NN_MODELS="../results/synthetic-train/mpc-6-train-subset/bitrate-6-train-20/nn_model_ep_24400.ckpt"

#NN_MODELS=(
#"../results/Norway-DR-train/multiply-norm-0-1-train/model_saved/nn_model_ep_31000.ckpt"
#"../results/Norway-DR-train/multiply-norm-0-1-train/model_saved/nn_model_ep_32200.ckpt"
#)
#RANDOM_SEED=41

TRACE_PATH="../data/synthetic-test/test_on_20_cut/"
SUMMARY_DIR="../results/synthetic-test/test_on_20_cut/"

#for i_folder in 2 10 20 30 40 50 60 70 80 90 100; do
#for i_folder in 10 20 30; do

#for i_folder in 2 20 60 100; do

#        TRACE_PATH="../data/synthetic-test/test_bitrate_new/test_on_${i_folder}"
#        SUMMARY_DIR="../results/synthetic-test/mpc-6-train-subset-test-new/mpc/test-on-${i_folder}"
##
#        TRACE_PATH="../data/synthetic-train-subset/val_${i_folder}_subset"
#        SUMMARY_DIR="../results/synthetic-test/mpc-6-subset-on-validation/bitrate-6-train-100/test-on-${i_folder}"

#        for ((i=0;i<${#NN_MODELS[@]};++i)); do
            python ${SIMULATOR_DIR}/rl_test.py \
                   --test_trace_dir ${TRACE_PATH} \
                   --summary_dir ${SUMMARY_DIR}/seed_1\
                   --model_path ${NN_MODELS} \
                   --random_seed=1 \
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
#
            python ${SIMULATOR_DIR}/mpc.py \
                 --test_trace_dir ${TRACE_PATH} \
                   --summary_dir ${SUMMARY_DIR}/seed_1\
                 --random_seed=1  \
                 --ROBUST_NOISE=0 \
                 --SAMPLE_LENGTH=0 \
                 --NUMBER_PICK=0 \
                 --duration ${DURATION}
###          #done

done