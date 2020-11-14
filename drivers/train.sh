#! /bin/bash

# immediately exit the bash if an error encountered
set -e

NOISE=0
DURATION=1
TEST_TRACE_PATH="../data/one-trace"

TRAIN_TRACE_PATH_1="../data/synthetic-train/train_maxBW_5"
VAL_TRACE_PATH_1="../data/synthetic-train/val_maxBW_5"

#TRAIN_TRACE_PATH_2="../data/Norway-DR-exp/train-2.2-noise"
#VAL_TRACE_PATH_2="../data/Norway-DR-exp/val-2.2-noise"

#TRAIN_TRACE_PATH_3="../data/Norway-DR-exp/train-2.3-noise"
#VAL_TRACE_PATH_3="../data/Norway-DR-exp/val-2.3-noise"
#
#TRAIN_TRACE_PATH_4="../data/Norway-DR-exp/train-2.4-noise"
#VAL_TRACE_PATH_4="../data/Norway-DR-exp/val-2.4-noise"

SIMULATOR_DIR="../sim"



SUMMARY_DIR_1="../results/synthetic-train/bitrate-10-train-5"
python ${SIMULATOR_DIR}/multi_agent.py \
    --A_DIM=10\
    --S_LEN=10\
    --NUM_AGENTS=8\
    --train_trace_dir ${TRAIN_TRACE_PATH_1} \
    --val_trace_dir ${VAL_TRACE_PATH_1} \
    --test_trace_dir ${TEST_TRACE_PATH} \
    --summary_dir ${SUMMARY_DIR_1} \
    --noise ${NOISE} \
    --duration ${DURATION}


#SUMMARY_DIR_2="../results/Norway-DR-train/multiply-1-2.2-train"
#python ${SIMULATOR_DIR}/multi_agent.py \
#    --train_trace_dir ${TRAIN_TRACE_PATH_2} \
#    --val_trace_dir ${VAL_TRACE_PATH_2} \
#    --test_trace_dir ${TEST_TRACE_PATH} \
#    --summary_dir ${SUMMARY_DIR_2} \
#    --noise ${NOISE} \
#    --duration ${DURATION}
