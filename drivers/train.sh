#! /bin/bash

# immediately exit the bash if an error encountered
set -e

NOISE=0
DURATION=1
# TRAIN_TRACE_PATH="../data/train"
# VAL_TRACE_PATH="../data/val"
# TEST_TRACE_PATH="../data/test"
TRAIN_TRACE_PATH="../data/exponential_traces/train"
VAL_TRACE_PATH="../data/exponential_traces/val"
TEST_TRACE_PATH="../data/exponential_traces/test"
SIMULATOR_DIR="../sim"


LOG_FILES=( '0' '1' '2' '3' )

for NOISE in 0 ; do
    # SUMMARY_DIR="../results/noise_exp/noise_${NOISE}"
    # SUMMARY_DIR="../results/tmp/noise_${NOISE}"
    SUMMARY_DIR="../results/exponential_traces/results_noise_${NOISE}_duration_${DURATION}"
    python ${SIMULATOR_DIR}/multi_agent.py \
        --train_trace_dir ${TRAIN_TRACE_PATH} \
        --val_trace_dir ${VAL_TRACE_PATH} \
        --test_trace_dir ${TEST_TRACE_PATH} \
        --summary_dir ${SUMMARY_DIR} \
        --noise ${NOISE} \
        --duration ${DURATION}
done
