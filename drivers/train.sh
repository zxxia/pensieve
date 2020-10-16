#! /bin/bash

# immediately exit the bash if an error encountered
set -e

NOISE=0
DURATION=1
TRAIN_TRACE_PATH="../data/train"
VAL_TRACE_PATH="../data/val"
TEST_TRACE_PATH="../data/test"
# TRAIN_TRACE_PATH="../data/exponential_traces/train"
# VAL_TRACE_PATH="../data/exponential_traces/val"
# TEST_TRACE_PATH="../data/exponential_traces/test"
# TRAIN_TRACE_PATH="../data/step_traces/train"
# VAL_TRACE_PATH="../data/step_traces/val"
# TEST_TRACE_PATH="../data/step_traces/test"
# TRAIN_TRACE_PATH="../data/step_traces_period20/train"
# VAL_TRACE_PATH="../data/step_traces_period20/val"
# TEST_TRACE_PATH="../data/step_traces_period20/test"
# TRAIN_TRACE_PATH="../data/step_traces_period40_changing_peak/train"
# VAL_TRACE_PATH="../data/step_traces_period40_changing_peak/val"
# TEST_TRACE_PATH="../data/step_traces_period40_changing_peak/test"
# TRAIN_TRACE_PATH="../data/step_traces_period50/train"
# VAL_TRACE_PATH="../data/step_traces_period50/val"
# TEST_TRACE_PATH="../data/step_traces_period50/test"
# TRAIN_TRACE_PATH="../data/constant_trace/train"
# VAL_TRACE_PATH="../data/constant_trace/val"
# TEST_TRACE_PATH="../data/constant_trace/test"
SIMULATOR_DIR="../sim"


LOG_FILES=( '0' '1' '2' '3' )

for NOISE in 0 ; do
    # SUMMARY_DIR="../results/noise_exp/noise_${NOISE}"
    # SUMMARY_DIR="../results/tmp/noise_${NOISE}"
    # SUMMARY_DIR="../results/exponential_traces/results_noise_${NOISE}_duration_${DURATION}"
    # SUMMARY_DIR="../results/step_traces_period20/results_noise_${NOISE}_duration_${DURATION}"
    # SUMMARY_DIR="../results/step_traces_period40_changing_peak/results_noise_${NOISE}_duration_${DURATION}"
    # SUMMARY_DIR="../results/step_traces_period50/results_noise_${NOISE}_duration_${DURATION}"
    # SUMMARY_DIR="../results/constant_trace/results_noise_${NOISE}_duration_${DURATION}"
    # SUMMARY_DIR="../results/eval_train_e2e/results_noise_${NOISE}_duration_${DURATION}"
    SUMMARY_DIR="../results/eval_train_e2e_new/results_noise_${NOISE}_duration_${DURATION}"
    python ${SIMULATOR_DIR}/multi_agent.py \
        --train_trace_dir ${TRAIN_TRACE_PATH} \
        --val_trace_dir ${VAL_TRACE_PATH} \
        --test_trace_dir ${TEST_TRACE_PATH} \
        --summary_dir ${SUMMARY_DIR} \
        --noise ${NOISE} \
        --duration ${DURATION}
done
