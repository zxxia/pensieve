import os
import random
import subprocess
import sys

from sympy import N, Symbol, solve

# 68 files with 2000 seconds, 205 files with 320 seconds

# TRAIN_TRACE_DIR = "../data/synthetic_Ts_train/train_Ts_100"
TEST_TRACE_DIR = "../data/synthetic_test/Ts_train_test"
# VAL_TRACE_DIR = "../data/synthetic_Ts_train/val_Ts_100"

# os.makedirs(TRAIN_TRACE_DIR, exist_ok=True)
# os.makedirs(VAL_TRACE_DIR, exist_ok=True)
os.makedirs(TEST_TRACE_DIR, exist_ok=True)

# T_s_min = 10
# T_s_max = 100
# T_l_min = 1
# T_l_max = 5
# cov_min = 0.05
# cov_max = 0.5
# duration_min = 320
# duration_max = 2000

# small range
# T_s_min = 10
# T_s_max = 20
# T_l_min = 1
# T_l_max = 3
# cov_min = 0.05
# cov_max = 0.1
# duration_min = 320
# duration_max = 500

# overlap
# T_s_min = 50
# T_s_max = 150
# T_l_min = 2
# T_l_max = 10
# cov_min = 0.3
# cov_max = 0.7
# duration_min = 1000
# duration_max = 3000

# large range
T_s_min = 1
#T_s_max = 100
T_l_min = 1
T_l_max = 10
cov_min = 0.01
cov_max = 0.7
duration_min = 1000
duration_max = 3000

# generate 66 files, each 2000 seconds for training
MAX_TASK_CNT = 32
MIN_THROUGHPUT = 0.2
MAX_THROUGHPUT = 10
STEPS = 20

cmds = []
processes = []
eq = -1
x = Symbol("x", positive=True)
for y in range(1, STEPS-1):
    eq += (1/x**y)
res = solve(eq, x)
switch_parameter = N(res[0])
# for i in range(0, 600):
#     name = os.path.join(TRAIN_TRACE_DIR, f"trace{i}.txt")
#     print("create ", name)
#     T_s = random.uniform(T_s_min, T_s_max)
#     T_l = random.uniform(T_l_min, T_l_max)
#     cov = random.uniform(cov_min, cov_max)
#     duration = random.uniform(duration_min, duration_max)
#     #max_throughput = round(random.uniform(MIN_THROUGHPUT, MAX_THROUGHPUT),1)
#     # for T_s experiment:
#     max_throughput = MAX_THROUGHPUT
#     min_throughput = random.uniform(MIN_THROUGHPUT, max_throughput)
#     cmd = "python synthetic_traces.py --T_l {} --T_s {} --cov {} " \
#         "--duration {} --steps {} --switch-parameter {} --max-throughput {} " \
#         "--min-throughput {} --output_file {}".format(
#                 T_l, T_s, cov, duration, STEPS, switch_parameter,
#                 max_throughput, min_throughput, name)
#     cmds.append(cmd.split(" "))
#
# for i in range(600, 800):
#     name = os.path.join(VAL_TRACE_DIR, f"trace{i}.txt")
#     print("create ", name)
#     T_s = random.uniform(T_s_min, T_s_max)
#     T_l = random.uniform(T_l_min, T_l_max)
#     cov = random.uniform(cov_min, cov_max)
#     duration = random.uniform(duration_min, duration_max)
#     #max_throughput = round(random.uniform(MIN_THROUGHPUT, MAX_THROUGHPUT),1)
#     # for T_s experiment:
#     max_throughput = MAX_THROUGHPUT
#     min_throughput = random.uniform(MIN_THROUGHPUT, max_throughput)
#     cmd = "python synthetic_traces.py --T_l {} --T_s {} --cov {} " \
#         "--duration {} --steps {} --switch-parameter {} --max-throughput {} " \
#         "--min-throughput {} --output_file {}".format(
#                 T_l, T_s, cov, duration, STEPS, switch_parameter,
#                 max_throughput, min_throughput, name)
#     cmds.append(cmd.split(" "))

#for x in range(10, 101, 10):
T_s_max = 2
x=2
for i in range(800, 1000):
        os.makedirs(TEST_TRACE_DIR+"/"+str(x), exist_ok=True)
        name = os.path.join(TEST_TRACE_DIR+"/"+str(x), f"trace{i}.txt")
        print("create ", name)
        T_s = random.uniform(T_s_min, T_s_max)
        T_l = random.uniform(T_l_min, T_l_max)
        cov = random.uniform(cov_min, cov_max)
        duration = random.uniform(duration_min, duration_max)
        #max_throughput = round(random.uniform(MIN_THROUGHPUT, MAX_THROUGHPUT),1)
        #for T_s experiment:
        max_throughput = MAX_THROUGHPUT
        min_throughput = random.uniform(MIN_THROUGHPUT, max_throughput)
        cmd = "python synthetic_traces.py --T_l {} --T_s {} --cov {} " \
            "--duration {} --steps {} --switch-parameter {} --max-throughput {} " \
            "--min-throughput {} --output_file {}".format(
                    T_l, T_s, cov, duration, STEPS, switch_parameter,
                    max_throughput, min_throughput, name)
        cmds.append(cmd.split(' '))


while True:
    while cmds and len(processes) < MAX_TASK_CNT:
        cmd = cmds.pop()
        processes.append(subprocess.Popen(cmd, stdin=open(os.devnull)))
    for p in processes:
        if p.poll() is not None:
            if p.returncode == 0:
                print(p.args, 'finished!')
                processes.remove(p)
            else:
                print(p.args, 'failed!')
                sys.exit(1)

    if not processes and not cmds:
        break