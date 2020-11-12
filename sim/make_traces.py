import os
import random
import subprocess
import sys

# 68 files with 2000 seconds, 205 files with 320 seconds

TRAIN_TRACE_DIR = "../data/synthetic_traces/train_large_range"
TEST_TRACE_DIR = "../data/synthetic_traces/test_large_range"
VAL_TRACE_DIR = "../data/synthetic_traces/val_large_range"

os.makedirs(TRAIN_TRACE_DIR, exist_ok=True)
os.makedirs(VAL_TRACE_DIR, exist_ok=True)
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
T_s_max = 200
T_l_min = 1
T_l_max = 10
cov_min = 0.01
cov_max = 0.7
duration_min = 1000
duration_max = 3000

# generate 66 files, each 2000 seconds for training
MAX_TASK_CNT = 32
cmds = []
processes = []
for i in range(0, 600):
    name = os.path.join(TRAIN_TRACE_DIR, f"trace{i}.txt")
    print("create ", name)
    T_s = random.uniform(T_s_min, T_s_max)
    T_l = random.uniform(T_l_min, T_l_max)
    cov = random.uniform(cov_min, cov_max)
    duration = random.uniform(duration_min, duration_max)
    cmd = "python synthetic_traces.py --T_l {} --T_s {} --cov {} " \
        "--duration {} --output_file {}".format(T_s, T_l, cov, duration, name)
    cmds.append(cmd.split(" "))

for i in range(600, 800):
    name = os.path.join(VAL_TRACE_DIR, f"trace{i}.txt")
    print("create ", name)
    T_s = random.uniform(T_s_min, T_s_max)
    T_l = random.uniform(T_l_min, T_l_max)
    cov = random.uniform(cov_min, cov_max)
    duration = random.uniform(duration_min, duration_max)
    cmd = "python synthetic_traces.py --T_l {} --T_s {} --cov {} " \
        "--duration {} --output_file {}".format(T_s, T_l, cov, duration, name)
    cmds.append(cmd.split(" "))

for i in range(800, 1000):
    name = os.path.join(TEST_TRACE_DIR, f"trace{i}.txt")
    print("create ", name)
    T_s = random.uniform(T_s_min, T_s_max)
    T_l = random.uniform(T_l_min, T_l_max)
    cov = random.uniform(cov_min, cov_max)
    duration = random.uniform(duration_min, duration_max)
    cmd = "python synthetic_traces.py --T_l {} --T_s {} --cov {} " \
        "--duration {} --output_file {}".format(T_s, T_l, cov, duration, name)
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
