import os
import numpy as np


INPUT_TRAIN_PATH = "./Norway-DR-exp/train"
OUTPUT_TRAIN_PATH = "./Norway-DR-exp/train-norm-1-0.5-noise"
INPUT_VAL_PATH = "./Norway-DR-exp/val"
OUTPUT_VAL_PATH = "./Norway-DR-exp/val-norm-1-0.5-noise"

os.makedirs(OUTPUT_TRAIN_PATH, exist_ok=True)
os.makedirs(OUTPUT_VAL_PATH, exist_ok=True)


def rewrite_traces(input_trace_dir, output_trace_dir):
    np.random.seed(41)
    trace_files = os.listdir( input_trace_dir )
    for trace_file in trace_files:
        file_input_path = os.path.join( input_trace_dir, trace_file )
        file_output_path = os.path.join( output_trace_dir, trace_file )

        with open( file_input_path, 'r' ) as fr, open( file_output_path, 'w' ) as fw:
            for line in fr:
                ts, bw = line.split()
                #noise = np.random.uniform( -1, 2.4, 1 )
                noise = np.random.normal(1, 0.5, 1)
                delta = 1 + noise
                if delta < 0:
                    delta = 0.1
                new_bw = line.replace( bw, str(float(bw) * float(delta)))
                fw.write(new_bw)

    return output_trace_dir


def main():
    # rewrite trace_dir with uniform.noise adding
    rewrite_traces(INPUT_TRAIN_PATH, OUTPUT_TRAIN_PATH)
    rewrite_traces(INPUT_VAL_PATH, OUTPUT_VAL_PATH)




if __name__ == '__main__':
    main()


