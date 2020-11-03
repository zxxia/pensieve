import os
import numpy as np


INPUT_PATH = "./Norway-DR-exp/val"
OUTPUT_PATH = "./Norway-DR-exp/val-2.5-noise"
os.makedirs(OUTPUT_PATH, exist_ok=True)

def rewrite_traces(input_trace_dir, output_trace_dir):
    np.random.seed(41)
    trace_files = os.listdir( input_trace_dir )
    for trace_file in trace_files:
        file_input_path = os.path.join( input_trace_dir, trace_file )
        file_output_path = os.path.join( output_trace_dir, trace_file )

        with open( file_input_path, 'r' ) as fr, open( file_output_path, 'w' ) as fw:
            for line in fr:
                ts, bw = line.split()
                noise = np.random.uniform( -1, 2.5, 1 )
                new_bw = line.replace( bw, str(float(bw) * float(1 + noise)) )
                fw.write(new_bw)

    return output_trace_dir


def main():
    # rewrite trace_dir with uniform.noise adding
    rewrite_traces(INPUT_PATH, OUTPUT_PATH)



if __name__ == '__main__':
    main()


