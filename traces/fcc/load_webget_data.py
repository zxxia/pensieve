""" Parse fcc webget csv file and output traces for simulation."""
import argparse
import csv
from datetime import datetime
import os

import numpy as np

NUM_LINES = np.inf
TIME_INTERVAL = 5  # seconds
BITS_IN_BYTE = 8  # 8 bits per byte
MICROSEC_IN_SEC = 1000000.0


def parse_args():
    '''
    Parse arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Parse FCC data.")
    parser.add_argument("--input_file", type=str, required=True,
                        help='path to a fcc data csv file.')
    parser.add_argument("--output_dir", type=str, required=True,
                        help='output directory.')

    return parser.parse_args()


def main():
    args = parse_args()
    input_file = args.input_file
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    time_origin = datetime.utcfromtimestamp(0)
    line_counter = 0
    bw_measurements = {}
    unique_targets = set()
    unique_addresses = set()
    with open(input_file, 'r') as f:
        csvreader = csv.reader(f)
        # refer to
        # http://data.fcc.gov/download/measuring-broadband-america/2016/data-dictionary-sept2015.xls
        # for each column name
        for row in csvreader:

            uid = row[0]  # unique identifier for an individual unit
            dtime = (datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S') -
                     time_origin).total_seconds()  # time test finished in sec
            target = row[2]  # URL to fetch
            # IP addres connected to to featch content from initial URL
            address = row[3]
            unique_targets.add(target)
            unique_addresses.add(address)

            # Average speed of downloading HTML content and then concurrently
            # downloading all resources (Units: bytes/sec)
            throughput = row[6]

            k = (uid, target)
            if k not in bw_measurements:
                bw_measurements[k] = []
            bw_measurements[k].append(
                float(throughput) * BITS_IN_BYTE / MICROSEC_IN_SEC)

            line_counter += 1
            if line_counter >= NUM_LINES:
                break
    print(len(unique_targets))
    print(len(unique_addresses))

    for k, bw_list in bw_measurements.items():
        # filter out inappropriate traces as the paper mentions
        out_file = 'trace_' + '_'.join(k)
        out_file = out_file.replace(':', '-')
        out_file = out_file.replace('/', '-')
        out_file = os.path.join(output_dir, out_file)
        if min(bw_list) < 0.2 or np.mean(bw_list) > 6 or len(bw_list) < 5:
            print('skip', out_file)
            continue
        with open(out_file, 'w') as f:
            csvwriter = csv.writer(f, delimiter='\t')
            for i, bw in enumerate(bw_list):
                csvwriter.writerow([i * TIME_INTERVAL, bw])


if __name__ == '__main__':
    main()
