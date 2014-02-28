#!/usr/bin/env python
import argparse
from tree_lib.parse import parse_csv_columns

def interface():
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input-file', help='Input data matrix', required=True)
    args = args.parse_args()
    return args

if __name__=="__main__":
    args = interface()
    labels, matrix = parse_csv_columns(args.input_file, labeled=True)
    print len(labels)
    print matrix.shape

