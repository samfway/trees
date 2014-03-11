#!/usr/bin/env python
import argparse
from tree_lib.parse import parse_csv_columns

def interface():
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input-file', help='Input data matrix', required=True)
    args.add_argument('--labeled', help='Data contains label column', action='store_true', \
        default=False)
    args = args.parse_args()
    return args

if __name__=="__main__":
    args = interface()
    labels, data = parse_csv_columns(args.input_file, args.labeled)
    print data
    
