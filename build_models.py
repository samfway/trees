#!/usr/bin/env python
import argparse
from tree_lib.parse import parse_csv_columns
from tree_lib.predict import build_models, load_pickle

def interface():
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input-file', help='Input data matrix', required=True)
    args.add_argument('-o', '--output-file', help='Models file (.pkl)', default='models.pkl')
    args.add_argument('-s', '--scale-file', help='Scaling file (.pkl)', default='scale.pkl')
    args = args.parse_args()
    return args

if __name__=="__main__":
    args = interface()
    build_models(args.input_file, args.output_file, args.scale_file)
    
