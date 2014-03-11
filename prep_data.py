#!/usr/bin/env python
import argparse
from tree_lib.parse import parse_csv_columns
import matplotlib.pyplot as plt
from numpy import sin, cos, array, deg2rad


def interface():
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input-file', help='Input file', required=True)
    args.add_argument('-o', '--output-file', help='Output file', required=False)
    args.add_argument('--show', help='Show transformation', action='store_true', \
        default=False)
    args = args.parse_args()
    return args

if __name__=="__main__":
    args = interface()
    labels, data = parse_csv_columns(args.input_file, False)

    # Grab the degrees column & transform
    v = data[:,1]

    if args.show:
        plt.hist(v, 35)
        plt.show()

    vt = [ 360-a if a > 180 else a for a in v ]
    vt = [ cos(deg2rad(a)) for a in vt ]
    data[:,1] = vt

    if args.show:
        plt.hist(vt, 35)
        plt.show()

    if args.output_file:
        output = open(args.output_file, 'w')
        for k in xrange(data.shape[0]):
            v = data[k,:]
            output.write(','.join([ str(vx) for vx in v ])+'\n')
        output.close()

