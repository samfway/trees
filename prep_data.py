#!/usr/bin/env python
import argparse
from tree_lib.parse import parse_csv_columns
import matplotlib.pyplot as plt
from numpy import sin, cos, array, deg2rad


def interface():
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input-file', help='Input file', required=True)
    args.add_argument('-o', '--output-file', help='Output file', required=False)
    args.add_argument('-k', '--index', help='Prep index', default=1, type=int)
    args.add_argument('--show', help='Show transformation', action='store_true', \
        default=False)
    args = args.parse_args()
    return args

if __name__=="__main__":
    args = interface()
    labels, data = parse_csv_columns(args.input_file, False)

    # Grab the degrees column & transform
    k = args.index
    v = data[:,k]

    if args.show:
        plt.hist(v, 35)
        plt.show()

    if args.output_file:
        output = open(args.output_file, 'w')
        for i in xrange(data.shape[0]):
            output.write(str(data[i][0]))
            output.write(',%.3f,%.3f' % (cos(deg2rad(data[i][1])), \
                sin(deg2rad(data[i][1]))))
            for j in xrange(2, data.shape[1]):
                output.write(',%s' % (str(data[i][j])))
            output.write('\n')
        output.close()

