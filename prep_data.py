#!/usr/bin/env python
import argparse
from tree_lib.parse import parse_csv_columns
import matplotlib.pyplot as plt
from numpy import sin, cos, array, deg2rad, linspace, log2

def interface():
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input-file', help='Input file', required=True)
    args.add_argument('-o', '--output-file', help='Output file', required=False)
    args.add_argument('-k', '--index', help='Prep index', default=1, type=int)
    args.add_argument('--show', help='Show transformation', \
        action='store_true', default=False)
    args.add_argument('--labeled', help='Input file is labeled', \
        action='store_true', default=False)
    args = args.parse_args()
    return args

if __name__=="__main__":
    args = interface()
    labels, data = parse_csv_columns(args.input_file, args.labeled)

    # Grab the degrees column & transform
    k = args.index
    v = data[:,k]

    if args.labeled:
        spruce = [ data[i,k] for i in xrange(data.shape[0]) if labels[i] == 1]
        lodge = [ data[i,k] for i in xrange(data.shape[0]) if labels[i] == 2]

        if k > 9:   
            for k in xrange(10, 51):
                print '\n%d' % (k)
                spruce = [ data[i,k] for i in xrange(data.shape[0]) if labels[i] == 1]
                lodge = [ data[i,k] for i in xrange(data.shape[0]) if labels[i] == 2]
                sp = float(sum(spruce))/len(spruce)
                lo = float(sum(lodge))/len(lodge)
                print 'spruce', float(sum(spruce))/len(spruce)
                print 'lodge', float(sum(lodge))/len(lodge)
                if lo > 2*sp:
                    print '*************'
            exit()

        min_val = min([min(spruce), min(lodge)])
        max_val = max([max(spruce), max(lodge)])
        bins = linspace(min_val, max_val, 30)
        plt.hist(spruce, bins, alpha=0.5, normed=True, label='Spruce/Fir')
        plt.hist(lodge, bins, alpha=0.5, normed=True, label='Lodgepole Pine')
        plt.legend(loc='upper left')
        plt.show()

    if args.show:
        plt.hist(v, 35)
        plt.show()

    if args.output_file:
        selected = [11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 25, 26, 34, 35, 27, 37, 43, 45]
        output = open(args.output_file, 'w')
        for i in xrange(data.shape[0]):
            output.write(str(data[i][0]))
            output.write(',' + str(int(data[i][0] > 3050)))
            output.write(',' + str(data[i][0]**2))
            #output.write(',' + str(sum(data[i][selected])))
            output.write(',%.3f,%.3f' % (cos(deg2rad(data[i][1])), \
                sin(deg2rad(data[i][1]))))
            for j in xrange(2, data.shape[1]):
                output.write(',%s' % (str(data[i][j])))
            output.write('\n')
        output.close()

