#!/usr/bin/env python

__author__ = "Sam Way"
__credits__ = ["Sam Way"]
__license__ = "GPL"
__version__ = "0.0.1-dev"
__maintainer__ = "Sam Way"
__email__ = "samfway@gmail.com"
__status__ = "Development"

from numpy import array

def parse_csv_columns(csv_file, labeled=False):
    """ Parse csv file to numpy matrix. If labeled is 
        set to True, the last column of the matrix is 
        interpretted as the label column and returned
        separately. 
    """
    labels = [] 
    matrix = []
    
    for line in open(csv_file, 'rU'):
        temp_list = [ float(x) for x in line.split(',') ] 
        if len(temp_list) < 1: 
            continue 
        if labeled: 
            matrix.append(temp_list[:-1]) 
            labels.append(int(temp_list[-1]))
        else:
            matrix.append(temp_list)

    matrix = array(matrix)
    return labels, matrix 

def save_model_predictions(predictions, filename):
    """ Save model predictions to an output csv """ 
    output = open(filename, 'w')
    output.write('Id,Prediction\n')
    for i in xrange(len(predictions)):
        output.write('%d,%d\n' % (i+1, predictions[i]))
    output.close()
