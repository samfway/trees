#!/usr/bin/env python
import argparse
from tree_lib.parse import parse_csv_columns, load_model_predictions
from ml_utils.plot_confusion_matrix import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

labels = ['Spruce/Fir', 
          'Lodgepole Pine',
          'Ponderosa Pine',
          'Cottonwood/Willow',
          'Aspen',
          'Douglas-fir',
          'Krummholz']

def interface():
    args = argparse.ArgumentParser()
    args.add_argument('-t', '--true-matrix-file', help='True matrix', required=True)
    args.add_argument('-p', '--predictions-file', help='Model predictions', required=True)
    args.add_argument('-o', '--output-file', help='Output image', default='cm.pdf')
    args = args.parse_args()
    return args

if __name__=="__main__":
    args = interface()
    true_labels, matrix = parse_csv_columns(args.true_matrix_file, labeled=True)
    predictions = load_model_predictions(args.predictions_file)
    conf_matrix = confusion_matrix(true_labels, predictions)
    labels = [ str(k+1) for k in xrange(len(conf_matrix)) ]
    plot_confusion_matrix(conf_matrix, labels, args.output_file)

