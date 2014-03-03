#!/usr/bin/env python
import argparse
from sklearn.metrics import accuracy_score
from tree_lib.parse import parse_csv_columns, save_model_predictions
from tree_lib.predict import load_pickle

def interface():
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input-file', help='Input data matrix', required=True)
    args.add_argument('-m', '--models-file', help='Input models file (.pkl)', required=True)
    args.add_argument('-s', '--scale-file', help='Scaling file (.pkl)', default='scale.pkl', required=True)
    args.add_argument('-o', '--output-prefix', help='Output file prefix', default='predictions_')
    args.add_argument('--validation', help='Input is labeled', action='store_true', default=False)
    args = args.parse_args()
    return args

if __name__=="__main__":
    args = interface()
    labels, matrix = parse_csv_columns(args.input_file, labeled=args.validation)
    models = load_pickle(args.models_file)
    scaler = load_pickle(args.scale_file)
    matrix = scaler.transform(matrix)

    for model_name, model in models:
        predictions = model.predict(matrix)
        if args.validation:
            #print model.best_params_
            print accuracy_score(labels, predictions)
        save_model_predictions(predictions, args.output_prefix + model_name + '.csv') 
