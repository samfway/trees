#!/usr/bin/env python
import argparse
from sklearn.metrics import accuracy_score
from tree_lib.parse import parse_csv_columns, save_model_predictions, yield_csv_columns
from tree_lib.predict import load_pickle
from numpy import array

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
    input_csv = yield_csv_columns(args.input_file, labeled=args.validation)
    models = load_pickle(args.models_file)
    scaler = load_pickle(args.scale_file)

    predictions = { model_name:[] for model_name, model in models } 
    labels = [] 
    more_input = True

    while more_input:
        vectors = [] 
        vector_labels = [] 
        for i in xrange(10000):
            try:
                label, vector = input_csv.next()
                vectors.append(vector)
                vector_labels.append(label)
            except:
                more_input = False

        if len(vectors) < 1: break

        print 'Predicting 10k...'
        vectors = array(vectors)
        vectors = scaler.transform(vectors)
        labels += vector_labels

        for model_name, model in models:
            predictions[model_name] += model.predict(vectors).tolist()

    if args.validation:
        for model_name in predictions.keys():
            print model_name, accuracy_score(labels, predictions[model_name])

    for model_name in predictions.keys():
        save_model_predictions(predictions[model_name], args.output_prefix + model_name + '.csv') 
