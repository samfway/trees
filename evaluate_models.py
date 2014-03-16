#!/usr/bin/env python
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from tree_lib.parse import parse_csv_columns
from ml_utils.util import convert_labels_to_int
from ml_utils.evaluation import make_evaluation_report
from ml_utils.cross_validation import get_test_sets

def interface():
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input-file', help='Input data matrix', required=True)
    args.add_argument('-o', '--output-file', help='Report file', default='report.txt')
    args.add_argument('-k', '--k-folds', help='Number of CV folds', default=10, type=int)
    args = args.parse_args()
    return args

if __name__=="__main__":
    args = interface()
    labels, matrix = parse_csv_columns(args.input_file, labeled=True)

    # Preprocess the data
    scaler = StandardScaler()
    matrix = scaler.fit_transform(matrix)
    label_legend, labels = convert_labels_to_int(labels)

    # Load up all desired models
    models =  [] 
    #models.append(('Most Freq', DummyClassifier(strategy='most_frequent')))
    #models.append(('Stratified', DummyClassifier(strategy='stratified')))
    #models.append(('Uniform', DummyClassifier(strategy='uniform')))
    #models.append(('Log. Regr', LogisticRegression()))
    models.append(('Naive Bayes', GaussianNB()))
    #models.append(('SVM (RBF)', SVC(kernel='rbf', C=50, gamma=0.01)))
    #models.append(('Random Forest', RandomForestClassifier(n_estimators=10, \
    #    criterion='entropy', max_features=10, bootstrap=False)))
    models.append(('Extra Trees', ExtraTreesClassifier(n_estimators=10, \
        criterion='entropy', max_features=10, bootstrap=False)))

    # Load up all desired performance metrics
    metrics = []   
    metrics.append(('Accuracy', accuracy_score))

    # Generate CV train/test sets
    test_sets = get_test_sets(labels, kfold=args.k_folds)

    # Perform analysis
    make_evaluation_report(models, matrix, labels, test_sets, metrics, args.output_file)
