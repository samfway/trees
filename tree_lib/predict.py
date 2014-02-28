#!/usr/bin/env python

__author__ = "Sam Way"
__credits__ = ["Sam Way"]
__license__ = "GPL"
__version__ = "0.0.1-dev"
__maintainer__ = "Sam Way"
__email__ = "samfway@gmail.com"
__status__ = "Development"

from .parse import parse_csv_columns
from sklearn import grid_search
from sklearn.svm import SVC
import pickle

def load_models(model_pickle_file):
    """ Retrieve models from a file """ 
    return pickle.load(open(model_pickle_file, "rb"))

def build_models(training_file, output_file=None):
    """ Build the models desired for testing """ 
    models = [] 
    labels, matrix = parse_csv_columns(training_file, labeled=True)
    
    # 1 - SVM 
    #svm_params = {'kernel':('linear', 'rbf'), 'C':[1, 5, 10]}
    #clf = grid_search.GridSearchCV(SVC(C=1), svm_params, cv=5)
    clf = SVC(kernel='rbf')
    clf.fit(matrix, labels)
    models.append(clf)
    
    if output_file is not None:
        pickle.dump(models, open(output_file, "wb"))
    
    return models

