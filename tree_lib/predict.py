#!/usr/bin/env python

__author__ = "Sam Way"
__credits__ = ["Sam Way"]
__license__ = "GPL"
__version__ = "0.0.1-dev"
__maintainer__ = "Sam Way"
__email__ = "samfway@gmail.com"
__status__ = "Development"

from .parse import parse_csv_columns
from sklearn import grid_search, preprocessing
from sklearn.svm import SVC
import pickle

def load_pickle(model_pickle_file):
    """ Retrieve models/scaler from a file """ 
    return pickle.load(open(model_pickle_file, "rb"))

def build_models(training_file, output_file=None, scale_file=None):
    """ Build the models desired for testing """ 
    models = [] 
    labels, matrix = parse_csv_columns(training_file, labeled=True)
   
    # Preprocess data first!
    scaler = preprocessing.StandardScaler()
    matrix = scaler.fit_transform(matrix)
 
    # 1 - SVM 
    #svm_params = {'kernel':['rbf'], 'C':[8, 9, 10, 11, 12]}
    #clf = grid_search.GridSearchCV(SVC(), svm_params, cv=5)
    clf = SVC(kernel='rbf', C=10)
    clf.fit(matrix, labels)
    models.append( ('svm',clf) )
    
    if output_file is not None:
        pickle.dump(models, open(output_file, "wb"))

    if scale_file is not None:
        pickle.dump(scaler, open(scale_file, "wb"))
    
    return models, scaler

