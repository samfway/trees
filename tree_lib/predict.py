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
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
import pickle

def load_pickle(model_pickle_file):
    """ Retrieve models/scaler from a file """ 
    return pickle.load(open(model_pickle_file, "rb"))

def build_models(training_file, output_file=None, scale_file=None):
    """ Build the models desired for testing """ 
    models = [] 
    labels, matrix = parse_csv_columns(training_file, labeled=True)
  
    priors = [0.365, 0.488, 0.062, 0.005, 0.016, 0.030, 0.035]
    class_w = { i+1:priors[i] for i in xrange(len(priors)) }
 
    # Preprocess data first!
    scaler = preprocessing.StandardScaler()
    matrix = scaler.fit_transform(matrix)
    #redux = TruncatedSVD(25) 
    #matrix = redux.fit_transform(matrix)    
    data_prep = Pipeline([('scaling', scaler)]) #, ('ksvd', redux)])
 
    # 1 - SVM 
    #svm_params = {'kernel':['rbf', 'poly', 'sigmoid'], 'C':[8, 10, 12]}
    #svm_params = [
    #    {'C': [1, 10, 100, 1000], 'kernel': ['poly']},
    #    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    #]
    #clf = grid_search.GridSearchCV(SVC(), svm_params, cv=5)
    clf = SVC(kernel='rbf', C=12) 
    #clf = SVC(kernel='poly', C=100, class_weight='auto')
    clf.fit(matrix, labels)
    models.append( ('svm',clf) )
    
    if output_file is not None:
        pickle.dump(models, open(output_file, "wb"))

    if scale_file is not None:
        pickle.dump(data_prep, open(scale_file, "wb"))
    
    return models, data_prep 

