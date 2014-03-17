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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import TruncatedSVD, NMF, RandomizedPCA
from sklearn.feature_selection import RFE
from numpy import arange
import cPickle as pickle

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
    #scaler = preprocessing.StandardScaler()
    scaler = preprocessing.MinMaxScaler()
    #matrix = scaler.fit_transform(matrix)
    # Neither NMF/kSVD seemed to work well
    #redux = RandomizedPCA() #NMF(7) #TruncatedSVD(35) 
    #matrix = redux.fit_transform(matrix)
    #data_prep = Pipeline([('scaling', scaler), ('dim_redux', redux)])
    #extract = RFE()
    #matrix = extract.fit_transform(matrix)
    #data_prep = FeatureUnion([('scaling', scaler), ('rfe', extract)])
    data_prep = FeatureUnion([('scaling', scaler)])
    matrix = data_prep.fit_transform(matrix)
 
    # 1 - SVM 
    #svm_params = {'kernel':['rbf', 'poly', 'sigmoid'], 'C':[8, 10, 12]}
    #svm_params = [
    #    {'C': [1, 10, 100, 1000], 'kernel': ['poly']},
    #    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    #]
    #C_range = 10. ** arange(-2, 9)
    #gamma_range = 10. ** arange(-5, 4)
    #svm_params = dict(gamma=gamma_range, C=C_range)
    #clf = grid_search.GridSearchCV(SVC(), svm_params, cv=5)

    # 77.7
    #clf = SVC(kernel='rbf', C=50, gamma=0.01)
    #clf.fit(matrix, labels)
    #models.append( ('svm',clf) )

    # 83.98 ()
    # 86.87 (n_estimators=100)
    # 87.09 (n_estimators=1000)
    # 87.54 (n_estimators=100, criterion='entropy')
    # 87.86 ((n_estimators=1000, criterion='entropy')
    # 89.11 (n_estimators=100, criterion='entropy', max_features=9, bootstrap=False)
    # 89.64 (n_estimators=1000, criterion='entropy', max_features=9, bootstrap=False)
    # 89.07 (n_estimators=50, criterion='entropy', max_features=9, bootstrap=False)
    #clf = RandomForestClassifier(n_estimators=50, criterion='entropy', max_features=9, bootstrap=False)
    #clf.fit(matrix, labels)
    #models.append( ('rf50',clf) )

    # Extra trees 
    # 90.8 (n_estimators=1000, criterion='entropy', max_features=9, bootstrap=False)
    # 90.73 (n_estimators=2000, criterion='entropy', max_features=9, bootstrap=False)
    # 90.78 (n_estimators=500, criterion='entropy', max_features=9, bootstrap=False)
    # 90.87 (n_estimators=1000, criterion='entropy', max_features=10, bootstrap=False)
    # 90.79 (n_estimators=1000, criterion='entropy', max_features=15, bootstrap=False)
    clf = ExtraTreesClassifier(n_estimators=100, criterion='entropy', max_features=10, bootstrap=False)
    clf.fit(matrix, labels)
    models.append( ('ext1k',clf) )

    # 83.38 (weights='distance')
    #clf = KNeighborsClassifier(weights='distance')
    #clf.fit(matrix, labels)
    #models.append( ('knn5',clf) )

    # 60.12
    #clf = MultinomialNB(class_prior=priors, alpha=10)
    #clf.fit(abs(matrix), labels)
    #models.append( ('MNB', clf) )

    # 85.07  AdaBoostClassifier(RandomForestClassifier(n_estimators=10, criterion='entropy'), n_estimators=10)
    # 87.73  AdaBoostClassifier(RandomForestClassifier(n_estimators=100, criterion='entropy'), n_estimators=100)
    # 87.7  AdaBoostClassifier(RandomForestClassifier(n_estimators=100, criterion='entropy'), n_estimators=1000)
    # 87.91 AdaBoostClassifier(RandomForestClassifier(n_estimators=1000, criterion='entropy'), n_estimators=100)
    # 89.24 AdaBoostClassifier(RandomForestClassifier(n_estimators=100, criterion='entropy', max_features=9, bootstrap=False), n_estimators=10)
    # 89.42 AdaBoostClassifier(RandomForestClassifier(n_estimators=100, criterion='entropy', max_features=9, bootstrap=False), n_estimators=100)
    # 89.41 AdaBoostClassifier(RandomForestClassifier(n_estimators=100, criterion='entropy', max_features=9, bootstrap=False), n_estimators=50, learning_rate=0.1)
    # 89.47 AdaBoostClassifier(RandomForestClassifier(n_estimators=500, criterion='entropy', max_features=9, bootstrap=False), n_estimators=50, learning_rate=5)
    # 89.47
    #clf = AdaBoostClassifier(SVC(kernel='linear', C=100, gamma=0.01), learning_rate=10)
    #clf = AdaBoostClassifier(RandomForestClassifier(n_estimators=50, criterion='entropy', max_features=9, bootstrap=False), n_estimators=100, learning_rate=5)
    #clf.fit(matrix, labels)
    #models.append( ('ada', clf) )
    
    if output_file is not None:
        pickle.dump(models, open(output_file, "wb"))

    if scale_file is not None:
        pickle.dump(data_prep, open(scale_file, "wb"))
    
    return models, data_prep 

