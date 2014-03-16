#!/usr/bin/env python

__author__ = "Sam Way"
__credits__ = ["Sam Way"]
__license__ = "GPL"
__version__ = "0.0.1-dev"
__maintainer__ = "Sam Way"
__email__ = "samfway@gmail.com"
__status__ = "Development"

from numpy import array, zeros, sqrt, arange, bincount, log2
from numpy.random import choice, randint, seed, random, randn
from random import getrandbits
from scipy.stats import mode

def coin_flip():
    return (random > 0.5)

def entropy(x):
    probs = bincount(x)/float(len(x))
    return sum([-p*log2(p) if p > 0 else 0 for p in probs])

class decision_tree_node:
    def __init__(self):
        self.feature_index = -1
        self.feature_threshold = -1
        self.min_split = 2
        self.left = None
        self.right = None
        self.pick = None 

    #@profile
    def grow(self, matrix, labels, max_features, threshold_tries=1):
        num_elements = matrix.shape[0]
        num_unique = len(set(labels))

        if num_elements < self.min_split or num_unique == 1:
            items, counts = mode(labels)
            self.pick = items[0]
            return 

        num_features = matrix.shape[1]
        feature_subset = choice(num_features, max_features, replace=False)

        original_entropy = entropy(labels)
        best_feature = -1
        best_threshold = -1
        best_gain = -1
        best_left = []
        best_right = []
        best_len_left = -1
        best_len_right = -1
    
        for feature in feature_subset:
            #feature_min = min(matrix[:, feature])
            #feature_max = max(matrix[:, feature])
            #if feature_min == feature_max: continue 

            # Pick thresholds uniformly from the interval 
            thresholds = random(threshold_tries) # assumes 0-1 range applied
            #thresholds = random(threshold_tries)*(feature_max-feature_min) + feature_min
            
            for random_threshold in thresholds:
                # Pick a random threshold for the split 
                left = [] 
                right = []
                for k in xrange(num_elements):
                    if matrix[k][feature] <= random_threshold:
                        left.append(k)
                    else:
                        right.append(k) 
                entropy_left = entropy(labels[left])
                info_gain = 2*(original_entropy - entropy_left) / (original_entropy + entropy_left) 

                len_left = len(left)
                len_right = len(right)
                if info_gain > best_gain and len_left > self.min_split and len_right > self.min_split: 
                    best_gain = info_gain
                    best_feature = feature
                    best_threshold = random_threshold
                    best_left = left[:]
                    best_right = right[:]
                    best_len_left = len_left
                    best_len_right = len_right

        if best_len_left < self.min_split or best_len_right < self.min_split:
            # Give up... couldn't split the data
            items, counts = mode(labels)
            self.pick = items[0]
            return 

        self.feature_index = best_feature
        self.feature_threshold = best_threshold
        self.left = decision_tree_node()
        self.right = decision_tree_node()
        self.left.grow(matrix[best_left], labels[best_left], max_features, threshold_tries)
        self.right.grow(matrix[best_right], labels[best_right], max_features, threshold_tries)  

class random_decision_tree:
    def __init__(self, max_features=10):
        self.root = None
        self.max_features = max_features
    
    def fit(self, matrix, labels):
        self.root = decision_tree_node()
        self.root.grow(matrix, labels, self.max_features)

    def predict(self, row):
        current_node = self.root
        while current_node.pick is None:
            if row[current_node.feature_index] <= current_node.feature_threshold:
                current_node = current_node.left
            else:
                current_node = current_node.right
        return current_node.pick

class extreme_tree_classifier:
    def __init__(self, n_estimators=50, max_features=10):
        self.trees = [random_decision_tree(max_features) for n in xrange(n_estimators)]

    def fit(self, matrix, labels):
        M = matrix.shape[0] # Number of examples
        all_samples = arange(M)
        self.num_samples = int(M/7)

        for tree in self.trees:
            sample_subset = choice(all_samples, self.num_samples, replace=False)
            sub_matrix = matrix[sample_subset]
            sub_labels = labels[sample_subset]
            tree.fit(sub_matrix, sub_labels)

    def predict(self, matrix):
        M = matrix.shape[0] # Number of examples
        predictions = zeros(M)
        for idx, row in enumerate(matrix):
            item, count = mode(array([ tree.predict(row) for tree in self.trees ]))
            predictions[idx] = int(item[0])
        return predictions

