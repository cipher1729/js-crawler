import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
import sklearn.preprocessing
from sklearn.datasets import load_iris
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import cross_validation
from sklearn.feature_selection import *
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn import tree
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from unbalanced_dataset.under_sampling import UnderSampler
from unbalanced_dataset.under_sampling import TomekLinks
from unbalanced_dataset.under_sampling import ClusterCentroids
from unbalanced_dataset.under_sampling import NearMiss
from unbalanced_dataset.under_sampling import CondensedNearestNeighbour
from unbalanced_dataset.under_sampling import OneSidedSelection
from unbalanced_dataset.under_sampling import NeighbourhoodCleaningRule
from unbalanced_dataset.ensemble_sampling import EasyEnsemble
from unbalanced_dataset.ensemble_sampling import BalanceCascade
from unbalanced_dataset.over_sampling import OverSampler
from unbalanced_dataset.over_sampling import SMOTE
from unbalanced_dataset.pipeline import SMOTETomek

from sklearn.naive_bayes import GaussianNB



#

import os
import csv
import sys
import shutil
import pickle
from os import environ
from os.path import dirname
from os.path import join
from os.path import exists
from os.path import expanduser
from os.path import isdir
from os.path import splitext
from os import listdir
from os import makedirs

## end

grid = None
n_samples= None
n_features= None
class Bunch(dict):
    
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        # Bunch pickles generated with scikit-learn 0.16.* have an non
        # empty __dict__. This causes a surprising behaviour when
        # loading these pickles scikit-learn 0.17: reading bunch.key
        # uses __dict__ but assigning to bunch.key use __setattr__ and
        # only changes bunch['key']. More details can be found at:
        # https://github.com/scikit-learn/scikit-learn/issues/6196.
        # Overriding __setstate__ to be a noop has the effect of
        # ignoring the pickled __dict__
        pass

#for loading and testing in the same file
def load_and_test(grid):
	with open("data/external_test.csv") as csv_file:
		data_file = csv.reader(csv_file)
		temp = next(data_file)
		n_samples = int(temp[0])
		n_features = int(temp[1])
		target_names = np.array(temp[2:4])
		data = np.empty((n_samples, n_features))
		target = np.empty((n_samples,), dtype=np.int)

		for count, value in enumerate(data_file):
			data[count] = np.asarray(value[:-1], dtype=np.float)
			target[count] = np.asarray(value[-1], dtype=np.int)
			#print "data is " + str(data[count])
			#print "target is " + str(target[count])
		print "Number of target records is  " + str(len(target))

		scaled_data = scaler.transform(data)
		X_test_new = out.transform(scaled_data)
		print str(grid.score(X_test_new, target))
		

#for loading data from the training dataset
def load_crawl():
	"""Load and return the malicious scripts dataset (classification).
	Returns
	-------
	data : Bunch
	Dictionary-like object, the interesting attributes are:
	'data', the data to learn, 'target', the classification labels,
	'target_names', the meaning of the labels, 'feature_names', the
	meaning of the features, and 'DESCR', the
	full description of the dataset.

	Examples
	--------
	Let's say you are interested in the samples 10, 25, and 50, and want to
	know their class name.

	>>> from sklearn.datasets import load_iris
	>>> data = load_iris()
	>>> data.target[[10, 25, 50]]
	array([0, 0, 1])
	>>> list(data.target_names)
	['setosa', 'versicolor', 'virginica']
	"""

	module_path = dirname(__file__)
	with open(join(module_path, 'data', 'train2.csv')) as csv_file:
		data_file = csv.reader(csv_file)
		temp = next(data_file)
		global n_samples
		n_samples = int(temp[0])
		global n_features
		n_features = int(temp[1])
		print "n samples " + str((n_samples))
		print "n_features"  + str((n_features))
		target_names = np.array(temp[2:4])
		data = np.empty((n_samples, n_features))
		target = np.empty((n_samples,), dtype=np.int)

		for count, value in enumerate(data_file):
			data[count] = np.asarray(value[:-1], dtype=np.float)
			target[count] = np.asarray(value[-1], dtype=np.int)
			#print "data is " + str(data[count])
			#print "target is " + str(target[count])
		print "Number of target records is  " + str(len(target))
	#with open(join(module_path, 'descr', 'train.rst')) as rst_file:
	#	fdescr = rst_file.read()

	return Bunch(data=data, target=target,
			 target_names=target_names,
			 DESCR=None,
			 feature_names = ['evalCount', 'setInterval', 'setTimeout', 'link', 
							  'search', 'exec','escape', 'unescape', 'ratio', 
							  'emtropyAvg', 'entropyScript', 'longStrings', 
							  'maxEntropy', 'stringAvg', 'maxLength', 'longVarFunc', 
							  'stringAssignments', 'stringModFuncsCount', 'eventFuncsCount', 
							  'domModFuncsCounter', 'suspStrings', 'whiteSpaceRatio', 
							  'hexaStrings', 'maxNonPrintableCharactersinString', 'lineAvg', 
							  'iframeCount', 'malTagCount', 'jsLength'])

#stratify shuffle split
def stratify_and_shuffle(X,y,num_train, num_test):
    sss = StratifiedShuffleSplit(y, 10, train_size=num_train, test_size = num_test)
    for train_index, test_index in sss:
        X_train = X[train_index]
        X_test = X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
    return X_train, X_test, y_train, y_test

#variance thresholding on features
def doVarianceThreshold(X, threshold):
	sel= VarianceThreshold(threshold = threshold)
	print "Most important features"  + str((sel.fit(X, threshold)))
	print "best features are" + str(sel.get_support(True))

	#print  "optimal features"  + str(sel.n_features_)
	return sel

def doKBest(X, y, k):
	#do K best feature selection using the chi squared stats
	fitted = SelectKBest(chi2, k=k)	
	print "Most important features"  + str(fitted.fit(X,y))
	return 	fitted.transform(X,y)

def doKBestPercentilet(X, y, percentile):
	#do K best feature selection using the chi squared stats
	fitted = SelectPercentile(chi2, percentile =percentile)	
	print "Most important features"  + str(fitted.fit(X,y))
	return 	fitted.transform(X,y)

def doSelectFpr(X, y, alpha):
	#do K best feature selection using the chi squared stats
	fitted = SelectFpr(chi2, alpha = alpha)	
	print "Most important features"  + str(fitted.fit(X,y))
	return 	fitted.transform(X,y)

def doRFEVC(estimator, X, y):
	rfecv = RFECV(estimator, step=1, cv=2,scoring='accuracy') 
	#print  "Most important features " +  str(rfecv.fit(X, y))
	rfecv.fit(X,y)
	print("Optimal number of features : %d" % rfecv.n_features_)

	"""
	# Plot number of features VS. cross-validation scores
	plt.figure()
	plt.xlabel("Number of features selected")
	plt.ylabel("Cross validation score (nb of correct classifications)")
	plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
	plt.show()
	"""
	
	return rfecv
	
def doGridSearch(X,y,myX_test, myy_test, estimator, param_grid, cv):
	
	
	grid = GridSearchCV(estimator, param_grid = param_grid, cv=cv)
	#print str(len(x_train))
	#print str(y_train)
	
	
	grid.fit(X, y)
	print "grid score "  +  str(grid.score(myX_test, myy_test))


	
	y_pred = grid.predict(myX_test)
	#y_pred_roc =  grid.predict_proba(myX_test)

	y_true = myy_test
	
	print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

	
	f =open("trainedSvm.pkl","w")
	pickle.dump(grid,f)
	f.close()

	#load_and_test(grid)


def doTreeFeatureSelection(estimator, X, y):
	clf = ExtraTreesClassifier()
	clf = clf.fit(X, y)
	#print "Features before fitting " + str(clf.n_features_)
	#print "Features after fitting " + str(clf.n_outputs_)
	
	print "Feature ranking "
	print str(100000* clf.feature_importances_)	
	imp = 100000* clf.feature_importances_
	imp.sort()
	print
	#print imp	


	model =  SelectFromModel(clf, prefit=True)
	return model


def doPCA(X):
	pca = PCA()
	pca.fit(X)
	return pca


#print "GRID CV DONE !"


#read the data from the file    
dataset = load_crawl() 

data = dataset.data
target = dataset.target

#apply standard scaler on it
scaler = StandardScaler() 
scaled_data = scaler.fit_transform(data) 	#C = 1000, gamma = 1.0
print "Standard Scaler"


#print "mean:", scaled_data.mean(axis= 0)
#print "std:", scaled_data.std(axis=0)

##############################################################################
# Train classifiers
#####################################################


#parameters for SVM
C_range = np.logspace(start=-5,stop= 15,num = 20, base =2)
gamma_range = np.logspace(start=-5,stop= 3,num = 20, base = 2)
param_grid_svm = dict(gamma=gamma_range, C=C_range)


#parameters for random forest
nTrees = range(5,30)
param_grid_rf = dict(n_estimators = nTrees)

#parameters for adaboost
param_grid_adaboost = dict(n_estimators = range(25,75), learning_rate=[0.5, 1, 1.5])

#parameters for decision tree
param_grid_DT = dict(max_features=["auto","sqrt","log2",None],class_weight=["balanced", None] , presort=[True, False])




#stratified shuffle split
X_train, X_test, y_train, y_test = stratify_and_shuffle(scaled_data, target, 0.99,0.01)


verbose= True


#undersample the scaled data
#US = UnderSampler(verbose=True)
US= TomekLinks(verbose=True)
#US = ClusterCentroids(verbose=True)
#US = NearMiss(version=1, verbose=verbose)
#US = OneSidedSelection(verbose=verbose)
#US = EasyEnsemble(verbose=verbose)
#US = NeighbourhoodCleaningRule(verbose=verbose)
#US = BalanceCascade(verbose=verbose)
#svm_args={'class_weight': 'auto'}
#US = SMOTE(kind='svm', verbose=verbose, **svm_args)
##US = SMOTE(kind='regular', verbose=verbose)
#US = SMOTETomek(verbose =verbose)
usx, usy =  US.fit_transform((X_train), y_train)
#usx = X_train
#usy = y_train


X_train_new = usx
#X_test_new =  out.transform(X_test)
X_test_new =  (X_test)
	
#feature selection
#print str(doTreeFeatureSelection(SVC(), X_train, y_train).transform().shape)
#out = doRFEVC(SVC(kernel= 'linear'), X_train, y_train)
#out = doPCA(X_train)
#out = doVarianceThreshold(X_train, 1)
#doTreeFeatureSelection(SVC(), X_train_new, usy)


#write the picked scaling object todisk
f =open("scale.pkl","w")
pickle.dump(scaler,f)
f.close()


#perform grid search
doGridSearch( X_train_new, usy,X_test_new, y_test, SVC(), param_grid_svm, 10)


