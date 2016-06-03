import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from scipy.interpolate import spline
from scipy import interpolate
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


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
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import MultinomialNB



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
    """Container object for datasets

    Dictionary-like object that exposes its keys as attributes.

    >>> b = Bunch(a=1, b=2)
    >>> b['b']
    2
    >>> b.b
    2
    >>> b.a = 3
    >>> b['a']
    3
    >>> b.c = 6
    >>> b['c']
    6

    """

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
	with open(join(module_path, 'data', 'train.csv')) as csv_file:
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

def stratify_and_shuffle(X,y,num_train, num_test):
    sss = StratifiedShuffleSplit(y, 10, train_size=num_train, test_size = num_test)
    for train_index, test_index in sss:
        X_train = X[train_index]
        X_test = X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
    return X_train, X_test, y_train, y_test


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


	# Plot number of features VS. cross-validation scores
	plt.figure()
	plt.xlabel("Number of features selected")
	plt.ylabel("Cross validation score (nb of correct classifications)")
	plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
	plt.show()
	
	return rfecv
	
def doGridSearch(X,y,myX_test, myy_test, estimator, param_grid, cv):
	
	print "Num folds: ", cv
	
	grid = GridSearchCV(estimator, param_grid = param_grid, cv=cv)
	#print str(len(x_train))
	#print str(y_train)
	
	
	grid.fit(X, y)
	print "grid score "  +  str(grid.score(myX_test, myy_test))

	y_pred = grid.predict(myX_test)
	#y_pred_roc =  grid.predict_proba(myX_test)

	print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

	
	pickle.dump(grid, open("SVM_grid.pkl", "wb"))

	return grid.best_params_

	
def plot_roc(X, y, X_test, y_test, estimator, cv, print_outputs = False):

	estimator.fit(X, y)
	y_score = estimator.predict(X_test)#.decision_function(X_test)
	
		
	print y_test, y_score
	false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_score)

	roc_auc = auc(false_positive_rate, true_positive_rate)
	
	if (print_outputs == True):

		precision_score = metrics.precision_score(y_test, y_score)
		recall_score =   metrics.recall_score(y_test, y_score)
		print "Precision score:\t " + str(precision_score)
		print "Recall score:\t " + str(recall_score)
		print "F1 score:\t " + str(metrics.f1_score(y_test, y_score))
		print "AUC:\t "  + str(roc_auc)
		print "false positivie:\t " + str(false_positive_rate)
		print "NPP"
		true_negatives=0	
		total_negatives=0
		for  index,y in enumerate(y_score):
			if y==0 and y_test[index]==0:
				true_negatives+=1
				total_negatives+=1
			elif y_test[index]==0:			
				total_negatives+=1
		print 	( float(true_negatives)/total_negatives)
		print "F2 score:" + str(  float (5* precision_score * recall_score) / (4*precision_score+ recall_score))	
	
		#print "true positive rate:\t " +  str(true_positive_rate)
		#print "thresholds:\t " + str(thresholds)

	plt.title('Receiver Operating Characteristic')
	plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
	plt.legend(loc='lower right')
	plt.plot([0,1],[0,1],'r--')
	plt.xlim([-0.1,1.2])
	plt.ylim([-0.1,1.2])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.show()
	
	

def doTreeFeatureSelection(estimator, X, y):
	clf = ExtraTreesClassifier()
	clf = clf.fit(X, y)
		
	#print str(clf.feature_importances_)	
	model =  SelectFromModel(clf, prefit=True)
	return model


def doPCA(X):
	pca = PCA()
	pca.fit(X)
	return pca




##############################################################################
# Load and scale data
##############################################################################
dataset = load_crawl() 

data = dataset.data
target = dataset.target

scaler = StandardScaler() 
scaled_data = scaler.fit_transform(data)

#normalizer = Normalizer() 			
#scaled_data = normalizer.transform(data)

#min_max_scaler = MinMaxScaler()
#scaled_data = min_max_scaler.fit_transform(data)

print "Means scaled data: ", scaled_data.mean(axis= 0)
print "STD scaled data: ", scaled_data.std(axis=0)

##############################################################################
# Train classifiers
#
# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.


#####################################################
#SVM

C_range = np.logspace(start=-5,stop= 15,num = 20, base =2)
gamma_range = np.logspace(start=-5,stop= 15,num = 20, base = 2)
 
param_grid_svm = dict(gamma=gamma_range, C=C_range)

nTrees = range(5,30)
param_grid_rf = dict(n_estimators = nTrees)
param_grid_adaboost = dict(n_estimators = range(25,75), learning_rate=[0.5, 1, 1.5])


X_train, X_test, y_train, y_test = stratify_and_shuffle(scaled_data, target, 0.75,0.25)

print "Training set: {} samples".format(X_train.shape[0])
print "Test set: {} samples".format(X_test.shape[0])


"""
cv = StratifiedShuffleSplit(target, n_iter=10, test_size=0.25, random_state=42)
"""



tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 1000]}]
                    #{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

"""
#get the benign samples
X_benign=[]
y_benign=[]
for i in range(1,len(X_train)):
	if y_train[i]==0:
		X_benign.append(X_train[i])
		y_benign.append(y_train[i])

"""
#do feature selection

#out =  doTreeFeatureSelection(SVC(), X_train, y_train)
verbose= True


#US = UnderSampler(verbose=True)
US= TomekLinks(verbose=True)
#US = ClusterCentroids(verbose=True)
#US = NearMiss(version=1, verbose=verbose)
#US = OneSidedSelection(verbose=verbose)
#US = EasyEnsemble(verbose=verbose)
#US = NeighbourhoodCleaningRule(verbose=verbose)

#svm_args={'class_weight': 'auto'}
#US = SMOTE(kind='svm', verbose=verbose, **svm_args)
usx, usy =  US.fit_transform((X_train), y_train)





print "Size pf over sampled samples"  + str(usx.shape)

"""
X_mal=[]
y_mal=[]
for i in range(1,len(X_train)):
	if y_train[i]==1:
		X_mal.append(X_train[i])
		y_mal.append(y_train[i])

X_all=[]
y_all=[]
for i in range(1,len(X_mal)):
	X_all.append(X_mal[i])
	y_all.appned(y_mal[i])
for i in range(1,len(usx)):
	X_all.append(usx[i])
	y_all.appned(usy[i])
"""



	

#print str(doTreeFeatureSelection(SVC(), X_train, y_train).transform().shape)


#out = doRFEVC(SVC(kernel= 'linear'), X_train, y_train)
#out = doPCA(X_train)
#out = doVarianceThreshold(X_train, 1)

#svc = SVC()
#svc.fit(out.transform(X_train), y_train)

#X_test= scaler.transform(X_test)

#this is the transformed training data


#print "initial features"  + str(X_train.shape)
#X_train_new = out.transform(X_train)
X_train_new = usx


#print "after_features"  + str(X_train_new.shape)
#this is the transformed testing data
#X_test_new =  out.transform(X_test)
X_test_new =  (X_test)


f =open("scale.pkl","w")
pickle.dump(scaler,f)
f.close()


"""
f =open("feature.pkl","w")
pickle.dump(out,f)
f.close()
"""

#svm
# Option 1: run a new grid search
#best_params = doGridSearch( X_train_new, usy,X_test_new, y_test,  SVC(), param_grid_svm, 10)

# Option 2: load an old grid search
#grid = pickle.load(open("SVM_grid.pkl", "rb"))
#best_params = grid.best_params_

#tunedSVC = SVC(C = best_params["C"], gamma = best_params["gamma"])
#plot_roc(X_train_new, usy, X_test_new, y_test, tunedSVC, 2, print_outputs = True)

#random forest
#best_params = doGridSearch( X_train_new, usy,X_test_new, y_test,  RandomForestClassifier(), param_grid_rf, 10)
#tunedRFC = RandomForestClassifier(n_estimators = best_params["n_estimators"])
#plot_roc(X_train_new, usy, X_test_new, y_test, tunedRFC, 2, print_outputs = True)

#adaboost
#best_params = doGridSearch( X_train_new, usy,X_test_new, y_test,  sklearn.ensemble.AdaBoostClassifier(), param_grid_adaboost, 10)
#tunedABC = sklearn.ensemble.AdaBoostClassifier(n_estimators = best_params["n_estimators"], learning_rate= best_params["learning_rate"])
#plot_roc(X_train_new, usy, X_test_new, y_test, tunedABC, 2, print_outputs = True)

# Gaussiannaive bayes
#param_grid_nb = dict()
#best_params = doGridSearch( X_train_new, usy,X_test_new, y_test,  GaussianNB(), param_grid_nb, 10)
#tunedNB = GaussianNB()
#plot_roc(X_train_new, usy, X_test_new, y_test, tunedNB, 2, print_outputs = True)

#multinomial naive bayes
#param_grid_nb = dict(alpha = range(1,10), fit_prior =[True, False])
#best_params = doGridSearch( X_train_new, usy,X_test_new, y_test,  MultinomialNB(), param_grid_nb, 10)
#tunedNB = MultinomialNB(alpha = best_params["alpha"], fit_prior = best_params["fit_prior"])
#plot_roc(X_train_new, usy, X_test_new, y_test, tunedNB, 2, print_outputs = True)


#decision tree
param_grid_DT = dict(max_features=["auto","sqrt","log2",None],class_weight=["balanced", None] , presort=[True, False])
best_params = doGridSearch( X_train_new, usy,X_test_new, y_test,  tree.DecisionTreeClassifier(), param_grid_DT, 10)
tunedDT = tree.DecisionTreeClassifier(max_features =best_params["max_features"], class_weight= best_params["class_weight"] , presort = best_params["presort"])
plot_roc(X_train_new, usy, X_test_new, y_test, tunedDT, 2, print_outputs = True)








print "Best  params" 
for key in best_params.keys():
	print key +" = " + str(best_params[key])


#load_and_test()
#print "svc testing score` "  +  str(svc.score(X_test_new, y_test))



#print str(out.score(X_test, y_test))
#print str(doKBest(X_train, y_train, 5).shape)   			#does not work for negative values
#print str(doSelectFpr(X_train, y_train, 0.5).shape)			#does not work for negative values
#doRFEVC(SVC(kernel = 'linear'), X_train, y_train)


#doGridSearch(X_train, y_train, SVC(), param_grid, 10)


#print str(doPCA(X_train).explained_variance_ratio_)


#svc = SVC(kernel = 'linear')
#svc.fit(X_train, y_train)

#print "svc testing score` "  +  str(svc.score(X_test, y_test))

###f =open("trainedSvm.pkl","w")
###pickle.dump(svc,f)
###f.close()



#do feature selection


"""
svmScores = cross_validation.cross_val_score(svc, scaled_data, target, cv = cv)
print "Cross validated scores:", svmScores
print 'Average:', sum(svmScores)/10
"""
#Decision Trees
clf = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, class_weight=None, presort=False)


#Random forest
clf= sklearn.ensemble.RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)


#Ada Boost
clf =sklearn.ensemble.AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None)

#Multinomial naive baiyes
#clf  = sklearn.naive_bayes.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)






