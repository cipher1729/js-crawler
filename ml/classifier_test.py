import pickle
import csv
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
from sklearn import metrics

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#from matplotlib.colors import Normalize

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
from sklearn.metrics import roc_curve, auc




## end
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

#Read the testing file from disk
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
	with open(sys.argv[1]) as csv_file:
		data_file = csv.reader(csv_file)
		temp = next(data_file)
		n_samples = int(temp[0])
		n_features = int(temp[1])
		target_names = np.array(temp[2:4])
		data = np.empty((n_samples, n_features))
		target = np.empty((n_samples,), dtype=np.int)

		for count, value in enumerate(data_file):
			if len(value)<28:
				continue
			try:
				data[count] = np.asarray(value[:28], dtype=np.float)
				target[count] = np.asarray(value[28], dtype=np.int)	
			except:
				continue
			#print "data is " + str(data[count])
			#print "target is " + str(target[count])

	with open(join(module_path, 'descr', 'crawl.rst')) as rst_file:
		fdescr = rst_file.read()

	return Bunch(data=data, target=target,
			 target_names=target_names,
			 DESCR=fdescr,
			 feature_names = ['evalCount', 'setInterval', 'setTimeout', 'link', 
							  'search', 'exec','escape', 'unescape', 'ratio', 
							  'emtropyAvg', 'entropyScript', 'longStrings', 
							  'maxEntropy', 'stringAvg', 'maxLength', 'longVarFunc', 
							  'stringAssignments', 'stringModFuncsCount', 'eventFuncsCount', 
							  'domModFuncsCounter', 'suspStrings', 'whiteSpaceRatio', 
							  'hexaStrings', 'maxNonPrintableCharactersinString', 'lineAvg', 
							  'iframeCount', 'malTagCount', 'jsLength'])




#read the testing data from disk
dataset = load_crawl()

data = dataset.data


#scaler = StandardScaler() 
#scaled_data = scaler.fit_transform(data) 	#C = 1000, gamma = 1.0


f= open("trainedSvm.pkl","r");
trainedObj = pickle.load(f);
f.close()


#open the scaling object from disk
#read the scaler object
f= open("scale.pkl","r")
scaler = pickle.load(f)


#scale testing data
f.close()
scaled_data = scaler.transform(data)

"""
#read the feature object and apply
f= open("feature.pkl","r")
out = pickle.load(f)
X_test_new = out.transform(scaled_data)
f.close()
"""

X_test_new = scaled_data

#print str(trainedObj.predict(scaled_data))

y_true = dataset.target
#y_pred = trainedObj.predict(X_test_new)
#print str(y_pred)


#predict the results
y_pred = (trainedObj.predict(X_test_new))
#print str(y_pred)


y_score = trainedObj.score(X_test_new, y_true)
print "RUNNING SVM + UNDERSAMPLER\n"
print("Analyzed %d scripts" %(len(y_pred)))
print("%0.2f Correct" %(y_score*100))


csvFile = open(sys.argv[1])
csvFile.readline()
records= csvFile.read().split("\n")

f= open("hashes_mal.txt","w")
f2 = open("hashes_benign.txt", "w")
f3 = open("passedScriptsFile.txt","r")
f4 = open("status.txt","w")

#write prediction correctness of scripts to status.txt
txt =f3.read().split("##########+")
htmlText = ""
#print "no of scripts " + str(len(txt))
#print txt[0]

d=0
for index,y in enumerate(y_pred):
	record = records[d].split(",")
	if len(record) == 31:
		continue
	else:
		if index<=6:
			htmlText += txt[index]
			#print htmlText
			htmlText += "\n"
			htmlText = htmlText+  "Prediction :" + str(y) + " \n"
			htmlText = htmlText + "Actual Label :"+ str(y_true[index]) + "\n"		
			if y == y_true[index]:
				htmlText += "Correct prediction! \n"
			else:
				htmlText += "Incorrect prediction \n"
	d+=1

f4.write(htmlText)
f.close()
f2.close() 
f3.close()
f4.close()

"""
print "Precision score: " + str(metrics.precision_score(y_true, y_pred))
print "Recall score: " + str(metrics.recall_score(y_true, y_pred))
print "F1 score: " + str(metrics.f1_score(y_true, y_pred))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)

print "AUC is"  + str(roc_auc)
print "false positivie " + str(false_positive_rate)
print "true positive rate" +  str(true_positive_rate)


plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

"""

