#Prove02
import sys
import math
import operator
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()

class KNearestModel:
    def __init__(self, d_train, t_train):
        self.datatrain = d_train
        self.targetstrain = t_train

    def predict(self, datatest, k):

    def getDistance(self, sPoint, ePoint, length):

    def nearest(self, point, k):

class KNearestClassifier:
    def fit(self, x_train, y_train):
        model = KNearestModel(x_train, y_train)
        return model

def getPercentage(predicted, y_test):
	count = 0
	successes = 0
	for x in predicted:
		if x == y_test[count]:
			successes = successes + 1
		count = count + 1
	print float('{0:.2f}'.format((float(successes)/len(y_test)) * 100)), "% correct (", successes,'/',len(y_test),")"


def knn(k, iris,):


classifier = KNeighborsClassifier(n_neighbors=3)
model = classifier.fit(train_data, train_target)
predictions = model.predict(test_data)
