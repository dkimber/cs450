import sys
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

class HardCodedModel:
    def __init__(self, d_train, t_train):
        self.datatrain = d_train
        self.targetstrain = t_train

    def predict(self, datatest):
        target = []
        for x in datatest:
            target.append(0)
        return target

class HardCodedClassifier:
    def fit(self, x_train, y_train):
        model = HardCodedModel(x_train, y_train)
        return model

def findPercentage(predicted, y_test):
	count = 0
	successes = 0
	for x in predicted:
		if x == y_test[count]:
			successes = successes + 1
		count = count + 1
	print float('{0:.2f}'.format((float(successes)/len(y_test)) * 100)), "% correct (", successes,'/',len(y_test),")"


x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=42)


gClassifier = GaussianNB()
gModel = gClassifier.fit(x_train, y_train)
gTargets_predicted = gModel.predict(x_test)
print "Gaussian Estimate:"
findPercentage(gTargets_predicted, y_test)

hClassifier = HardCodedClassifier()
hModel = hClassifier.fit(x_train, y_train)
hTargets_predicted = hModel.predict(x_test)
print "My Estimate:"
findPercentage(hTargets_predicted, y_test)
