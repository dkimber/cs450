import sys
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    main(sys.argv)

class HardCodedModel:
    def __init__(self, d_train, t_train):
        self.datatrain = datatrain
		self.targetstrain = targetstrain

    def predict(self, datatest):
        target = []
        for x in datatest:
            target.append(0)
        return target

class HardCodedClassifier:
    def fit(self, x_train, y_train):
        model = HardCodedModel(xtrain, ytrain)
		return model

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
iris = datasets.load_iris()

# Show the data (the attributes of each instance)
print(iris.data)

# Show the target values (in numeric format) of each instance
print(iris.target)

# Show the actual target names that correspond to each number
print(iris.target_names)


classifier = GaussianNB()
model = classifier.fit(data_train, targets_train)
targets_predicted = model.predict(data_test)
