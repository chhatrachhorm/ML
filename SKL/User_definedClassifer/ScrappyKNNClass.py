import random
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from scipy.spatial import distance


# random choice


class ScrappyKNN():
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        predicts = []
        for _ in x_test:
            label = random.choice(self.y_train)
            predicts.append(label)
        return predicts


# kNN = kth nearest neighbor
def euc(a, b):
    return distance.euclidean(a, b)


class KnnAlgorithm():
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        predictions = []
        for row in x_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_distance = euc(row, self.x_train[0])
        best_index = 0
        # hard code k = 1
        for i in range(1, len(self.x_train)):
            dist = euc(row, self.x_train[i])
            if dist < best_distance:
                best_distance = dist
                best_index = i
        return self.y_train[best_index]


# split test data
iris = datasets.load_iris()
x = iris.data
y = iris.target

X_test, X_train, Y_test, Y_train = train_test_split(
    x, y, train_size=.5
)

# tree classifier
tree_clf = tree.DecisionTreeClassifier()
tree_clf.fit(X_train, Y_train)
print(accuracy_score(Y_test, tree_clf.predict(X_test)))

# user_defined classifier
srcappy = ScrappyKNN()
srcappy.fit(X_train, Y_train)
print(accuracy_score(Y_test, srcappy.predict(X_test)))

# kNN classifier
knn = KnnAlgorithm()
knn.fit(X_train, Y_train)
print(accuracy_score(Y_test, knn.predict(X_test)))
