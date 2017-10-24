from sklearn import datasets
from sklearn import svm
from sklearn.externals import joblib

iris = datasets.load_iris()
X, y = iris.data[:-1], iris.target[:-1]
clf = svm.SVC(X, y)
joblib.dump(clf, "iris.pkl")


