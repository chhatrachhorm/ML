from sklearn import datasets
from sklearn import svm
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
digits = datasets.load_digits()

clf = svm.SVC(gamma=0.001, C=100.)
# [:-1] all except the last one
# [-1:] take the last one only
clf.fit(digits.data[:-1], digits.target[:-1])
print(accuracy_score(digits.target[-1:], clf.predict(digits.data[-1:])))
