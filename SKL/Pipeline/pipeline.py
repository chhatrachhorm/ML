from sklearn import datasets

iris = datasets.load_iris()

# f(x) = y
x = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=.5
    #     half of data will be used for testing
)
# Classifier 1 : tree
from sklearn import tree
tree_clf = tree.DecisionTreeClassifier()
tree_clf.fit(X_train, Y_train)

tree_predictions = tree_clf.predict(X_test)
print(tree_predictions)

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, tree_predictions))

# Classifier 2 : KNeightborClassifier
from sklearn.neighbors import KNeighborsClassifier
kn_clf = KNeighborsClassifier()
kn_clf.fit(X_train, Y_train)
print(accuracy_score(Y_test, kn_clf.predict(X_test)))
