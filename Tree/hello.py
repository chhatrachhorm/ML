from sklearn import tree
# before starting
# configure pycharm to work with Anaconda first
# change the project interpreter path into Anaconda/python.exe

# feature
# 0 is for smooth, 1 is for bumpy
features = [
    [140, 0],
    [130, 0],
    [150, 1],
    [170, 1]
]
# label
# 0 is for apple
# 1 is for orange
labels = [0, 0, 1, 1]

# classifier = box of rules
# learning algorithm
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print(clf.predict([[150, 1]]))

