from sklearn import tree

X = [
  [150, 0],
  [170, 0],
  [140, 1],
  [130, 1]
]

y = [0, 0, 1, 1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

predict = clf.predict([ [135, 1], [160, 0] ])

print(predict)