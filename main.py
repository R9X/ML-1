from sklearn import tree
import matplotlib.pyplot as plt 

X_names = ['Peso', 'Textura']
X = [
  [150, 0],
  [170, 0],
  [140, 1],
  [130, 1],
  [90, 0],
  [80, 0]
]

y_names = ['Laranja', 'Maçã', 'Limão']
y = [0, 0, 1, 1, 2, 2]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

fig = plt.figure()

tree.plot_tree(
  clf,
  feature_names=X_names,
  class_names=y_names,
  filled=True,
  impurity=False,
  rounded=True
)

plt.savefig('fruta.png')