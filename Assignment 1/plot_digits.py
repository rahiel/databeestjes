import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

digits = datasets.load_digits()

images_and_labels = list(zip(digits.images, digits.target))

nsamples = len(digits.images)
data = digits.images.reshape((nsamples, -1))

Xtr, Xte, ytr, yte = train_test_split(data, digits.target, test_size = 0.5, random_state = 42)

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}

clf = GridSearchCV(svm.SVC(kernel='rbf'), param_grid)
clf.fit( Xtr, ytr)
exp = yte
pred = clf.predict(Xte)

print metrics.classification_report(exp, pred)
print metrics.confusion_matrix(exp, pred)
