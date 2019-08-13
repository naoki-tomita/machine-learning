from sklearn import svm
import pickle

X = [
  [0, 0],
  [1, 1]
]
y = ['red', 'blue']

model = svm.SVC()

model.fit(X, y)

with open('model.pkl', 'wb') as fout:
        pickle.dump(model, fout)
