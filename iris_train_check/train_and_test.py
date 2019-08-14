import csv
import random
from sklearn import svm
import pickle

def getXy(input):
  X = []
  y = []
  for d in train:
    X.append([float(v) for v in d[:-1]])
    y.append(d[-1])
  return (X, y)

with open('iris.csv') as fin:
  reader = csv.reader(fin)
  header = next(reader)
  data = list(reader)

random.shuffle(data)
n_train = int(len(data) * 0.7)
train = data[:n_train]
test = data[n_train:]

(train_X, train_y) = getXy(train)
(test_X, test_y) = getXy(test)

model = svm.SVC()
model.fit(train_X, train_y)
result_y = model.predict(test_X)

print(len(result_y))

for (result, test) in zip(result_y, test_y):
  if result != test:
    print(result, test, result == test)
