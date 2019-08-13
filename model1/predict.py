import pickle

X = [[0, 0.5], [2, 2], [0.6, 0.5], [0.4, 0.5]]

with open('model.pkl', 'rb') as fin:
  model = pickle.load(fin)

print(model.predict(X))
