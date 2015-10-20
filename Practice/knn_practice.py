import pandas as pd
import numpy as np
import sklearn.cross_validation
import math

# df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
# Y = df['quality'].values
# df = df.drop('quality',1)
# x_train, x_test, y_train, y_test = sklearn.cross_validation.train_test_split(df, Y, test_size = .20, random_state=42)

iris = pd.read_csv('iris.data', header=None)
y = iris[4]
iris = iris.drop([4], 1)
x_train, x_test, y_train, y_test = sklearn.cross_validation.train_test_split(iris, y, test_size = .20, random_state=42)


def minkowski_distance(vector1, vector2, q = 2):
	total = 0
	for component in (vector1-vector2):
		total += abs(component ** q)
	return total ** (1.0/q)

def find_neighbors(x_train, test_datapoint, k, q=2):
	neighbors = []
	for datapoint in range(len(x_train)):
		distance = minkowski_distance(x_train[datapoint], test_datapoint, q)
		neighbors.append([datapoint, distance, x_train[datapoint]])

	neighbors.sort(key= lambda i: i[1])

	indices = []
	for i in range(k):
		indices.append(neighbors[i][0])
	return indices

def outcome(indices, y_train):
	outcome_dict = {}
	for index in indices:
		try:
			outcome_dict[y_train[index]] += 1
		except KeyError:
			outcome_dict[y_train[index]] = 1
	result = 0
	for index, value in outcome_dict.iteritems():
		if value > result:
			result = value
			outcome = y_train[index]
	return outcome

def preprocess(data):
	means = []
	for i in data.T:
		means.append(np.mean(i))

	for i, work in enumerate(data):
		for index, element in enumerate(work):
			element = element-means[index]
	return data

def knn(x_train, y_train, x_test, k, q=2):
	x_train = preprocess(x_train)
	x_test = preprocess(x_test)
	predictions = []
	for datapoint in x_test:
		neighbors = find_neighbors(x_train, datapoint, k)
		result = outcome(neighbors, y_train)
		predictions.append(result)
	return predictions	

predictions =  knn(x_train, y_train, x_test, 8, .04)

count = 0
for i in range(len(predictions)):
	if predictions[i] == y_test[i]:
		count += 1
print count/float(len(y_test))