import numpy as np
import pandas as pd
import math
import sklearn.cross_validation

iris = pd.read_csv('iris.data', header=None)
y = iris[4]
iris = iris.drop([4], 1)
x_train, x_test, y_train, y_test = sklearn.cross_validation.train_test_split(iris, y, test_size = .50, random_state=42)

def minkowski_distance(vector1, vector2, q=2):
	vector = vector1 - vector2
	value = 0
	for element in vector:
		value += element ** q
	return value ** 1.0/q

def get_neighbors(x_train, test_datapoint, k, q=2):
	distances = []
	for index, datapoint in enumerate(x_train):
		distances.append([index, minkowski_distance(datapoint, test_datapoint, q), datapoint])

	distances.sort(key= lambda i: i[1])

	neighbors = []
	for i in range(k):
		neighbors.append(distances[i][0])

	return neighbors

def get_outcome(y_train, neighbors):
	outcome_counts = {}
	for neighbor in neighbors:
		try:
			outcome_counts[y_train[neighbor]] += 1
		except KeyError:
			outcome_counts[y_train[neighbor]] = 1
	valu = 0
	for index, value in outcome_counts.iteritems():
		if value > valu:
			valu = value
			index_final = index
	return index_final

def knn(x_train, y_train, x_test, k, q=2):
	predictions = []
	for i in x_test:
		neighbors = get_neighbors(x_train, i, k, q)
		predictions.append(get_outcome(y_train, neighbors))
	return predictions

correct_predictions = [(1 if a==b else 0) for a, b in zip(y_test, knn(x_train, y_train, x_test, 5))]
print float(sum(correct_predictions))/ len(y_test) 