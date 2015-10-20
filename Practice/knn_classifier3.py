import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.cross_validation


iris = pd.read_csv('iris.data', header=None)
y = iris[4]
iris = iris.drop([4], 1)
x_train, x_test, y_train, y_test = sklearn.cross_validation.train_test_split(iris, y, test_size = .20, random_state=42)


def minkowski_distance(vector1, vector2, q = 2):
	vector = vector1-vector2
	total = 0
	for i in vector:
		total += i ** q
	return total ** (1.0/q)

def get_neighbors(x_train, test_datapoint, k, q=2):
	distances = []
	for i in range(len(x_train)):
		distances.append([i, minkowski_distance(x_train[i], test_datapoint), x_train[i]])

	distances.sort(key=lambda t: t[1])

	neighbors = []
	for i in range(k):
		neighbors.append(distances[i][0])
	return neighbors

def get_outcome(neighbors, y_train):
	outcome_counts = {}
	for i in neighbors:
		try:
			outcome_counts[y_train[i]] += 1
		except KeyError:
			outcome_counts[y_train[i]] = 1

	store = 0
	for key, value in outcome_counts.iteritems():
		if value > store:
			store = value
			outcome = key
	return outcome

def knn(x_train, y_train, x_test, k, q=2):
	predictions = []
	for datapoint in x_test:
		neighbors = get_neighbors(x_train, datapoint, k, q)
		outcome = get_outcome(neighbors, y_train)
		predictions.append(outcome)
	return predictions

print knn(x_train, y_train, x_test, 4)