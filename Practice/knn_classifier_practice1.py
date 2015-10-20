from __future__ import division
import math
import numpy as np
import pandas as pd
import sklearn.cross_validation
from operator import itemgetter

def minkowski_distance(vector1, vector2, q=2):
	distance = 0
	for component in (vector1-vector2):
		distance += abs(component) ** q
	return distance ** (1.0/q)

def nearest_neighbors(x_train, test_datapoint, k, q=2):
	distances = []
	for datapoint in range(len(x_train)):
		distance = minkowski_distance(x_train[datapoint], test_datapoint, q)
		distances.append((distance, x_train[datapoint], datapoint))

	distances.sort(key=lambda tup: tup[0])

	neighbors = []
	for neighbor in range(k):
		neighbors.append(distances[neighbor][2])
	return neighbors

def outcome(neighbors, y_train):
	neighbor_classes = {}
	for neighbor in range(len(neighbors)):

		response = y_train[neighbors[neighbor]]

		if response in neighbor_classes:
			neighbor_classes[response] += 1
		else:
			neighbor_classes[response] = 1

	votes = sorted(neighbor_classes.iteritems(), key=lambda x: x[1])
	return votes[0][0]

def KNN_Classifier(x_train, y_train, x_test, k, q=2):
	x_train = np.array(x_train)
	y_train = np.array(y_train)
	x_test = np.array(x_test)

	predictions = []
	for i in range(len(x_test)):
		neighbors = nearest_neighbors(x_train, x_test[i], k, q)
		prediction = outcome(neighbors, y_train)
		predictions.append(prediction)
	return predictions

iris = pd.read_csv('iris.data', header=None)
y = iris[4]
iris = iris.drop([4], 1)
x_train, x_test, y_train, y_test = sklearn.cross_validation.train_test_split(iris, y, test_size = .20, random_state=42)

predictions = KNN_Classifier(x_train, y_train, x_test, 4, q=2)
print predictions








