import numpy as np
import pandas as pd
import math
import sklearn.cross_validation	
from sklearn import neighbors
from sklearn import ensemble

def minkowski_distance(vector1, vector2, q):
	vector = vector1 - vector2
	total = 0
	for component in vector:
		total += abs(component) ** q
	return total * (1.0/q)

def find_neighbors(x_train, test_datapoint, k, q = 2):
	distances = []
	for index, train_datapoint in enumerate(x_train):
		distances.append([index, minkowski_distance(train_datapoint, test_datapoint, q), test_datapoint])

	distances.sort(key = lambda t: t[1])

	neighbors = []
	for i in range(k):
		neighbors.append(distances[i][0])

	return neighbors

def calculate_outcome(neighbors, y_train):
	outcomes = {}
	for neighbor in neighbors:
		try:
			outcomes[y_train[neighbor]] += 1
		except KeyError:
			outcomes[y_train[neighbor]] = 1

	store = 0
	for result, value in outcomes.iteritems():
		if value > store:
			store = value
			outcome = result

	return outcome

def preprocess(data):
	means = []
	for i in data.T:
		means.append(np.mean(i))
	for datapoint in data:
		for index, value in enumerate(datapoint):
			value = value-means[index]
	return data

def kNN_classifier(x_train, y_train, x_test, k, q=2):
	x_train = np.array(x_train)
	x_test = np.array(x_test)
	
	y_train = np.array(y_train)
	x_test = preprocess(np.array(x_test))
	x_train = preprocess(np.array(x_train))

	predictions = []
	for test in x_test:
		neighbors = find_neighbors(x_train, test, k, q)
		predictions.append(calculate_outcome(neighbors, y_train))
	return predictions

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
Y = df['quality'].values
df = df.drop('quality',1)
x_train, x_test, y_train, y_test = sklearn.cross_validation.train_test_split(df, Y, test_size = .20, random_state=42)

#clf = neighbors.KNeighborsClassifier(n_neighbors = 8)
clf = ensemble.GradientBoostingClassifier(n_estimators = 1000)
score = sklearn.cross_validation.cross_val_score(clf, df, Y, cv = 10)
print np.mean(score)
# predictions = clf.predict(x_test)

# #predictions = kNN_classifier(x_train, y_train, x_test, 8, q=.4)
# count = 0
# for index, value in enumerate(y_test):
# 	if value == predictions[index]:
# 		count += 1
# print count/float(len(y_test))
