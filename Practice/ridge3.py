import numpy as np
import pandas as pd
import sklearn.cross_validation

def ridge_regression(x_train, y_train, lam):
	x_train = np.array(x_train)
	y_train = np.array(y_train)


	x = np.column_stack((np.ones(len(x_train)), x_train))
	y = y_train

	xt = np.transpose(x)
	xtx = np.dot(xt, x)
	lamI = lam * np.identity(len(xt))
	inverse = np.linalg.inv(xtx + lamI)
	xty = np.dot(xt, y)
	weights = np.dot(inverse, xty)

	return weights


file = "/Users/jeremynixon/Dropbox/python/Ultimate Frisbee/Indoor.csv"
dataset = pd.read_csv(file)
for_cluster = dataset[['Games','Goals','Assists','Ds','Turns','Drops']]
x_train, x_test, y_train, y_test = sklearn.cross_validation.train_test_split(for_cluster['Goals'], 
	for_cluster['Assists'], test_size = .20, random_state = 42)

weights = ridge_regression(x_train, y_train, 0)
print weights

def predict(x_test, weights):
	
	predictions = []
	for datapoint in x_test:
		result = weights[0] + weights[1] * datapoint
		predictions.append(result)
	return predictions

predictions = predict(x_test, weights)

for i, v in enumerate(y_test):
	print v
	print predictions[i]
