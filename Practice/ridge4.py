import numpy as np
import pandas as pd
import sklearn.cross_validation


file = "/Users/jeremynixon/Dropbox/python/Ultimate Frisbee/Indoor.csv"
dataset = pd.read_csv(file)
for_cluster = dataset[['Games','Goals','Assists','Ds','Turns','Drops']]
x_train, x_test, y_train, y_test = sklearn.cross_validation.train_test_split(for_cluster['Goals'], 
	for_cluster['Assists'], test_size = .20, random_state = 42)


def ridge_regression(x_train, y_train, x_test, lam):
	x_train = np.array(x_train)
	y_train = np.array(y_train)

	x = np.column_stack((np.ones(len(x_train)), x_train))
	y = y_train

	xt = np.transpose(x)
	xtx = np.dot(xt, x)
	lamI = np.identity(len(xt))
	inverse = np.linalg.inv(xtx + lamI)
	xty = np.dot(xt, y)
	weights = np.dot(inverse, xty)
	
	predictions = []
	for test_point in x_test:
		value = test_point * weights[1:]
		predictions.append(float(value + weights[0]))

	return predictions




print ridge_regression(x_train, y_train, x_test, 0)