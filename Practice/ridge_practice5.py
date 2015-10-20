import numpy as np
import pandas as pd

file = "/Users/jeremynixon/Dropbox/python/Ultimate Frisbee/Indoor.csv"
dataset = pd.read_csv(file)
for_cluster = dataset[['Games','Goals','Assists','Ds','Turns','Drops']]
for_cluster = np.array(for_cluster)


def ridge_regression(x_train, y_train, lam):
	x_train = np.array(x_train)
	y_train = np.array(y_train)

	x = np.column_stack((np.ones(len(x_train)), x_train))
	y = y_train

	xt = np.transpose(x)
	xtx = np.dot(xt, x)
	lIdentity = lam * np.identity(len(xt))
	inverse = np.linalg.inv(xtx + lIdentity)
	weights = np.dot(inverse, np.dot(xt, y))
	print weights

ridge_regression(dataset['Goals'], dataset['Assists'], 1)