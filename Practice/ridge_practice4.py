import pandas as pd
import numpy as np


file = "/Users/jeremynixon/Dropbox/python/Ultimate Frisbee/Indoor.csv"
dataset = pd.read_csv(file)
for_cluster = dataset[['Games','Goals','Assists','Ds','Turns','Drops']]
for_cluster = np.array(for_cluster)

def ridge_regression(x_train, y_train, lam):
	x_train = np.array(x_train)
	y_train = np.array(y_train)

	x = np.column_stack((np.ones(len(x_train)), x_train))
	y = y_train

	# (xtx + lam*I)^-1 xty

	xt = np.transpose(x)
	product = np.dot(xt, x)
	lambdaI = lam * np.identity(len(xt))
	inverse = np.linalg.inv(product + lambdaI)
	weights = np.dot(inverse, np.dot(xt, y))
	print weights

ridge_regression(dataset['Goals'], dataset['Assists'], 0)
ridge_regression(dataset['Goals'], dataset['Assists'], 1)