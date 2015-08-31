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

	xt = np.transpose(x)
	xtx = np.dot(xt, x)
	lambdaI = lam * np.identity(len(xt))
	xty = np.dot(xt, y)
	weights = np.dot(np.linalg.inv(xtx + lambdaI), xty)
	print weights

ridge_regression(dataset['Goals'], dataset['Assists'], 0)