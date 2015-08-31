import numpy as np
import pandas as pd

file = "/Users/jeremynixon/Dropbox/python/Ultimate Frisbee/Indoor.csv"
dataset = pd.read_csv(file)
for_cluster = dataset[['Games','Goals','Assists','Ds','Turns','Drops']]
for_cluster = np.array(for_cluster)




def ridge_regression(x_train, y_train, lam):
	x_train = np.array(x_train)
	y_train = np.array(y_train)
	#x_test = np.array(x_test)

	x = np.column_stack((np.ones(len(x_train)), x_train))
	y = y_train

	# want to compute ((X^TX + lambda I)^-1 X^Ty ) 
	xt = np.transpose(x)
	product = np.dot(xt, x)
	lambda_identity = lam*np.identity(x.shape[1])
	inverse = np.linalg.inv(product+lambda_identity)
	weights = np.dot(np.dot(inverse, xt), y)
	print weights
	return weights

ridge_regression(dataset['Goals'], dataset['Assists'], 1)