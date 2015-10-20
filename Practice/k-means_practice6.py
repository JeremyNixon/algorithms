import numpy as np
import pandas as pd
import math
import random

def l2_norm(vector1, vector2):
	vector = vector1-vector2
	total = 0
	for i in vector:
		total += i ** 2
	return math.sqrt(total)

def cluster(data, means):
	clusters = {}
	for datapoint in data:
		store = float("inf")
		for index, mean in enumerate(means):
			distance = l2_norm(datapoint, mean)
			if distance < store:
				mean_index = index
				store = distance

		try:
			clusters[mean_index].append(datapoint)
		except KeyError:
			clusters[mean_index] = [datapoint]

	return clusters

def adjust_means(clusters):
	means = []
	for i, datapoints in clusters.iteritems():
		means.append(np.mean(datapoints))
	return means

def terminated(means, old_means):
	for i in means:
		for j in old_means:
			if (i == j).all():
				return True
			else:
				return False

def k_means(data, K):
	means = random.sample(data, K)
	old_means = random.sample(data, K)
	while(not terminated(means, old_means)):
		old_means = means
		clusters = cluster(data, means)
		means = adjust_means(clusters)
	print clusters

file = "/Users/jeremynixon/Dropbox/python/Ultimate Frisbee/Indoor.csv"
dataset = pd.read_csv(file)
for_cluster = dataset[['Games','Goals','Assists','Ds','Turns','Drops']]
for_cluster = np.array(for_cluster)


print k_means(for_cluster, 6)