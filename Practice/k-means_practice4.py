import numpy as np
import pandas as pd
import random
import math


def l2_norm(vector1, vector2):
	vector = vector1 - vector2
	total = 0
	for component in vector:
		total += component ** 2
	return math.sqrt(total)

def cluster(data, means):
	clusters = {}
	for datapoint in data:
		store = float("inf")
		for index, mean in enumerate(means):
			distance = l2_norm(mean, datapoint)
			if distance < store:
				store = distance
				closest_mean = index
		try:
			clusters[closest_mean].append(datapoint)
		except KeyError:
			clusters[closest_mean] = [datapoint]
	return clusters

def assign_means(clusters):
	means = []
	for key, datapoints in clusters.iteritems():
		means.append(np.mean(datapoints))
	return means

def terminate(means, old_means):
	for i in means:
		for j in old_means:
			if (i == j).all():
				return True
			else:
				return False

def k_means(data, K):
	old_means = random.sample(data, K)
	means = random.sample(data, K)
	while(not terminate(means, old_means)):
		old_means = means
		clusters = cluster(data, means)
		means = assign_means(clusters)

	cluster_list = []
	for index, datapoints in clusters.iteritems():
		for datapoint in datapoints:
			cluster_list.append([index, datapoint])
	return cluster_list



file = "/Users/jeremynixon/Dropbox/python/Ultimate Frisbee/Indoor.csv"
dataset = pd.read_csv(file)
for_cluster = dataset[['Games','Goals','Assists','Ds','Turns','Drops']]
for_cluster = np.array(for_cluster)


print k_means(for_cluster, 6)