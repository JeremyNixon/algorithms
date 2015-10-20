import numpy as np
import pandas as pd
import random
import math

file = "/Users/jeremynixon/Dropbox/python/Ultimate Frisbee/Indoor.csv"
dataset = pd.read_csv(file)
for_cluster = dataset[['Games','Goals','Assists','Ds','Turns','Drops']]
for_cluster = np.array(for_cluster)


def distance_metric(vector1, vector2):
	total = 0
	for i in vector1-vector2:
		total += i ** 2
	return math.sqrt(total)

def assign_clusters(data, means):
	clusters = {}
	for datapoint in data:
		distance = float("inf")
		for index, mean in enumerate(means):
			d = distance_metric(datapoint, mean)
			if d < distance:
				distance = d
				assignment = index
		try:
			clusters[assignment].append(datapoint)
		except KeyError:
			clusters[assignment] = [datapoint]

	return clusters

def assign_means(clusters):
	means = []
	for index, datapoint in clusters.iteritems():
		means.append(np.mean(datapoint))
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
	oldmeans = random.sample(data, K)
	while (not terminated(means, oldmeans)):
		oldmeans = means
		clusters = assign_clusters(data, means)
		means = assign_means(clusters)

	print clusters
k_means(for_cluster, 4)
