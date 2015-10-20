import numpy as np
import math
import pandas as pd

file = "/Users/jeremynixon/Dropbox/python/Ultimate Frisbee/Indoor.csv"
dataset = pd.read_csv(file)
for_cluster = dataset[['Games','Goals','Assists','Ds','Turns','Drops']]
for_cluster = np.array(for_cluster)

def l2_norm(vector1, vector2):
	return math.sqrt(sum(v**2 for v in (vector1 - vector2)))

def cluster(data, means):
	clusters = {}
	for datapoint in data:
		store = float('inf')
		for index, mean in enumerate(means):
			if l2_norm(mean, datapoint) < store:
				min_index = index
				store = l2_norm(mean, datapoint)

		try:
			clusters[min_index].append(datapoint)
		except KeyError:
			cousters[min_index] = [datapoint]

	return clusters

def reassign_means(clusters):
	means = []
	for index, datapoint in enumerate(clusters):
		means.append(np.mean(datapoint))
	return means

def terminated(means, old_means):
	for i in old_means:
		for j in means:
			if (i == j).all():
				return True
			else:
				return False

def k_means(data, k):
	means = random.sample(data, k)
	old_means = random.sample(data, k)
	while (not Terminated):
		old_means = means
		clusters = cluster(data, means)
		means = reassign_means(clusters)

	return clusters

print k_means(for_cluster, 5)