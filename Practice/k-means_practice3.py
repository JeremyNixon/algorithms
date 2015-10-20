import pandas as pd
import numpy as np
import random
import math

def L2_norm(vector1, vector2):
	total = 0
	vector = vector1 - vector2
	for component in vector:
		total += component ** 2
	#print math.sqrt(total)
	return math.sqrt(total)

def cluster(data, means):
	clusters = {}
	for datapoint in data:
		distances = []
		store = float("inf")
		mean_index = 0
		for index, mean in enumerate(means):
			if L2_norm(datapoint, mean) < store:
				store = L2_norm(datapoint, mean)
				mean_index = index

		try:
			clusters[mean_index].append(datapoint)
		except KeyError:
			clusters[mean_index] = [datapoint]

	return clusters

def adjust_means(clusters):
	means = []
	for key, datapoint in clusters.iteritems():
		mean = np.mean(clusters[key])
		means.append(mean)
	return means



def terminated(means, old_means):
	for i in means:
		for j in old_means:
			if (i == j).all():
				return True
			else:
				return False


def k_means(data, K):
	old_means = random.sample(data, K)
	means = random.sample(data, K)
	while(not terminated(means, old_means)):
		old_means = means
		clusters = cluster(data, means)
		adjust_means(clusters)

	clusters_list = []
	for cluster_index, elements in clusters.iteritems():
		for element in elements:
			clusters_list.append([cluster_index, element])
	print clusters_list
	return clusters_list




file = "/Users/jeremynixon/Dropbox/python/Ultimate Frisbee/Indoor.csv"
dataset = pd.read_csv(file)
for_cluster = dataset[['Games','Goals','Assists','Ds','Turns','Drops']]
for_cluster = np.array(for_cluster)


k_means(for_cluster, 4)