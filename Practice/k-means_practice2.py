import numpy as np
import pandas as pd
import math
import random

""" L2 norm, assign points to clusters, assign cluster means based on points, 
	check for termination, create main function to iterate between until termination
"""

def L2_norm(vector1, vector2):
	total = 0
	for i in range(len(vector1)):
		total += (vector1[i] - vector2[i]) ** 2
	return math.sqrt(total)

def assign_clusters(data, means):
	clusters = {}
	for datapoint in data:
		closest_mean = min([(i[0], L2_norm(datapoint, means[i[0]]))  for i in enumerate(means)], key = lambda t:t[1])[0]
		print clusters
		try:
			clusters[closest_mean].append(datapoint)
		except KeyError:
			clusters[closest_mean] = [datapoint]

	return clusters

def assign_means(clusters):
	means = []
	for i in sorted(clusters.keys()):
		means.append(np.mean(clusters[i], axis=0))
	return means

def terminated(means, old_means):
	return set([tuple(a) for a in means]) == set([tuple(a) for a in old_means])

def k_means(data, K):
	old_means = random.sample(data, K)
	means = random.sample(data, K)
	while(not terminated(means, old_means)):
		old_means = means
		clusters = assign_clusters(data, means)
		means = assign_means(clusters)

	cluster_list = []
	for i, j in clusters.iteritems():
		for k in j:
			cluster_list.append([i, k])
	return cluster_list

file = "/Users/jeremynixon/Dropbox/python/Ultimate Frisbee/Indoor.csv"
dataset = pd.read_csv(file)
for_cluster = dataset[['Games','Goals','Assists','Ds','Turns','Drops']]
for_cluster = np.array(for_cluster)

print k_means(for_cluster, 5)