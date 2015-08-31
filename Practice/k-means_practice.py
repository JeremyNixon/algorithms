import numpy as np
import random
import pandas as pd

def L2_norm(vector):
	total = 0
	for component in vector:
		total += component**2
	return np.sqrt(total)

def assign_clusters(data, means):
	clusters = {}
	for datapoint in data:
		closest_mean = min([(i[0], L2_norm(datapoint-means[i[0]])) \
			for i in enumerate(means)], key = lambda t:t[1])[0]
		try:
			clusters[closest_mean].append(datapoint)
		except KeyError:
			clusters[closest_mean] = [datapoint]
	return clusters

def reevaluate_centers(clusters):
	new_means = []
	keys = sorted(clusters.keys())
	for k in keys:
		new_means.append(np.mean(clusters[k], axis=0))
	return new_means

def terminated(means, old_means):
	return (set([tuple(a) for a in means]) == set([tuple(a) for a in old_means]))

def k_means(data, K):
	old_means = random.sample(data, K)
	means = random.sample(data, K)
	while not terminated(means, old_means):
		old_means = means
		clusters = assign_clusters(data, means)
		means = reevaluate_centers(clusters)
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