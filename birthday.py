#import numpy as np

# Calculate the probability that nobody shares a birthday. Take 1-x.

# Probability that nobody shares a birthday:
r = []
for i in range(1, 30):
	n = 1
	for j in range(1, i):
		n = n * (365.0-j)/365.0
	r.append(1-n)
print r


# 	p = 1 * 364.0/365.0 * 363/365 * 362/365

# prob = 1-p

# print prob
