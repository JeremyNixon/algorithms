
import numpy as np

s = np.random.randint(0,20,15)
print s


def quicksort(s, l, h):
    if ((h-1) > 0):
        p = partition(s,l,h)
        return quicksort(s,1,p-1)
        return quicksort(s,p+1, h)


def partition(s, l, h):
    p = h
    firsthigh = 1
    for i in range(1,h):
        if s[i] < s[p]:
            temp = s[firsthigh]
            s[firsthigh] = s[i]
            s[i] = temp
            firsthigh = firsthigh + 1
    temp = s[firsthigh]
    s[firsthigh] = s[p]
    s[p] = temp
    return(firsthigh)

quicksort(s, 0, len(s)-1)
print s

