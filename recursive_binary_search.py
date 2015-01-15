import numpy as np

array = np.random.randint(1,20,15)

for i in range(0,len(array)):
    for j in range(0, len(array)-1):
        if array[j] > array[j+1]:
            temp = array[j]
            array[j] = array[j+1]
            array[j+1] = temp

def bsearch(array, element, floor, ceiling):
    mid = (floor + ceiling)//2
    if floor == ceiling:
        if array[floor] == element:
            return floor
        else:
            print -1
    if array[mid] > element:
        print 1
        bsearch(array, element, floor, mid-1)
    elif array[mid] < element:
        print 2
        bsearch(array, element, mid+1, ceiling)
    else:
        print "true"
        return mid

for i in range(0,20):
	bsearch(array, i, 0, len(array) - 1)