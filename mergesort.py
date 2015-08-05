
import numpy as np

# Reasonable test with 200,000 elements to be sorted (don't try this with an n**2 algorithm)
l = np.random.randint(0,200000,50)
print l

def mergesort(l):
    # Convert from numpy
    l = list(l) 
    result = []
    if len(l) <= 1:
        return l
    
    # Divide array recursively
    mid = len(l)//2
    right = l[mid:len(l)]
    left = l[0:mid]
    
    left = mergesort(left)
    right = mergesort(right)
    
    # Merge
    while (len(left) > 0) or (len(right) > 0):
        
        # Camparison with content in both arrays
        if len(left) > 0 and len(right) > 0:
            if left[0] > right[0]:
                result.append(right[0])
                right.pop(0)
            else:
                result.append(left[0])
                left.pop(0)
                
        # Case where there is a single array left with elements
        elif len(right) > 0:
            for i in right:
                result.append(i)
                right.pop(0)
        else:
            for i in left:
                result.append(i)
                left.pop(0)
                
    return result

print mergesort(l)