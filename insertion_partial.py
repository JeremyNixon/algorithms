import sys

def insertionSort(ar):
    insert = ar[-1]
    i = 1
    while(ar[-(i+1)] > insert):
        ar[-(i)] = ar[-(i+1)]
        sys.stdout.write(ar)
        i += 1
    ar[-(i)] = insert
    sys.stdout.write(ar)
    return ""

m = input()
ar = [int(i) for i in raw_input().strip().split()]
insertionSort(ar)