N = int(raw_input())
string = raw_input()
array = []
# for i in range(len(string)):
#     if string[i] != " ":
#         if i < len(string)-1:
#             if string[i+1] != " ":
#                 a = int(string[i])
#                 b = int(string[i+1])
#                 array.append(10*a+b)
#             elif string[i-1] == " " or i == 0:
#                 array.append(int(string[i]))

array = map(int, string.split(' '))

indicator = 0
while(indicator !=1):
    minimum = 1000
    count = 0
    for i in range(len(array)):
        if array[i] < minimum and array[i] != 0:
            minimum = array[i]
    for j in range(len(array)):
        if array[j] != 0:
            array[j] = array[j] - minimum
            count += 1
    print count

    indicator = 1
    for l in range(len(array)):
        if array[l] != 0:
            indicator = 0
