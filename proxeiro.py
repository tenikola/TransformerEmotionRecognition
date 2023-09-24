import sys

def geomean(list0):
    list1 = []
    for string in list0:
        is_digit = string.isdigit()
        if is_digit:
            myInt = int(string)
            list1.append(myInt)
    print(list1)
    total = 1
    n = len(list1)
    for i in range(n):
        total = total * list1[i]
    print(total)
    reslt = total**(1/n)
    return reslt

print(geomean(sys.argv))

