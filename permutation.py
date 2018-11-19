import itertools
a = [1,2,3,4,5,6,7,8,9,10]


# print sub_lists(a)


# a = [1,2,3]

def sub_lists(res, tmp, a, n):
	print "calling with res ",res
	if n==1:
		for i in range(len(a)):
			tmp.append(a[i])
			tmp2 = tmp[:]
			res.append(tmp2)
			tmp.pop()
		tmp.pop()
		return res
	else:
		for i in range(len(a)-1):
			print "i: ", i
			tmp.append(a[i])
			x = a[(i + 1):]
			sub_lists(res,tmp,x,n-1)
	

def rec(a,n):
	if n == 1:
		a.append(0)
		return 
	else:
		a.append(1)
		rec(a,n-1)

	return a


res = [[]]
tmp = []

# print sub_lists(res, tmp, a, 3)

print set(itertools.combinations(a,2))



# def findsubsets(S,m):
#     return set()