import TMNIST as fs
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import time


cov = np.cov(np.array(fs.train_data).T)

U, S, V = np.linalg.svd(cov)

def PCA(data,V,S):
	ans=[]
	for i in range(len(V)):
		s=S[i]
		v=V[i]
		ans.append(np.dot(data,v)/np.sqrt(s))
	return np.array(ans)


new_data = []
for data in fs.train_data:
	new_data.append(PCA(data,V,S))
new_data = np.array(new_data)


mu = np.mean(new_data)

for i in range(len(new_data)):
	new_data[i] -= mu


train_classified = [[] for i in range(10)]
for label, data in zip(fs.train_labels, fs.train_data):
	train_classified[label].append(data)

def SW(train_classified,n_class,n_features):
	res = np.zeros([n_features,n_features])
	for i in range(n_class):
		mu = np.mean(train_classified[i],axis=0)
		for x in train_classified[i]:
			res += np.dot((x-mu).reshape(n_features,1),(x-mu).reshape(1,n_features))
	return res

def SB(mu,train_classified,n_class,n_features):
	res = np.zeros([n_features,n_features])
	for i in range(n_class):
		m = np.mean(train_classified[i],axis=0)
		res += np.dot((m-mu).reshape(n_features,1),(m-mu).reshape(1,n_features))*len(train_classified[i])
	return res

new_cov=np.cov(new_data.T)
print new_cov.shape[0]

sb = SB(np.mean(new_data),train_classified,10,62)
sw = SW(train_classified,10,62)
sw_inv = np.linalg.inv(sw)
sep = np.dot(sw_inv,sb)

U2, S2, V2 = np.linalg.svd(sep)

plt.plot(S2)
plt.show()


#4.b

def toarray(a):
	b=[]
	for i in a:
		b.append(i)
	return b


trace=[]
n=[]
for num in range(63):
	sub=[]
	for data in new_data:
		sub.append(np.dot(data,V2[:num].T))


	train_classified = [[] for i in range(10)]
	for j in range(10):
		for i in range(len(sub)):
			if fs.train_labels[i]==j:
				train_classified[j].append(sub[i])

	sb = SB(np.mean(sub), train_classified, 10, num)
	sw = SW(train_classified, 10, num)
	sw_inv = np.linalg.inv(sw)
	sep = np.dot(sw_inv,sb)
	n.append(num)
	trace.append(np.trace(sep))



plt.plot(n,trace)
plt.show()

