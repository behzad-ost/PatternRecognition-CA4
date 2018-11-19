import TMNIST as fs
import numpy as np
import matplotlib.pyplot as plt
import prettytable
from scipy.stats import multivariate_normal


normalized_data = fs.train_data-np.mean(fs.train_data,axis=0)

# U, s, V = np.linalg.svd(normalized_data, full_matrices=True)

# np.save("5.U", U)
# np.save("5.s", s)
# np.save("5.V", V)

U = np.load("5.U.npy")
s = np.load("5.s.npy")
V = np.load("5.V.npy")

def pca(data, v, c):
	transform = []
	for i in range(c):
		transform.append(v[i])
	transform = np.array(transform)
	new_data = []
	for d in data:
		new_data.append(np.dot(d, transform.T))
	return np.array(new_data)

new_train_data = pca(fs.train_data,V,62)
new_test_data = pca(fs.test_data,V,62)

plt.imshow(np.cov(new_train_data.T))
plt.show()

class GaussianBayes:
	def __init__(self, n_labels, n_features, train_set, test_set):
		self.n_labels = np.array(n_labels)
		self.n_features = np.array(n_features)
		self.train_set = train_set
		self.test_set = test_set

		self.train_classified = [[] for i in range(10)]
		for label, data in zip(train_set[1], train_set[0]):
			self.train_classified[label].append(data)
		self.priors= [len(self.train_classified[n])/float(len(train_set[0])) for n in range(10)]

	def classify(self, data):
		results = [self.gauss_pdf(data, self.mu[y], self.cov[y], self.priors[y]) for y in range(self.n_labels)]
		label = np.argmax(results)
		return label

	def train(self):
		self.cov=[]
		self.mu=[]
		for train_class in self.train_classified:
			self.cov.append(np.cov(np.array(train_class).T))
			self.mu.append(np.mean(np.array(train_class),axis=0))  
		self.cov = np.array(self.cov)
		self.mu = np.array(self.mu)


	def get_parameters(self):
		return ([self.mean, self.var], self.pi)

	def gauss_pdf(self, x, mu, cov, prior):
		try:
			p = multivariate_normal(mean=mu, cov=cov)
		except Exception as e:
			return 0 
		return p.pdf(x)*prior


if __name__=="__main__":
	n_labels = 10
	ccrs = []
	features = []
	features_CCR = prettytable.PrettyTable(["n", "Features", "ccr"])
	tmp_ccrs = []
	tmp_fs = []

	for q in xrange(1,62) :
		for f in range(1,62):
			if f in features:
				continue
			tmp_fetures = features
			tmp_fetures.append(f)
			n_features = f
			train_data = new_train_data[:,tmp_fetures]
			train_labels = fs.train_labels
			test_data = new_test_data[:,tmp_fetures]
			test_labels = fs.test_labels
			tmp_fetures.pop()
			train_set = [train_data, train_labels]
			test_set = [test_data, test_labels]

			mnist_model = GaussianBayes(n_labels, n_features, train_set, test_set)
			mnist_model.train()

			limit = len(fs.test_data)
			limit = 100
			test_data, test_labels = test_data[:limit], test_labels[:limit]
			results = np.arange(limit, dtype=np.int)
			for n in range(limit):
				results[n] = mnist_model.classify(test_data[n])
				# print "%d : predicted %s, correct %s" % (n, results[n], test_labels[n])

			tmp_ccrs.append((results == test_labels).mean())
			tmp_fs.append(f)

			print "feature", f , "recognition rate: ", (results == test_labels).mean()

		best_feature = np.argmax(tmp_ccrs)
		features.append(tmp_fs[best_feature])
		# print best_feature + 1, ccrs[best_feature]
		# print features
		ccrs.append(tmp_ccrs[best_feature])
		features_CCR.add_row([str(q) ,str(features), tmp_ccrs[best_feature]])
		print features_CCR
		tmp_ccrs = []
		tmp_fs = []
	print np.argmax(ccrs)

	res = features_CCR.get_string()
	with open('features_CCR.data', 'wb') as f:
		f.write(res)
		f.close()

	with open('features_CCR.data', 'a') as f:
		f.write("\nMaximum ccr is in row " + str(np.argmax(ccrs)+1))