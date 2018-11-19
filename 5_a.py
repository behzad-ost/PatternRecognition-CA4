
# coding: utf-8

# In[82]:

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import numpy as np
from sklearn.decomposition import PCA
import time


# In[83]:

import numpy as np
from sklearn.feature_selection import VarianceThreshold

# Loading Dataset
train_data = np.loadtxt('TinyMNIST/trainData.csv', dtype=np.float32, delimiter=',')
train_labels = np.loadtxt('TinyMNIST/trainLabels.csv', dtype=np.int32, delimiter=',')
test_data = np.loadtxt('TinyMNIST/testData.csv', dtype=np.float32, delimiter=',')
test_labels = np.loadtxt('TinyMNIST/testLabels.csv', dtype=np.int32, delimiter=',')
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Feature Selection
tr_samples_size, _ = train_data.shape
all_data = np.vstack((train_data,test_data))
sel = VarianceThreshold(threshold=0.90*(1-0.90))
all_data = sel.fit_transform(all_data)
train_data = all_data[:tr_samples_size]
test_data = all_data[tr_samples_size:]

tr_samples_size, feature_size = train_data.shape
te_samples_size, _ = test_data.shape
print('Train Data Samples:',tr_samples_size,
      ', Test Data Samples',te_samples_size,
      ', Feature Size(after feature-selection):', feature_size)


# In[84]:

miu=np.mean(train_data,axis=0)
new_data=train_data-miu


# In[85]:

U, s, V = np.linalg.svd(new_data, full_matrices=True)


# In[86]:

cov = np.cov(new_data.T)


# In[87]:

W=[]
for v in V:
    W.append(np.dot(np.dot(cov,v),v)/ np.linalg.norm(v)**2)


# In[88]:

def pca(data,v,w,ncomp):
    v=np.real(v)
    w=np.real(w)
    
    transfer=[]
    for i in range(ncomp):
        transfer.append(v[i])
    new_data=[]
    for d in data:
        new_data.append(np.dot(transfer,d.T))
    return np.array(new_data)


# In[96]:

new_data=pca(train_data,V,W,62)
new_test=pca(test_data,V,W,62)


# In[91]:

plt.imshow(np.cov(new_data.T))
plt.show()


# In[92]:

from scipy.stats import multivariate_normal

def bayse(train_data,train_lables,nclass,test_data,test_labels):
    types=[]
    for i in range(10):
        types.append([])
    
    for j in range(len(types)):
        for i in range(len(train_labels)):
            if train_labels[i]==j:
                types[j].append(train_data[i])
    cov=[]
    miu=[]
    for i in range(len(types)):
        cov.append(np.cov(np.array(types[i]).T))
        miu.append(np.mean(np.array(types[i]),axis=0))
    prior=[]
    for i in types:
        prior.append(len(i)/len(train_data))
        
        
        
        
    choose=[]
    for n  in range(len(test_data)):
        pmax=0
        data=test_data[n]
        org=test_labels[n]
        label=float('nan')
        for i in range(10):
            try:
                p = multivariate_normal(mean=miu[i], cov=cov[i]).pdf(data)*prior[i]
            except:
                p=0
            if p > pmax:
                label=i
                pmax=p
        choose.append(label)
    c=[]
    for i in range(len(test_data)):
        c.append(choose[i]==test_labels[i])
    return np.mean(c)


# In[93]:

def select_feat(data,select):
    Data=[]
    for i in data:
        temp=[]
        for j in select:
            temp.append(i[j])
        Data.append(temp)
    return Data


# In[95]:

select=[]
tstart=time.time()
rank=0
RANK=[]
feature=np.arange(62)
for n in range(10): #num of feature select
    S=float('nan')
    for m in feature: # search on feature to find best
        tr_data=select_feat(new_data,select+[m])
        te_data=select_feat(new_test,select+[m])
        r=bayse(tr_data,train_labels,10,te_data[0:100],test_labels[0:100])
        if r > rank:
            S = m
            rank = r
    select.append(S)
    np.delete(feature,S)
    print(select)
    RANK.append(rank)
    np.save('./rank_a/rank'+str(n),RANK)
    np.save('./rank_a/feature'+str(n),select)
    print(time.time()-tstart)


# In[ ]:

plt.plot(RANK)
plt.savefig('5_a.png')
plt.show()
