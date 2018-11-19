import matplotlib.pyplot as plt
import numpy as np
import time 	
from os import listdir
import matplotlib.image as mpimg
from sklearn.decomposition import PCA

face_data = []
for i in range(1,41):
    for j in range(1,11):
        face_data.append(mpimg.imread('face/s'+str(i)+'/'+str(j)+'.png')[:,:,0])

face_data = np.array(face_data)
print face_data.shape
face_labels = np.array(face_labels)
face_vector = face_data.reshape(400,112*92)

print face_vector.shape
face_vector-=np.mean(face_vector,axis=0)


U, s, V = np.linalg.svd(face_vector, full_matrices=True)

s.shape

for i in range(0,100,10):
    im=V[i].reshape(112,92)
    plt.imshow(im)
    plt.savefig('./svd_plot/'+str(i))
    plt.close()


plt.plot(s)
plt.show()

transport = V[:400].T


# In[103]:

new_data=np.dot(face_vector,transport)


# In[104]:

new_data.shape


# In[ ]:


# In[46]:

def invers(data,V):
    ans=np.zeros(V.shape[1])
    for i in range(len(data)):
        ans+=V[i]*data[i]
    return ans


# In[47]:

a=invers(new_data[0],transport.T)


# In[48]:

b=a.reshape(112,92)


# In[49]:

plt.imshow(b)
plt.show()


# In[50]:

plt.imshow(face_data[0])
plt.show()


# In[108]:

for i in range(100,401,20):
    transport=V[:i].T
    new_data=np.dot(face_vector,transport)
    a=invers(new_data[2],transport.T)
    b=a.reshape(112,92)
    plt.imshow(b)
    plt.title('num of vector:'+str(i))
    plt.savefig('./plot_num_of_vector/'+str(i)+'.png')
    plt.close()


U, s, V = np.linalg.svd(cov , full_matrices = True)


# In[11]:

V.shape


# In[58]:

a=np.dot(cov,V[100])


# In[62]:

plt.scatter(a,V[100])
plt.show()


# In[40]:

a/V[10000]


# In[75]:

s[:399]


# In[ ]:



