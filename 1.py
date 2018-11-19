from scipy import misc
import imageio
import numpy as np
import matplotlib.pyplot as plt

im = ('face/s1/1.png')

faces_matrix_3d = []
for i in range(40):
	for j in range(10):
		faces_matrix_3d.append(imageio.imread('face/s'+str(i+1)+'/'+str(j+1)+'.png')[:,:,0])
faces_matrix_3d = np.array(faces_matrix_3d)

faces_matrix_2d = faces_matrix_3d.reshape(400, 112*92)

normalized_face_vector = faces_matrix_2d - np.mean(faces_matrix_2d,axis=0)

# U, s, V = np.linalg.svd(normalized_face_vector, full_matrices=True)

U = np.load("U.npy")
s = np.load("s.npy")
V = np.load("V.npy")


print V[0]
print V[500]

print V.shape

# np.save("U", U)
# np.save("s", s)
# np.save("V", V)


# for i in range(0,10):
#     im=V[i].reshape(112,92)
#     plt.imshow(im)
#     plt.savefig('./eigen-faces/'+str(i/10))
#     plt.close()

# plt.plot(s)
# plt.savefig("singular")
# # plt.show()


# # print 

print normalized_face_vector.shape
transform = V[:400].T
print transform.shape

# new_data = np.dot(normalized_face_vector, transform)

# def vector2image(data,eigen):
#     res = np.zeros(eigen.shape[1])
#     for i in range(len(data)):
#         res += eigen[i]*data[i]
#     return res.reshape(112,92)

# # a = vector2image(new_data[0],transform.T)
# # plt.imshow(a)
# # plt.show()

# print new_data.shape

# # for i in range(400):
# #     # transform = V[:i].T
# #     # new_data = np.dot(face_vector,transform)
# #     a = vector2image(new_data[i],transform.T)
# #     plt.imshow(a)
# #     plt.title('new_faces:'+str(i))
# #     plt.savefig('./new_faces/'+str(i/10 + 1)+'/'+str(i%10 + 1)+'.png')
# #     plt.close()