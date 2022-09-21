from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2

IMAGE_PATH = '/Users/muneebmuhammad/Documents/Machine Learning/Germany Internship/Crop desease/easy data/1.jpeg'

# read image and convert to LAB color space
img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# get 'a' and 'b' channel features
a_chl = img[:,:,1]
b_chl = img[:,:,2]

a_chl = a_chl.flatten()
b_chl = b_chl.flatten()

features = [[a,b] for a,b in zip(a_chl, b_chl)]
features = np.array(features)

# Clustering with Kmeans
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(features)

output = kmeans.labels_
output = output.reshape((img.shape[0], img.shape[1]))
plt.imshow(output)