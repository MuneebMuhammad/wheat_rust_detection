import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import Model
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.keras.layers import Activation, Dense, Flatten
from sklearn.cluster import KMeans
import sys
# import torch
# import torch.nn as nn
import matplotlib.pyplot as plt
import glob
from tensorflow.keras.models import Sequential
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.vgg16 import VGG16
import pickle
import tensorflow_addons as tfa
from tqdm import tqdm

DATASET_LOCATION = '/scratch/muneebm/Dataset/Unlabeled'

# returns list of pixel wise indices of each patch
def create_patches(fname):
    x = 0
    y = 0
    patches = []
    img = cv2.imread(fname)
    while (y + 128 < img.shape[0]):
        if (x + 128 >= img.shape[1]):
            x = 0
            y += 128
        if y + 128 < img.shape[0]:
            patches.append([x, y])
        x += 128

    return patches

# Select those patches which feature vectors are close to teh centroids
def get_reliable_images(kmeans, features):
    threshold = 0.95
    # select centers
    centers = kmeans.cluster_centers_

    # calculate similarity matrix
    similarities = cosine_similarity(centers, features)  # NUM_CLUSTER * num images
    print(similarities.shape)
    # select reliable images
    reliable_image_ids = np.unique(np.argwhere(similarities > threshold)[:, 1])

    while (len(reliable_image_ids) < 50 and threshold>0.5):
        threshold = threshold - 0.05
        reliable_image_ids = np.unique(np.argwhere(similarities > threshold)[:, 1])

    return reliable_image_ids

def invert(array):
  inverted_array = [1 if(array[i] == 0) else 0 for i in range(len(array))]
  return inverted_array

# return psudo labels based on kmeans clustering. Cluster with more number of patches assigned to it are considered rust
def get_pseudo_labels(kmeans, reliable_image_ids):
  kmeans_labels = kmeans.labels_
  label = [kmeans_labels[i] for i in reliable_image_ids]
  if np.count_nonzero(label) > len(label)//2:      # 0 is healthy. 1 is rust
    labels = tf.keras.utils.to_categorical(invert(label), num_classes=2)
  else:
    labels = tf.keras.utils.to_categorical(label, num_classes=2)
  return labels

# make vgg16 model
def create_model():
    # load VGG16 pre-trained model on imagenet and add flatten layer and change the output layer
    model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    feature2 = Flatten()(model.layers[-2].output) # add flatten layers after convolution
    model2 = Model(model.input, feature2)
    features = Dense(2, activation='softmax')(model2.layers[-1].output)
    final_model = Model(model.input, features)  # pretrained vgg16, which last layer is replaced with a layer containing 2 nodes
    return final_model

final_model = create_model()
feature_model = Model(final_model.input, final_model.layers[-2].output)

# model to get features (second last layer)
print(final_model.summary())
# load image locations

rusts = glob.glob(f'{DATASET_LOCATION}/*')
for epoch in range(10):
  loop = tqdm(rusts)
  kmeans = KMeans(n_clusters=2)
  for i in loop:
    features = []
    img_patches = np.zeros((1, 128, 128, 3))

    print("image: ", i)
    zero = []
    one = []
    img = cv2.imread(i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    patches = create_patches(i)

    for cc, patch in enumerate(patches):
      p = img[patch[1]:patch[1]+128, patch[0]:patch[0]+128, :]
      p = np.array([p])
      img_patches = np.vstack((img_patches, p))
      x = feature_model.predict(p)
      features.append(np.squeeze(x))  # 2D array
      # print(final_model(p))
    kmeans.fit(features)
    reliable_images_ids = get_reliable_images(kmeans, features)
    print("number of reliable images: ", len(reliable_images_ids))

    for layer in final_model.layers:
        layer.trainable = True

    final_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=["acc"])
    labels = get_pseudo_labels(kmeans, reliable_images_ids)
    np.delete(img_patches, 0)
    images = np.array([img_patches[jj] for jj in reliable_images_ids])
    history = final_model.fit(images, labels, batch_size=16, validation_split=0.1, epochs=1, verbose = False, shuffle=True)
    print("Accuracy: ", history.history['acc'])
    loop.set_description(f"epoch={epoch} Accruracy={history.history['acc']}  image={i}")
      
  final_model.save_weights(f"./weights/ckpt{epoch}")
  with open(f"./weights/kmeans_model{epoch}.pkl", "wb") as f:
    pickle.dump(kmeans, f)









