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

DATASET_LOCATION = '/scratch/muneebm/easy data'

# Invert the labels of patches
def invert(array):
    inverted_array = [1 if (array[i] == 0) else 0 for i in range(len(array))]
    return inverted_array

# create patch's pixel wise indices
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

# Selects patches which are close to centroids
def get_reliable_images(kmeans, features):
    threshold = 0.95
    # select centers
    centers = kmeans.cluster_centers_

    # calculate similarity matrix
    similarities = cosine_similarity(centers, features)  # NUM_CLUSTER * num images
    print(similarities.shape)
    # select reliable images
    reliable_image_ids = np.unique(np.argwhere(similarities > threshold)[:, 1])
    return reliable_image_ids

# return psudo labels based on K-means clustering
def get_pseudo_labels(kmeans):
    label = kmeans.labels_
    print("labels: ", kmeans.labels_)
    if len(label) != 0:
        if np.count_nonzero(label) > len(label) // 2:  # 0 is healthy. 1 is rust
            labels = tf.keras.utils.to_categorical(invert(label), num_classes=2)
        else:
            labels = tf.keras.utils.to_categorical(label, num_classes=2)
    return labels

# create VGG16 model
def create_model():
    # load VGG16 pre-trained model on imagenet and add flatten layer and change the output layer
    model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    feature2 = Flatten()(model.layers[-2].output)  # add flatten layers after convolution
    model2 = Model(model.input, feature2)
    features = Dense(2, activation='softmax')(model2.layers[-1].output)
    final_model = Model(model.input,
                        features)  # pretrained vgg16, which last layer is replaced with a layer containing 2 nodes
    return final_model


final_model = create_model()
final_model.load_weights("./weights/ckpt9")
feature_model = Model(final_model.input, final_model.layers[-2].output)

rusts = glob.glob(f'{DATASET_LOCATION}/*')

for i in rusts:

    with open("./weights/kmeans_model9.pkl", "rb") as f:
        old_kmeans = pickle.load(f)
        print(old_kmeans.cluster_centers_.shape)
        kmeans = KMeans(n_clusters=2, init=old_kmeans.cluster_centers_)

    features = []
    img_patches = np.zeros((1, 128, 128, 3))
    all_patches = np.array([[0, 0]])

    print("image: ", i)
    zero = []
    one = []
    img = cv2.imread(i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    patches = create_patches(i)

    for j, patch in enumerate(patches):
        p = img[patch[1]:patch[1] + 128, patch[0]:patch[0] + 128, :]
        p = np.array([p])
        img_patches = np.vstack((img_patches, p))
        x = feature_model.predict(p)
        features.append(np.squeeze(x))


    kmeans.fit(features)
    labels = get_pseudo_labels(kmeans)
    if (len(labels) != 0):
        
        np.delete(img_patches, 0)
        images = np.array([img_patches[i] for i in range(len(img_patches))])
        for zz, l in enumerate(labels):
            if l[0] == 0:
                # print(len(patches), patches[i])
                cv2.rectangle(img, (patches[zz][0], patches[zz][1]), (patches[zz][0] + 128, patches[zz][1] + 128), (255, 0, 0),
                              20)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(img.shape)
        print("i: ", i)
        name = i.split('/')[-1].split('.')[0]
        print("name: ", name)
        cv2.imwrite(f'/scratch/muneebm/results/{name}.png', img)
