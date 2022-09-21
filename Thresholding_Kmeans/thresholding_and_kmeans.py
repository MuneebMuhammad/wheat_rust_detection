from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2

IMAGE_PATH = '/Users/muneebmuhammad/Documents/Machine Learning/Germany Internship/Crop desease/easy data/146.jpg'
MASK_PATH = '/Users/muneebmuhammad/Documents/Machine Learning/Germany Internship/Crop desease/easy mask/146.jpg'
PATCH_SIZE = 64

# return pixel wise indices of each patch
def create_patches(fname):
    x = 0
    y = 0
    patches = []
    img = cv2.imread(fname)
    while (y + PATCH_SIZE < img.shape[0]):
        if (x + PATCH_SIZE >= img.shape[1]):
            x = 0
            y += PATCH_SIZE
        if y + PATCH_SIZE < img.shape[0]:
            patches.append([x, y])
        x += PATCH_SIZE
    return patches

# return f1 score given tp, tn, fp, and fn
def get_scores(tp, tn, fp, fn):
    if (tp + fp == 0):
        precision = 1
    else:
        precision = (tp) / (tp + fp)
    if (tp + fn == 0):
        recall = 1
    else:
        recall = (tp) / (tp + fn)
    if (precision + recall == 0):
        f1 = 0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1

patches = create_patches(IMAGE_PATH)

label_mask = cv2.imread(MASK_PATH)
print("Label mask shape:", label_mask.shape)
img = cv2.imread(IMAGE_PATH)
original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# make boundries for color channels
hsv_low = np.array([0, 0, 0], np.uint8)
hsv_high = np.array([66, 255, 255], np.uint8)

rgb_low = np.array([115, 0, 0], np.uint8)
rgb_high = np.array([255, 199, 255], np.uint8)

lab_low = np.array([0, 113, 0], np.uint8)
lab_high = np.array([255, 255, 255], np.uint8)

# making mask for image
mask = cv2.inRange(hsv, hsv_low, hsv_high)
mask = mask & cv2.inRange(lab, lab_low, lab_high)
img = mask & cv2.inRange(rgb, rgb_low, rgb_high)

features = []
targets = []
print("number of patches:", len(patches))
for p_num, p in enumerate(patches):
  # Check if with thresholding the patch has rust in it
  if (img[p[1]:p[1] + PATCH_SIZE, p[0]:p[0] + PATCH_SIZE].max() == 255): 
    features.append(img[p[1]:p[1] + PATCH_SIZE, p[0]:p[0] + PATCH_SIZE].flatten())
    if (label_mask[p[1]:p[1] + PATCH_SIZE, p[0]:p[0] + PATCH_SIZE].max() > 200):
      targets.append(1)
    else:
      targets.append(0)

features = np.array(features)
targets = np.array(targets)

kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
output = kmeans.labels_

if (np.count_nonzero(output)*2>len(output)):
    output = 1-output

tp, tn, fp, fn = 0,0,0,0
for p,l in zip(output, targets):
  if (p == 1 and l == 1): tp +=1 
  elif (p == 0 and l == 0): tn += 1
  elif (p==1 and l == 0): fp += 1
  elif (p == 0 and l == 1): fn += 1

precision, recall, f1 = get_scores(tp,tn,fp,fn)

print(f"Precision={precision}, Recall={recall}, F1={f1}")