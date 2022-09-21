import random
from typing import final
import cv2
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils import shuffle
import sys
import glob
from skimage import io, color, filters, util, exposure
import csv


PATCH_SIZE = 256

IMAGE_DATA = '/scratch/muneebm/easy data'
MASK_DATA = '/scratch/muneebm/easy_mask'

rusts = glob.glob(f'{IMAGE_DATA}/*')
masks = glob.glob(f'{MASK_DATA}/*')
mask_ids = [i.split('/')[-1].split('.')[0] for i in masks]
rusts = [i for i in rusts if (i.split('/')[-1].split('.')[0] in mask_ids)]
rusts.sort()
masks.sort()


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

def f1_scores(images, labels, mask, original, lab):  # label is the actual ground truth (in form of true and false), and mask (in form of 0 and 255) is the prediction
  tp = 0
  tn = 0
  fp = 0
  fn = 0

  for count, i in enumerate(images):
    if ((mask[i[1]:i[1] + PATCH_SIZE, i[0]: i[0] + PATCH_SIZE].max()==255) and (True in labels[i[1]:i[1] + PATCH_SIZE, i[0]: i[0] + PATCH_SIZE])):
        tp += 1
    elif not(mask[i[1]:i[1] + PATCH_SIZE, i[0]: i[0] + PATCH_SIZE].max()==255) and not(True in labels[i[1]:i[1] + PATCH_SIZE, i[0]: i[0] + PATCH_SIZE]):
        tn += 1
    elif (mask[i[1]:i[1] + PATCH_SIZE, i[0]: i[0] + PATCH_SIZE].max()==255 and not(True in labels[i[1]:i[1] + PATCH_SIZE, i[0]: i[0] + PATCH_SIZE])):
        fp += 1
    elif (not(mask[i[1]:i[1] + PATCH_SIZE, i[0]: i[0] + PATCH_SIZE].max()==255) and (True in labels[i[1]:i[1] + PATCH_SIZE, i[0]: i[0] + PATCH_SIZE])):
        fn += 1

  accuray = (tp + tn) / (tp + tn + fp + fn)
  if (tp+fp != 0):
    precission = (tp) / (tp + fp)
  else:
    precission = 1
  if (tp + fn !=0):
    recall = (tp) / (tp + fn)
  else:
    recall = 1
  if (precission + recall >0):
    f1 = 2 * ((precission * recall) / (precission + recall))
  else: 
    f1 = 1
  print("in fuction: ")
  print("accuracy=", accuray, "  precission=", precission, "  recall=", recall, "  f1=", f1, "  tp=", tp, "  tn=", tn, "  fp=", fp, "  fn=", fn)
  return accuray, precission, recall, f1, tp, tn, fp, fn, original


sprc, sacc, srec, sf1, stp, stn, sfp, sfn, spatches = 0, 0, 0, 0, 0, 0, 0, 0, 0
for l, r in zip(masks, rusts):
    print(r)
    patches = create_patches(r)
    spatches += len(patches)
    img = cv2.imread(r)
    original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.GaussianBlur(img, (5, 5), 6)

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
    # img = cv2.inRange(lab, lab_low, lab_high)

    for p in patches:
        if (img[p[1]:p[1] + PATCH_SIZE, p[0]:p[0] + PATCH_SIZE].max() == 255):
          cv2.rectangle(original, (p[0], p[1]), (p[0] + PATCH_SIZE, p[1] + PATCH_SIZE), (255, 0, 0), 20)
            

    accuracy, precission, recall, f1, tp, tn, fp, fn, x = f1_scores(patches, io.imread(l) > 200, img, original, rgb)

    print("accuracy=", accuracy, "  precission=", precission, "  recall=", recall, "  f1=", f1, "  tp=", tp, "  tn=",
          tn, "  fp=", fp, "  fn=", fn)

    sacc += accuracy
    sprc += precission
    srec += recall
    sf1 += f1
    stp += tp
    stn += tn
    sfp += fp
    sfn += fn

final_recall = srec / len(rusts)
final_prec = sprc / len(rusts)
print("Average values:   accuracy=", sacc / len(rusts), "  precission=", sprc / len(rusts), "  recall=",
      srec / len(rusts), "  f1=", (2 * final_recall * final_prec) / (final_recall + final_prec), "  tp=",
      stp / len(rusts), "  tn=", stn / len(rusts), "  fp=", sfp / len(rusts), "  fn=", sfn / len(rusts))
