import random
from typing import final
import cv2
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import sys
import matplotlib.pyplot as plt
import glob
import pickle
from skimage import io, color, filters, util, exposure
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
from torchsummary import summary
from scipy.special import softmax
from tqdm import tqdm
import csv



graph_file = "outputs/correct_unet_RGB_curriculum_300_patch.csv"
PATCH_SIZE = 300
resnet_weights_directory_list = ["/scratch/muneebm/weights/resnet18_curriculum_patches_224_patch/8.pth.tar"]
unet_weights_directory_list = [i for i in glob.glob('/scratch/muneebm/weights/easy_medium/correct_unet_RGB_curriculum_300_patch_weights_rust_notrust/*') if int(i.split('/')[-1].split('.')[0]) %2 == 0]
testing_directory = "/scratch/muneebm/correct_unet_RGB_testing"

raw_dir = (os.path.join(testing_directory, 'raws'))
mask_dir = (os.path.join(testing_directory, 'masks'))

rusts = glob.glob('/scratch/muneebm/easy data/*')
masks = glob.glob('/scratch/muneebm/easy_mask/*')
mask_ids = [i.split('/')[-1].split('.')[0] for i in masks]
rusts = [i for i in rusts if (i.split('/')[-1].split('.')[0] in mask_ids)]
rusts.sort()
masks.sort()

with open(graph_file, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Precision', 'Recall', 'F1'])

print("availability:", torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device=", device)
resnet_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
resnet_model.fc = nn.Linear(512, 2)


unet_model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=False)
unet_model.to(device)


transform_val_resnet = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((128, 128)),
                                    transforms.ToTensor()])
transform_val_unet = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((256, 256)),
                                    transforms.ToTensor()])

# given a file name, it givesn list of pixel level index of patches
def create_patches_unet(fname):
    x = 0
    y = 0
    patches = []
    img = cv2.imread(fname)
    while (y + PATCH_SIZE < img.shape[0]):
        if (x + PATCH_SIZE > img.shape[1]):
            x = 0
            y += PATCH_SIZE
        if y + PATCH_SIZE < img.shape[0]:
            patches.append([x, y])
        x += PATCH_SIZE

    return patches

# return f1 scores given tp,tn,fp, and fn
def get_scores(tp,tn,fp,fn):
  if (tp + fp == 0):
    precision = 1
  else:
    precision = (tp)/(tp+fp)
  if (tp + fn == 0):
    recall = 1
  else:
    recall = (tp)/(tp+fn)
  if (precision+recall==0):
    f1 =0
  else:
    f1 = (2*precision*recall)/(precision+recall)
  return precision, recall, f1

# given color space mask, patches, input image, and label it predicts f1 scores based on 3 step model
def f1_score_unet(patches, input_img, thresh_mask, label):
  name = input_img.split('/')[-1]
  patch_raw_dir = os.path.join(raw_dir,name.split('.')[0])
  patch_mask_dir = os.path.join(mask_dir,name.split('.')[0])
  os.makedirs(patch_mask_dir, exist_ok=True)
  os.makedirs(patch_raw_dir, exist_ok=True)

  input_img = cv2.cvtColor(cv2.imread(input_img), cv2.COLOR_BGR2RGB)
  label = cv2.imread(label, 0)
  # print("Input_img_shape, label_shape",input_img.shape, label.shape)
  img = np.zeros((8000,8000))
  original_img = np.zeros((8000,8000,3))
  xu = 0
  yu = 0
  xr = 0
  yr = 0

  stp = 0
  stn = 0
  sfn = 0
  sfp = 0

  for p_num, patch in enumerate(patches):
    tp,tn,fn,fp = 0,0,0,0
    # second step of the technique: Resnet18 classification
    classifier_pred = (resnet_model(transform_val_resnet(input_img[patch[1]:patch[1]+PATCH_SIZE, patch[0]:patch[0]+PATCH_SIZE]).unsqueeze(0).to(device)).argmax(axis=1) == 1)
    if ((thresh_mask[patch[1]:patch[1] + PATCH_SIZE, patch[0]: patch[0] + PATCH_SIZE].max()==255) and classifier_pred):
      # third step of the technique: unet model segmentation
      pred = torch.round((unet_model(transform_val_unet(input_img[patch[1]:patch[1]+PATCH_SIZE, patch[0]:patch[0]+PATCH_SIZE]).unsqueeze(0).to(device))))
      pred_np = pred[0].detach().to('cpu')
      img[yu:yu+256,xu:xu+256] = torch.round(pred_np)
      new_label = transform_val_unet(label[patch[1]:patch[1]+PATCH_SIZE, patch[0]:patch[0]+PATCH_SIZE])
      new_label = torch.round(new_label)
      tp = np.count_nonzero(np.logical_and(pred_np[0], (new_label)))
      tn = np.count_nonzero(np.logical_and(np.logical_not(pred_np[0]), np.logical_not((new_label))))
      fn = np.count_nonzero(np.logical_and(np.logical_not(pred_np[0]), (new_label)))
      fp = np.count_nonzero(np.logical_and(pred_np[0], np.logical_not((new_label))))
      precision, recall, f1 = get_scores(tp, tn, fp, fn)
    else:
      new_label = transform_val_unet(label[patch[1]:patch[1]+PATCH_SIZE, patch[0]:patch[0]+PATCH_SIZE])
      new_label = torch.round(new_label)
      tn = np.count_nonzero(np.logical_not((new_label)))
      fn = np.count_nonzero((new_label))

    precision, recall, f1 = get_scores(tp,tn,fp,fn)


    stp += tp
    stn += tn
    sfn += fn
    sfp += fp

    original_img[yu:yu+256,xu:xu+256,:] = cv2.resize(input_img[yr:yr+300,xr:xr+300,:], (256,256))
    xu += 256
    xr += 300
    if xr >= label.shape[1]:
      length_x = xu
      length_y = yu
      xu=0
      xr=0
      yu+=256
      yr+=300

  patch_taken = original_img[:length_y+256,:length_x]
  patch_taken = cv2.cvtColor(patch_taken.astype('uint8'), cv2.COLOR_RGB2BGR)
  save_to = os.path.join(raw_dir,name)

  image_precision, image_recall, image_f1= get_scores(stp,stn,sfp,sfn)
  # print("image score: ", image_precision, image_recall, image_f1)
  return image_precision, image_recall, image_f1


for resnet_model_name in resnet_weights_directory_list:
    for unet_model_name in unet_weights_directory_list:
        sum_p = 0
        sum_r = 0
        sum_f = 0
        
        resnet_model = torch.load(resnet_model_name)
        resnet_model.to(device)
        resnet_model.eval()
        unet_model = torch.load(unet_model_name)
        unet_model.eval()
        print(resnet_model_name, unet_model_name)
        for l, r in zip(masks, rusts):
            # print(l, r)
            patches = create_patches_unet(r)
            img = cv2.imread(r)
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

            # 1st step of technique: making mask for image based on hsv, lab and rgb color spaces
            mask = cv2.inRange(hsv, hsv_low, hsv_high)
            mask = mask & cv2.inRange(lab, lab_low, lab_high)
            img = mask & cv2.inRange(rgb, rgb_low, rgb_high)

            precision, recall, f1 = f1_score_unet(patches, r, img, l)
            sum_p += precision
            sum_r += recall
            sum_f += f1

        prcs = sum_p/len(rusts)
        rcl = sum_r/len(rusts)
        last_f1 = (2*prcs*rcl)/(prcs+rcl)
        print(f"precision={prcs} recall={rcl} f1={last_f1}")
        with open(graph_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([unet_model_name.split('/')[-1].split('.')[0], (prcs), (rcl), (last_f1)])