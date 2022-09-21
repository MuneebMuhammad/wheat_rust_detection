import random
from typing import final
import cv2
import os
import numpy as np
import pandas as pd
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


store_weights = "/scratch/muneebm/weights/10_epoch_curriculum/easy_medium_unet"

# load pretrained model
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=4, out_channels=1, init_features=32, pretrained=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("device=",device)

# load data
rust_easy = glob.glob('/scratch/muneebm/curriculum_patches_300_patch/*/easy/*')
rust_medium = glob.glob('/scratch/muneebm/curriculum_patches_300_patch/*/medium/*')
rust_hard = glob.glob('/scratch/muneebm/curriculum_patches_300_patch/*/hard/*')
not_rust = glob.glob('/scratch/muneebm/curriculum_patches_300_patch/*/green/*')

easy_label = glob.glob('/scratch/muneebm/curriculum_patches_300_maskpatch/*/easy/*')
medium_label = glob.glob('/scratch/muneebm/curriculum_patches_300_maskpatch/*/medium/*')
hard_label = glob.glob('/scratch/muneebm/curriculum_patches_300_maskpatch/*/hard/*')
not_rust_label = glob.glob('/scratch/muneebm/curriculum_patches_300_maskpatch/*/green/*')

easy_label.sort()
rust_easy.sort()
medium_label.sort()
rust_medium.sort()
hard_label.sort()
rust_hard.sort()
not_rust.sort()
not_rust_label.sort()
random.seed(4)
random.shuffle(not_rust)
random.seed(4)
random.shuffle(not_rust_label)
not_rust = not_rust[:2000]
not_rust_label = not_rust_label[:2000]

print(f"rust_easy={len(rust_easy)} rust_medium={len(rust_medium)} rust_hard={len(rust_hard)} not_rust={len(not_rust)}")

# devide data between easy, medium, and hard data
rust_medium.extend(rust_easy)
medium_label.extend(easy_label)
rust_hard.extend(rust_medium)
hard_label.extend(medium_label)

arranging = list(zip(rust_medium, medium_label))
random.shuffle(arranging)
rust_medium, medium_label = zip(*arranging)
rust_medium = list(rust_medium)
medium_label = list(medium_label)
arranging = list(zip(rust_hard, hard_label))
random.shuffle(arranging)
rust_hard, hard_label = zip(*arranging)
rust_hard = list(rust_hard)
hard_label = list(hard_label)

print(f"rust_easy={len(rust_easy)} rust_medium={len(rust_medium)} rust_hard={len(rust_hard)} not_rust={len(not_rust)}")

re_train = int(len(rust_easy)*0.8)
re_val = len(rust_easy)-re_train
rm_train = int(len(rust_medium)*0.8)
rm_val = len(rust_medium)-rm_train
rh_train = int(len(rust_hard)*0.8)
rh_val = len(rust_hard) - rh_train
nr_train = int(len(not_rust)*0.8)
nr_val = len(not_rust) - nr_train

xe_train = rust_easy[:re_train]
xe_train.extend(not_rust[:int(nr_train*0.2)])
xm_train = rust_medium[:rm_train]
xm_train.extend(not_rust[:int(nr_train*0.4)])
xh_train = rust_hard[:rh_train]
xh_train.extend(not_rust[:nr_train])
xe_val = rust_easy[re_train:]
xe_val.extend(not_rust[nr_train:])
xm_val = rust_medium[rm_train:]
xm_val.extend(not_rust[nr_train:])
xh_val = rust_hard[rh_train:]
xh_val.extend(not_rust[nr_train:])

ye_train = easy_label[:re_train]
ye_train.extend(not_rust_label[:int(nr_train*0.2)])
ym_train = medium_label[:rm_train]
ym_train.extend(not_rust_label[:int(nr_train*0.4)])
yh_train = hard_label[:rh_train]
yh_train.extend(not_rust_label[:nr_train])
ye_val = easy_label[re_train:]
ye_val.extend(not_rust_label[nr_train:])
ym_val = medium_label[rm_train:]
ym_val.extend(not_rust_label[nr_train:])
yh_val = hard_label[rh_train:]
yh_val.extend(not_rust_label[nr_train:])

xe_train,ye_train,xe_val,ye_val = np.array(xe_train),np.array(ye_train),np.array(xe_val),np.array(ye_val)
xm_train,ym_train,xm_val,ym_val = np.array(xm_train),np.array(ym_train),np.array(xm_val),np.array(ym_val)
xh_train,yh_train,xh_val,yh_val = np.array(xh_train),np.array(yh_train),np.array(xh_val),np.array(yh_val)

# make boundries for color channels
hsv_low = np.array([0, 0, 0], np.uint8)
hsv_high = np.array([66, 255, 255], np.uint8)

rgb_low = np.array([115, 0, 0], np.uint8)
rgb_high = np.array([255, 199, 255], np.uint8)

lab_low = np.array([0, 113, 0], np.uint8)
lab_high = np.array([255, 255, 255], np.uint8)

# making a generator
class CData(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return self.y.shape[0]

    # performs color space thresholding on an image
    def get_threshold_mask(self, raw_image):
        lab = cv2.cvtColor(raw_image, cv2.COLOR_RGB2LAB)
        rgb = raw_image
        hsv = cv2.cvtColor(raw_image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, hsv_low, hsv_high)
        mask = mask & cv2.inRange(lab, lab_low, lab_high)
        mask = mask & cv2.inRange(rgb, rgb_low, rgb_high)
        return mask

    def __getitem__(self, index):
        img_path = self.x[index]
        image = io.imread(img_path)
        label = self.y[index]
        label = io.imread(label)
        thres_mask = self.get_threshold_mask(image)

        if self.transform:
            thres_mask = self.transform(thres_mask)
            image = self.transform(image)
            label = self.transform(label)
            channel_4_image = torch.concat((image, thres_mask), axis=0)
        return (channel_4_image, label)



# Datasets
train_val_transform = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize((256, 256)),
                                         transforms.ToTensor()
                                         ])
easy_dataset_train = CData(xe_train, ye_train, train_val_transform)
easy_dataset_val = CData(xe_val, ye_val, train_val_transform)

medium_dataset_train = CData(xm_train, ym_train, train_val_transform)
medium_dataset_val = CData(xm_val, ym_val, train_val_transform)

hard_dataset_train = CData(xh_train, yh_train, train_val_transform)
hard_dataset_val = CData(xh_val, yh_val, train_val_transform)

#Dataloader
easy_train_loader = DataLoader(dataset=easy_dataset_train, batch_size=4, shuffle=True)
easy_val_loader = DataLoader(dataset=easy_dataset_val, batch_size=4, shuffle=True)

medium_train_loader = DataLoader(dataset=medium_dataset_train, batch_size=4, shuffle=True)
medium_val_loader = DataLoader(dataset=medium_dataset_val, batch_size=4, shuffle=True)

hard_train_loader = DataLoader(dataset=hard_dataset_train, batch_size=4, shuffle=True)
hard_val_loader = DataLoader(dataset=hard_dataset_val, batch_size=4, shuffle=True)

#Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def get_scores(tp,tn,fp,fn):
  if (tp + fp == 0):
    precision = 0
  else:
    precision = (tp)/(tp+fp)
  if (tp + fn == 0):
    recall = 0
  else:
    recall = (tp)/(tp+fn)
  if (precision+recall==0):
    f1 =0
  else:
    f1 = (2*precision*recall)/(precision+recall)
  return precision, recall, f1

#Training PipeLine
weight_loc = store_weights
os.makedirs(weight_loc, exist_ok=True)
step_train=0
step_val=0

for epoch in range(300):
  if (epoch<2):
      train_loader = easy_train_loader
      val_loader = easy_val_loader
  elif (epoch<4):
      train_loader = medium_train_loader
      val_loader = medium_val_loader
  else:
      train_loader = hard_train_loader
      val_loader = hard_val_loader
  losses=[]
  acces=[]
  ioues=[]
  precisions = []
  recalls = []
  f1s = []
  model.train()
  loop = tqdm(enumerate(train_loader), total=len(train_loader))
  for batch_idx, (data, targets) in loop:
    data = data.to(device)
    targets = targets.to(device)
    #Forward Pass
    pred = model(data)
    loss = criterion(pred, targets)
    losses.append(loss.item())
    pred_np = torch.round(pred)
    pred_np = pred.detach().to('cpu')
    new_target = targets.detach().to('cpu')
    tp = np.count_nonzero(np.logical_and(pred_np.flatten(), (new_target.flatten())))
    fn = np.count_nonzero(np.logical_and(np.logical_not(pred_np.flatten()), (new_target.flatten())))
    fp = np.count_nonzero(np.logical_and(pred_np.flatten(), np.logical_not((new_target.flatten()))))
    tn = np.count_nonzero(np.logical_and(np.logical_not(pred_np.flatten()), np.logical_not((new_target.flatten()))))
    acc = (tp+tn)/(tp+tn+fp+fn)
    acces.append(acc)
    precision, recall, f1 = get_scores(tp, tn, fp, fn)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    iou = (tp)/(tp+fp+fn)
    ioues.append(iou)
    #Backward Pass
    optimizer.zero_grad()
    loss.backward()
    step_train+=1

    #step
    optimizer.step()

    loop.set_description(
        "Epoch=" + str(epoch) + " precision=None" + " recall=None"  + " f1=None" + " Acc=None"  + " Train_Loss=None"  + " IOU=None" )

  train_acc = np.mean(acces)
  train_loss = np.mean(losses)
  train_iou = np.mean(ioues)
  train_prec = np.mean(precisions)
  train_recall = np.mean(recalls)
  train_f1 = np.mean(f1s)
  torch.save(model, weight_loc+'/{}.pth.tar'.format(epoch))

  model.eval()
  losses=[]
  acces=[]
  ioues=[]
  precisions = []
  recalls = []
  f1s = []

  loop = tqdm(enumerate(val_loader), total=len(val_loader))
  with torch.no_grad():
    for batch_idx, (data, targets) in loop:
      data = data.to(device)
      targets = targets.to(device)

      #Forward Pass
      pred = model(data)
      loss = criterion(pred, targets)
      losses.append(loss.item())
      pred_np = torch.round(pred)
      pred_np = pred.detach().to('cpu')
      new_target = targets.detach().to('cpu')

      tp = np.count_nonzero(np.logical_and(pred_np.flatten(), (new_target.flatten())))
      fn = np.count_nonzero(np.logical_and(np.logical_not(pred_np.flatten()), (new_target.flatten())))
      fp = np.count_nonzero(np.logical_and(pred_np.flatten(), np.logical_not((new_target.flatten()))))
      tn = np.count_nonzero(np.logical_and(np.logical_not(pred_np.flatten()), np.logical_not((new_target.flatten()))))
      acc = (tp + tn) / (tp + tn + fp + fn)
      acces.append(acc)
      precision, recall, f1 = get_scores(tp, tn, fp, fn)
      precisions.append(precision)
      recalls.append(recall)
      f1s.append(f1)
      iou = (tp) / (tp + fp + fn)
      ioues.append(iou)
      loop.set_description(
        "Epoch=" + str(epoch) + " precision=None" + " recall=None"  + " f1=None" + " Acc=None"  + " Train_Loss=None"  + " IOU=None" )

      step_val+=1
