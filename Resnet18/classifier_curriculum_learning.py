import random
from typing import final
import tensorflow as tf
import cv2
import os
import numpy as np
from tensorflow.keras.models import Model
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.keras.layers import Activation, Dense, Flatten
from sklearn.cluster import KMeans
import sys
import matplotlib.pyplot as plt
import glob
from tensorflow.keras.models import Sequential
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.vgg16 import VGG16
import pickle
from skimage import io, color, filters, util, exposure
from skimage.filters import threshold_multiotsu
from skimage.feature import greycomatrix, greycoprops
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

ENCODER_FILE = '/scratch/muneebm/weights/SimSiam_dict/56.pth.tar'

fine_tune = False
dataset_directory = 'curriculum_patches_300_patch'  # in /scratch/muneebm/...
graph_file = 'outputs/resnet/simsiam_resnet50_curriculum_patches_224_patch_graph.csv'  # in /home/muneebm/...  for just resnet
# graph_file = 'outputs/resnet/simsiam_pretrained_resnet50_curriculum_patches_224_patch_graph.csv'  # for simsiam pretrained
weights_directory = 'resnet50_curriculum_patches_224_patch' # in /scratch/muneebm/weights/...   for just resnet
# weights_directory = 'simsiam_pretrained_resnet50_curriculum_patches_224_patch' # for simsiam pretrained

os.makedirs(weights_directory, exist_ok=True)

# load pretrained model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
if fine_tune:
    model.load_state_dict(torch.load(ENCODER_FILE), strict=False)

model.fc = nn.Linear(512, 2)   # for resnet18
# model.fc = nn.Linear(2048, 2)    # for resnet50


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# load data
rust_easy = glob.glob(f'/scratch/muneebm/{dataset_directory}/*/easy/*')
rust_medium = glob.glob(f'/scratch/muneebm/{dataset_directory}/*/medium/*')
rust_hard = glob.glob(f'/scratch/muneebm/{dataset_directory}/*/hard/*')
not_rust = glob.glob(f'/scratch/muneebm/{dataset_directory}/*/green/*')
random.seed(4)
random.shuffle(not_rust)
not_rust = not_rust[:3000]

rust_medium.extend(rust_easy)
rust_hard.extend(rust_medium)

print(f"rust_easy={len(rust_easy)} rust_medium={len(rust_medium)} rust_hard={len(rust_hard)} not_rust={len(not_rust)}")

# devide the data into 3 parts: easy, medium, and hard
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

ye_train = [1]*re_train
ye_train.extend([0]*int(nr_train*0.2))
ym_train = [1]*rm_train
ym_train.extend([0]*int(nr_train*0.4))
yh_train = [1]*rh_train
yh_train.extend([0]*nr_train)
ye_val = [1]*re_val
ye_val.extend([0]*nr_val)
ym_val = [1]*rm_val
ym_val.extend([0]*nr_val)
yh_val = [1]*rh_val
yh_val.extend([0]*nr_val)

print(f"xe_train:{len(xe_train)} ye_train:{len(ye_train)} xe_val:{len(xe_val)} ye_val:{len(ye_val)} xm_train:{len(xm_train)} ym_train:{len(ym_train)} xm_val:{len(xm_val)} ym_val:{len(ym_val)} xh_train:{len(xh_train)} yh_train:{len(yh_train)} xh_val:{len(xh_val)} yh_val:{len(yh_val)}")


xe_train,ye_train,xe_val,ye_val = np.array(xe_train),np.array(ye_train),np.array(xe_val),np.array(ye_val)
xm_train,ym_train,xm_val,ym_val = np.array(xm_train),np.array(ym_train),np.array(xm_val),np.array(ym_val)
xh_train,yh_train,xh_val,yh_val = np.array(xh_train),np.array(yh_train),np.array(xh_val),np.array(yh_val)

# making a generator
class CData(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        img_path = self.x[index]
        image = io.imread(img_path)
        label = int(self.y[index])
        if self.transform:
            image = self.transform(image)
        return (image, label)

# Dataset
transform_train = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize((224, 224)),
                                     transforms.RandomHorizontalFlip(p=0.20),
                                     transforms.ColorJitter(brightness=0.25),
                                     transforms.GaussianBlur(5),
                                     transforms.ToTensor()
                                     ])
transform_val = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor()])
easy_dataset_train = CData(xe_train, ye_train, transform_train)
easy_dataset_val = CData(xe_val, ye_val, transform_val)

medium_dataset_train = CData(xm_train, ym_train, transform_train)
medium_dataset_val = CData(xm_val, ym_val, transform_val)

hard_dataset_train = CData(xh_train, yh_train, transform_train)
hard_dataset_val = CData(xh_val, yh_val, transform_val)

#Dataloader
easy_train_loader = DataLoader(dataset=easy_dataset_train, batch_size=4, shuffle=True)
easy_val_loader = DataLoader(dataset=easy_dataset_val, batch_size=4, shuffle=True)

medium_train_loader = DataLoader(dataset=medium_dataset_train, batch_size=4, shuffle=True)
medium_val_loader = DataLoader(dataset=medium_dataset_val, batch_size=4, shuffle=True)

hard_train_loader = DataLoader(dataset=hard_dataset_train, batch_size=4, shuffle=True)
hard_val_loader = DataLoader(dataset=hard_dataset_val, batch_size=4, shuffle=True)

#Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

#Training PipeLine
weight_loc = f"/scratch/muneebm/weights/{weights_directory}"
os.makedirs(weight_loc, exist_ok=True)
step_train=0
step_val=0
with open(graph_file, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch','Train_acc', 'Train_loss', 'Val_acc', 'Val_loss'])

for epoch in range(50):
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
  model.train()
  loop = tqdm(enumerate(train_loader), total=len(train_loader))
  for batch_idx, (data, targets) in loop:
    data = data.to(device)
    targets = targets.to(device)

    #Forward Pass
    pred = model(data)
    loss = criterion(pred, targets)
    losses.append(loss.item())
    acc = sum(pred.argmax(axis=1)==targets)/float(targets.shape[0])
    acces.append(acc.item())
    #Backward Pass
    optimizer.zero_grad()
    loss.backward()
    step_train+=1

    #step
    optimizer.step()
    loop.set_description("Epoch "+str(epoch)+" Acc "+str(np.mean(acces))+" Train Loss "+str(np.mean(losses)))

  train_acc = np.mean(acces)
  train_loss = np.mean(losses)
  torch.save(model, weight_loc+'/{}.pth.tar'.format(epoch))

  model.eval()
  losses = []
  acces = []

  loop = tqdm(enumerate(val_loader), total=len(val_loader))
  with torch.no_grad():
    for batch_idx, (data, targets) in loop:
      data = data.to(device)
      targets = targets.to(device)

      #Forward Pass
      pred = model(data)
      loss = criterion(pred, targets)
      losses.append(loss.item())
      acc = sum(pred.argmax(axis=1)==targets)/float(targets.shape[0])
      acces.append(acc.item())
      loop.set_description("Epoch "+str(epoch)+" Acc "+str(np.mean(acces))+" Val Loss "+str(np.mean(losses)))
      step_val+=1

  with open(f"{graph_file}", 'a') as f:
      writer = csv.writer(f)
      writer.writerow([epoch, train_acc, train_loss, f'{np.mean(acces)}', f'{np.mean(losses)}'])