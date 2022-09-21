import yaml
import os
import random
import time
import cv2
import numpy as np
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist

from model.PFENet import PFENet
from util import dataset
from util import transform,config
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU

# load config file
def get_config_file(fname):
    with open(fname, 'r') as f:
        cfg_from_file = yaml.safe_load(f)
    cfg = config.CfgNode(cfg_from_file)
    return cfg

def main():
    args = get_config_file('pfenet.yaml')
    BatchNorm = nn.BatchNorm2d
    model = PFENet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, \
                   criterion=nn.CrossEntropyLoss(), BatchNorm=BatchNorm, \
                   pretrained=True, shot=args.shot, ppm_scales=args.ppm_scales, vgg=args.vgg)
    model = torch.nn.DataParallel(model.cuda())
    criterion = nn.CrossEntropyLoss()
    # load pretrained model
    if args.weight:
        if os.path.isfile(args.weight):
            print("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded weight '{}'".format(args.weight))
        else:
            print("=> no weight found at '{}'".format(args.weight))

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    if args.resized_val:
        val_transform = transform.Compose([
            transform.Resize(size=args.val_size),
            transform.ToTensor()])  # removed normalization part
    else:
        val_transform = transform.Compose([
            transform.test_Resize(size=args.val_size),
            transform.ToTensor()])   # removed normalization part

    val_data = dataset.SemData(shot=args.shot, transform=val_transform, mode='val', \
                test_image_paths='/scratch/muneebm/few_shot_learning/test_data', \
                support_image_paths='/scratch/muneebm/few_shot_learning/support_data')

    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False)

    loss_val = validate(val_loader, model, criterion)
    print("validation loss:", loss_val)

# runs inference on test data
def validate(val_loader, model, criterion):
    model.eval()
    loss = 0
    count = 0
    for e in range(1):
        for i, (input, target, s_input, s_mask) in enumerate(val_loader):
            count += 1
            print(f"input_max:{input.max()}, target_max:{target.max()}, s_input_max:{s_input.max()}, s_mask:{s_mask.max()}")
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = model(s_x=s_input, s_y=s_mask, x=input, y=target)
            print(f"output_max:{output.max()}")
            print(f"target_shape:{target.shape}")
            print(f"output_type:{type(output)} output_shape:{output.shape}")
            output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
            # loss += criterion(output, target)
            write_to = f'/scratch/muneebm/few_shot_learning/results/{i}'
            os.makedirs(write_to, exist_ok=True)
            print(f"input type:{type(input[0])}, input_shape:{input[0].shape}")
            # The output contains two channels, one for each class, so store both outputs
            cv2.imwrite(f'{write_to}/0output.jpg', output[0][0].detach().cpu().numpy())
            cv2.imwrite(f'{write_to}/1output.jpg', output[0][1].detach().cpu().numpy())

    loss /= count

    return loss



if __name__ == '__main__':
    main()
