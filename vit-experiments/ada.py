import torch
from torch.utils import data
import numpy as np

from torchvision import transforms
import os 
from PIL import Image
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop as cc
import torchvision.utils as vutils
from torch.nn import ReLU
import torch.nn as nn

import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from captum.attr import Saliency, IntegratedGradients, LayerGradCam, GuidedGradCam, GuidedBackprop, InputXGradient, LayerAttribution

import models
import datasets

import random


def pd(matrix, gt, width=5, height=5):
    if len(gt.shape) == 3:
        gt = gt.mean(0)
        
    if len(matrix.shape) != 2 or len(gt.shape) != 2:
        raise Exception("matrix and gt dim should be 2")

    sum = np.zeros(matrix.shape, dtype=np.float32)
    mask = np.ones(matrix.shape, dtype=np.float32)
    w, h = matrix.shape[0], matrix.shape[1]

    for i in range(w):
        for j in range(h):
            sum[i][j] = matrix[i][j]
            if i > 0: sum[i][j] += sum[i - 1][j]
            if j > 0: sum[i][j] += sum[i][j - 1]
            if i > 0 and j > 0: sum[i][j] -= sum[i - 1][j - 1]
            
    high = 0
    pos = (0, 0)

    for i in range(w):
        for j in range(h):
            k, l = i + height, j + width
            if k >= w or l >= h: continue 
            sub = sum[k][l]
            if i > 0: sub -= sum[i - 1][l]
            if j > 0: sub -= sum[k][j - 1]
            if i > 0 and j > 0: sub += sum[i - 1][j - 1]
            if sub > high:
                high = sub
                pos = (i, j)

    cont_ones = 0
    out = 0

    for i in range(pos[0], pos[0] + height + 1):
        for j in range(pos[1], pos[1] + width + 1):    
            if i >= w or j >= h: 
                out += 1
                continue
            if gt[i][j] == 1:
                cont_ones += 1
            else:
                mask[i][j] = 0.0
                
    return sum, high, pos, mask, (cont_ones, gt.sum())


def pd_pytorch(matrix, gt, width=5, height=5):
    if len(matrix.shape) != 2 or len(gt.shape) != 2:
        raise Exception("matrix and gt dim should be 2")

    mask = np.ones(matrix.shape, dtype=np.float32)
    w, h = matrix.shape[0], matrix.shape[1]
    
    high = 0
    pos = (0, 0)

    for i in range(w):
        for j in range(h):
            part_sum = matrix[i:i+height,j:j+width].cuda().sum()
            if part_sum > high:
                high = part_sum
                pos = (i, j)

    cont_ones = 0
    out = 0

    for i in range(pos[0], pos[0] + height + 1):
        for j in range(pos[1], pos[1] + width + 1):    
            if i >= w or j >= h: 
                out += 1
                continue
            if gt[i][j] == 1:
                cont_ones += 1
            else:
                mask[i][j] = 0.0
    
    return sum, high, pos, mask, (cont_ones, gt.sum())


def build_new_ada_image(x, att_maps, gt, width, height):
    sum, high, pos, mask, (cont_ones, gt_sum) = pd(att_maps, gt, width, height)
    #sum, high, pos, mask, (cont_ones, gt_sum) = pd_pytorch(att_maps, gt, width, height)
    x_new = x * mask

    return x_new


def explain(model, x, y, att_method, shape_img):
    attribution = None
    #print(x.shape)
    if att_method == 'saliency':
        sal = Saliency(model)
        #print('x.shape', x.shape)
        
        attribution = sal.attribute(x, target=y, abs=True)
        #print('attribution shape', attribution.shape)
        attribution = attribution.mean(1)

    elif att_method == 'guidedbackprop':
        gbp = GuidedBackprop(model)
        attribution = gbp.attribute(x, target=y).abs().mean(1)

    elif att_method == 'gradcam' or att_method == 'gradcam4':
        layer_gc = LayerGradCam(model, model.layer4)
        attribution = layer_gc.attribute(x, y)
        attribution = LayerAttribution.interpolate(attribution, shape_img)

    elif att_method == "gradcam3":
        layer_gc = LayerGradCam(model, model.layer3)
        attribution = layer_gc.attribute(x, y)
        attribution = LayerAttribution.interpolate(attribution, shape_img)

    elif att_method == "gradcam2":
        layer_gc = LayerGradCam(model, model.layer2)
        attribution = layer_gc.attribute(x, y)
        attribution = LayerAttribution.interpolate(attribution, shape_img)

    elif att_method == "gradcam1":
        layer_gc = LayerGradCam(model, model.layer1)
        attribution = layer_gc.attribute(x, y)
        attribution = LayerAttribution.interpolate(attribution, shape_img)

    elif att_method == 'guidedgradcam':
        guided_gc = GuidedGradCam(model, model.layer4)
        attribution = guided_gc.attribute(x, y)

    elif att_method == 'integratedgradients':
        ig = IntegratedGradients(model)
        attribution = ig.attribute(x, target=y)

    elif att_method == 'inputxgradient':
        input_x_gradient = InputXGradient(model)
        attribution = input_x_gradient.attribute(x, target=y)
    else:
        print('attribution method not defined.')
        return None

    return attribution


class AdaDataset(data.Dataset):

    def __init__(self):
        self.imgs = []
        self.gts = []
        self.targets = []

    def add(self, img, gt, y):
        self.imgs.append(img)
        self.gts.append(gt)
        self.targets.append(y)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        return self.imgs[index], self.gts[index], self.targets[index]


def ada(dataset, model, att_method, block_size=20):
    ada_dataset = AdaDataset()
    print("... BUILDING ADA DATASET ...")
    indexes = random.sample(range(1, len(dataset)), 50)

    for idx in tqdm(range(len(dataset))):
        x, blob, y = dataset[idx]
        x_shape = (x.shape[1], x.shape[2])
        
        att = explain(model, x.unsqueeze(0).cuda(), y, att_method, x_shape)
        if len(att.shape) == 4:
            att = att.squeeze(0)
        
        x_new = build_new_ada_image(
            x, 
            att.squeeze(0), 
            blob.squeeze(0), 
            width=block_size, 
            height=block_size
            )
        
        ada_dataset.add(x_new, blob, y)
        ada_dataset.add(x, blob, y)

    return ada_dataset

if __name__ == "__main__":
    model = models.get_resnet18(num_classes=2, pretrained=False)
    model = model.cuda()

    isic_dataset = datasets.IsicSkinDataset()
    
    ada(isic_dataset, model, "saliency")