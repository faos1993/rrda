import os
import sys

import  time
import copy
import torch
import torch.nn as nn
import torchvision
import numpy as np

from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import torch.nn.functional as F

data_mean = (0.485, 0.456, 0.406)
data_std = (0.229, 0.224, 0.225)

mu = torch.tensor(data_mean).view(3,1,1).cuda()
std = torch.tensor(data_std).view(3,1,1).cuda()

upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def attack_fgsm(model, X, y):
    epsilon = (4/255.)/std
    alpha = (1/255.)/std
    #print(4, 255, 1, 255)
    delta = torch.zeros_like(X).cuda()
    for i in range(len(epsilon)):
        delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
    delta.data = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    X.requires_grad = True
    output = model(X + delta[:X.size(0)])
            
    loss = F.cross_entropy(output, y)
    loss.backward()
    
    grad = delta.grad.detach()
    delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
    delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
    delta = delta.detach()
    
    return delta

def evaluate_fgsm(test_loader, model, opt=None):
    epsilon = (8 / 255.) / std
    alpha = (2 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    print(len(test_loader))
    for i, (X, mask, y) in tqdm(enumerate(test_loader)):
        X, y = X.cuda(), y.cuda()
        fgsm_delta = attack_fgsm(model, X, y, epsilon, alpha)
        with torch.no_grad():
            output = model(X + fgsm_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return pgd_loss/n, pgd_acc/n
