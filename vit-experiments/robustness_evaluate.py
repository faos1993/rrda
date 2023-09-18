import os
import datasets
import models
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation


from captum.attr import Saliency, IntegratedGradients, LayerGradCam, GuidedGradCam, GuidedBackprop, InputXGradient, LayerAttribution

from tqdm.notebook import tqdm
import pandas as pd

import torch.multiprocessing

from sklearn.metrics import accuracy_score
NUM_WORKERS = 2

def shuffle_outside_mask(X, seg):
    """Mask data by randomizing the pixels outside of the mask."""
    X_masked = torch.clone(X)

    # Loop through batch images individually
    for b in range(seg.shape[0]):

        # Get all of the relevant values using this mask.
        b_seg = seg[b, :, :, :]
        tmp = X[b, b_seg]

        # Randomly permute those values.
        b_idx = torch.randperm(tmp.nelement())
        tmp = tmp[b_idx]
        X_masked[b, b_seg] = tmp

    return X_masked

def plot_batch(x, blob):
    print(blob.max(), blob.min())
    fig, axs = plt.subplots(2, x.shape[0], figsize=(10, 5))
    for i in range(x.shape[0]):
        axs[0, i].imshow(x[i].detach().cpu().squeeze(0).permute(1, 2, 0))
        axs[1, i].imshow(blob[i].detach().cpu().squeeze(0).permute(1, 2, 0) * 255)
    plt.show()

def evaluate_shuffle(data, model, sign=False):
    model.eval()
    
    train_dataloader = DataLoader(
        data, 
        batch_size=32, 
        num_workers=NUM_WORKERS, 
        shuffle=False
    )
    preds = []
    targets = []
    for x, mask, y in tqdm(train_dataloader):
        mask[mask > 0.0] = 1.0 #make mask with only 0 and 1
        if sign:
            background = mask
        else:
            background = 1 - mask
        x = shuffle_outside_mask(x, background.type(torch.BoolTensor))
        #print(x.shape, mask.shape)
        #plot_batch(x, mask)
        out = model(x.cuda()).detach().cpu()
        #pred_category = out.argmax(1).numpy().tolist()[0]
        prob, _ = F.softmax(out, dim=1).max(1)
        preds.extend(out.argmax(1).detach().cpu().numpy().tolist())
        targets.extend(y.detach().cpu().numpy().tolist())
        
    acc_score = accuracy_score(y_true=targets, y_pred=preds)
    print(f'\tSHUFFLE {sign} acc;', acc_score)
    
    return acc_score


class PatchTransform(object):
    def __init__(self, k = 2):
        self.k = k

    def __call__(self, xtensor:torch.Tensor):
        '''
        X: torch.Tensor of shape(c, h, w)   h % self.k == 0
        :param xtensor:
        :return:
        '''
        patches = []
        c, h, w = xtensor.size()
        dh = h // self.k
        dw = w // self.k

        #print(dh, dw)
        sh = 0

        for i in range(h // dh):
            eh = sh + dh
            eh = min(eh, h)
            sw = 0
            for j in range(w // dw):
                ew = sw + dw
                ew = min(ew, w)
                patches.append(xtensor[:, sh:eh, sw:ew])

                #print(sh, eh, sw, ew)
                sw = ew
            sh = eh

        random.shuffle(patches)

        start = 0
        imgs = []

        for i in range(self.k):
            end = start + self.k
            imgs.append(torch.cat(patches[start:end], dim = 1))
            start = end

        img = torch.cat(imgs, dim = 2)

        return img


from sklearn.metrics import accuracy_score
import random
import attack_utils  
#from sklearn.metrics import accuracy_score
#import random

import importlib
#importlib.reload(attack_utils)

def shuffle_outside_mask(X, seg):
    """Mask data by randomizing the pixels outside of the mask."""
    X_masked = torch.clone(X)

    # Loop through batch images individually
    for b in range(seg.shape[0]):

        # Get all of the relevant values using this mask.
        b_seg = seg[b, :, :, :]
        tmp = X[b, b_seg]

        # Randomly permute those values.
        b_idx = torch.randperm(tmp.nelement())
        tmp = tmp[b_idx]
        X_masked[b, b_seg] = tmp

    return X_masked

def plot_batch(x, blob):
    print(blob.max(), blob.min())
    fig, axs = plt.subplots(2, x.shape[0], figsize=(20, 5))
    for i in range(x.shape[0]):
        axs[0, i].imshow(x[i].detach().cpu().squeeze(0).permute(1, 2, 0))
        axs[1, i].imshow(blob[i].detach().cpu().squeeze(0).permute(1, 2, 0) * 255)
    plt.show()

def evaluate_patchtransform(data, model, k=8):
    model.eval()
    
    train_dataloader = DataLoader(
        data, 
        batch_size=32, 
        num_workers=NUM_WORKERS, 
        shuffle=False
    )
    preds = []
    targets = []
    pt = PatchTransform(k=k)
    for x, mask, y in tqdm(train_dataloader):
        mask[mask > 0.0] = 1.0 #make mask with only 0 and 1
        background = 1 - mask
        #x = shuffle_outside_mask(x, background.type(torch.BoolTensor))
        #print(x.shape, mask.shape)
        #plot_batch(x, mask)
        for i in range(x.shape[0]):
            x[i] = pt(x[i])
        #plot_batch(x, mask)
        out = model(x.cuda()).detach().cpu()
        #pred_category = out.argmax(1).numpy().tolist()[0]
        prob, _ = F.softmax(out, dim=1).max(1)
        preds.extend(out.argmax(1).detach().cpu().numpy().tolist())
        targets.extend(y.detach().cpu().numpy().tolist())
        
    acc_score = accuracy_score(y_true=targets, y_pred=preds)
    print('\t PATCHTRANSFORM', acc_score)
    
    return acc_score


def plot_batch(x, blob):
    print(blob.max(), blob.min())
    fig, axs = plt.subplots(2, x.shape[0], figsize=(20, 5))
    for i in range(x.shape[0]):
        axs[0, i].imshow(x[i].detach().cpu().squeeze(0).permute(1, 2, 0))
        axs[1, i].imshow(blob[i].detach().cpu().squeeze(0).permute(1, 2, 0) * 255)
    plt.show()


def evaluate_fgsm(data, model):
    model.eval()
    
    train_dataloader = DataLoader(
        data, 
        batch_size=32, 
        num_workers=NUM_WORKERS, 
        shuffle=False
    )
    preds = []
    targets = []
    
    for x, mask, y in tqdm(train_dataloader):
        mask[mask > 0.0] = 1.0 #make mask with only 0 and 1
        background = 1 - mask
        
        delta = attack_utils.attack_fgsm(model, x.cuda(), y.cuda())
        out = model((x.cuda() + delta).cuda()).detach().cpu()
        #pred_category = out.argmax(1).numpy().tolist()[0]
        prob, _ = F.softmax(out, dim=1).max(1)
        preds.extend(out.argmax(1).detach().cpu().numpy().tolist())
        targets.extend(y.detach().cpu().numpy().tolist())
        
    acc_score = accuracy_score(y_true=targets, y_pred=preds)
    print('\t FGSM accuracy:', acc_score)
    
    return acc_score


# +
def _eval_attack_batch(x, delta, model_item):
    mname, model = model_item
    out = model((x.cuda() + delta).cuda())
        #pred_category = out.argmax(1).numpy().tolist()[0]
    prob, _ = F.softmax(out, dim=1).max(1)
    return out.argmax(1).detach().cpu().numpy().tolist()

def multi_evaluate_fgsm(data, model, models):
    model.eval()
    
    train_dataloader = DataLoader(
        data, 
        batch_size=32, 
        num_workers=NUM_WORKERS, 
        shuffle=False
    )
    
    preds = {}
    targets = []
    
    for key in models.keys():
        preds[key] = []
    
    for x, mask, y in tqdm(train_dataloader):
        mask[mask > 0.0] = 1.0 #make mask with only 0 and 1
        background = 1 - mask
        
        delta = attack_utils.attack_fgsm(model, x.cuda(), y.cuda())
        #out = model((x.cuda() + delta).cuda()).detach().cpu()
        #pred_category = out.argmax(1).numpy().tolist()[0]
        #prob, _ = F.softmax(out, dim=1).max(1)
        #preds.extend(out.argmax(1).detach().cpu().numpy().tolist())
        
        for mname, m in models.items():
            ans = _eval_attack_batch(x, delta, (mname, m))
            preds[mname].extend(ans)
        
        targets.extend(y.detach().cpu().numpy().tolist())
    
    accs = []
    for mname in models.keys():    
        acc_score = accuracy_score(y_true=targets, y_pred=preds[mname])
        accs.append(acc_score)
        print(f'\t FGSM accuracy on {mname}:', acc_score)
    
    return accs
    


# -

def build_model_summary(data, model, att_method):
    model.eval()
    
    train_dataloader = DataLoader(
        data, 
        batch_size=16, 
        num_workers=NUM_WORKERS, 
        shuffle=False
    )
    preds = []
    targets = []
    for x, mask, y in tqdm(train_dataloader):
        mask[mask > 0.0] = 1.0 #make mask with only 0 and 1
        background = 1 - mask
        
        #print(x.shape, mask.shape)
        out = model((x * mask).cuda()).detach().cpu()
        #pred_category = out.argmax(1).numpy().tolist()[0]
        prob, _ = F.softmax(out, dim=1).max(1)
        preds.extend(out.argmax(1).detach().cpu().numpy().tolist())
        targets.extend(y.detach().cpu().numpy().tolist())
        
    print(accuracy_score(y_true=targets, y_pred=preds))
    
    return 

