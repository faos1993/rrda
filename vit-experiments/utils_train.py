import torch
import sys
import numpy as np
import pickle as pkl
import os
from os.path import join as oj

import torch.nn.functional as F
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, ConcatDataset
from PIL import Image
from tqdm import tqdm
from torch import nn
from numpy.random import randint

import torchvision.transforms as transforms
import torchvision.models as models
import time
import copy

from skimage.morphology import dilation
from skimage.morphology import square
from sklearn.metrics import f1_score

from scores import cd, score_funcs
from scores import cd_architecture_specific 

import attack_utils

import datasets
import ada
import models
from datetime import datetime


device = 'cuda'


def get_bbox(size, pos, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    elif len(size) == 2:
        W = size[0]
        H = size[1]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_rat = np.random.uniform(low=0.0, high=1.0)
    
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    
    posy1 = max(pos[0].item() - cut_h // 2, 0)
    posy2 = min(pos[0].item() + cut_h // 2, H)

    posx1 = max(pos[1].item() - cut_w // 2, 0)
    posx2 = min(pos[1].item() + cut_w // 2, W)

    return (posy1, posx1), (posy2, posx2)


def get_fixed_bbox(size, pos, cut_box=32):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    elif len(size) == 2:
        W = size[0]
        H = size[1]
    else:
        raise Exception
    
    cut_w = cut_box
    cut_h = cut_box
    
    posy1 = max(pos[0] - cut_h // 2, 0)
    posy2 = min(pos[0] + cut_h // 2, H)

    posx1 = max(pos[1] - cut_w // 2, 0)
    posx2 = min(pos[1] + cut_w // 2, W)

    return (posy1, posx1), (posy2, posx2)


def build_mask_from_poslist(size, poslist, blobs, cut_box):
    mask = torch.ones(size).cuda()
    
    for pos in poslist:
        bbox = get_fixed_bbox(size, pos, cut_box)
        mask[:, :, bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1]] = 0
    
    mask = torch.clamp(mask + blobs, max=1) #keep foreground 1

    return mask


def from_argsort2bbox(att_flatten_idx, k):
    bestk = att_flatten_idx[0, -k:]
    #print(bestk)
    return bestk


def plot(x, blob, mask):
    fig, axs = plt.subplots(1, 4, figsize=(10, 20))

    axs[0].imshow(x.detach().cpu().squeeze(0).permute(1, 2, 0))
    axs[1].imshow(blob.detach().cpu().squeeze(0).permute(1, 2, 0) * 255)
    axs[2].imshow(mask.detach().cpu().squeeze(0).permute(1, 2, 0))
    axs[3].imshow(1 - mask.detach().cpu().squeeze(0).permute(1, 2, 0))

    plt.show()


def plot_batch(x, blob):
    print(blob.max(), blob.min())
    fig, axs = plt.subplots(2, x.shape[0], figsize=(10, 20))
    for i in range(x.shape[0]):
        axs[0, i].imshow(x[i].detach().cpu().squeeze(0).permute(1, 2, 0))
        axs[1, i].imshow(blob[i].detach().cpu().squeeze(0).permute(1, 2, 0) * 255)
    plt.show()


def build_mask(shape, bbox, blobs):
    #print('BBOX')
    #print('\t',bbox)
    #print(bbox.max(), bbox.min())
    #print(shape)
    #print(blobs.shape)
    #print(blobs.max(), blobs.min())
    #print()
    mask = torch.ones(shape) 
    #print('\t\t', bbox[0][0], bbox[1][0])
    #print('\t\t', bbox[0][1], bbox[1][1])
    mask[:, :, bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1]] = 0
    mask = mask.cuda() + blobs
    #print('\t\t', mask.shape, blobs.shape)
    #print('\t\t', (mask == 0).sum())
    return torch.clamp(mask, max=1) 


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


def rand_bbox(size, blob, lam, center=False, attcen=None):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    elif len(size) == 2:
        W = size[0]
        H = size[1]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    nonzero_pos = torch.nonzero(blob).cpu()
    #print()
    #print('argwhere shape', nonzero_pos.shape)
    
    #print('nonzero pos sample', nonzero_pos[idx_nonzero_pos])
    #print()

    if attcen is None:
        # uniform
        #cx = 0
        #cy = 0
        if nonzero_pos.shape[0] > 0:
            idx_nonzero_pos = np.random.randint(low=0, high=nonzero_pos.shape[0])
            _, cx, cy = nonzero_pos[idx_nonzero_pos]
        elif W>0 and H>0:
            cx = np.random.randint(W)
            cy = np.random.randint(H)
            #print('shape 0')

        if center:
            cx = int(W/2)
            cy = int(H/2)
    else:
        cx = attcen[0]
        cy = attcen[1]

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def plot_batch(batch1, batch2):
    num_imgs = batch1.shape[0]
    fig, axs = plt.subplots(2, num_imgs, figsize=(num_imgs*5, 10))

    for i in range(num_imgs):
        img = batch1[i].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        img = (img - img.min())/(img.max() - img.min())
        axs[0][i].imshow(img)

    for i in range(num_imgs):
        img = batch2[i].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        img = (img - img.min())/(img.max() - img.min())
        axs[1][i].imshow(img)

    plt.show()


def compare_activations(act_a, act_b):
    """
    Calculates the mean l2 norm between the lists of activations
    act_a and act_b.
    source: https://github.com/josephdviviano/saliency-red-herring/blob/d697f87068bf7576e191e709fee6ec4306242165/activmask/models/loss.py#L6
    """
    assert len(act_a) == len(act_b)
    dist = torch.nn.modules.distance.PairwiseDistance(p=2)
    all_dists = []

    # Gather all L2 distances between the activations
    #for a, b in zip(act_a, act_b):
    #    all_dists.append(dist(a, b).view(-1))

    #all_dists = torch.cat(all_dists)
    all_dists = dist(act_a, act_b)
    actdiff_loss = all_dists.sum() / len(all_dists)

    return (actdiff_loss)


def plot_xmasked(a, b):
    fig, axs = plt.subplots(2, a.shape[0], figsize=(5, 20))
    for i in range(a.shape[0]):
        axs[0, i].imshow(a[i].permute(1, 2, 0).detach().cpu().numpy())
        axs[1, i].imshow(b[i].permute(1, 2, 0).detach().cpu().numpy())
    plt.show()


def plot_xmasked_save(a, b, filename):
    os.makedirs('tmp-plots', exist_ok=True)

    fig, axs = plt.subplots(2, a.shape[0])
    for i in range(a.shape[0]):
        a1 = a[i].permute(1, 2, 0).detach().cpu().numpy()
        a1max = a1.max()
        a1min = a1.min()
        
        b1 = b[i].permute(1, 2, 0).detach().cpu().numpy()
        b1max = b1.max()
        b1min = b1.min()

        axs[0, i].imshow((a1 - a1min)/(a1max - a1min))        
        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])
        axs[0, i].set_xticklabels([])
        axs[0, i].set_yticklabels([])

        axs[1, i].imshow((b1 - b1min)/(b1max - b1min))
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])
        axs[1, i].set_xticklabels([])
        axs[1, i].set_yticklabels([])

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f"tmp-plots/{filename}.jpg", dpi=600)
    plt.show()


def plot_xmaskedp(a, b, c):
    fig, axs = plt.subplots(3, a.shape[0], figsize=(15, 5))
    for i in range(a.shape[0]):
        axs[0, i].imshow(a[i].permute(1, 2, 0).detach().cpu().numpy())
        axs[1, i].imshow(b[i].permute(1, 2, 0).detach().cpu().numpy())
        axs[2, i].imshow(c[i].permute(1, 2, 0).detach().cpu().numpy())
    plt.show()


def get_grad_contrast(im, y_pred, labels):
    """
    Gradmask: Simple Constrast Loss. d(y_0-y_1)/dx
    based on: https://github.com/josephdviviano/saliency-red-herring/blob/d697f87068bf7576e191e709fee6ec4306242165/activmask/models/loss.py#L146
    """

    #oh = torch.nn.functional.one_hot(labels, y_pred.shape[1])
    oh = torch.zeros_like(y_pred)
    oh[:, labels] = 1.0
    oh_reverse = 1 - oh

    #contrast = torch.abs(y_pred[:, 0] - y_pred[:, 1])

    contrast = torch.abs((oh*y_pred).sum(1) - (oh_reverse*y_pred).sum(1)).sum().cuda()
    im.requires_grad = True

    #print('labels shape:', labels.shape, oh.shape, oh_reverse.shape, contrast.shape)
    #print('y_pred shpae', y_pred.shape)
    #print('contrast shape:', contrast.shape)

    labels.cuda()

    # This is always a list of length 1, so we remove the element from the list.
    gradients = torch.autograd.grad(
        outputs=contrast, 
        inputs=im, 
        allow_unused=True, 
        create_graph=True
    )[0]

    #print('gradients shape:', gradients.shape, len(gradients))

    return gradients


def get_patches(img, patch_size):
    #patch_size = 8
    unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
    fold = nn.Fold(output_size=(224, 224), kernel_size=patch_size, stride=patch_size)

    img = img.unsqueeze(0)
    patches = unfold(img)
    #patches.shape, fold(patches).shape
    rebuild = fold(patches)

    return patches, rebuild


def extract_background(x, mask):
    mask[mask > 0.0] = 1.0
    #print(x.shape, mask.shape, y)
    #print("set of pixel values in input mask:",set(mask.flatten().numpy().tolist()))

    patch_size = 32

    unfold = nn.Unfold(
        kernel_size=patch_size, 
        stride=patch_size
    )
    fold = nn.Fold(
        output_size=(224, 224),
        kernel_size=patch_size, 
        stride=patch_size
    )

    if len(x.shape) == 3:
        img = x.unsqueeze(0)
    elif len(x.shape) == 4:
        img = x
    else:
        raise Exception('x shape must have 3 or 4 dimensions.')
    
    img_max = img.max()
    img_min = img.min()

    #img = (img - img_min)/(img_max - img_min)

    patches = unfold(img) #build the patches from the raw image
    #print(patches.shape, fold(patches).shape)

    mask_fore_patch, _ = get_patches(
        mask, 
        patch_size=patch_size
    ) #extract patches from the raw mask
    mask_back_patch, _ = get_patches(
        (1 - mask), 
        patch_size=patch_size
    ) #extract patches from the background mask

    #print(mask_patch, mask_patch.sum(1).shape)
    #print(mask_patch.shape, mask_patch0.sum(1).shape)
    #print(mask_patch0.sum(1))

    mask_fore_01 = mask_fore_patch.sum(1).nonzero() # get the indexes of patches whose sum if different of zero
    mask_back_01 = torch.nonzero(mask_back_patch.sum(1) == (patch_size**2)*3) #get indexes of patches from background only

    #print(m2.shape, m20.shape)
    #print(f'background indexes shape {0, m20.shape[0]}', )

    background_indexes = torch.randint(0, mask_back_01.shape[0], (mask_fore_01.shape[0],))
    patches[:, :, mask_fore_01[:, 1]] = patches[:, :, mask_back_01[background_indexes, 1]]

    background_end = fold(patches)

    """
    print('background shape:', background_end.shape)
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    axs[0].imshow(background_end.squeeze(0).permute(1, 2, 0))
    axs[1].imshow(fold(unfold(img)).squeeze(0).permute(1, 2, 0))
    axs[2].imshow(mask.squeeze(0).permute(1, 2, 0))
    axs[3].imshow((1 - mask.squeeze(0).permute(1, 2, 0)))
    axs[4].imshow((1-mask.squeeze(0).permute(1, 2, 0)) * fold(unfold(img)).squeeze(0).permute(1, 2, 0))
    plt.show()
    """
    
    return background_end


def batch_transform(x, mask):
    x_background = torch.zeros(x.shape)
    for i in range(x.shape[0]):
        #print(x[i].shape)
        try:
            background_tmp = extract_background(x[i], mask[i])
            x_background[i] = background_tmp
        except Exception as e:
            #print(e)
            x_background[i] = x[i]
        
    return x_background


def adak(model, x, blob, y):
    masks = []
    input_shape = (x.shape[1], x.shape[2])

    for idx in range(x.shape[0]):
        chosed_x = x[idx].unsqueeze(0)
        
        # 1. GET THE INTERPRETABILITY AND SELECT THE POSITION OF THE MOST IMPORTANT BACKGROUND PIXEL 
        att = ada.explain(
            model, 
            x=chosed_x, 
            att_method='saliency', 
            y=y[idx],
            shape_img=input_shape,
            ).abs()

        #print("x and att:", chosed_x.shape, att.shape)
        new_blob = torch.clamp(((blob[idx] > 0.0) * 1).sum(dim=0), max=1).unsqueeze(0)
        #print('new_blob', new_blob.shape)
        att = att * (1 - new_blob) #keep only the interpretability of background pixels
        
        att_flatten = att.flatten(start_dim=1, end_dim=2)
        att_flatten_idx = att_flatten.argsort()
        #print(att_flatten.shape, att_flatten_idx.shape)

        #posx, posy = pos_max // inputs.shape[2], pos_max % inputs.shape[3]
        
        beta = 1.0

        # 2. BUILD THE MASK TO CUTOFF THE REGION AROUND THE PIXEL SELECTED IN 1
        
        #out = get_bbox(
        #    chosed_x_shape, 
        #    pos=(posy, posx), 
        #    lam=np.random.beta(beta, beta)
        #)
        cut_box = 20
        kpositions = from_argsort2bbox(att_flatten_idx, k=5)
        posx = kpositions // chosed_x.shape[2]
        posy = kpositions % chosed_x.shape[3]
        positions = list(zip(posx.cpu().detach().tolist(), posy.cpu().detach().tolist()))
        #print('background positions:', positions)
        mask = build_mask_from_poslist(chosed_x.shape, positions, blob[idx].unsqueeze(0), cut_box)
        #mask = build_mask(
        #        shape=chosed_x_shape, 
        #        bbox=out, 
        #        blobs=(blob[idx] > 0.0) * 1
        #        )
        #print('mask:', mask.shape)
        # 3. CUTOFF THE PIXEL WITH MASK BUILT IN 2
        #x[idx] = chosed_x * mask
        #print()
        masks.append(mask)
    masks = torch.cat(masks, dim=0)
    #plot_xmasked_save(x, masks, int(datetime.timestamp(datetime.now())))
    del model

    return masks


def rrr_mix(inputs, blob, labels, dbg=False):
    bs = inputs.shape[0]
    #print('random permutation:'.upper(), torch.randperm(bs))
    beta = 1.0
    blob = 1 - blob*1 #1 in relevant and 0 in irrelevant
    inputs2 = inputs.detach().clone()

    perm1 = torch.randperm(bs)
    perm2 = torch.randperm(bs)
        
    target_b = labels.clone()
    target_b_weights = []
        
    for i in range(inputs.shape[0]):
        blob_idx = perm1[i]
        blob_idx1 = perm2[i]
        
        lam = np.random.beta(beta, beta)
        lam1 = np.random.beta(beta, beta)

        bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), (1 - blob[blob_idx]*1), lam)
        bbx1_1, bby1_1, bbx2_1, bby2_1 = rand_bbox(inputs.size(), blob[blob_idx1]*1, lam1)

        area = (bby2 - bby1)*(bbx2 - bbx1)
        area1 = (bby2_1 - bby1_1)*(bbx2_1 - bbx1_1)
            
        if  area1 > 0 and  area > 0:
            ncont = inputs[blob_idx1, :, bbx1_1:bbx2_1, bby1_1:bby2_1].clone().unsqueeze(0)
            ncont = F.interpolate(ncont, size=(bbx2-bbx1, bby2-bby1), mode='bilinear', align_corners=True)
                
            inputs2[blob_idx, :, bbx1:bbx2, bby1:bby2] = ncont
            p1 = blob[blob_idx1, :, bbx1_1:bbx2_1, bby1_1:bby2_1].sum()
            p2 = blob[blob_idx1, :, :, :].sum()
            
            target_b_weights.append(p1/p2)
            target_b[i] = labels[blob_idx1]
        else:
            target_b_weights.append(1.0)
            target_b[i] = labels[blob_idx]
    if dbg:
        print('Original labels:', labels)
        print('Labels after replace:', target_b)
        print('New labels weightes:', target_b_weights)
        plot_batch(inputs, inputs2)
    
    return inputs2, target_b, target_b_weights


def train_with_gradmask(model, dataloader, criterion, optimizer, regularizer_rate):            
    model.train()  

    dataset_size = 0
    running_loss = 0.0
    running_loss_cd = 0.0
    running_corrects = 0
    
    target_labels = []
    cat_preds = []

    bar = tqdm(dataloader)
    for (inputs, blob, labels) in bar:
        dataset_size += inputs.shape[0]
        optimizer.zero_grad()
        
        inputs = inputs.to(device)
        labels = labels.to(device)
            
        blob = torch.FloatTensor(blob.float()).to(device)
        
        add_loss = torch.zeros(1,).cuda()
        inputs.requires_grad = True

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            blob = (1 - blob).type(torch.BoolTensor) #blob tem 1 nos relevantes, no caso do rrr, actdiff e gradmask o 1 deve ser a informacao irrelevante
            

            gradients = get_grad_contrast(inputs, outputs, labels)
            #print(gradients.shape, blob.shape)
            grad_loss = gradients * blob.float().cuda()
            add_loss = regularizer_rate * grad_loss.abs().sum() 

            joined_loss = loss + add_loss
            joined_loss.backward()
            # print how much memory is used
            #print(torch.cuda.memory_allocated()/(np.power(10,9)))
            optimizer.step()

        # statistics
        running_loss += float(loss.detach().cpu().data) * int(inputs.size(0))
        running_loss_cd += float(add_loss.detach().cpu().data) * int(inputs.size(0))
        running_corrects += float(torch.sum(preds.cpu().detach().data == labels.cpu().detach().data))
        
        cat_preds.extend(preds.cpu().detach().data.tolist())
        target_labels.extend(labels.cpu().detach().data.tolist())

        del add_loss
        del loss
        del preds                
        del joined_loss

    epoch_loss = running_loss / dataset_size
    epoch_rrr_loss = running_loss_cd / dataset_size
    epoch_acc = running_corrects / dataset_size

    summary = {
        'epoch_loss':epoch_loss,
        'epoch_rrr_loss':epoch_rrr_loss,
        'epoch_acc':epoch_acc,
        'target_labels':target_labels,
        'predictions':cat_preds
        }
    
    return summary


def train_with_actdiff(model, dataloader, criterion, optimizer, regularizer_rate):            
    model.train()  

    dataset_size = 0
    running_loss = 0.0
    running_loss_cd = 0.0
    running_corrects = 0
    
    target_labels = []
    cat_preds = []

    bar = tqdm(dataloader)
    for (inputs, blob, labels) in bar:
        dataset_size += inputs.shape[0]
        optimizer.zero_grad()
        
        inputs = inputs.to(device)
        labels = labels.to(device)
            
        blob = torch.FloatTensor(blob.float()).to(device)
        
        add_loss = torch.zeros(1,).cuda()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            #inputs_activations = model.avgpool.avgoutput
            inputs_activations = model.forward_features(inputs.detach())
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            blob = (1 - blob).type(torch.BoolTensor) #blob tem 1 nos relevantes, no caso do rrr e actdiff o 1 deve ser a informacao irrelevante
            inputs.requires_grad = True
            
            xmasked = shuffle_outside_mask(inputs, blob)
            #xmasked_outputs = model(xmasked)
            #xmasked_activations = model.avgpool.avgoutput
            xmasked_activations = model.forward_features(xmasked.detach())
            #print(inputs_activations.shape, xmasked_activations.shape, compare_activations(inputs_activations, xmasked_activations))
            #plot_xmasked(inputs, xmasked)
            #print(inputs_activations.shape, xmasked_activations.shape)
            add_loss = compare_activations(inputs_activations, xmasked_activations)
            joined_loss = loss + regularizer_rate*add_loss
            joined_loss.backward()
            # print how much memory is used
            #print(torch.cuda.memory_allocated()/(np.power(10,9)))
            optimizer.step()
        #print(inputs.shape)
        # statistics
        running_loss += float(loss.detach().cpu().data) * int(inputs.size(0))
        running_loss_cd += float(add_loss.detach().cpu().data) * int(inputs.size(0))
        running_corrects += float(torch.sum(preds.cpu().detach().data == labels.cpu().detach().data))
        
        cat_preds.extend(preds.cpu().detach().data.tolist())
        target_labels.extend(labels.cpu().detach().data.tolist())

        del add_loss
        del loss
        del preds                
        del joined_loss

    epoch_loss = running_loss / dataset_size
    epoch_rrr_loss = running_loss_cd / dataset_size
    epoch_acc = running_corrects / dataset_size

    summary = {
        'epoch_loss':epoch_loss,
        'epoch_rrr_loss':epoch_rrr_loss,
        'epoch_acc':epoch_acc,
        'target_labels':target_labels,
        'predictions':cat_preds
        }
    
    return summary


def train_with_actdiffbackground(model, dataloader, criterion, optimizer, regularizer_rate):            
    model.train()  

    dataset_size = 0
    running_loss = 0.0
    running_loss_cd = 0.0
    running_corrects = 0
    
    target_labels = []
    cat_preds = []

    bar = tqdm(dataloader)
    for (inputs, blob, labels) in bar:
        dataset_size += inputs.shape[0]
        optimizer.zero_grad()
        
        inputs = inputs.to(device)
        labels = labels.to(device)
            
        blob = torch.FloatTensor(blob.float()).to(device)
        
        add_loss = torch.zeros(1,).cuda()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            #inputs_activations = model.avgpool.avgoutput
            inputs_activations = model.forward_features(inputs.detach())
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            #blob = (1 - blob).type(torch.BoolTensor) #blob tem 1 nos relevantes, no caso do rrr e actdiff o 1 deve ser a informacao irrelevante
            inputs.requires_grad = True
            blob[blob > 0] = 1.0
            #xmasked = shuffle_outside_mask(inputs, blob)
            x = inputs
            backgrounds = batch_transform(x, blob).cuda()
            perm = torch.randperm(x.shape[0])
            xmasked = x*blob + (1 - blob)*backgrounds[perm]
            #xmasked_outputs = model(xmasked)
            #xmasked_activations = model.avgpool.avgoutput
            xmasked_activations = model.forward_features(xmasked.detach())
            #print(inputs_activations.shape, xmasked_activations.shape, compare_activations(inputs_activations, xmasked_activations))
            #plot_xmasked(inputs, xmasked)
            add_loss = compare_activations(inputs_activations, xmasked_activations)
            joined_loss = loss + regularizer_rate*add_loss
            joined_loss.backward()
            # print how much memory is used
            #print(torch.cuda.memory_allocated()/(np.power(10,9)))
            optimizer.step()
        #print(inputs.shape)
        # statistics
        running_loss += float(loss.detach().cpu().data) * int(inputs.size(0))
        running_loss_cd += float(add_loss.detach().cpu().data) * int(inputs.size(0))
        running_corrects += float(torch.sum(preds.cpu().detach().data == labels.cpu().detach().data))
        
        cat_preds.extend(preds.cpu().detach().data.tolist())
        target_labels.extend(labels.cpu().detach().data.tolist())

        del add_loss
        del loss
        del preds                
        del joined_loss

    epoch_loss = running_loss / dataset_size
    epoch_rrr_loss = running_loss_cd / dataset_size
    epoch_acc = running_corrects / dataset_size

    summary = {
        'epoch_loss':epoch_loss,
        'epoch_rrr_loss':epoch_rrr_loss,
        'epoch_acc':epoch_acc,
        'target_labels':target_labels,
        'predictions':cat_preds
        }
    
    return summary


def train_with_standardbackground(model, dataloader, criterion, optimizer, regularizer_rate):            
    model.train()  

    dataset_size = 0
    running_loss = 0.0
    running_loss_cd = 0.0
    running_corrects = 0
    
    target_labels = []
    cat_preds = []

    bar = tqdm(dataloader)
    for (inputs, blob, labels) in bar:
        dataset_size += inputs.shape[0]
        optimizer.zero_grad()
        
        inputs = inputs.to(device)
        labels = labels.to(device)
            
        blob = torch.FloatTensor(blob.float()).to(device)
        
        add_loss = torch.zeros(1,).cuda()

        with torch.set_grad_enabled(True):
            #outputs = model(inputs)
            #inputs_activations = model.avgpool.avgoutput
            #inputs_activations = model.forward_features(inputs.detach())
            #_, preds = torch.max(outputs, 1)
            #blob = (1 - blob).type(torch.BoolTensor) #blob tem 1 nos relevantes, no caso do rrr e actdiff o 1 deve ser a informacao irrelevante
            inputs.requires_grad = True
            blob[blob > 0] = 1.0

            #xmasked = shuffle_outside_mask(inputs, blob)
            x = inputs

            backgrounds = batch_transform(x, blob).cuda()
            perm = torch.randperm(x.shape[0])
            xmasked = x*blob + (1 - blob)*backgrounds[perm]

            outputs = model(xmasked)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            add_loss = 0.0

            joined_loss = loss + regularizer_rate*add_loss
            joined_loss.backward()

            # print how much memory is used
            #print(torch.cuda.memory_allocated()/(np.power(10,9)))

            optimizer.step()
        #print(inputs.shape)
        # statistics
        running_loss += float(loss.detach().cpu().data) * int(inputs.size(0))
        running_loss_cd += 0.0
        running_corrects += float(torch.sum(preds.cpu().detach().data == labels.cpu().detach().data))
        
        cat_preds.extend(preds.cpu().detach().data.tolist())
        target_labels.extend(labels.cpu().detach().data.tolist())

        del add_loss
        del loss
        del preds                
        del joined_loss

    epoch_loss = running_loss / dataset_size
    epoch_rrr_loss = running_loss_cd / dataset_size
    epoch_acc = running_corrects / dataset_size

    summary = {
        'epoch_loss':epoch_loss,
        'epoch_rrr_loss':epoch_rrr_loss,
        'epoch_acc':epoch_acc,
        'target_labels':target_labels,
        'predictions':cat_preds
        }
    
    return summary


def train_with_rrr(model, dataloader, criterion, optimizer, regularizer_rate):            
    model.train()  

    dataset_size = 0
    running_loss = 0.0
    running_loss_cd = 0.0
    running_corrects = 0
    
    target_labels = []
    cat_preds = []

    bar = tqdm(dataloader)
    for (inputs, blob, labels) in bar:
        dataset_size += inputs.shape[0]
        optimizer.zero_grad()
        
        inputs = inputs.to(device)
        labels = labels.to(device)
            
        blob = torch.FloatTensor(blob.float()).to(device)
        
        add_loss = torch.zeros(1,).cuda()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            blob = 1 - blob #blob tem 1 nos relevantes, no caso do rrr o 1 deve ser a informacao irrelevante
            inputs.requires_grad = True
            add_loss = score_funcs.gradient_sum(inputs, labels, blob, model, criterion)
            if add_loss > 1000.0:
                print('ADD LOSS:', add_loss)
                #plot_batch(inputs, blob)
                add_loss = 0.0*add_loss
            joined_loss = loss + regularizer_rate*add_loss
            joined_loss.backward()
            # print how much memory is used
            #print(torch.cuda.memory_allocated()/(np.power(10,9)))
            optimizer.step()

        # statistics
        running_loss += float(loss.detach().cpu().data) * int(inputs.size(0))
        running_loss_cd += float(add_loss.detach().cpu().data) * int(inputs.size(0))
        running_corrects += float(torch.sum(preds.cpu().detach().data == labels.cpu().detach().data))
        
        cat_preds.extend(preds.cpu().detach().data.tolist())
        target_labels.extend(labels.cpu().detach().data.tolist())

        del add_loss
        del loss
        del preds                
        del joined_loss

    epoch_loss = running_loss / dataset_size
    epoch_rrr_loss = running_loss_cd / dataset_size
    epoch_acc = running_corrects / dataset_size

    summary = {
        'epoch_loss':epoch_loss,
        'epoch_rrr_loss':epoch_rrr_loss,
        'epoch_acc':epoch_acc,
        'target_labels':target_labels,
        'predictions':cat_preds
        }
    
    return summary


def train_with_cdep(model, dataloader, criterion, optimizer, regularizer_rate):            
    model.train()  

    dataset_size = 0
    running_loss = 0.0
    running_loss_cd = 0.0
    running_corrects = 0
    
    target_labels = []
    cat_preds = []

    bar = tqdm(dataloader)
    for inputs, blob, labels in bar:
        dataset_size += inputs.shape[0]
        optimizer.zero_grad()
        
        inputs = inputs.to(device)
        labels = labels.to(device)
            
        blob = torch.FloatTensor(blob.float()).to(device)
        
        relevant = blob * inputs
        relevant = relevant.to(device)
        relevant.requires_grad = False

        irrelevant = (1 - blob) * inputs
        irrelevant = irrelevant.to(device)
        irrelevant.requires_grad = False

        add_loss = torch.zeros(1,).cuda()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            mask = torch.tensor([blob[kx].sum() > 0 for kx in range(inputs.size(0))]).cuda()
            if mask.any():
                #print(mask.shape, type(mask), mask)
                #rel, irrel = cd.cd_vgg_classifier(rel, irrel, inputs_feat, model.classifier)
                rel, irrel = cd_architecture_specific.cd_propagate_resnet(relevant, irrelevant, model)
                cur_cd_loss1 = torch.nn.functional.softmax(torch.stack((rel[:,0].masked_select(mask), irrel[:,0].masked_select(mask)), dim =1), dim = 1)[:,0].mean() 
                cur_cd_loss2 = torch.nn.functional.softmax(torch.stack((rel[:,1].masked_select(mask), irrel[:,1].masked_select(mask)), dim =1), dim = 1)[:,0].mean()
                add_loss = (cur_cd_loss1 + cur_cd_loss2)/2

            joined_loss = loss + regularizer_rate*add_loss
            joined_loss.backward()
            # print how much memory is used
            #print(torch.cuda.memory_allocated()/(np.power(10,9)))
            optimizer.step()

        # statistics
        running_loss += float(loss.detach().cpu().data) * int(inputs.size(0))
        running_loss_cd += float(add_loss.detach().cpu().data) * int(inputs.size(0))
        running_corrects += float(torch.sum(preds.cpu().detach().data == labels.cpu().detach().data))
        
        cat_preds.extend(preds.cpu().detach().data.tolist())
        target_labels.extend(labels.cpu().detach().data.tolist())

        del add_loss
        del loss
        del preds                
        del joined_loss

    epoch_loss = running_loss / dataset_size
    epoch_rrr_loss = running_loss_cd / dataset_size
    epoch_acc = running_corrects / dataset_size

    summary = {
        'epoch_loss':epoch_loss,
        'epoch_rrr_loss':epoch_rrr_loss,
        'epoch_acc':epoch_acc,
        'target_labels':target_labels,
        'predictions':cat_preds
        }
    
    return summary


def train_with_ada(model, dataloader, criterion, optimizer):            
    model.train()  

    dataset_size = 0
    running_loss = 0.0
    running_corrects = 0
    
    target_labels = []
    cat_preds = []

    bar = tqdm(dataloader)
    for inputs, blob, labels in bar:
        dataset_size += inputs.shape[0]
        optimizer.zero_grad()
        
        inputs = inputs.to(device)
        labels = labels.to(device)
            
        blob = torch.FloatTensor(blob.float()).to(device)
        
        input_shape = (inputs.shape[1], inputs.shape[2])
        idx = torch.randint(0, inputs.shape[0], size=(1,)) #chose one image in batch
        
        chosed_x = inputs[idx]
        chosed_x_shape = chosed_x.shape
        
        # 1. GET THE INTERPRETABILITY AND SELECT THE POSITION OF THE MOST IMPORTANT BACKGROUND PIXEL 
        att = ada.explain(
            copy.deepcopy(model), 
            x=chosed_x, 
            att_method='saliency' if inputs.shape[-1] != 32 else 'gradcam', 
            y=labels[idx],
            shape_img=input_shape,
            ).abs()
                
        new_blob = torch.clamp(
            ((blob[idx] > 0.0) * 1).sum(dim=1), 
            max=1
            )
        att = att * (1 - new_blob) #keep only the interpretability of background pixels
        
        pos_max = att.flatten(start_dim=1, end_dim=2).argmax().detach().cpu().int()
        posx, posy = pos_max // inputs.shape[2], pos_max % inputs.shape[3]
        
        beta = 1.0

        # 2. BUILD THE MASK TO CUTOFF THE REGION AROUND THE PIXEL SELECTED IN 1
        out = get_bbox(
            chosed_x_shape, 
            pos=(posy, posx), 
            lam=np.random.beta(beta, beta))
        
        mask = build_mask(
                shape=chosed_x_shape, 
                bbox=out, 
                blobs=(blob[idx] > 0.0) * 1
                )
        
        # 3. CUTOFF THE PIXEL WITH MASK BUILT IN 2
        inputs[idx] = chosed_x * mask
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        # statistics
        running_loss += float(loss.detach().cpu().data) * int(inputs.size(0))
        running_corrects += float(torch.sum(preds.cpu().detach().data == labels.cpu().detach().data))
        
        cat_preds.extend(preds.cpu().detach().data.tolist())
        target_labels.extend(labels.cpu().detach().data.tolist())

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects / dataset_size

    summary = {
        'epoch_loss':epoch_loss,
        'epoch_acc':epoch_acc,
        'target_labels':target_labels,
        'predictions':cat_preds
        }
    
    return summary


def train_with_adak(model, dataloader, criterion, optimizer):            
    model.train()  

    dataset_size = 0
    running_loss = 0.0
    running_corrects = 0
    
    target_labels = []
    cat_preds = []

    bar = tqdm(dataloader)
    for inputs, blob, labels in bar:
        dataset_size += inputs.shape[0]
        optimizer.zero_grad()
        
        inputs = inputs.to(device)
        labels = labels.to(device)
            
        blob = torch.FloatTensor(blob.float()).to(device)
        
        input_shape = (inputs.shape[1], inputs.shape[2])
        idx = torch.randint(0, inputs.shape[0], size=(1,)) #chose one image in batch
        
        chosed_x = inputs[idx]
        chosed_x_shape = chosed_x.shape
        
        masks = adak(copy.deepcopy(model), inputs, blob, labels)
        x = inputs
        backgrounds = batch_transform(x, blob).cuda()
        perm = torch.randperm(x.shape[0])
        xmasked = x*masks + (1 - masks)*backgrounds[perm] #remove the most important patch in background and insert others from batch images
        
        #plot_xmasked_save(x, xmasked, int(datetime.timestamp(datetime.now())))

        outputs = model(xmasked)
        _, preds = torch.max(outputs, 1)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        # statistics
        running_loss += float(loss.detach().cpu().data) * int(inputs.size(0))
        running_corrects += float(torch.sum(preds.cpu().detach().data == labels.cpu().detach().data))
        
        cat_preds.extend(preds.cpu().detach().data.tolist())
        target_labels.extend(labels.cpu().detach().data.tolist())

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects / dataset_size

    summary = {
        'epoch_loss':epoch_loss,
        'epoch_acc':epoch_acc,
        'target_labels':target_labels,
        'predictions':cat_preds
        }
    
    return summary


def train_with_std(model, dataloader, criterion, optimizer):            
    model.train()  

    dataset_size = 0
    running_loss = 0.0
    running_corrects = 0
    
    target_labels = []
    cat_preds = []

    bar = tqdm(dataloader)
    for inputs, blob, labels in bar:
        dataset_size += inputs.shape[0]
        optimizer.zero_grad()
        
        inputs = inputs.to(device)
        labels = labels.to(device)
            
        blob = torch.FloatTensor(blob.float()).to(device)
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += float(loss.detach().cpu().data) * int(inputs.size(0))
        running_corrects += float(torch.sum(preds.cpu().detach().data == labels.cpu().detach().data))
        
        cat_preds.extend(preds.cpu().detach().data.tolist())
        target_labels.extend(labels.cpu().detach().data.tolist())

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects / dataset_size

    summary = {
        'epoch_loss':epoch_loss,
        'epoch_acc':epoch_acc,
        'target_labels':target_labels,
        'predictions':cat_preds
        }
    
    return summary


def train_with_fgsm(model, dataloader, criterion, optimizer):            
    model.train()  

    dataset_size = 0
    running_loss = 0.0
    running_corrects = 0
    
    target_labels = []
    cat_preds = []

    bar = tqdm(dataloader)
    for inputs, blob, labels in bar:
        dataset_size += inputs.shape[0]
        optimizer.zero_grad()
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        y = labels
        blob = torch.FloatTensor(blob.float()).to(device)
        
        delta = attack_utils.attack_fgsm(model, inputs.cuda(), y.cuda())
        outputs = model((inputs + delta.cuda()))

        #outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += float(loss.detach().cpu().data) * int(inputs.size(0))
        running_corrects += float(torch.sum(preds.cpu().detach().data == labels.cpu().detach().data))
        
        cat_preds.extend(preds.cpu().detach().data.tolist())
        target_labels.extend(labels.cpu().detach().data.tolist())

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects / dataset_size

    summary = {
        'epoch_loss':epoch_loss,
        'epoch_acc':epoch_acc,
        'target_labels':target_labels,
        'predictions':cat_preds
        }
    
    return summary


def evaluate_std(model, dataloader, criterion):            
    model.eval()  

    dataset_size = 0
    running_loss = 0.0
    running_corrects = 0
    
    target_labels = []
    cat_preds = []

    bar = tqdm(dataloader)
    for inputs, blob, labels in bar:
        dataset_size += inputs.shape[0]
        
        inputs = inputs.to(device)
        labels = labels.to(device)
            
        blob = torch.FloatTensor(blob.float()).to(device)
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()

        # statistics
        running_loss += float(loss.detach().cpu().data) * int(inputs.size(0))
        running_corrects += float(torch.sum(preds.cpu().detach().data == labels.cpu().detach().data))
        
        cat_preds.extend(preds.cpu().detach().data.tolist())
        target_labels.extend(labels.cpu().detach().data.tolist())

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects / dataset_size

    summary = {
        'epoch_loss':epoch_loss,
        'epoch_acc':epoch_acc,
        'target_labels':target_labels,
        'predictions':cat_preds
        }
    
    return summary


def train_model_generic(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()
    val_acc_history = []
    val_loss_history = []
    train_loss_history = []
    
    train_acc_history = []
    train_cd_history= []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_f1 = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            dataset_size = 0
            loss_curve = []
            if phase == 'train':
                optimizer.zero_grad()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_loss_cd = 0.0
            running_corrects = 0
            running_count_samples = 0
            # Iterate over data.
            target_labels = []
            cat_preds = []
            bar = tqdm(dataloaders[phase])
            for (inputs, blob, labels) in bar:
                dataset_size += inputs.shape[0]
                optimizer.zero_grad()
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                    
                optimizer.zero_grad()

                #blob = blob.to(device)
                #blob = blob.unsqueeze(1)
                blob = torch.FloatTensor(blob.float()).to(device)
                #print(blob.shape, inputs.shape)
                relevant = blob * inputs
                relevant = relevant.to(device)
                relevant.requires_grad = False

                irrelevant = (1 - blob) * inputs
                irrelevant = irrelevant.to(device)
                irrelevant.requires_grad = False
                
                add_loss = torch.zeros(1,).cuda()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    if phase == 'val':
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train' and TRAIN_METHOD == 'CDEP':
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        #plt.imshow(mask)
                        #plt.show()
                        mask = torch.tensor([blob[kx].sum() > 0 for kx in range(inputs.size(0))]).cuda()
                        if mask.any():
                            #print(mask.shape, type(mask), mask)
                            #rel, irrel = cd.cd_vgg_classifier(rel, irrel, inputs_feat, model.classifier)
                            rel, irrel = cd_architecture_specific.cd_propagate_resnet(relevant, irrelevant, model)
                            cur_cd_loss1 = torch.nn.functional.softmax(torch.stack((rel[:,0].masked_select(mask), irrel[:,0].masked_select(mask)), dim =1), dim = 1)[:,0].mean() 
                            cur_cd_loss2 = torch.nn.functional.softmax(torch.stack((rel[:,1].masked_select(mask), irrel[:,1].masked_select(mask)), dim =1), dim = 1)[:,0].mean()
                            add_loss = (cur_cd_loss1 + cur_cd_loss2)/2

                        joined_loss = loss + regularizer_rate*add_loss
                        joined_loss.backward()
                        # print how much memory is used
                        #print(torch.cuda.memory_allocated()/(np.power(10,9)))
                        optimizer.step()
                    
                    if phase == 'train' and TRAIN_METHOD == 'RRR':
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        blob = 1 - blob #blob tem 1 nos relevantes, no caso do rrr o 1 deve ser a informacao irrelevante
                        inputs.requires_grad = True
                        add_loss = score_funcs.gradient_sum(inputs, labels, blob, model, criterion)
                        joined_loss = loss + regularizer_rate*add_loss
                        joined_loss.backward()
                        # print how much memory is used
                        #print(torch.cuda.memory_allocated()/(np.power(10,9)))
                        optimizer.step()

                    if phase == 'train' and TRAIN_METHOD == 'ADA':
                        input_shape = (inputs.shape[1], inputs.shape[2])
                        idx = torch.randint(0, inputs.shape[0], size=(1,))
                        chosed_x = inputs[idx]
                        chosed_x_shape = chosed_x.shape
                        #print('chosed x', chosed_x_shape)
                        att = ada.explain(
                            copy.deepcopy(model), 
                            x=chosed_x, 
                            att_method='saliency', 
                            y=labels[idx],
                            shape_img=input_shape,
                            ).abs()
                        #print('Attribution shape:', att.shape, att.max(), att.min(), att.abs().max(), att.abs().min())
                        
                        new_blob = torch.clamp(
                            ((blob[idx] > 0.0) * 1).sum(dim=1), 
                            max=1
                            )
                        #print("Blob shape:", ((blob[idx] > 0.0) * 1).shape, new_blob.shape, new_blob.unique(), (1 - new_blob).shape, att.shape, (att * (1 - new_blob)).shape)
                        att = att * (1 - new_blob)
                        
                        #fig, axs = plt.subplots(1, 3, figsize=(5, 10))
                        #axs[0].imshow(att.detach().cpu().permute(1, 2, 0))
                        #axs[1].imshow((1 - new_blob).detach().cpu().permute(1, 2, 0))
                        #axs[2].imshow((att * (1 - new_blob)).detach().cpu().permute(1, 2, 0))
                        #plt.show()

                        pos_max = att.flatten(start_dim=1, end_dim=2).argmax().detach().cpu().int()
                        #print("POS MAX:", pos_max, att.flatten(start_dim=1, end_dim=2).max())

                        
                       
                        #pos_max = att.flatten(start_dim=2, end_dim=3).argmax().detach().cpu().int()
                        #print("POS MAX AFTER:", pos_max)
                        
                        #att = att.unsqueeze(1)
                        #pos_max = att.flatten(start_dim=2, end_dim=3).argmax().detach().cpu().int()
                        posx, posy = pos_max // inputs.shape[2], pos_max % inputs.shape[3]
                        #print('POS:')
                        #print('\t', posx, posy)
                        beta = 1.0

                        out = get_bbox(
                            chosed_x_shape, 
                            pos=(posy, posx), 
                            lam=np.random.beta(beta, beta))
                        #print('BBOX:')
                        #print('\t',out)
                        mask = build_mask(
                                shape=chosed_x_shape, 
                                bbox=out, 
                                blobs=(blob[idx] > 0.0) * 1
                                )
                        #print(mask.shape)
                        #print(blob[idx].max(), blob[idx].min())
                        #print(mask.max(), mask.min())
                        #print(chosed_x.shape, mask.shape)
                        #plot(chosed_x, (blob[idx] > 0.0) * 1, mask)
                        inputs[idx] = chosed_x * mask
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += float(loss.detach().cpu().data) * int(inputs.size(0))
                running_loss_cd += float(add_loss.detach().cpu().data) * int(inputs.size(0))
                running_corrects += float(torch.sum(preds.cpu().detach().data == labels.cpu().detach().data))
                running_count_samples += int(inputs.size(0))
                
                cat_preds.extend(preds.cpu().detach().data.tolist())
                target_labels.extend(labels.cpu().detach().data.tolist())

                del add_loss
                del loss
                del preds

                if phase == "train" and (TRAIN_METHOD == 'RRR' or TRAIN_METHOD == 'CDEP'):
                    del joined_loss
                loss_curve.append(round(running_loss/running_count_samples, 3))
                bar.set_description('{} Loss: {:.4f} Acc: {:.4f} CD Loss : {:.4f}'.format(
                phase, round(running_loss/running_count_samples, 3), round(running_corrects/running_count_samples, 3), round(running_loss_cd/running_count_samples, 3)))

            #plt.plot(loss_curve)
            #plt.show()
            epoch_loss = running_loss / dataset_size
            epoch_cd_loss = running_loss_cd / dataset_size
            epoch_acc = running_corrects / dataset_size
  
            
            print('{} Loss: {:.4f} Acc: {:.4f} CD Loss : {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_cd_loss))

            # deep copy the model
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
                val_f1 = f1_score(target_labels, cat_preds, average="weighted")
                print("F1 WEIGHTED:", val_f1)

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_cd_history.append(epoch_cd_loss)
                train_acc_history.append(epoch_acc)
                
            if phase == 'val':
                if val_f1 > best_val_f1:
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_val_f1 = val_f1
                print("BEST VAL F1 {}".format(best_val_f1))
             
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val loss: {:4f}'.format(best_loss))
    print('Best val f1: {:4f}'.format(best_val_f1))
  
    hist_dict = {}
    hist_dict['val_acc_history'] = val_acc_history
    hist_dict['val_loss_history'] = val_loss_history
    hist_dict['train_acc_history'] = train_acc_history
    hist_dict['train_loss_history'] = train_loss_history
    hist_dict['train_cd_history'] = train_cd_history
    model.load_state_dict(best_model_wts)

    return model, hist_dict 
