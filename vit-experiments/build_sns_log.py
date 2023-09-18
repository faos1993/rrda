import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import datasets
import models
import robustness_evaluate as re
from tqdm.notebook import tqdm
import timm
import random
from PIL import Image, ImageFilter
import random
import torch.nn as nn
from captum.attr import Saliency, IntegratedGradients
import seaborn as sns
import torch.nn.functional as F
from torchvision.transforms import Resize
import cv2
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image

IMAGENET9_DIR = "path_to_imagenet9_dir"
IMAGENET9_BGCHALLENGE_DIR = os.path.join(IMAGENET9_DIR, 'bg_challenge')
TMETHODS = ['standard', 'standardbackground', 'rrr', 'ada', 'actdiff', 'actdiffbackground', 'gradmask', 'fgsm']
DATASETS = ['OxfordFlower', 'CUB', 'Cars']

device = 'cuda'

def transforms_test(image, img_size):
    if img_size == 448:
        sizs = [512, 448]
    elif img_size == 224:
        sizs = [256, 224]
    elif img_size == 128:
        sizs = [160, 128]
    elif img_size == 96:
        sizs = [128, 96]
    elif img_size == 32:
        sizs = [48, 32]

    resize = transforms.Resize(size=(sizs[0], sizs[0]))
    image = resize(image)

    #if random.random():
    # Random crop
    ccrop = transforms.CenterCrop(size=(sizs[1], sizs[1]))
    image = ccrop(image)

    # Transform to tensor
    image = TF.to_tensor(image)

    if image.shape[0] == 1:
        image = torch.cat([image, image, image], dim=0)

    image = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    return image


class Imagenet9Challenge():
    
    def __init__(self, split='train', img_size=224):
        self.dir = IMAGENET9_BGCHALLENGE_DIR
        self.original = os.path.join(self.dir, split, 'val')
        self.split = split
        self.img_size = img_size
        self.paths = []
        self.imgs = []
        self.masks = []
        self.targets = []
        self.class_weights = None
        self.load()
    
    def compute_class_weights(self):
        self.class_weights = np.zeros(len(set(self.targets)))
        one_based = 1 * (np.array(self.targets).min() == 1)
        
        if one_based:
            for i in range(len(self.targets)):
                self.targets[i] = self.targets[i] - one_based

        for c in self.targets:
            self.class_weights[c] += 1

        total = self.class_weights.sum()
        
        self.class_weights = self.class_weights/total

    def load(self):
        path_images = self.original
        for idx, folder in enumerate(sorted(os.listdir(path_images))):
            #print(idx, folder)
            class_dir = os.path.join(path_images, folder)
            for img in os.listdir(class_dir):
                path_img_orig = os.path.join(class_dir, img)
                self.paths.append(path_img_orig)
            
                if '.npy' in path_img_orig:
                    img2 = np.load(path_img_orig, allow_pickle=True)
                elif '.jpg' in path_img_orig:
                    img2 = Image.open(path_img_orig)
                    img2 = np.asarray(img2)
                elif '.JPEG' in path_img_orig:
                    img2 = Image.open(path_img_orig)
                    img2 = np.asarray(img2)
                    
                
                self.imgs.append(img2)
                self.targets.append(idx)
        
        self.num_of_categories = len(set(self.targets))
        self.compute_class_weights()
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        img = Image.fromarray(self.imgs[idx])
        y = self.targets[idx]
        
        
        img = transforms_test(img, self.img_size)

        if img.shape[0] == 1:
            img = torch.cat([img, img, img], 0)
            
        return img, 0, y


class IN9Joint():
    
    def __init__(self, img_size=224):
        self.dir = IMAGENET9_BGCHALLENGE_DIR
        self.img_size = img_size
        self.modes = ['original', 'fg_mask', 'only_fg', 'mixed_rand', 'mixed_same', 'mixed_next']
        
        print('Loading original ...')
        original = Imagenet9Challenge('original', 224)
        
        print('Loading fg mask ...')
        fg_mask = Imagenet9Challenge('fg_mask', 224)
        
        print('Loading only fg ...')
        fg_dataset = Imagenet9Challenge('only_fg', 224)
        
        print('Loading mixed rand ...')
        mr_dataset = Imagenet9Challenge('mixed_rand', 224)
        
        print('Loading mixed same ...')
        ms_dataset = Imagenet9Challenge('mixed_same', 224)
        
        print('Loading mixed next ...')
        mn_dataset = Imagenet9Challenge('mixed_next', 224)
        
        self.property_dataset = {
            'original':sorted(original.paths),
            'fg_mask':sorted(fg_mask.paths),
            'only_fg':sorted(fg_dataset.paths),
            'mixed_rand':sorted(mr_dataset.paths),
            'mixed_same':sorted(ms_dataset.paths),
            'mixed_next':sorted(mn_dataset.paths),
            'target':original.targets,
        }
        
        self.targets = original.targets
        self.df = pd.DataFrame.from_dict(self.property_dataset)
        self.class_weights = None
    
    def compute_class_weights(self):
        self.class_weights = np.zeros(len(set(self.targets)))
        one_based = 1 * (np.array(self.targets).min() == 1)
        
        if one_based:
            for i in range(len(self.targets)):
                self.targets[i] = self.targets[i] - one_based

        for c in self.targets:
            self.class_weights[c] += 1

        total = self.class_weights.sum()
        #for i in range(self.class_weights.shape[0]):
        #    self.class_weights[i] = self.class_weights[i]/total
        self.class_weights = self.class_weights/total

    def load_img(self, path):
        path_img_orig = path
        
        if '.npy' in path_img_orig:
            img2 = np.load(path_img_orig, allow_pickle=True)
        elif '.jpg' in path_img_orig:
            img2 = Image.open(path_img_orig)
            img2 = np.asarray(img2)
        elif '.JPEG' in path_img_orig:
            img2 = Image.open(path_img_orig)
            img2 = np.asarray(img2)
            
        return img2
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        imgs = []
        
        for m in self.modes:
            img = self.load_img(self.property_dataset[m][idx])
            img = Image.fromarray(img)
            img = transforms_test(img, self.img_size)
            
            if img.shape[0] == 1:
                img = torch.cat([img, img, img], 0)
            
            if m == 'fg_mask': #guarantee pixels in {0, 1} after normalization
                img[img > 0] = 1.0
                img[img < 0] = 0.0
                
            imgs.append(img)
            
        y = self.property_dataset['target'][idx]
            
        return imgs, y

def get_best(dataset_name, tmethod, img_size=224):
    results_dir = f'results-{dataset_name}/'
    best_path = None
    best_eval = 0.0
    for file in [f for f in os.listdir(results_dir) if tmethod in f]:
        print(file)
        try:
            log_path = os.path.join(results_dir, file)
            log = torch.load(log_path)
            if str(img_size) in file and log['train_method'] == (tmethod+'-baseline'):
                if log['best_eval_acc'] > best_eval:
                    best_eval = log['best_eval_acc']
                    best_path = log_path
                    
        except Exception as e:
            print('Error:',e)
            continue
    print()
    print('best path'.upper(), best_path)
    print('accuracy:'.upper(), best_eval)

    return best_path, best_eval


def get_model_from_path(path, num_of_categories):
    summary = torch.load(path)
    keys = summary.keys()
    #print(keys)
    print('Train method:', summary['train_method'])
    print('Best eval acc:', summary['best_eval_acc'])
    print('regularizer_rate', summary['regularizer_rate'])
    
    if summary['regularizer_rate'] == 100.0:
        r = 'r3'
    elif summary['regularizer_rate'] == 10.0:
        r = 'r2'
    else:
        r = 'r1'
    device = 'cuda'

    def printnorm(self, input, output):
        self.avgoutput = output
        
    # Set model
    #model = models.get_resnet18(num_classes=num_of_categories, pretrained=False)
    #model.load_state_dict(summary['best_ckp'])
    #model.avgpool.register_forward_hook(printnorm)
    #model.to(device)
    
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_of_categories)
    model.load_state_dict(summary['best_ckp'])
    model = model.to(device)
    
    return model, r



dataset_name = 'Imagenet9'
name2models = {}
num_of_categories = 9
models = ['standard', 'standardbackground', 'actdiff', 'gradmask', 'actdiffbackground', 'ada', 'rrr']
for m in models:
    try:
        path, acc = get_best(dataset_name, m)
        name2models[m] = get_model_from_path(path, num_of_categories)[0]
        name2models[m].eval()
    except Exception as e:
        print(e)

def norm(x):
    xmin = x.min()
    xmax = x.max()
    x = (x - xmin)/(xmax - xmin)
    return x

def process(img):
    return norm(img.permute((1, 2, 0)))

class SNIN9():
    
    def __init__(self, ch_name, edge_width, img_size=224):
        assert ch_name in ['original', 'mixed_rand', 'mixed_same', 'mixed_next']
        self.dir = IMAGENET9_BGCHALLENGE_DIR
        self.img_size = img_size
        self.modes = [ch_name, 'fg_mask']
        self.ch_name = ch_name
        
        print(f'Loading {ch_name} ...')
        original = Imagenet9Challenge(ch_name, 224)
        
        print('Loading fg mask ...')
        fg_mask = Imagenet9Challenge('fg_mask', 224)
        
        self.property_dataset = {
            self.ch_name:sorted(original.paths),
            'fg_mask':sorted(fg_mask.paths),
            'target':original.targets,
        }
        
        self.targets = original.targets
        self.df = pd.DataFrame.from_dict(self.property_dataset)
        
        self.class_weights = None
        self.edge_width = edge_width

    def load_img(self, path):
        path_img_orig = path
        
        if '.npy' in path_img_orig:
            img2 = np.load(path_img_orig, allow_pickle=True)
        elif '.jpg' in path_img_orig:
            img2 = Image.open(path_img_orig)
            img2 = np.asarray(img2)
        elif '.JPEG' in path_img_orig:
            img2 = Image.open(path_img_orig)
            img2 = np.asarray(img2)
            
        return img2
    
    def edge_process(self, imgs):
        """
        Expect everyone as numpy array
        """
        img = imgs[0]
        mask = imgs[1]
        #mask = imgs[1].numpy()
        mask = mask * 255
        mask = mask.astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        kernel_dilate = np.ones((self.edge_width, self.edge_width), np.uint8)
        
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        image = Image.fromarray(closing.transpose((1, 2, 0)), 'RGB')
        image = image.filter(ImageFilter.ModeFilter(size=13))
        
        closing = np.asarray(image).transpose((2, 0, 1))
        mask_after_filter = closing.copy()
                
        image_erode = cv2.erode(closing, kernel)
        image_dilate = cv2.dilate(closing, kernel_dilate)

        erode01 = image_erode.copy()
        erode01[erode01 > 0] = 1
        erode01[erode01 < 0] = 0


        dilate01 = image_dilate.copy()
        dilate01[dilate01 > 0] = 1
        dilate01[dilate01 < 0] = 0

        borda = erode01 * imgs[0]

        background_dilate = (1 - dilate01) * imgs[0]
        foreground_dilate = dilate01 * imgs[0]

        #print(borda.max(), borda.min())

        edge = (1 - erode01) * dilate01 
        edge_img = edge * imgs[0]
        no_edge_img = (1 - edge) * imgs[0] 
        imgs[1] = mask_after_filter
        
        return imgs[0], imgs[1], no_edge_img, edge_img, edge
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        imgs = []
        
        for m in self.modes:
            img = self.load_img(self.property_dataset[m][idx])
            
            if len(img.shape) == 3 and img.shape[0] == 1:
                img = np.concatenate([img, img, img], 0)
            elif len(img.shape) == 2:
                img = img[np.newaxis, :, :]
                img = np.concatenate([img, img, img], 0)
                
            if m == 'fg_mask': #guarantee pixels in {0, 1} after normalization
                img[img > 0] = 1.0
                img[img < 0] = 0.0
                
            if img.shape == (224, 224, 3):
                img = img.transpose((2, 0, 1))
                
            imgs.append(img)
        
        img, mask, noedge, edge, edge_mask = self.edge_process(imgs)
        mask = mask * 1.0
        mask[mask > 0] = 255.0
        mask[mask < 0] = 0.0
        edge_mask[edge_mask > 0] = 255
        edge_mask[edge_mask < 0] = 0
        
        img = transforms_test(Image.fromarray(img.transpose((1, 2, 0))), self.img_size)
        noedge = transforms_test(Image.fromarray(noedge.transpose((1, 2, 0))), self.img_size)
        edge = transforms_test(Image.fromarray(edge.transpose((1, 2, 0))), self.img_size)
        mask = transforms_test(Image.fromarray(mask.transpose((1, 2, 0)).astype(np.uint8)), self.img_size)
        edge_mask = transforms_test(Image.fromarray(edge_mask.transpose((1, 2, 0)).astype(np.uint8)), self.img_size)
        
        mask[mask > 0] = 1.0
        mask[mask < 0] = 0.0
        edge_mask[edge_mask > 0] = 1.0
        edge_mask[edge_mask < 0] = 0.0
            
        y = self.property_dataset['target'][idx]
        
        return img, mask, noedge, edge, edge_mask, y

challenge_name = 'mixed_next'
challenge_name = 'original'
challenge_name = 'mixed_same'
challenge_name = 'mixed_rand'

e8 = SNIN9(ch_name=challenge_name, edge_width=25)

loader8 = torch.utils.data.DataLoader(
        e8,
        batch_size=8,
        shuffle=True, 
        num_workers=15,
        #pin_memory=False
    )

from captum.attr import InputXGradient
from tqdm.notebook import tqdm
import seaborn
import random
from datetime import datetime

data = {
    'img_pos':[],
    'signal_rate':[],
    'back_rate':[],
    'signal_count':[],
    'back_count':[],
    'sn':[],
    'model':[],
    'target':[],
}

processed = 0
interpretability_method = 'saliency'

for idx in range(len(e8)):
    img, mask, noedge, edge, edge_mask, y = e8[idx]

    img = img.unsqueeze(0)
    mask = mask.unsqueeze(0)

    mask = mask.cuda()
    mask = mask.mean(1)

    mask[mask > 0] = 1.0
    mask[mask < 0] = 0.0

    signal_count = mask.sum((1, 2)).cpu().item()
    back_count = (1 - mask).sum((1, 2)).cpu().item()
    pos = 0

    for model_name in name2models.keys():
        model = name2models[model_name]
        ixg = Saliency(model)
        sns = []

        att = ixg.attribute(img.cuda(), y)
        att = att.sum(1).abs()

        signal_rate = (att * mask).sum((1, 2)).cpu().item()
        back_rate = (att * (1 - mask)).sum((1, 2)).cpu().item()

        try:
            sn = (signal_rate/signal_count) / (back_rate/back_count)
        except:
            sn = signal_rate/signal_count
        #print(f'\t{sn}')

        del ixg
        pos += 1

        data['img_pos'].append(idx)
        data['signal_rate'].append(signal_rate)
        data['back_rate'].append(back_rate)
        data['signal_count'].append(signal_count)
        data['back_count'].append(back_count)
        data['sn'].append(sn)
        data['model'].append(model_name)
        data['target'].append(y)
    processed += 1
    if (processed % 500) == 0:
        print(processed, datetime.now())

os.makedirs('signal2noise-log', exist_ok=True)

pd.DataFrame.from_dict(data).to_pickle(f"{interpretability_method}-{challenge_name}.pkl")
