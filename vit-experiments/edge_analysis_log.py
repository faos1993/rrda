import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import datasets
import models
import robustness_evaluate as re
from tqdm import tqdm
import timm
import random
from PIL import Image, ImageFilter
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

in9 = IN9Joint()
tmp = Imagenet9Challenge('fg_mask', 224)

dataset_name = 'Imagenet9'
name2models = {}
num_of_categories = 9
#for m in ['standardbackground']:
models_names = ['standard', 'actdiff', 'gradmask', 'ada', 'rrr', 'standardbackground', 'actdiffbackground']
for m in models_names:
    try:
        path, acc = get_best(dataset_name, m)
        name2models[m] = get_model_from_path(path, num_of_categories)[0]
        name2models[m].eval()
    except Exception as e:
        print(e)

import random
import torch.nn as nn
from captum.attr import Saliency, IntegratedGradients
import seaborn as sns

def norm(x):
    xmin = x.min()
    xmax = x.max()
    x = (x - xmin)/(xmax - xmin)
    return x

import torch.nn.functional as F
from torchvision.transforms import Resize
import cv2

def process(img):
    return norm(img.permute((1, 2, 0)))


class EdgeIN9():
    
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

        #print(mask.dtype, mask.max(), mask.min())


        kernel = np.ones((self.edge_width, self.edge_width), np.uint8)
        kernel_dilate = np.ones((self.edge_width, self.edge_width), np.uint8)
        
        #print('mask shape:', mask.shape)
        plt.imshow(mask.transpose((1, 2, 0)))
        plt.show()
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        
        
        #print(closing.shape, closing.max(), closing.min(), closing.dtype)
        image = Image.fromarray(closing.transpose((1, 2, 0)), 'RGB')
        image = image.filter(ImageFilter.ModeFilter(size=13))
        
        closing = np.asarray(image).transpose((2, 0, 1))
        mask_after_filter = closing.copy()
        
        plt.imshow(closing.transpose((1, 2, 0)))
        plt.show()
        
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

        edge = (1 - erode01) * dilate01 
        edge_img = edge * imgs[0]
        no_edge_img = (1 - edge) * imgs[0] 
        imgs[1] = mask_after_filter
        
        return imgs[0], imgs[1], no_edge_img, edge_img
        
    def __len__(self):
        return len(self.targets)
    
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
        
        img, mask, noedge, edge = self.edge_process(imgs)
        mask = mask * 1.0
        mask[mask > 0] = 255.0
        mask[mask < 0] = 0.0
        
        #print('mask after', mask.max(), mask.min())
        
        img = transforms_test(Image.fromarray(img.transpose((1, 2, 0))), self.img_size)
        noedge = transforms_test(Image.fromarray(noedge.transpose((1, 2, 0))), self.img_size)
        edge = transforms_test(Image.fromarray(edge.transpose((1, 2, 0))), self.img_size)
        mask = transforms_test(Image.fromarray(mask.transpose((1, 2, 0)).astype(np.uint8)), self.img_size)
        
        mask[mask > 0] = 1.0
        mask[mask < 0] = 0.0
            
        y = self.property_dataset['target'][idx]
            
        return img, mask, noedge, edge, y

def evaluate_std(model, dataloader):            
    model.eval()  

    dataset_size = 0
    running_loss = 0.0
    running_corrects = 0
    
    target_labels = []
    cat_preds = []

    bar = tqdm(dataloader)
    for inputs, mask, noedge, edge, labels in bar:
        dataset_size += inputs.shape[0]
        inputs = noedge
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        # statistics
        running_corrects += float(torch.sum(preds.cpu().detach().data == labels.cpu().detach().data))
        
        cat_preds.extend(preds.cpu().detach().data.tolist())
        target_labels.extend(labels.cpu().detach().data.tolist())

    epoch_acc = running_corrects / dataset_size

    summary = {
        'epoch_acc':epoch_acc,
        'target_labels':target_labels,
        'predictions':cat_preds
        }
    
    return summary

os.makedirs('edge-smooth', exist_ok=True)
edge_size = 49

for edge_size in np.arange(40, 55, step=5):
    print(f'PROCESSING {edge_size} edge width...')
    if os.path.exists(f"edge-smooth/edgewidth-{edge_size}.pkl"): continue

    edge_summary = {}
    modes = ['original', 'mixed_next', 'mixed_same', 'mixed_rand']
    mode2dataset = {}

    for mode in modes:
        print(mode.upper())
        edge_summary[mode] = []
        e = EdgeIN9(mode, edge_size)
        mode2dataset[mode] = e
        
    for mode in modes:
        print(mode.upper())
        edge_summary[mode] = []
        edge_dataloader = torch.utils.data.DataLoader(
            mode2dataset[mode],
            batch_size=16,
            shuffle=True, 
            num_workers=1
        )
        
        for name in name2models.keys():
            print('\t', name.upper())
            ans = evaluate_std(name2models[name], edge_dataloader)
            edge_summary[mode].append(ans['epoch_acc'])
            
        del edge_dataloader
    
    

    try:
        edge_summary['method'] = list(name2models.keys())

        os.makedirs('edge-smooth', exist_ok=True)
        eg = pd.DataFrame.from_dict(edge_summary)[['method', 'original', 'mixed_same', 'mixed_rand', 'mixed_next']].sort_values('method')
        eg.to_pickle(f"edge-smooth/edgewidth-{edge_size}.pkl")

        print(eg)
    except Exception as e:
        print(e)

    try:
        in9_results = pd.read_pickle("challenge-results.pkl").sort_values('method')
        og = in9_results[['method', 'original', 'mixed_same', 'mixed_rand', 'mixed_next']].drop([1])
        print(og)

        dif = {}
        for col in og.columns:
            if col == 'method':
                dif[col] = og[col].tolist()
            else:
                dif[col] = (og[col].values - eg[col].values).tolist()
        #og['method'].tolist(), eg['method'].tolist(),
        print(dif)
        pd.DataFrame.from_dict(dif)[['method', 'original', 'mixed_same', 'mixed_rand', 'mixed_next']].to_pickle(f"edge-smooth/edgewidth-{edge_size}-diff.pkl")
    except Exception as e:
        print(e)
