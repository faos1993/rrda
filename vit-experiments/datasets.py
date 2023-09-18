import os 
import scipy.io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import torch
from tqdm import tqdm

import torchvision.transforms.functional as TF
from torchvision import transforms

ROOT_DATASETS = "/home/arthur/Documents/datasets/"
ROOT_DATASETS = "/home/work/datafolder/rrr/"

def transforms_train(image, mask, img_size):
    if img_size == 448:
        sizs = [512, 448]
    elif img_size == 224:
        sizs = [256, 224]
    elif img_size == 96:
        sizs = [128, 96]
    elif img_size == 128:
        sizs = [160, 128]
    elif img_size == 32:
        sizs = [48, 32]

    resize = transforms.Resize(size=(sizs[0], sizs[0]))

    image = resize(image)
    mask = resize(mask)

    if random.random() > 0.5:
        angle = random.randint(-30, 30)
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)

    #if random.random():
    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(
        image, 
        output_size=(sizs[1], sizs[1])
        )
    image = TF.crop(image, i, j, h, w)
    mask = TF.crop(mask, i, j, h, w)

    # Random horizontal flipping
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    # Random vertical flipping
    #if random.random() > 0.5:
    #    image = TF.vflip(image)
    #    mask = TF.vflip(mask)

    # Transform to tensor
    image = TF.to_tensor(image)
    
    if image.shape[0] == 1:
        image = torch.cat([image, image, image], dim=0)

    image = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    mask = TF.to_tensor(mask)

    return image, mask


def transforms_test(image, mask, img_size):
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
    mask = resize(mask)

    #if random.random():
    # Random crop
    ccrop = transforms.CenterCrop(size=(sizs[1], sizs[1]))
    image = ccrop(image)
    mask = ccrop(mask)

    # Transform to tensor
    image = TF.to_tensor(image)

    if image.shape[0] == 1:
        image = torch.cat([image, image, image], dim=0)

    image = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    mask = TF.to_tensor(mask)

    return image, mask


class CarsDataset():
    """
    source: https://ai.stanford.edu/~jkrause/cars/car_dataset.html
    """
    def __init__(self, split, img_dir, annot_path, img_size):
        self.img_size = img_size
        self.split = split
        self.img_dir = img_dir
        self.annot_path = annot_path
        self.bboxs = []
        self.fnames = []
        self.categories = []
        self.img_paths = []
        self.num_of_categories = 196
        self.class_weights = None
        self._load_images()
    
    def _load_images(self):
        mat = scipy.io.loadmat(self.annot_path)
        num_imgs = mat['annotations'].shape[1]
        for i in range(num_imgs):
            x1, x2, y1, y2, cat, fname = mat['annotations'][0, i]
            img_path = os.path.join(self.img_dir, fname[0])
            if not os.path.exists(img_path): continue
            
            self.bboxs.append([(x1[0][0], y1[0][0]), (x2[0][0], y2[0][0])])
            self.categories.append(cat[0][0])
            self.fnames.append(fname[0])
            self.img_paths.append(os.path.join(self.img_dir, fname[0]))

        self.class_weights = np.zeros(len(set(self.categories)))

        one_based = 1 * (np.array(self.categories).min() == 1)
        for i in range(len(self.categories)):
            self.categories[i] = self.categories[i] - one_based

        for c in self.categories:
            self.class_weights[c] += 1

        total = len(self.categories)
        #for i in range(self.class_weights.shape[0]):
        #    self.class_weights[i] = self.class_weights[i]/total
        self.class_weights = self.class_weights/total

    def build_mask(self, bbox, shape):
        bw = np.zeros((shape[1], shape[0], 3))
        bw[bbox[1][0]:bbox[1][1], bbox[0][0]:bbox[0][1], :] = 1
        return bw.astype(np.uint8)

    def show_sample(self, idx):
        img = Image.open(self.img_paths[idx])
        bbox = self.bboxs[idx]
        print(bbox)
        print(img.size)
        print(self.categories[idx])
        bw = self.build_mask(bbox, img.size)
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 20))
        
        axs[0].imshow(img)
        axs[1].imshow(bw * 255)
        #axs[1, 0].imshow(img * bw)
        #axs[1, 1].imshow((1 - bw) * img)
        
        plt.show()

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        bbox = self.bboxs[idx]
        bw = self.build_mask(bbox, img.size)
        y = np.int_(self.categories[idx])
        #print('bbox: ', bbox)
        #print('Raw img shape:', img.size)
        #print('Img category:', y)
        bw = Image.fromarray(bw)

        if self.split == 'train':
            img, bw = transforms_train(img, bw, self.img_size)
        else:
            img, bw = transforms_test(img, bw, self.img_size)

        if img.shape[0] == 1:
            img = torch.cat([img, img, img], 0)

        bw[bw > 0.0] = 1.0
            
        return img, bw, y

    def __len__(self):
        return len(self.img_paths)


class CaltechBirds():
    """
    source: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
    """
    def __init__(self, split, img_size):
        self.img_size = img_size
        self.split = split
        self.id2info = {}
        self.images = f'{ROOT_DATASETS}Caltech-UCSD Birds-200-2011/CUB_200_2011/CUB_200_2011/images.txt'
        self.train_test_splits = f'{ROOT_DATASETS}Caltech-UCSD Birds-200-2011/CUB_200_2011/CUB_200_2011/train_test_split.txt'
        self.bbox_path = f'{ROOT_DATASETS}Caltech-UCSD Birds-200-2011/CUB_200_2011/CUB_200_2011/bounding_boxes.txt'
        self.imgs_label = f'{ROOT_DATASETS}Caltech-UCSD Birds-200-2011/CUB_200_2011/CUB_200_2011/image_class_labels.txt'
        self.seg_folder = f'{ROOT_DATASETS}Caltech-UCSD Birds-200-2011/segmentations'
        self.ids = []
        self.num_of_categories = 200
        self.class_weights = None
        self._load_dataset()
        
    def _load_dataset(self):
        categories = []
        for line in open(self.train_test_splits).readlines():
            tokens = line.replace("\n", "").split()
            img_id = tokens[0]
            split = tokens[1]
            #print(split, self.split)
            if int(split) == self.split:
                self.id2info[img_id] = {}
                self.ids.append(img_id)

        for line in open(self.images).readlines():
            tokens = line.replace("\n", '').split()
            img_id = tokens[0]
            path = tokens[1]
            if img_id in self.id2info:
                self.id2info[img_id]['path'] = os.path.join(
                    f'{ROOT_DATASETS}Caltech-UCSD Birds-200-2011/CUB_200_2011/CUB_200_2011/images', 
                    path
                    )

        for line in open(self.imgs_label).readlines():
            tokens = line.replace("\n", '').split()
            img_id = tokens[0]
            label = tokens[1]
            if img_id in self.id2info:
                self.id2info[img_id]['label'] = np.int_(label)
                categories.append(np.int_(label))

        for line in open(self.bbox_path).readlines():
            tokens = line.replace("\n", '').split()
            img_id = tokens[0]
            x = math.trunc(float(tokens[1]))
            y = math.trunc(float(tokens[2]))
            width = math.trunc(float(tokens[3]))
            height = math.trunc(float(tokens[4]))
            if img_id in self.id2info:
                self.id2info[img_id]['bbox'] = (x, y, width, height)

        self.class_weights = np.zeros(len(set(categories)))
        
        one_based = 1 * (np.array(categories).min() == 1)
        for i in range(len(categories)):
            categories[i] = categories[i] - one_based
        
        if one_based:
            for img_id in self.id2info.keys():
                self.id2info[img_id]['label'] = self.id2info[img_id]['label'] - one_based


        for c in categories:
            self.class_weights[c] += 1

        total = len(categories)
        self.class_weights = self.class_weights/total

    def build_mask(self, shape, bbox):
        x, y, w, h = bbox

        #print('bbox', bbox)
        #print('shape', shape)

        bw = np.zeros((shape[1], shape[0], 3))
        bw[y:y+h, x:x+w, :] = 1
        #print(f"{y}:{y+h}, {x}:{x+w}")

        return bw.astype(np.uint8)

    def __getitem__(self, idx):
        id = self.ids[idx]
        info = self.id2info[id]

        path = info['path']
        seg_path = os.path.join(
            self.seg_folder, 
            path.split('/')[-2], 
            path.split('/')[-1]
            ).replace('.jpg', '.png')
        bbox = info['bbox']
        y = info['label']

        img = Image.open(path).convert('RGB')

        #bw = self.build_mask(img.size, bbox)
        #bw = Image.fromarray(bw)

        seg = Image.open(seg_path).convert('RGB')
        #m = np.asarray(seg) == [255, 255, 255]
        m = np.asarray(seg) != [0, 0, 0]
        #m = np.asarray(seg) == [255, 255, 255]
        m = (m * 255).astype(np.uint8)
        #m = (m * 1.0).astype(np.uint8)
        #print(m.max(), m.min())
        bw = Image.fromarray(m // 255)

        #print(path, bbox, y)
        if self.split == 1: #train
            img, bw = transforms_train(img, bw, self.img_size)
        else:
            img, bw = transforms_test(img, bw, self.img_size)
        #print(img.shape, bw.shape, y)
        bw[bw > 0] = 1
        #print(bw.max(), bw.min())
        #print()
        if img.shape[0] == 1:
            img = torch.cat([img, img, img], 0)

        return img, bw, y

    def __len__(self):
        return len(self.ids)

    def visualize_sample(self, idx):
        id = self.ids[idx]
        info = self.id2info[id]

        path = info['path']
        bbox = info['bbox']
        y = info['label']

        img = Image.open(path)
        bw = self.build_mask(img.size, bbox)

        fig, axs = plt.subplots(1, 2, figsize=(10, 20))
        print(bw.max(), bw.min(), (bw == 1).sum(), y)
        axs[0].imshow(img)
        axs[1].imshow(bw * 255)

        plt.show()


class FgvcAircraft():
    """
    source: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/
    """
    def __init__(self, split, img_size):
        self.split = split
        self.img_size = img_size
        assert split in ['train', 'val', 'test']
        
        self.imgs_dir = f'{ROOT_DATASETS}FGVC/fgvc-aircraft-2013b/data/images'
        self.imgs_list = f'{ROOT_DATASETS}FGVC/fgvc-aircraft-2013b/data/images_manufacturer_{self.split}.txt'
        self.imgs_bbox = f'{ROOT_DATASETS}FGVC/fgvc-aircraft-2013b/data/images_box.txt' 
        self.id2info = {}
        self.ids = []
        self.label2id = {}
        self.id2label = {}
        self.class_weights = None
        self._load_dataset()

    def _load_dataset(self):
        labels = []
        for line in open(self.imgs_list).readlines():
            tokens = line.replace('\n', '').split()
            img_id = tokens[0]
            label = tokens[1]
            self.id2info[img_id] = {'label':label}
            labels.append(label)
            self.ids.append(img_id)
        
        for line in open(self.imgs_bbox).readlines():
            tokens = line.replace('\n', '').split()
            img_id = tokens[0]
            x1 = int(tokens[1])
            y1 = int(tokens[2])
            x2 = int(tokens[3])
            y2 = int(tokens[4])
            if img_id in self.id2info:
                self.id2info[img_id]['bbox'] = [(x1, y1), (x2, y2)]

        for id, label in enumerate(list(set(labels))):
            self.label2id[label] = id
            self.id2label[id] = label

        self.num_of_categories = len(self.label2id)

        categories = []
        for c in labels:
            y = self.label2id[c]
            categories.append(y)

        self.class_weights = np.zeros(len(set(categories)))
        
        for c in categories:
            self.class_weights[c] += 1

        total = len(categories)
        self.class_weights = self.class_weights/total

    def build_mask(self, shape, bbox):
        p1, p2 = bbox

        #print('bbox', bbox)
        #print('shape', shape)

        bw = np.zeros((shape[1], shape[0], 3))
        bw[p1[1]:p2[1], p1[0]:p2[0], :] = 1
        #print(f"p1 {p1}, p2 {p2}")

        return bw.astype(np.uint8)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.id2info[img_id]
        img_path = os.path.join(self.imgs_dir, f'{img_id}.jpg')
        img = Image.open(img_path)
        img_vector = np.asarray(img)
        img = Image.fromarray(img_vector[0:-20, :, :]) #cut of bias information

        y = self.label2id[info['label']]
        #print(img.size, y)
        bw = self.build_mask(img.size, info['bbox'])
        bw = Image.fromarray(bw)

        if self.split == 'train':
            img, bw = transforms_train(img, bw, self.img_size)
        else:
            img, bw = transforms_test(img, bw, self.img_size)
        
        return img, bw, y

    def visualize_sample(self, idx):
        img_id = self.ids[idx]
        info = self.id2info[img_id]
        img_path = os.path.join(self.imgs_dir, f'{img_id}.jpg')
        img = Image.open(img_path)
        img_vector = np.asarray(img)
        print('img vector shape:', img_vector.shape)
        img = Image.fromarray(img_vector[0:-20, :, :])

        y = info['label']
        print(img.size, y)
        bw = self.build_mask(img.size, info['bbox'])
        print('bw:', bw.max(), bw.min())
        fig, axs = plt.subplots(1, 2, figsize=(10, 20))
        
        axs[0].imshow(img)
        axs[1].imshow(bw * 255)

        plt.show()
        

    def __len__(self):
        return len(self.ids)


class Oxford102Flowers():
    """
    source: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
    """
    def __init__(self, split, img_size):
        self.split = split
        self.img_size = img_size
        self.img_labels = scipy.io.loadmat(f"{ROOT_DATASETS}Orford-102-flower/imagelabels.mat")['labels'].flatten().tolist()
        self.img_split = scipy.io.loadmat(f"{ROOT_DATASETS}Orford-102-flower/setid.mat")[self._get_split_code()].flatten().tolist()
        self.img_dir = f'{ROOT_DATASETS}Orford-102-flower/102flowers/jpg'
        self.seg_dir = f'{ROOT_DATASETS}Orford-102-flower/102segmentations/segmim'
        self.num_of_categories = 102
        self.class_weights = None
        self.compute_class_weights()
    
    def compute_class_weights(self):
        self.class_weights = np.zeros(len(set(self.img_labels)))
        
        one_based = 1 * (np.array(self.img_labels).min() == 1)
        
        if one_based:
            for i in range(len(self.img_labels)):
                self.img_labels[i] = self.img_labels[i] - one_based

        for c in self.img_labels:
            self.class_weights[c] += 1

        total = self.class_weights.sum()
        #for i in range(self.class_weights.shape[0]):
        #    self.class_weights[i] = self.class_weights[i]/total
        self.class_weights = self.class_weights/total

    def __len__(self):
        return len(self.img_split)

    def _get_split_code(self):
        if self.split == 'train':
            return 'trnid'
        elif self.split == 'val':
            return 'valid'
        elif self.split == 'test':
            return 'tstid'
        else:
            raise Exception("split should be train, val, or test.")

    def _get_imgname(self, id):
        id_str = str(id)
        code = "0" * (5 - len(id_str)) + id_str
        img_file = f'image_{code}.jpg'

        return img_file

    def _get_segname(self, id):
        id_str = str(id)
        code = "0" * (5 - len(id_str)) + id_str
        img_file = f'segmim_{code}.jpg'

        return img_file

    def _build_mask(self, seg):
        m = np.asarray(seg) != [0, 0, 254]
        m = m * 255
        #m[m != 0] = 1
        m = (m.mean(2) != 0) * 255
        m = np.expand_dims(m, axis=2)
        m = np.concatenate([m, m, m], axis=2)
        
        #print(m.shape)

        return m.astype(np.uint8)

    def __getitem__(self, id):
        idx = self.img_split[id]
        img_path_file = os.path.join(self.img_dir, self._get_imgname(idx))
        seg_path_file = os.path.join(self.seg_dir, self._get_segname(idx))

        img = Image.open(img_path_file).convert('RGB')
        seg = Image.open(seg_path_file)

        seg2 = self._build_mask(seg) // 255
        bw = Image.fromarray(seg2)
        try:
            y = self.img_labels[idx - 1]
        except:
            print(idx)
            print(idx)
            print(idx)

        if self.split == 'train':
            img, bw = transforms_train(img, bw, self.img_size)
        else:
            img, bw = transforms_test(img, bw, self.img_size)

        bw[bw > 0.0] = 1.0

        return img, bw, y

    def visualize_sample(self, id):
        idx = self.img_split[id]
        img_path_file = os.path.join(self.img_dir, self._get_imgname(idx))
        seg_path_file = os.path.join(self.seg_dir, self._get_segname(idx))

        img = Image.open(img_path_file)
        seg = Image.open(seg_path_file)
        seg2 = self._build_mask(seg)

        y = self.img_labels[idx]

        print(y, seg2.shape)

        fig, axs = plt.subplots(1, 3, figsize=(10, 20))
        axs[0].imshow(img)
        axs[1].imshow(seg)
        axs[2].imshow(seg2)
        plt.show()


class Imagenet9():
    
    def __init__(self, split='train', img_size=224):
        self.dir = '/home/work/datafolder/imagenet9'
        self.original = os.path.join(self.dir, 'original')
        self.onlyfg = os.path.join(self.dir, 'only_fg')
        self.split = split
        self.img_size = img_size
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
        #for i in range(self.class_weights.shape[0]):
        #    self.class_weights[i] = self.class_weights[i]/total
        self.class_weights = self.class_weights/total

    def load(self):
        
        path_images = os.path.join(self.original, self.split)
        path_images_fg = os.path.join(self.onlyfg, self.split)
        
        for idx, folder in enumerate(sorted(os.listdir(path_images))):
            print(idx, folder)
            class_dir = os.path.join(path_images, folder)
            class_dir_fg = os.path.join(path_images_fg, folder)
            for img in tqdm(os.listdir(class_dir)):
                path_img_fg = os.path.join(class_dir_fg, img)
                path_img_orig = os.path.join(class_dir, img)
                
                img1 = Image.open(path_img_fg)
                v1 = np.asarray(img1)
                
                img2 = Image.open(path_img_orig)
                v2 = np.asarray(img2)
                
                mask = ((v1 > 0) * 1).astype(np.uint8)
                mask = Image.fromarray(mask)
                #eq = ((v1.mean(2) > 0) * 255)
                #print(v1.shape, v2.shape, eq.shape, set(eq.flatten().tolist()))
                
                #fig, axs = plt.subplots(1, 3)
                #axs[0].imshow(img1)
                #axs[1].imshow(img2)
                #axs[2].imshow(eq, cmap='magma')
                #plt.show()
                self.imgs.append(img2)
                self.masks.append(mask)
                self.targets.append(idx)
        
        self.num_of_categories = len(set(self.targets))
        self.compute_class_weights()
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        bw = self.masks[idx]
        y = self.targets[idx]
        
        if self.split == 'train':
            img, bw = transforms_train(img, bw, self.img_size)
        else:
            img, bw = transforms_test(img, bw, self.img_size)

        if img.shape[0] == 1:
            img = torch.cat([img, img, img], 0)

        bw[bw > 0.0] = 1.0
            
        return img, bw, y


import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
import os
import glob
import json
import random
import matplotlib.image as mpimg
from PIL import Image
from pathlib import Path
import pickle
from binascii import a2b_base64
from tqdm import tqdm

# UPDATE _DATA_ROOT to '{path to dir where rival10.zip is unzipped}/RIVAL10/'
_DATA_ROOT = '/home/work/datafolder/RIVAL10/{}/'
_LABEL_MAPPINGS = '/home/work/datafolder/RIVAL10/meta/label_mappings.json'
_WNID_TO_CLASS = '/home/work/datafolder/RIVAL10/meta/wnid_to_class.json'

_ALL_ATTRS = ['long-snout', 'wings', 'wheels', 'text', 'horns', 'floppy-ears',
              'ears', 'colored-eyes', 'tail', 'mane', 'beak', 'hairy', 
              'metallic', 'rectangular', 'wet', 'long', 'tall', 'patterned']

def attr_to_idx(attr):
    return _ALL_ATTRS.index(attr)

def idx_to_attr(idx):
    return _ALL_ATTRS[idx]

def resize(img): 
    return np.array(Image.fromarray(np.uint8(img)).resize((224,224))) / 255

def to_3d(img):
    return np.stack([img, img, img], axis=-1)

def save_uri_as_img(uri, fpath='tmp.png'):
    ''' saves raw mask and returns it as an image'''
    binary_data = a2b_base64(uri)
    with open(fpath, 'wb') as f:
        f.write(binary_data)
    img = mpimg.imread(fpath)
    img = resize(img)
    # binarize mask
    img = np.sum(img, axis=-1)
    img[img != 0] = 1
    img = to_3d(img)
    return img


class LocalRIVAL10(Dataset):
    
    def __init__(self, train=True, masks_dict=True, include_aug=False):
        '''
        Set masks_dict to be true to include tensor of attribute segmentations when retrieving items.
        See __getitem__ for more documentation. 
        '''
        self.train = train
        self.data_root = _DATA_ROOT.format('train' if self.train else 'test')
        self.masks_dict = masks_dict

        self.instance_types = ['ordinary']
        # NOTE: 
        if include_aug:
            self.instance_types += ['superimposed', 'removed']
        
        self.instances = self.collect_instances()
        self.resize = transforms.Resize((224,224))

        with open(_LABEL_MAPPINGS, 'r') as f:
            self.label_mappings = json.load(f)
        with open(_WNID_TO_CLASS, 'r') as f:
            self.wnid_to_class = json.load(f)

    def get_rival10_og_class(self, img_url):
        wnid = img_url.split('/')[-1].split('_')[0]
        inet_class_name = self.wnid_to_class[wnid]
        classname, class_label = self.label_mappings[inet_class_name]
        return classname, class_label

    def collect_instances(self):
        self.instances_by_type = dict()
        self.all_instances = []
        for subdir in self.instance_types:
            instances = []
            dir_path = self.data_root + subdir
            for f in tqdm(glob.glob(dir_path+'/*')):
                if '.JPEG' in f and 'merged_mask' not in f:
                    img_url = f
                    label_path = f[:-5] + '_attr_labels.npy'
                    merged_mask_path = f[:-5] + '_merged_mask.JPEG'
                    mask_dict_path = f[:-5] + '_attr_dict.pkl'
                    instances.append((img_url, label_path, merged_mask_path, mask_dict_path))
            self.instances_by_type[subdir] = instances.copy()
            self.all_instances.extend(self.instances_by_type[subdir])

    def __len__(self):
        return len(self.all_instances)

    def transform(self, imgs):
        transformed_imgs = []
        i, j, h, w = transforms.RandomResizedCrop.get_params(imgs[0], scale=(0.8,1.0),ratio=(0.75,1.25))
        coin_flip = (random.random() < 0.5)
        for ind, img in enumerate(imgs):
            if self.train:
                img = TF.crop(img, i, j, h, w)

                if coin_flip:
                    img = TF.hflip(img)

            img = TF.to_tensor(self.resize(img))
            
            if img.shape[0] == 1:
                img = torch.cat([img, img, img], axis=0)
            
            transformed_imgs.append(img)

        return transformed_imgs

    def merge_all_masks(self, mask_dict):
        merged_mask = np.zeros((224,224,3))
        for attr in mask_dict:
            if attr == 'entire-object':
                continue
            mask_uri = mask_dict[attr]
            mask = save_uri_as_img(mask_uri)
            merged_mask = mask if merged_mask is None else mask + merged_mask
        merged_mask[merged_mask > 0] = 1
        return merged_mask

    def __getitem__(self, i):
        '''
        Returns dict with following keys:
            img
            attr_labels: binary vec with 1 for present attrs
            changed_attr_labels: binary vec with 1 for attrs that were removed or pasted (not natural)
            merged_mask: binary mask with 1 for any attribute region
            attr_masks: tensor w/ mask per attribute. Masks are empty for non present attrs
        '''
        img_url, label_path,  merged_mask_path, mask_dict_path = self.all_instances[i]

        # get rival10 info for original image (label may not hold for attr-augmented images)
        class_name, class_label = self.get_rival10_og_class(img_url)

        # load img
        img = Image.open(img_url)
        if img.mode == 'L':
            img = img.convert("RGB")

        # load labels
        labels = np.load(label_path)
        attr_labels = torch.Tensor(labels[0]).long()
        changed_attrs = torch.Tensor(labels[1]).long() # attrs that were added or removed

        merged_mask_img = Image.open(merged_mask_path)
        """
        imgs = [img, merged_mask_img]
        if self.masks_dict:
            try:
                with open(mask_dict_path, 'rb') as fp:
                    mask_dict = pickle.load(fp)
            except:
                mask_dict = dict()
            for attr in mask_dict:
                mask_uri = mask_dict[attr]
                mask = save_uri_as_img(mask_uri)
                imgs.append(Image.fromarray(np.uint8(255*mask)))
        
        transformed_imgs = self.transform(imgs)
        img_transformed = transformed_imgs.pop(0)
        merged_mask = transformed_imgs.pop(0)
        out = dict({'img':img, 
                    'attr_labels': attr_labels, 
                    'changed_attrs': changed_attrs,
                    #'merged_mask' :merged_mask,
                    'merged_mask' :merged_mask_img,
                    'og_class_name': class_name,
                    'og_class_label': class_label})
        if self.masks_dict:
            attr_masks = [torch.zeros(img.shape) for i in range(len(_ALL_ATTRS)+1)]
            for i, attr in enumerate(mask_dict):
                # if attr == 'entire-object':
                ind = -1 if attr == 'entire-object' else attr_to_idx(attr)
                attr_masks[ind] = transformed_imgs[i]
            out['attr_masks'] = torch.stack(attr_masks)
        """
        
        return img, merged_mask_img, class_label


class Rival10():

    def __init__(self, split='train', img_size=224):
        self.split = split
        self.img_size = img_size
        self.data = LocalRIVAL10(self.split == 'train')
        self.num_of_categories = 10
        self.class_weights = np.ones(self.num_of_categories)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, bw, y = self.data[idx]

        if self.split == 'train':
            img, bw = transforms_train(img, bw, self.img_size)
        else:
            img, bw = transforms_test(img, bw, self.img_size)

        if img.shape[0] == 1:
            img = torch.cat([img, img, img], 0)

        bw[bw > 0.0] = 1.0
            
        return img, bw, y


def get_dataset(name, split, img_size):
    if name == 'Cars':
        if split == 'test':
            return CarsDataset(
                split=split, 
                img_dir=f'{ROOT_DATASETS}car-dataset/cars_{split.lower()}',
                annot_path=f'{ROOT_DATASETS}car-dataset/car_devkit/devkit/cars_{split}_annos_withlabels.mat',
                img_size=img_size,
            )
        else:
            return CarsDataset(
                split=split, 
                img_dir=f'{ROOT_DATASETS}car-dataset/cars_{split.lower()}',
                annot_path=f'{ROOT_DATASETS}car-dataset/car_devkit/devkit/cars_{split}_annos.mat',
                img_size=img_size,
            )
    elif name == 'CUB':
        split = 1 if split == 'train' else 0
        return CaltechBirds(split=split, img_size=img_size,)
    elif name == 'FGVC':
        return FgvcAircraft(split=split, img_size=img_size,)
    elif name == 'OxfordFlower':
        return Oxford102Flowers(split=split, img_size=img_size,)
    elif name == 'Imagenet9':
        if split == 'test':
            return Imagenet9(
                split='val', 
                img_size=img_size,
            )
        else:
            return Imagenet9(
                split=split, 
                img_size=img_size,
            )
    elif name == 'RIVAL10':
        return Rival10(split=split, img_size=img_size)
    else:
        raise Exception(f'Dataset {name} does not exists')


def test_cars_dataset():
    cd = CarsDataset(
        split='test', 
        img_dir=f'{ROOT_DATASETS}car-dataset/cars_test',
        annot_path=f'{ROOT_DATASETS}car-dataset/car_devkit/devkit/cars_test_annos_withlabels.mat',
        )
    cd.show_sample(10)

    cd = CarsDataset(
        split='train', 
        img_dir=f'{ROOT_DATASETS}car-dataset/cars_train',
        annot_path=f'{ROOT_DATASETS}car-dataset/car_devkit/devkit/cars_train_annos.mat',
        )
    cd.show_sample(10)


def test_caltech_birds():
    cb = CaltechBirds(split=0)
    cb = CaltechBirds(split=1)


def test_fgvc_airfract():
    fa = FgvcAircraft(split='train')
    fa = FgvcAircraft(split='test')
    fa = FgvcAircraft(split='val')


if __name__ == "__main__":
    """

    cd = CarsDataset(
        split='train', 
        img_dir='car-dataset/cars_train',
        annot_path='car-dataset/car_devkit/devkit/cars_train_annos.mat',
        )
    cb = CaltechBirds(split=0)

    of = Oxford102Flowers(split='train')
    of = Oxford102Flowers(split='val')
    of = Oxford102Flowers(split='test')
    
    """

    fa = FgvcAircraft(split='val', img_size=224)

    
