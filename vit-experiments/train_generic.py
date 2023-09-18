import torch
import numpy as np

import torch.optim as optim
import argparse
import matplotlib.pyplot as plt

from torch import nn
from numpy.random import randint
from tqdm import tqdm

import torchvision.transforms as transforms
import torchvision.models as models

from skimage.morphology import dilation
from skimage.morphology import square
from sklearn.metrics import f1_score

from scores import cd, score_funcs
from scores import cd_architecture_specific 

import datasets
import ada
import models
import utils_train
import os
import timm

IMG_SIZE = 224

# Training settings
parser = argparse.ArgumentParser(description='RRR')
parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate needs to be extremely small, otherwise loss nans (default: 0.00001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument(
    '--seed', 
    type=int, 
    default=42, 
    metavar='S',
    help='random seed (default: 42)')
parser.add_argument(
    '--img_size', 
    type=int, 
    default=224, 
    metavar='N',
    help=' image size (default: 224)')
parser.add_argument(
    '--regularizer_rate', 
    type=float, 
    default=10.0, 
    metavar='N',
    help='hyperparameter for RRR weight - higher means more regularization')
parser.add_argument(
    '--dataset',
    type=str,
    help='Dataset to be used in the experiment. It must be Cars, CUB, FGVC, or OxfordFlower'
)
parser.add_argument(
    '--pretrained_method',
    type=str,
    default='baseline',
    help='Dataset to be used in the experiment. It must be Cars, CUB, FGVC, or OxfordFlower'
)
parser.add_argument(
    '--ckp_path',
    type=str,
    default='',
    help='Dataset to be used in the experiment. It must be Cars, CUB, FGVC, or OxfordFlower'
)
parser.add_argument(
    '--train_method',
    type=str,
    help='train method must be standard, rrr, cdep, or ada.'
)

args = parser.parse_args()

results_dir = f'results-{args.dataset}'
os.makedirs(results_dir, exist_ok=True)

from time import time

path2save = os.path.join(results_dir, '{}-{}-{}-{}.pth'.format(int(time()), args.train_method, args.epochs, args.img_size))

print('Logging dir and files:')
print('\t', results_dir)
print('\t', path2save)

regularizer_rate = args.regularizer_rate
num_epochs = args.epochs
device = "cuda"

MODEL_NAME = 'resnet18'
#BATCH_SIZE = 8
#LR = 0.01

# Build train dataloader
train_dataset = datasets.get_dataset(
    name=args.dataset, 
    split='train', 
    img_size=args.img_size
)
x, mask, y = train_dataset[0]
print()
print('input shape:', x.shape)
print('mask shape:', mask.shape)

cweights = train_dataset.class_weights.astype(np.float32) #np vector of class weights - useful for weighted loss
num_of_categories = train_dataset.num_of_categories

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True, 
    num_workers=6
)

# Build test dataloader
test_dataset = datasets.get_dataset(name=args.dataset, split='test', img_size=args.img_size)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=True, 
    num_workers=6
)
x, mask, y = train_dataset[0]

print('input shape:', x.shape)
print('mask shape:', mask.shape)
print()

# Set model
if MODEL_NAME == 'resnet18':
    model = models.get_resnet18(num_classes=num_of_categories, pretrained=True)
    model = model.to(device)
    if args.pretrained_method != "baseline":
        checkpoint = torch.load(
            args.ckp_path, 
            map_location=device)
        state_dict = checkpoint['state_dict']

        for k in list(state_dict.keys()):
            if k.startswith('backbone.'):
                if k.startswith('backbone') and not k.startswith('backbone.fc'):
                    # remove prefix
                    state_dict[k[len("backbone."):]] = state_dict[k]
                    del state_dict[k]
        log = model.load_state_dict(state_dict, strict=False)
else:
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_of_categories)
    model = model.to(device)

params_to_update = model.parameters()

#Loss function (does not consider class unbalance)
criterion = nn.CrossEntropyLoss()

#Optimizer
optimizer = optim.SGD(
    params_to_update, 
    lr=args.lr,
    momentum=args.momentum
    )

# scheduler = torch.optim.lr_scheduler.MultiStepLR(
#    optimizer,
#    milestones=[50, 80, 100], 
#    gamma=0.1, 
#    last_epoch=-1
#    )

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    gamma=0.1, 
    verbose=True,
    step_size=50
    )


exp_stats = {
    'train_acc_hist':[],
    'train_loss_hist':[],
    'val_acc_hist':[],
    'val_loss_hist':[],
    'epoch_rrr_loss':[]
}

best_eval_acc = 0.0

print('\tTRAIN METHOD:', args.train_method, args.train_method == 'adak')
print('\tREGULARIZER RATE', args.regularizer_rate)

for epoch in range(args.epochs):
    if args.train_method == 'standard':
        summary = utils_train.train_with_std(
            model=model, 
            dataloader=train_dataloader, 
            criterion=criterion, 
            optimizer=optimizer
            )
    elif args.train_method == 'standardbackground':
        summary = utils_train.train_with_standardbackground(
            model=model, 
            dataloader=train_dataloader, 
            criterion=criterion, 
            optimizer=optimizer,
            regularizer_rate=args.regularizer_rate,
            )

    elif args.train_method == 'fgsm':
        summary = utils_train.train_with_fgsm(
            model=model, 
            dataloader=train_dataloader, 
            criterion=criterion, 
            optimizer=optimizer
            )
    elif args.train_method == 'ada':
        summary = utils_train.train_with_ada(
            model=model, 
            dataloader=train_dataloader, 
            criterion=criterion, 
            optimizer=optimizer
            )
    elif args.train_method == 'rrr':
        summary = utils_train.train_with_rrr(
            model=model, 
            dataloader=train_dataloader, 
            criterion=criterion, 
            optimizer=optimizer, 
            regularizer_rate=args.regularizer_rate
            )
    elif args.train_method == 'cdep':
        summary = utils_train.train_with_cdep(
            model=model, 
            dataloader=train_dataloader, 
            criterion=criterion, 
            optimizer=optimizer, 
            regularizer_rate=args.regularizer_rate
            )
    elif args.train_method == 'actdiff':
        summary = utils_train.train_with_actdiff(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            regularizer_rate=regularizer_rate
        )
    elif args.train_method == 'gradmask':
        summary = utils_train.train_with_gradmask(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            regularizer_rate=regularizer_rate
        )

    elif args.train_method == 'actdiffbackground':
        summary = utils_train.train_with_actdiffbackground(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            regularizer_rate=regularizer_rate
        )
    elif args.train_method == 'adak':
        print(args.train_method)
        summary = utils_train.train_with_adak(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
        )
    else:
        raise Exception('train method must be standard, ada, rrr, or cdep.')

    scheduler.step()

    exp_stats['train_acc_hist'].append(summary['epoch_acc'])
    exp_stats['train_loss_hist'].append(summary['epoch_loss'])
    exp_stats['epoch_rrr_loss'].append(summary.get('epoch_rrr_loss', 0.0))

    print(f'\nEpoch {epoch}/{args.epochs}')
    print('\t Loss: ', summary['epoch_loss'])
    print('\t RRR Loss: ',summary.get('epoch_rrr_loss', 0.0))
    print('\t Accuracy: ', summary['epoch_acc'])


    summary_eval = utils_train.evaluate_std(
        model=model, 
        dataloader=test_dataloader, 
        criterion=criterion
        )
    
    exp_stats['val_acc_hist'].append(summary_eval['epoch_acc'])
    exp_stats['val_loss_hist'].append(summary_eval['epoch_loss'])
    
    print('')
    print('\t Loss test:', summary_eval['epoch_loss'])
    print('\t Accuracy test:', summary_eval['epoch_acc'])
    print('Best val acc:', np.array(exp_stats['val_acc_hist']).max())
    print('Best train acc:', np.array(exp_stats['train_acc_hist']).max())

    if summary_eval['epoch_acc'] > best_eval_acc:
        best_eval_acc = summary_eval['epoch_acc']
        best_ckp = model.state_dict()

    exp_stats['best_ckp'] = best_ckp
    exp_stats['best_eval_acc'] = best_eval_acc
    exp_stats['train_method'] = args.train_method + '-' + args.pretrained_method
    exp_stats['dataset'] = args.dataset
    exp_stats['regularizer_rate'] = args.regularizer_rate
    exp_stats['epochs'] = args.epochs
    exp_stats['batch_size'] = args.batch_size
    exp_stats['img_size'] = args.img_size

    torch.save(exp_stats, path2save)

exp_stats['best_ckp'] = best_ckp
exp_stats['best_eval_acc'] = best_eval_acc
exp_stats['train_method'] = args.train_method + '-' + args.pretrained_method
exp_stats['dataset'] = args.dataset
exp_stats['regularizer_rate'] = args.regularizer_rate
exp_stats['epochs'] = args.epochs
exp_stats['batch_size'] = args.batch_size
exp_stats['img_size'] = args.img_size

torch.save(exp_stats, path2save)
