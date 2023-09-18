import os
import argparse
import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import datasets
import models
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation
import os
import saliency_utils
from captum.attr import Saliency, IntegratedGradients, LayerGradCam, GuidedGradCam, GuidedBackprop, InputXGradient, LayerAttribution
import timm
from tqdm import tqdm

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


TMETHODS_ALL = ['standard', 'rrr', 'ada', 'actdiff', 'gradmask', 'actdiffgrad', 'actdiffminback', 'actdiffgradminbackss', 'fgsm']
DATASETS = ['Cars', 'OxfordFlower', 'CUB']

os.makedirs('inplace-logs-bests', exist_ok=True)


def convert_regularization_rate(r):
    ans = None

    if r == 100.0:
        ans = 'r3'
    elif r == 10.0:
        ans = 'r2'
    else:
        ans = 'r1'

    return ans


def get_all_paths(dataset_name, tmethod, img_size, acc_threshold=0.50):
    print("get_all_paths".upper())
    paths = []
    dataset_name_lower = dataset_name.lower()
    results_dir = f'results-{dataset_name}/'
    for file in [f for f in os.listdir(results_dir) if tmethod in f]:
        print(file)
        try:
            log_path = os.path.join(results_dir, file)
            log = torch.load(log_path)
            #train_method = log['train_method']
            reg = convert_regularization_rate(log['regularizer_rate'])
            
            if (log['img_size'] == img_size) and (log['train_method'] == (tmethod+'-baseline')) and log['epochs'] > 40:
                
                print("acc", log['best_eval_acc'], log['regularizer_rate'], log['epochs'])
                if log['best_eval_acc'] > acc_threshold:
                    paths.append(log_path)
                    
        except Exception as e:
            print('Error:',e)

    return paths


def get_best(dataset_name, tmethod, img_size):
    results_dir = f'results-{dataset_name}/'
    best_path = None
    best_eval = 0.0
    for file in [f for f in os.listdir(results_dir) if tmethod in f]:
        try:
            log_path = os.path.join(results_dir, file)
            log = torch.load(log_path)
            #train_method = log['train_method']
            reg = convert_regularization_rate(log['regularizer_rate'])
            
            if (log['img_size'] == img_size) and (log['train_method'] == (tmethod+'-baseline')):
                
                print('TRAIN METHOD:', tmethod)
                print('IMG SIZE:', img_size)
                print('regularized_rate'.upper(), log['regularizer_rate'])
                print('acc:', log['best_eval_acc'])
                print('Num epoch:', log['epochs'])
                if log['best_eval_acc'] > best_eval:
                    best_eval = log['best_eval_acc']
                    best_path = log_path
                    
        except Exception as e:
            print('Error:',e)
    print()
    print('best path'.upper(), best_path)
    print('accuracy:'.upper(), best_eval)

    return best_path, best_eval


def explain(model, x, y, att_method):
    attribution = None
    relu_att = True
    sp = (x.shape[-2], x.shape[-1])
    
    if att_method == 'saliency':
        sal = Saliency(model)
        attribution = sal.attribute(x, target=y, abs=False).mean(1)

    elif att_method == 'guidedbackprop':
        gbp = GuidedBackprop(model)
        attribution = gbp.attribute(x, target=y).abs().mean(1)

    elif att_method == 'gradcam' or att_method == 'gradcam4':
        layer_gc = LayerGradCam(model, model.layer4)
        attribution = layer_gc.attribute(x, y, relu_attributions=relu_att).abs()
        attribution = LayerAttribution.interpolate(attribution, sp)

    elif att_method == 'guidedgradcam':
        guided_gc = GuidedGradCam(model, model.layer4)
        attribution = guided_gc.attribute(x, y, relu_attributions=relu_att).abs()
        attribution = LayerAttribution.interpolate(attribution, sp)

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


def build_model_summary(data, model, att_method):
    model.eval()
    
    train_dataloader = DataLoader(
        data, 
        batch_size=1, 
        num_workers=4, 
        shuffle=False
    )
    
    logits_list = [] 
    categories = []
    
    sns_list = []
    iou_list = []
    
    probs = []
    atts = []
    masks = []
    
    target_prob_list = []
    target_logit_list = []
    logits_sum_list = []
    pred_category_list = []
    mask_list = []  
    indexes = []
    grad_norms = []
    idx = 0
    
    for x, mask, y in tqdm(train_dataloader):
        mask[mask > 0.0] = 1.0
        mask_mean = mask.mean(dim=1).cuda()
        mask_mean[mask_mean > 0.0] = 1.0
        background = 1 - mask_mean
        
        num_signal_pixels = mask_mean.sum()
        num_noise_pixels = background.sum()
        
        if num_noise_pixels == 0: continue
            
        out = model(x.cuda()).detach().cpu()
        #pred_category = out.argmax(1).numpy().tolist()[0]
        probabilities = F.softmax(out, dim=1)

        att = explain(model, x.cuda(), y.cuda(), att_method=att_method)
        
        att_abs = att.abs()
        grad_sum = att_abs.sum().detach().cpu().numpy()
        att_abs = (att_abs - att_abs.min())/(att_abs.max() - att_abs.min())
        
        signal = (att_abs * mask_mean).sum()/num_signal_pixels
        noise = (att_abs * background).sum()/num_noise_pixels
        
        att = att.detach().cpu().numpy()
        att_min = att.min()
        att_max = att.max()
        att = (att - att_min) / (att_max - att_min)
        
        saliency = saliency_utils.clean_saliency(att[0], percentile=50.0, blur=True, absoloute=True)
        
        if False:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(saliency)
            axs[1].imshow(mask[0])
            axs[2].imshow(x.permute(1, 2, 0))
            plt.show()
            
        #print(saliency.shape, mask[0].shape, x.shape)
        metrics = saliency_utils._get_bin_loc_scores(saliency, mask_mean[0])
        iou = metrics['iou']
        
        logits = out.detach().cpu().numpy()
        category = y.detach().cpu().numpy()
        target_prob = probabilities[0, y].detach().cpu().numpy()
        target_logit = logits[0, category]
        
        logits_sum = logits.sum()
        pred_category = logits.argmax(1)
        masks = mask_mean.detach().cpu().numpy()
        sns = (signal/noise).detach().cpu().numpy()   
        
        if False:
            print('category:', category)
            print('target_prob', target_prob)
            print('target_logit', target_logit)
            print('logits_sum', logits_sum)
            print('pred_category', pred_category)
            print('sns', sns)
            print('metrics', metrics)
            print('')
        
        grad_norms.append(grad_sum)
        logits_list.append(logits)
        categories.append(category)
        pred_category_list.append(pred_category)
        
        sns_list.append(sns)
        iou_list.append(iou)
        
        target_prob_list.append(target_prob)
        target_logit_list.append(target_logit)
        logits_sum_list.append(logits_sum)
        
        indexes.append(idx)
        
        idx += 1
        
        #if idx > 10: break
        
    summary = {
        'grad_norm':grad_norms,
        'logits':logits_list, 
        'categories':categories,
        'pred_category':pred_category_list,
        'sns':sns_list,'iou_list':iou_list,
        'target_prob':target_prob_list,
        'target_logit':target_logit_list,
        'logits_sum':logits_sum_list,
        'img_index':indexes
    }
        
    return summary


def get_model_from_path(path, num_of_categories):
    summary = torch.load(path)
    if len(summary['val_acc_hist']) < 50: raise Exception("model trained during few epochs.")
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

    # Set model
    #model = models.get_resnet18(num_classes=num_of_categories, pretrained=False)
    #model.to(device)

    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_of_categories)
    model.load_state_dict(summary['best_ckp'])
    model = model.to(device)
    
    return model, r


parser = argparse.ArgumentParser(description='MODEL SUMMARY INPLACE')
parser.add_argument(
    '--dataset',
    type=str,
    help='Dataset to be used in the experiment. It must be Cars, CUB, FGVC, or OxfordFlower'
)
parser.add_argument(
    '--tmethod',
    type=str,
    default='baseline',
    help='TRAINING METHOD'
)
parser.add_argument(
    '--acc_th',
    type=float,
    default=0.85,
    help='Accuracy threshold'
)
args = parser.parse_args()

im_size = 224
acc_th = args.acc_th

print(args)

for dataset_name in tqdm([args.dataset]):
    for tm in tqdm([args.tmethod]):
        try:
            print()
            all_paths = get_all_paths(
                dataset_name, 
                tmethod=tm, 
                img_size=im_size, 
                acc_threshold=acc_th
            )
            for log_path in all_paths:
                #log_path, best_eval = get_best(dataset_name, tm, 224)
                try:
                    print('log_path', log_path)
                    test_data = datasets.get_dataset(
                        name=dataset_name,
                        split='test',
                        img_size=im_size
                    )
                    pth_name = log_path.split('/')[-1].replace('.pth', '')
                    print('pth name', pth_name)
                    model, reg = get_model_from_path(log_path, test_data.num_of_categories)
                    path2save = f'inplace-logs-bests/{dataset_name}-{pth_name}-snsnorm.pkl'

                    if os.path.exists(path2save): 
                        print(path2save, 'exists')
                        continue

                    summary = build_model_summary(test_data, model, 'saliency')
                    pd.DataFrame.from_dict(summary).to_pickle(path2save)
                except Exception as e:
                    print(e)
                    print(e)
                    print(e)
                    print()

            print()
        except Exception as e:
            print(e)
