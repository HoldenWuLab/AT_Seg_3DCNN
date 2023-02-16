# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.functional as f
import numpy as np
from torch.nn import functional as F
import _reduction as _Reduction
from torch._C import _infer_size, _add_docstr
import torch.nn.modules.loss
from torchvision import transforms, utils
from torch.autograd import Variable
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import filters
import matplotlib
import matplotlib.pyplot as plt


# 3D Weighted Dice Loss (WDL)
class DICELossMultiClass3D(nn.Module):

    def __init__(self):
        super(DICELossMultiClass3D, self).__init__()

    def forward(self, output, masks):
        
        num_classes = output.size(1)
        num_subj = output.size(0)
        nom_temp = torch.cuda.FloatTensor(num_classes,1)
        denom_temp = torch.cuda.FloatTensor(num_classes,1)
        temp_loss = torch.cuda.FloatTensor(num_subj,1)
        epsilon = 0.0000001
        for jj in range(num_subj):
            for ii in range(num_classes):
                class_weight = 1/((torch.sum(masks[jj,ii,:,:,:].view(-1)))**2+epsilon)
                probs = output[jj,ii,:,:,:]
                one_mask = masks[jj,ii,:,:,:]
                probs = probs.reshape(-1)
                one_mask = one_mask.reshape(-1)
                prob_sum1 = torch.matmul(torch.t(probs),one_mask)
                prob_sum2 = torch.sum(probs + one_mask, 0)
                nom_temp[ii] = class_weight*prob_sum1
                denom_temp[ii] = class_weight*prob_sum2+epsilon
            nom = torch.sum(nom_temp,0)
            denom = torch.sum(denom_temp,0)
            temp_loss[jj] = 1 - 2*nom/denom       
        loss = torch.sum(temp_loss,0)/num_subj
        
        return loss


# 3D Frequency Balancing Dice Loss (FBDL)
class LogWeightedDICELossMultiClass3D(nn.Module):

    def __init__(self):
        super(LogWeightedDICELossMultiClass3D, self).__init__()

    def forward(self, output, masks, loss_threshold):
        
        num_classes = output.size(1)
        num_subjects = output.size(0)
        Nsl = output.size(2)
        nom_temp = torch.cuda.FloatTensor(num_classes,1)
        denom_temp = torch.cuda.FloatTensor(num_classes,1)
        temp_loss = torch.cuda.FloatTensor(num_subjects,1)
        frequency = torch.cuda.FloatTensor(num_subjects, num_classes)
        w_0 = torch.cuda.FloatTensor(num_subjects,1)
        logistic_weights = torch.cuda.FloatTensor(num_subjects,num_classes,Nsl,192, 192)
        gradient = torch.cuda.FloatTensor(num_subjects,num_classes,Nsl,192, 192)
        epsilon = 0.0000001
        
        # frequency balancing for imbalanced class samples
        for jj in range(num_subjects):
            for ii in range(num_classes):
                frequency[jj,ii] = torch.sum(masks[jj,ii,:,:,:].reshape(-1),0)/(192*192*Nsl) 
            w_0[jj] = 2*torch.median(frequency[jj,:])/(torch.min(frequency[jj,:])+0.00001)

        # calculation of the logistics weights
        for jj in range(num_subjects):
            for ii in range(num_classes):
                training_seg = output[jj,ii,:,:,:]>loss_threshold
                gradient[jj,ii,:,:,:] = torch.from_numpy(filters.sobel(training_seg.cpu().numpy())).float().cuda()
                logistic_weights[jj,ii,:,:,:] = ((training_seg==masks[jj,ii,:,:,:])
                                               *torch.median(frequency[jj,:])/(frequency[jj,ii] + 0.00001)
                                               + w_0[jj]*(gradient[jj,ii,:,:,:]>0))
                class_weight = torch.sum(torch.sum(torch.sum(logistic_weights[jj,ii,:,:,:])))
                # calculation of logistic weighted Dice loss
                probs = output[jj,ii,:,:,:]
                mask = masks[jj,ii,:,:,:]
                probs = probs.view(-1)
                mask = mask.view(-1)
                prob_sum1 = torch.matmul(torch.t(probs),mask)
                prob_sum2 = torch.sum(probs + mask, 0)
                nom_temp[ii] = class_weight*prob_sum1
                denom_temp[ii] = class_weight*prob_sum2+epsilon
            nom = torch.sum(nom_temp,0)
            denom = torch.sum(denom_temp,0)
            temp_loss[jj] = 1 - 2*nom/denom
        loss = torch.sum(temp_loss,0)/num_subjects
        
        return loss