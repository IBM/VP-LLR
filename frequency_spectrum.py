### The code has been modified from https://github.com/kiranchari/Fourier-Sensitivity-Regularization ###
import sys
from PIL import Image

import torch.nn.functional as F
import torch.fft

import torchvision
from torchvision import transforms
from torchvision import datasets as torchvision_datasets

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

import argparse
from torchvision import transforms
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import manifold
from torch.nn import functional as F
import cv2
import os
from torch.cuda.amp import autocast

def rotavg(image):
    """
    Computes radial average at each distinct radius from centre.
    input is power spectral matrix
    """
    assert (image.shape[0] == image.shape[1]) and (image.shape[0] % 2 == 0), 'image must be square and even'

    L=np.arange(-image.shape[1]/2,image.shape[1]/2) # [-N/2, ... 0, ... N/2-1]
    x,y = np.meshgrid(L, L)
    R = np.round(np.sqrt(x**2+y**2))
    
    f = lambda r : image[R==r].mean()
    
    r  = np.unique(R)
    r = r[1:] # exclude DC component
    outputs = np.vectorize(f)(r)
    outputs /= outputs.sum()
        
    return outputs

def plot_spectrum(im, ylim=1, return_data=False, save=False, title=False, channel=True):
    """
    plot image spectrum as described in "Natural Image Statistics" (Hyvärinen et al.) pg. 117
    """
    vectors = []
    
    # input can be (N, c, h, w) or (N, h, w) or [ (h,w) .. (h,w) ] or [ (c,h,w), ... ]
    if hasattr(im, 'shape'):
        assert len(im.shape)==4, 'must be shape "(N,C,H,W)"'
        num_samples = im.shape[0]
            
    else:
        num_samples = len(im)
     
    R, radial_indices=None,None
    low_energy_sums, med_energy_sums, high_energy_sums=[],[],[]

    fft_all = []
    
    for _ in range(num_samples):
        im1 = im[_]
        
        # average channels, (C, H, W) -> (H,W) 
        if len(im1.shape) == 3 and channel == False: 
            im1=torch.mean(im1,axis=0)

        # rotavg expects even shaped input
        # if even shaped, pad width and height by 1 each
        if im1.shape[0] % 2 == 1 and channel == False:
            im1 = F.pad(im1,(0,1,0,1))

        # size of image
        fouriersize = im1.shape[1]
        
        # Partly adapted from code by Bruno Olshausen
        im1f=np.fft.fftshift(torch.fft.fftn(im1, norm='ortho').numpy())
        im1_ph = np.angle(im1f, deg=True) # phase
        im1_pf=np.abs(im1f)**2 # power
        fft_all.append(im1_pf)
            
        f=np.arange(-fouriersize/2, fouriersize/2)
        if channel == True:
            PF_all = []
            for ch in im1:
                Pf1=rotavg(ch) # power ratios in [1, ... , 23]
                freq = np.arange(1, len(Pf1)+1)
                PF_all.append(Pf1)
            vectors.append(PF_all)
        else:
            Pf1=rotavg(im1_pf) # power ratios in [1, ... , 23]
            freq = np.arange(1, len(Pf1)+1)  # [1, ..., 23] for cifar.  
            vectors.append(Pf1)

    fft_all = np.array(fft_all)

    mean_vector = np.mean(vectors, axis=0)
    mean_fft_all = np.mean(fft_all, axis=0)    
    return mean_fft_all

def get_input_jacobian(model, inp, label):
    """
    compute input-loss jacobian
    """
    model.input_perturbation.delta.requires_grad = True

    # clip need to resize by ourself 
    x = model.clip_rz_transform(inp)

    img_h = -1
    img_w = -1
    if(model.no_trainable_resize == 0):
        x, img_h, img_w = model.train_resize(x)
    else:
        x = model.train_resize(x)

    prompt_img = model.input_perturbation(x, img_h, img_w)

    if(model.model_name == "clip_ViT_B_32"):
        x = model.model.encode_image(prompt_img)
    elif(model.model_name[0:4] == "clip"):
        x = model.CLIP_network(prompt_img)
    else:
        x = model.model(prompt_img)
    output = model.output_mapping(x)

    
    if type(output) is tuple:
        pred, final_inp = output
    else:
        pred = output

    loss = nn.CrossEntropyLoss()(pred, label)
    dloss_dinp = torch.autograd.grad(loss, model.input_perturbation.delta, create_graph=True)[0].detach().cpu()
    dloss_dinp = torch.unsqueeze(dloss_dinp, 0) 
    
    return dloss_dinp

def plot_sfs(model, data_loader, device, transform=None, return_data=False, ylim=1, num_samples=1000, jacobian=True, save=False, title=False, channel=False, pixel_grad=False, wild_dataset=False):
    """
    Plot Fourier sensitivity of one model
    """
    model.eval()
    
    input_jacobians = None
    
    for idx, pb in enumerate(data_loader):
        if(wild_dataset == True):
            inp, label, _ = pb
        else:
            inp, label = pb

        inp = inp.to(device)
        label = label.to(device)

        if(transform != None): 
            inp = transform(inp)
        
        input_jacobian = get_input_jacobian(model, inp, label)            
        
        if input_jacobians is None:
            input_jacobians = input_jacobian
        else:
            input_jacobians = torch.cat((input_jacobians, input_jacobian), dim=0)
        
        if idx * inp.shape[0] > num_samples:
            break
    
    input_jacobians = input_jacobians.detach()
    if pixel_grad == False:
        mean_fft_all = plot_spectrum(input_jacobians, return_data=return_data, ylim=ylim, save=save, title=title, channel=channel, pixel_grad=pixel_grad)
    else:
        mean_fft_all = torch.mean(input_jacobians, dim=0)
    
    return mean_fft_all

def freq_spec(reprogram_model, trainloader, scale, device, channel=False, pixel_grad=False, wild_dataset=False, num_samples=1000):
    img_resize = 128
    img_resize = int(img_resize*scale)
    if(img_resize > 224):
        img_resize = 224

    transform =  transforms.Resize((img_resize,img_resize))
    mean_fft_all = plot_sfs(reprogram_model, trainloader, device, transform=transform, channel=channel, pixel_grad=pixel_grad, wild_dataset=wild_dataset, num_samples=num_samples)
    return mean_fft_all