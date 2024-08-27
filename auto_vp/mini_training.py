from auto_vp.wrapper import BaseWrapper
from auto_vp import programs
from auto_vp.const import CLASS_NUMBER, IMG_SIZE, SOURCE_CLASS_NUM, BATCH_SIZE, NETMEAN, NETSTD, DEFAULT_TEMPLATE, ENSEMBLE_TEMPLATES
from auto_vp.imagenet1000_classname import IMGNET_CLASSNAME
from auto_vp.utilities import Trainable_Parameter_Size

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
from torch.cuda.amp import autocast, GradScaler
import clip

def FreqLabelMap(model, trainloader, device,  wild_dataset=False):
    if(model.output_mapping.mapping_method == "frequency_based_mapping"):
        model.output_mapping.Frequency_mapping(model, trainloader, device, wild_dataset)
        print(model.output_mapping.self_definded_map)
    return

def Mini_Training(model, trainloader, testloader, class_names, num_samples, lr, device, freqmap_interval=None, wild_dataset=False, weight_decay=0.0, runs=1):
    if(model.output_mapping.mapping_method == "semantic_mapping"):
        source_labels = list(IMGNET_CLASSNAME.values())
        model.output_mapping.Semantic_mapping(source_labels, class_names)

    print("Params to learn:")
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

    # Update stretagy
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) 


    # Frequency mapping
    FreqLabelMap(model, trainloader, device)

    total_train_acc = 0

    # Training
    model.train()
    model.model.eval()
    train_loss = []
    train_accs = []
    total_train_acc_list = []
    
    for run in range(runs):
        pbar = tqdm(trainloader, total=len(trainloader),
                desc=f"Batch Training Lr {optimizer.param_groups[0]['lr']:.1e}", ncols=100)
        for ii, pb in enumerate(pbar):
            if(wild_dataset == True):
                imgs, labels, _ = pb
            else:
                imgs, labels = pb

            if ii * len(imgs) > num_samples:
                break

            if imgs.get_device() == -1:
                imgs = imgs.to(device)
                labels = labels.to(device)

            optimizer.zero_grad()
            with autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()
            if(model.no_trainable_resize == 0):
                with torch.no_grad():
                    model.train_resize.scale = model.train_resize.scale.clamp_(0.1, 5.0)

            acc = (logits.argmax(dim=-1) == labels).float().mean()
            train_loss.append(loss.item())
            train_accs.append(acc)

            total_train_loss = sum(train_loss) / len(train_loss)
            total_train_acc = sum(train_accs) / len(train_accs)
            total_train_acc_list.append(total_train_acc)
            if(model.no_trainable_resize == 0):
                pbar.set_postfix_str(f"ACC: {total_train_acc*100:.2f}%, Loss: {total_train_loss:.4f}, Scale: {model.train_resize.scale.item():.4f}")
            else:
                pbar.set_postfix_str(f"ACC: {total_train_acc*100:.2f}%, Loss: {total_train_loss:.4f}")

    return total_train_acc_list

def Mini_CLIP_Training(model, trainloader, testloader, class_names, num_samples, lr, device, freqmap_interval=None, wild_dataset=False, convergence=False, weight_decay=0.0, runs=1):
    if(model.output_mapping.mapping_method == "semantic_mapping"):
        print("CLIP not support semantic mapping!")
        return
    # loss
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    print("Params to learn:")
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

    # Frequency mapping
    FreqLabelMap(model, trainloader, device, wild_dataset=wild_dataset)

    
    total_train_acc = 0
    scaler = GradScaler()

    # Training
    model.train()
    model.model.eval()
    train_loss = []
    train_accs = []
    total_train_acc_list = []
    
    for run in range(runs):
        pbar = tqdm(trainloader, total=len(trainloader),
                desc=f"Batch Training Lr {optimizer.param_groups[0]['lr']:.1e}", ncols=100)
        for ii, pb in enumerate(pbar):
            if(wild_dataset == True):
                imgs, labels, _ = pb
            else:
                imgs, labels = pb
                
            if ii * len(imgs) > num_samples:
                break

            if imgs.get_device() == -1:
                imgs = imgs.to(device)
                labels = labels.to(device)
            
            optimizer.zero_grad()
            with autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)
                
            scaler.scale(loss).backward()

            # clip scale's gradient
            if(model.no_trainable_resize == 0): 
                nn.utils.clip_grad_value_(model.train_resize.scale, 0.001)


            if(model.output_mapping.mapping_method == "fully_connected_layer_mapping"):
                nn.utils.clip_grad_value_(model.output_mapping.layers.bias, 0.001) 
                nn.utils.clip_grad_value_(model.output_mapping.layers.weight, 0.001)
            

            scaler.step(optimizer)
            scaler.update()
            model.model.logit_scale.data = torch.clamp(model.model.logit_scale.data, 0, 4.6052) # Clamps all elements in input into the range in CLIP model 

            if(model.no_trainable_resize == 0):
                with torch.no_grad():
                    model.train_resize.scale = model.train_resize.scale.clamp_(0.1, 5.0)

            acc = (logits.argmax(dim=-1) == labels).float().mean()
            train_loss.append(loss.item())
            train_accs.append(acc)

            total_train_loss = sum(train_loss) / len(train_loss)
            total_train_acc = sum(train_accs) / len(train_accs)
            total_train_acc_list.append(total_train_acc)
            if(model.no_trainable_resize == 0):
                pbar.set_postfix_str(f"ACC: {total_train_acc*100:.2f}%, Loss: {total_train_loss:.4f}, Scale: {model.train_resize.scale.item():.4f}")
            else:
                pbar.set_postfix_str(f"ACC: {total_train_acc*100:.2f}%, Loss: {total_train_loss:.4f}")
    return total_train_acc_list