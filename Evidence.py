import argparse
from torchvision import transforms
import numpy as np
import torch
import random
from torch.nn.parameter import Parameter
import os
import torchvision.transforms.functional as F
from tqdm.auto import tqdm

import time
import warnings
import sys


import auto_vp.datasets as datasets
from auto_vp.utilities import setup_device, Trainable_Parameter_Size
from auto_vp.dataprepare import DataPrepare, Data_Scalability
from auto_vp import programs
from auto_vp.const import CLASS_NUMBER, IMG_SIZE, SOURCE_CLASS_NUM, BATCH_SIZE, NETMEAN, NETSTD
from auto_vp.load_model import Load_Reprogramming_Model
from auto_vp.programs import CLIP_network_module,  CLIP_encode_img
from auto_vp.mini_training import * 

from LogME import LogME
import math

def To_Frequency(image):
    rgb_fft = torch.fft.ifftshift(image) 
    rgb_fft = torch.fft.fft2(rgb_fft)
    rgb_fft = torch.fft.fftshift(rgb_fft)
    return rgb_fft

def To_Img(rgb_fft):
    rgb_ifft = torch.fft.ifftshift(rgb_fft)
    rgb_ifft = torch.fft.ifft2(rgb_ifft)
    rgb_ifft = torch.fft.fftshift(rgb_ifft).real
    return rgb_ifft
       
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True

def forward_pass(trainloader, model, fc_layer, device, wild_dataset=False, mode="gaussain"):
    features = []
    targets = []  

    if(mode == "lp" and model.model_name[0:4] != "clip"): # since clip will hook the feature after encode_img module  
        def hook_fn_forward(module, input, output):
            features.append(input[0].detach().cpu()) # input is tuple
    else:   
        def hook_fn_forward(module, input, output):
            features.append(output.detach().cpu())

    forward_hook = fc_layer.register_forward_hook(hook_fn_forward)

    train_accs = []
    model.eval()
    model.model.eval()
    pbar = tqdm(trainloader, total=len(trainloader), desc=f"Training...", ncols=120)
    for itr, pb in enumerate(pbar):
        if(wild_dataset == True):
            imgs, labels, _ = pb
        else:
            imgs, labels = pb
        
        targets.append(labels.detach().cpu())

        if imgs.get_device() == -1:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
        with torch.no_grad():
            logits = model(imgs)

        acc = (logits.argmax(dim=-1) == labels).float().mean()
        train_accs.append(acc)
        total_train_acc = sum(train_accs) / len(train_accs)
        pbar.set_postfix_str(f"ACC: {total_train_acc*100:.2f}%")


    forward_hook.remove()

    features = torch.cat([x for x in features])
    targets = torch.cat([x for x in targets])
    return features, targets 


def Trainable_Parameter_Size(model):
    total_num = 0
    print("Params to learn:")
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t", name,", number: ", param.numel(),", dtype: ", param.dtype)
            total_num += param.numel()
    print("Trainable Parameter: ", total_num)
    return 

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument(
        '--dataset', choices=["CIFAR10", "CIFAR10-C", "CIFAR100", "ABIDE", "Melanoma", "DR", "SVHN", "GTSRB", "Flowers102", "DTD", "Food101", "EuroSAT", "OxfordIIITPet", "StanfordCars", "SUN397", "UCF101", "FMoW"], default="SVHN")
    p.add_argument('--datapath', type=str, default="/home/joytsao/Reprogramming/SVHN")

    p.add_argument('--pretrained', choices=["resnet18", "ig_resnext101_32x8d", "vit_b_16", "swin_t", "clip"], default="clip")
    p.add_argument('--img_scale', type=float, default=1.0) 
    p.add_argument('--seed', type=int, default=7)

    p.add_argument('--mode', choices=["lp", "gaussain", "grad", "mini_finetune", "from_file", "no_prompt"], default="mini_finetune") 
    p.add_argument('--mean', type=float, default=0.0)
    p.add_argument('--std', type=float, default=0.1)
    p.add_argument('--ckpt_file', type=str, default="None") 

    p.add_argument('--runs', type=int, default=1)
    p.add_argument('--device', type=int, default=0) 


    print("====== The Start of Evidence Computation ======")
    start_time = time.time()
    args = p.parse_args()

    # RuntimeError: Too many open files. Communication with the workers is no longer possible.
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
        
    # set random seed
    set_seed(args.seed)
    
    # device setting
    device = args.device
    
    # set image scale
    if(args.mode == "lp" or args.mode == "no_prompt"):
        scale = 2.0
    else:
        scale = args.img_scale

    # set file path
    if(args.ckpt_file == "None"):
        file_path =  None 
    else:
        file_path = args.ckpt_file 

    # Load or build a reprogramming model
    if(file_path == None):
        reprogram_model = Load_Reprogramming_Model(args.dataset, device, mapping_method="fully_connected_layer_mapping", set_train_resize=False, pretrained_model=args.pretrained, scale=scale, weightinit=True)
    else:
        reprogram_model = Load_Reprogramming_Model(args.dataset, device, file_path=file_path)

    # Settings for Dataset
    wild_ds_list = ["Camelyon17", "Iwildcam", "FMoW"]
    if args.dataset in wild_ds_list:
        wild_dataset = True
    else:
        wild_dataset = False

    if(reprogram_model.model_name[0:4] == "clip"):
        clip_transform = reprogram_model.clip_preprocess
    else:
        clip_transform = None
    
    # set image size
    img_resize = IMG_SIZE[args.dataset]
    if(file_path != None):
        if(reprogram_model.no_trainable_resize == 1):
            set_train_resize = False
        else:
            set_train_resize = True
        scale = reprogram_model.init_scale
    else:
        set_train_resize = False

    if(set_train_resize == False):
        img_resize = int(IMG_SIZE[args.dataset]*scale)
        if(img_resize > 224):
            img_resize = 224
        logme_img_sz = img_resize
    else:
        logme_img_sz = int(IMG_SIZE[args.dataset]*reprogram_model.train_resize.scale)
        if(logme_img_sz > 224):
            logme_img_sz = 224

    batch_sz = BATCH_SIZE[args.dataset] // 4
    if(args.mode == "grad"):
        batch_sz = batch_sz // 2

    # Dataloader
    trainloader, testloader, class_names, trainset = DataPrepare(dataset_name=args.dataset, dataset_dir=args.datapath, target_size=(
        img_resize, img_resize), mean=NETMEAN[reprogram_model.model_name], std=NETSTD[reprogram_model.model_name], download=False, batch_size=batch_sz, random_state=args.seed, clip_transform=clip_transform)
    
    # prepare the text embedding for clip
    if(reprogram_model.model_name[0:4] == "clip"):
        template_number = 0 # use default template
        reprogram_model.CLIP_Text_Embedding(class_names, template_number) 
    
    # Load or set the visual prompt (delta)
    if(file_path != None): # Load the mask from file
        pbar = tqdm(testloader, total=len(testloader), desc=f"Testing", ncols=100)
        for pb in pbar:
            if(wild_dataset == True):
                x, y, _ = pb
            else:
                x, y = pb
            with torch.no_grad():
                # clip need to resize by ourself 
                xx = reprogram_model.clip_rz_transform(x.to(device))
                a = -1
                b = -1
                if(set_train_resize == True):
                    xx, a, b = reprogram_model.train_resize(xx)
                else:
                    xx = reprogram_model.train_resize(xx)
                xx = reprogram_model.input_perturbation(xx, a, b)  
                break

        delta_nan = np.where(np.float32(reprogram_model.input_perturbation.mask.cpu().detach().numpy()) > 0.5, np.float32(reprogram_model.input_perturbation.delta.cpu().detach().numpy()), np.nan)
        mean = np.nanmean(delta_nan)
        std = np.nanstd(delta_nan)
        print("prompts from file, mean: ", mean, ", std: ", std)
    else: # Set the prompt (delta) with mean and std from argument
        mean = args.mean
        std = args.std
        reprogram_model.input_perturbation.delta = torch.nn.Parameter(torch.normal(mean=mean, std=std, size=reprogram_model.input_perturbation.output_size, device=device), requires_grad=True).to(device)
        print("prompts from argument, mean: ", mean, ", std: ", std)


    # different models has different linear projection names
    if args.pretrained in ["resnet18", "resnet50", "ig_resnext101_32x8d", "googlenet", "resnet152"]:
        fc_layer = reprogram_model.model.fc
        # fc_layer = reprogram_model.model --> will be killed
    elif args.pretrained in ["densenet121", "vgg16_bn"]:
        fc_layer = reprogram_model.model.classifier
    elif args.pretrained == "vit_b_16":
        fc_layer = reprogram_model.model.heads.head
    elif args.pretrained == "swin_t":
        fc_layer = reprogram_model.model.head
    elif args.pretrained[0:4] == "clip":
        if (args.mode == "lp"):
            encode_image_module = CLIP_encode_img(reprogram_model.model.encode_image)
            reprogram_model.CLIP_network_module = CLIP_network_module(reprogram_model.txt_emb, encode_image_module, reprogram_model.model.logit_scale)
            fc_layer = reprogram_model.CLIP_network_module.encode_image
        else:
            reprogram_model.CLIP_network_module = CLIP_network_module(reprogram_model.txt_emb, reprogram_model.model.encode_image, reprogram_model.model.logit_scale)
            fc_layer = reprogram_model.CLIP_network_module
 
    # Set all the parameter frozen
    for name,param in reprogram_model.named_parameters():
        param.requires_grad = False
        

    if(args.mode == "grad"):
        from frequency_spectrum import freq_spec
        mean_fft_all = freq_spec(reprogram_model, trainloader, scale,  device, pixel_grad=True, channel=True, wild_dataset=wild_dataset, num_samples=1000)

        if(type(mean_fft_all) == np.ndarray):
            mean_fft_all = torch.from_numpy(mean_fft_all)  
        mean_fft_all = mean_fft_all.to(device)

        ### plot grad in rgb domain ###
        plt.figure()
        im = plt.imshow(np.float32(mean_fft_all.cpu()[0].detach().numpy()))
        cbar = plt.colorbar(im)
        plt.savefig("./images/grad_rgb_check")
        fft_masked = To_Frequency(mean_fft_all) 
    
        ### check mask prompt in frequency domain ###
        plt.figure()
        fft_img_mean = torch.mean(fft_masked, dim=0)
        im = plt.imshow(np.float32(torch.log(torch.abs(fft_img_mean)).cpu().detach().numpy()), cmap='gray')
        cbar = plt.colorbar(im)
        plt.savefig("./images/fft_filer_check")
        
        # Taylor expansion optimization: 0.0 or 1.0 prompt value 
        simulated_prompts = torch.where(mean_fft_all < 0, 1.0, 0.0) 

        # Assign the prompt variable (delta)
        reprogram_model.input_perturbation.delta = torch.nn.Parameter(data=simulated_prompts, requires_grad=True)
        
        # check prompt in rgb domain
        plt.figure()
        plt.imshow(np.transpose(np.float32((reprogram_model.input_perturbation.delta * reprogram_model.input_perturbation.mask).cpu().detach().numpy()), (1,2,0))) 
        plt.savefig("./images/ifft_filer_check")

    elif(args.mode == "mini_finetune"):
        # turn on require grad
        for name,param in reprogram_model.input_perturbation.named_parameters():
            param.requires_grad = True

        Trainable_Parameter_Size(reprogram_model)

        num_samples = 1000
        scalibility_rio = int(len(trainset)/num_samples)
        if(scalibility_rio == 0):
            scalibility_rio = 1
            
        print("scalibility_rio: ", scalibility_rio)
        if(args.runs > 1 and scalibility_rio != 1):
            new_trainloader = Data_Scalability(trainset, scalibility_rio, batch_sz, mode="random", random_state=args.seed, wild_dataset=wild_dataset) 
            if args.pretrained[0:4] == "clip":
                Mini_CLIP_Training(reprogram_model, new_trainloader, testloader, class_names, num_samples, 40, device, wild_dataset=wild_dataset, runs=args.runs)
            else:
                Mini_Training(reprogram_model, new_trainloader, testloader, class_names, num_samples, 0.1, device, wild_dataset=wild_dataset, runs=args.runs)
        else:
            if args.pretrained[0:4] == "clip":
                Mini_CLIP_Training(reprogram_model, trainloader, testloader, class_names, num_samples, 40, device, wild_dataset=wild_dataset, runs=args.runs)
            else:
                Mini_Training(reprogram_model, trainloader, testloader, class_names, num_samples, 0.1, device, wild_dataset=wild_dataset, runs=args.runs)
        
        # check the learned prompts
        plt.figure()
        plt.imshow(np.transpose(np.float32((reprogram_model.input_perturbation.delta * reprogram_model.input_perturbation.mask).cpu().detach().numpy()), (1,2,0))) 
        plt.savefig("./images/mini_learned_prompt")

        # check the learned prompts in FFT
        plt.figure()
        fft_img_mean = torch.mean(To_Frequency(reprogram_model.input_perturbation.delta * reprogram_model.input_perturbation.mask), dim=0)
        im = plt.imshow(np.float32(torch.log(torch.abs(fft_img_mean)).cpu().detach().numpy()), cmap='gray')
        cbar = plt.colorbar(im)
        plt.savefig("./images/mini_learned_prompt_fft")

        # turn off requires_grad of input_perturbation
        for name,param in reprogram_model.input_perturbation.named_parameters():
            param.requires_grad = False

    # Conducting features extraction
    features, targets = forward_pass(trainloader, reprogram_model, fc_layer, device, wild_dataset=wild_dataset, mode=args.mode)
    
    # Conducting evidence computation
    logme = LogME(regression=False, img_size=logme_img_sz)
    score, alphas, betas = logme.fit(features.numpy(), targets.numpy())

    print(f"The configurations: dataset: {args.dataset}, pretrained model: {args.pretrained}, mode: {args.mode}, image size: {logme_img_sz}, std: {args.std}") 
    print("The evidence score: ", score)
    print("    --- evidence terms: ", np.mean(logme.dominate_term[2], axis=0))
    print(f"    --- evidence dominating term:  {logme.dominate_term[0]}, {logme.dominate_term[1]}")
    print(f"Total Exection Time (second) : %s" % (time.time() - start_time))
    print("====== The End of Evidence Computation ======\n")
