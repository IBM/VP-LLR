# Melanoma Data: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FDBW86T
# ABIDE Data: https://www.nitrc.org/frs/?group_id=404
import os.path
import pickle
import pandas as pd
import nibabel as nib
import numpy as np
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from typing import Any
from PIL import Image
import numpy.ma as ma
import os
import urllib.request as request
from sklearn.model_selection import train_test_split


class Dataset(object):
    def __init__(self, data_path, mode, target_size, data_mean, data_std):
        self.data_path = data_path
        self.mode = mode
        self.train_list = []
        self.val_list = []
        self.test_list = []
        self.total_list = []
        self.data: Any = []
        self.label = []
        self.mean = data_mean
        self.std = data_std
        self.transformer = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def Decide_list(self):
        if self.mode == "train":
            return self.train_list
        elif self.mode == "validation":
            return self.val_list
        elif self.mode == "test":
            return self.test_list
        elif self.mode == "total":
            return self.total_list

class Melanoma(Dataset):
    def __init__(self, download=True, data_path=None, mode=None, target_size=(64, 64), data_mean=(0.5, 0.5, 0.5), data_std=(0.5, 0.5, 0.5), random_state=1, transformer=None):
        super().__init__(data_path, mode, target_size, data_mean, data_std)

        if(download):
            self.Download_dataset(out_dir=data_path)

        # Train/Test list
        self.train_list = ["data"] 
        self.val_list = ["data"]  
        self.test_list = ["data"]
        self.total_list = ["data"]

        # Decide the file list to be downloaded by the mode
        downloaded_list = self.Decide_list()

        if(transformer!=None):
            self.transformer = transformer

        # Lable dictionary
        self.lab_dic = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}

        # Read all the data
        total_data = []
        for file_name in self.total_list:
            file_path = os.path.join(self.data_path, file_name)
            number_list = sorted([x for x in os.listdir(file_path)])
            total_data += [os.path.join(file_path, number_list[x]) for x in range(len(number_list))]

        df = pd.read_csv(os.path.join(self.data_path, "HAM10000_metadata.csv")) # Unordered list in the csv file
        df =  df[['image_id', 'dx']]
        img_name = [(x.split("/"))[-1] for x in total_data]
        total_label = [self.lab_dic[df[df['image_id'] == x[0:-4]]['dx'].values[0]] for x in img_name] # File name end with ".jpg"

        # Resampling
        nv_count = 0
        idx = 0
        for i in range(len(total_data)):
            if(total_label[idx] == self.lab_dic['nv']):
                if( nv_count%3 != 0):
                    total_data.pop(idx)
                    total_label.pop(idx)
                    idx -= 1
                nv_count += 1
            idx += 1

        X_train, X_test, y_train, y_test = train_test_split(
            total_data, total_label, test_size=0.1, random_state=random_state)

        if(mode == "train"):
            self.data = X_train
            self.label = y_train
        elif(mode == "test"):
            self.data = X_test
            self.label = y_test
        elif(mode == "total"):
            self.data = total_data
            self.label = total_label

    def Download_dataset(self, out_dir):
        raise NotImplementedError("Melanoma download not implement")

    def __getitem__(self,idx):
        fname = self.data[idx]
        img = Image.open(fname)
        img = self.transformer(img)
        label = self.label[idx]
        return img,label


# ref: https://github.com/tanimutomo/cifar10-c-eval/blob/master/src/dataset.py
class CIFAR10_C(Dataset):
    def __init__(self, download=True, data_path=None, mode=None, target_size=(64, 64), data_mean=(0.5, 0.5, 0.5), data_std=(0.5, 0.5, 0.5), transformer=None):
        super().__init__(data_path, mode, target_size, data_mean, data_std)
        
        if(download):
            file_name = "CIFAR-10-C.tar"
            # tgz_md5 = "56bf5dcef84df0e2308c6dcbcbbd8499"
            file_out_path = os.path.join(data_path, file_name)
            try:
                if not os.path.exists(file_out_path):
                    request.urlretrieve("https://zenodo.org/record/2535967/files/CIFAR-10-C.tar", file_out_path) 
                    print('{0} download complete'.format(file_name))
                    # unzip file
                else:
                    print('File {0} already exists'.format(file_name))
            except Exception as exc:
                print('There was a problem downloading https://zenodo.org/record/2535967/files/CIFAR-10-C.tar\n')
                
        if(transformer!=None):
            self.transformer = transformer
        
        mode_list = ["gaussian_noise", "shot_noise", "speckle_noise", "impulse_noise", "defocus_blur", "gaussian_blur",
                     "motion_blur", "zoom_blur", "snow", "fog", "brightness", "contrast", "elastic_transform", "pixelate", 
                     "jpeg_compression", "spatter", "saturate", "frost"]
        
        img_path = os.path.join(data_path, mode + '.npy')
        target_path = os.path.join(data_path, 'labels.npy')
        
        self.data = np.load(img_path)
        self.label = np.load(target_path)
        self.classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    
    def __getitem__(self,idx):
        img = self.data[idx]
        img = Image.fromarray(img)
        img = self.transformer(img)
        label = self.label[idx]
        return img,label

class ImageNet1k(Dataset):
    def __init__(self, download=True, data_path=None, mode=None, target_size=(64, 64), data_mean=(0.5, 0.5, 0.5), data_std=(0.5, 0.5, 0.5), random_state=1, transformer=None):
        super().__init__(data_path, mode, target_size, data_mean, data_std)

        if(download):
            self.Download_dataset(out_dir=data_path)

        if(transformer!=None):
            self.transformer = transformer
        
        # Lable dictionary
        self.lab_dic = {}
        self.classes = []
        label_path = os.path.join(os.getcwd(),"/auto_vp/ImageNet_label.txt")
        label_file = open(label_path, 'r')
        for line in label_file.readlines():
            self.lab_dic[line.split()[0]] = int(line.split()[1])-1
            self.classes.append(line.split()[2])

        train_path = os.path.join(data_path, "imagenet-mini/train")
        if(mode == "train"):
            for category in sorted([y for y in os.listdir(train_path)]):
                category_path = os.path.join(train_path, category)
                for img in sorted([im for im in os.listdir(category_path)]):
                    self.data.append(os.path.join(category_path, img))
                    self.label.append(self.lab_dic[category])

        test_path = os.path.join(data_path, "imagenet-mini/val")
        if(mode == "test"):
            for category in sorted([y for y in os.listdir(test_path)]):
                category_path = os.path.join(test_path, category)
                for img in sorted([im for im in os.listdir(category_path)]):
                    self.data.append(os.path.join(category_path, img))
                    self.label.append(self.lab_dic[category])

    def Download_dataset(self, out_dir):
        # kaggle datasets download -d ifigotin/imagenetmini-1000
        # unzip imagenetmini-1000.zip -d ImageNet1k
        raise NotImplementedError("ImageNet1k download not implement")

    def __getitem__(self,idx):
        fname = self.data[idx]
        img = Image.open(fname) 
        img = self.transformer(img)
        label = self.label[idx]
        return img, label
