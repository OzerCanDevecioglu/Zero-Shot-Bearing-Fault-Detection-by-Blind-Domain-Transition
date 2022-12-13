import os, glob, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import shutil
from scipy.io import loadmat
import torch
from torchvision.utils import make_grid
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from Fastonn import SelfONNTranspose1d as SelfONNTranspose1dlayer
from Fastonn import SelfONN1d as SelfONN1dlayer
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"




class ECGDataset(Dataset):
    def __init__(self, base_img_paths, style_img_paths, phase='train'):

        self.base_img_paths = base_img_paths
        self.style_img_paths = style_img_paths   
        self.phase = phase
        path_data_ch1 = loadmat('tmats/clear_sig.mat')
        label = loadmat('tmats/noisy_sig.mat')
        self.data_ch1=path_data_ch1["clear_sig"]

        
        self.label=label["noisy_sig"]


        self.data=np.zeros((1,int(len(self.data_ch1)),4096))
        self.data2=np.zeros((1,int(len(self.label)),4096))

        self.data[0,:,:]=self.data_ch1
        self.data2[0,:,:]=self.label

        self.num_channels = self.data.shape[0]
        self.num_samples = self.data.shape[1]
        self.inputt = torch.tensor(self.data).float()
        self.labels = torch.tensor(self.data2).float()
        print("{0} samples with {1} channels each and {2} classes".format(self.num_samples,self.num_channels,self.labels.unique().numel()))

    def normalize(self,x):
        for idx,data in enumerate(x):
            # data =(data- data.mean(dim=-1).unsqueeze(-1))/data.std(dim=-1).unsqueeze(-1)
            data -= data.min(dim=-1)[0].unsqueeze(-1)
            data /= (data.max(-1)[0]-data.min(-1)[0]).unsqueeze(-1)
            # data2 = abs(data)
            # data /= (data2.max(-1)[0]).unsqueeze(-1)
            data *= 2
            data -= 1
            
            x[idx] = data
        return x

    def __len__(self): return self.num_samples
    
    def __getitem__(self,index): 
        base = self.labels[:,index,:]
        style = self.inputt[:,index,:]#.unsqueeze(0)
        # data = self.standartize(data)
        base = self.normalize(base)
        style = self.normalize(style)

        return base,style

class ECGDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, phase='train', seed=0):
        super(ECGDataModule, self).__init__()
        self.data_dir = data_dir
        
        self.batch_size = batch_size
        self.phase = phase
        self.seed = seed

    def prepare_data(self):
        self.base_img_paths = glob.glob(os.path.join(self.data_dir, 'clear_sig.mat'))
        self.style_img_paths = glob.glob(os.path.join(self.data_dir, 'noisy_sig.mat'))

    def train_dataloader(self):
        random.seed()
        random.shuffle(self.base_img_paths)
        random.shuffle(self.style_img_paths)
        random.seed(self.seed)
        self.train_dataset = ECGDataset(self.base_img_paths, self.style_img_paths, self.phase)
        
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          pin_memory=True
                          )    
    
    

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>
    
    
    
    
    
    

class TECGDataset(Dataset):
    def __init__(self, base_img_paths, style_img_paths, phase='train'):

        self.base_img_paths = base_img_paths
        self.style_img_paths = style_img_paths
   
        self.phase = phase
        
        

        path_data_ch1 = loadmat(self.base_img_paths[0])
        label = loadmat(self.style_img_paths[0])
        self.data_ch1=path_data_ch1["clear_sig"]
        # self.data_ch1=path_data_ch1["real_sig"]
        
        self.label=label["noisy_sig"]
        # self.label=label["gan_outputs"]

        self.data=np.zeros((1,int(len(self.data_ch1)),4096))
        self.data2=np.zeros((1,int(len(self.label)),4096))

        self.data[0,:,:]=self.data_ch1
        self.data2[0,:,:]=self.label

        self.num_channels = self.data.shape[0]
        self.num_samples = self.data.shape[1]
        self.inputt = torch.tensor(self.data).float()
        self.labels = torch.tensor(self.data2).float()
        print("{0} samples with {1} channels each and {2} classes".format(self.num_samples,self.num_channels,self.labels.unique().numel()))

    def normalize(self,x):
        for idx,data in enumerate(x):
            # data =(data- data.mean(dim=-1).unsqueeze(-1))/data.std(dim=-1).unsqueeze(-1)
            # data -= data.min(dim=-1)[0].unsqueeze(-1)
            # data /= (data.max(-1)[0]-data.min(-1)[0]).unsqueeze(-1)
            data2 = abs(data)
            data /= (data2.max(-1)[0]).unsqueeze(-1)
            # data /= 0.2
            with open('datamax.txt', 'a') as f:
                f.write("\n"+"max : "+str(data2.max(-1)[0].numpy()))
            
            # data *= 2
            # data -= 1
            
            x[idx] = data
        return x


    def __len__(self): return self.num_samples
    
    def __getitem__(self,index): 
        base = self.labels[:,index,:]
        style = self.inputt[:,index,:]#.unsqueeze(0)
        base = self.normalize(base)
        style = self.normalize(style)
        return base,style




class TECGDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, phase='train', seed=0):
        super(TECGDataModule, self).__init__()
        self.data_dir = data_dir
        
        self.batch_size = batch_size
        self.phase = phase
        self.seed = seed

    def prepare_data(self):
        self.base_img_paths = glob.glob(os.path.join(self.data_dir, 'clear_sig.mat'))
        self.style_img_paths = glob.glob(os.path.join(self.data_dir, 'noisy_sig.mat'))
        # self.base_img_paths = glob.glob(os.path.join(self.data_dir, 'real_sig1.mat'))
        # self.style_img_paths = glob.glob(os.path.join(self.data_dir, 'gan_outputs1.mat'))
        
    def train_dataloader(self):
        random.seed()
        random.shuffle(self.base_img_paths)
        random.shuffle(self.style_img_paths)
        random.seed(self.seed)
        self.train_dataset = TECGDataset(self.base_img_paths, self.style_img_paths, self.phase)
        
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          pin_memory=True
                          )    


