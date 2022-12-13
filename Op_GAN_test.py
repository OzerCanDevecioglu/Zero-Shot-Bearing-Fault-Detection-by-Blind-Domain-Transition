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
from utils import ECGDataset, ECGDataModule,init_weights,TECGDataset,TECGDataModule
from GAN_Arch_details import Upsample,Downsample,CycleGAN_Unet_Generator,CycleGAN_Discriminator

# epnum=["1420","1560","1600"]
# epnum=["1000","2000","2990"]
import timeit



epnum=["10"]
# for i in range(0,89):
#     epnum.append(str((i+1)*10))

for a in epnum:
    print(a)
    G_basestyle = CycleGAN_Unet_Generator()
    checkpoint =torch.load("weightsm2t/model_weights_"+str(a)+"_.pth")
    
    G_basestyle.load_state_dict(checkpoint)
    
    G_basestyle.eval()
    
    
    
    
    data_dir = "temats/"
    batch_size = 2000  
    dm = TECGDataModule(data_dir, batch_size, phase='test')
    dm.prepare_data()
    dataloader = dm.train_dataloader()
    base, style = next(iter(dataloader))
    print('Input Shape {}, {}'.format(base.size(), style.size())) 
    net = G_basestyle
    net.eval()
    predicted = []
    predicted=pd.DataFrame(data=predicted)
    actual = []
    actual=pd.DataFrame(data=actual)
    with torch.no_grad():
        for base, style in (dataloader):
            start = timeit.default_timer()

            output = net(base).squeeze()
            stop = timeit.default_timer()

            print('Time: ', stop - start) 
            ganoutput=output.detach().numpy()
            ganoutput=pd.DataFrame(data=ganoutput)
            predicted=pd.concat([predicted,ganoutput])
            ganacc=base.detach().numpy().squeeze()
            ganacc=pd.DataFrame(data=ganacc)
            actual=pd.concat([actual,ganacc])
     
    ch1array=pd.concat([actual,predicted])
    labelsarray=np.ones((2000,2))
    labelsarray[0:1000,1]=-1*labelsarray[0:1000,1]
    labelsarray[1000:2000,0]=-1*labelsarray[1000:2000,0]
     
    
    ch1array=ch1array.values.reshape(len(ch1array)*4096,1)
    labelarray=labelsarray.reshape(len(labelsarray)*2,1)
    import scipy.io as sio
    # sio.savemat('goutputs/ch1data_train_'+str(a)+'.mat', {'predicted':predicted.values})    
    # sio.savemat('goutputs/ch1data_test_'+str(a)+'.mat', {'actual':actual.values}) 
    # # sio.savemat('classifier_mats/ch1data_train_1.mat', {'ch1array':ch1array})    
    # # sio.savemat('classifier_mats/labels_train_1.mat', {'labelarray':labelarray})    
    
