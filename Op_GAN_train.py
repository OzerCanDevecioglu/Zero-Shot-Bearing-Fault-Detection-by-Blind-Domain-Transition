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
import seaborn as sn
from scipy.stats import norm
import scipy.signal as sig
import copy
import scipy.io as sio
from torch.autograd import Variable
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torchaudio
from scipy.fft import fft, fftfreq, fftshift
from torch_stoi import NegSTOILoss
sample_rate = 16000
loss_func = NegSTOILoss(sample_rate=sample_rate)
spectrogram = torchaudio.transforms.Spectrogram(n_fft = 256 ,win_length=256 ,hop_length=128)

data_dir = 'input/gan-getting-started'
batch_size = 8

# DataModule  -----------------------------------------------------------------    
dm = ECGDataModule(data_dir, batch_size, phase='test')
dm.prepare_data()
dataloader = dm.train_dataloader()

G = CycleGAN_Unet_Generator().cuda()
D = CycleGAN_Discriminator().cuda()

# Init Weight  --------------------------------------------------------------
# for net in [G_basestyle, D_style]:
#     init_weights(net, init_type='normal')
    
    
    
num_epoch = 3000
# lr=0.01
# lr=0.000001

lr=0.0001

betas=(0.5, 0.999)


G_params = list(G.parameters())
D_params = list(D.parameters())

optimizer_g = torch.optim.Adam(G_params, lr=lr, betas=betas)

optimizer_d = torch.optim.Adam(D_params, lr=lr*2, betas=betas)
criterion_mae = nn.L1Loss()
criterion_bce = nn.BCEWithLogitsLoss()
total_loss_d, total_loss_g = [], []
result = {}
    
E=0.00001    
  
for e in range(1,num_epoch):
    print("Epoch: "+str(e))
    G.train()
    D.train()
    LAMBDA = 100.0
    total_loss_g, total_loss_d = [], []
    for input_img, real_img in (dataloader): 
      if(0):
          # check beats
          plt.subplot(211)
          plt.plot(input_img[4,0,:].cpu().detach())
          plt.title("Noisy Audio Signal/Clear Audio Signal")
          plt.subplot(212)
          plt.plot(real_img[4,0,:].cpu().detach())
        

      input_img=input_img.cuda()
      real_img=real_img.cuda()
      real_label = torch.ones(input_img.size()[0], 1, 1).cuda()
      fake_label = torch.zeros(input_img.size()[0],  1, 1).cuda()
      # Generator 
      fake_img = G(input_img).cuda()
      fake_img_ = fake_img.detach() # commonly using 
      out_fake = D(fake_img).cuda()
      loss_g_bce = criterion_bce(out_fake, real_label) # binaryCrossEntropy
      loss_g_mae = criterion_mae(fake_img, real_img) # MSELoss
      
      rspre = spectrogram(torch.tensor(fake_img.cpu()))+E
      ispre = spectrogram(torch.tensor(input_img.cpu()))+E
      loss_g_dim = criterion_mae(rspre.log10(), ispre.log10()) # MSELoss

      
      loss_batch = loss_func(torch.tensor(fake_img.cpu()), torch.tensor(input_img.cpu()))  
      
      loss_g = loss_g_bce + LAMBDA * loss_g_mae +LAMBDA *loss_g_dim
      
      # loss_grg = Variable(loss_g.data, requires_grad=True)
      
      total_loss_g.append(loss_g.item())
    
      optimizer_g.zero_grad()
      optimizer_d.zero_grad()
      loss_g.backward()
      optimizer_g.step()
      # Discriminator
      out_real = D(real_img)
      loss_d_real = criterion_bce(out_real, real_label)
      out_fake = D(fake_img_)
      loss_d_fake = criterion_bce(out_fake, fake_label)
      loss_d = loss_d_real + loss_d_fake +loss_batch.mean()
      total_loss_d.append(loss_d.item())
      # loss_drg = Variable(loss_d.data, requires_grad=True)

      optimizer_g.zero_grad()
      optimizer_d.zero_grad()
      loss_d.backward()
      optimizer_d.step()
      loss_g, loss_d, fake_img=np.mean(total_loss_g), np.mean(total_loss_d), fake_img.detach().cpu()
   
      total_loss_d.append(loss_d)
      total_loss_g.append(loss_g)

    if e%10 == 0:
  
      # Sanity Check
        data_dir = "vmatsm2t"
        batch_size = 100    
        dm2 = TECGDataModule(data_dir, batch_size, phase='test')
        dm2.prepare_data()
        dataloader2 = dm2.train_dataloader()
        base, style = next(iter(dataloader2))
        net = G
        net.eval()
        predicted = []
        predicted=pd.DataFrame(data=predicted)
        actual = []
        actual=pd.DataFrame(data=actual)
        ractual = []
        ractual=pd.DataFrame(data=ractual)
        m=sio.loadmat("vmats/noisy_sig.mat")["noisy_sig"].max(axis=1)
        m2=sio.loadmat("vmats/clear_sig.mat")["clear_sig"].max(axis=1)
        
        with torch.no_grad():
          for base, style in (dataloader2):     
              output = net(base.cuda()).squeeze().cpu()
              ganoutput=output.detach().numpy()
              ganoutput=pd.DataFrame(data=ganoutput)
              predicted=pd.concat([predicted,ganoutput])
              ganacc=base.detach().numpy().squeeze()
              reall=style.detach().numpy().squeeze()
              reall=pd.DataFrame(data=reall)
              ganacc=pd.DataFrame(data=ganacc)
              actual=pd.concat([actual,ganacc])
              ractual=pd.concat([ractual,reall])
      
        gan_outputs=predicted.values.reshape(len(predicted)*4096,1)
        real_outputs=actual.values.reshape(len(actual)*4096,1)
        ractual=ractual.values.reshape(len(ractual)*4096,1)
      
        gan_outputs=gan_outputs[:len(gan_outputs)]
        real_outputs=real_outputs[:len(real_outputs)]
        ractual=ractual[:len(ractual)]
      
        gan_outputs1=gan_outputs.reshape(int(len(gan_outputs)/4096),4096)
        real_outputs1=real_outputs.reshape(int(len(real_outputs)/4096),4096)
        ractual1=ractual.reshape(int(len(ractual)/4096),4096)
       
        from random import randrange
        a=randrange(len(gan_outputs1))
        plt.figure()
        plt.subplot(221)
        plt.plot(real_outputs1[a,:])
        plt.grid()
        plt.title("Input")
        
        plt.subplot(222)
        plt.plot(gan_outputs1[a,:]) 
        plt.grid()
        plt.title("Output")
        
        # number of signal points
        plt.subplot(223)
        plt.plot(np.abs(fftshift(fft(real_outputs1[a,:]))))
        plt.grid()
        plt.title("Input")
        
        plt.subplot(224)
        plt.plot(np.abs(fftshift(fft(gan_outputs1[a,:]))))
        plt.grid()
        plt.title("Output")
        plt.savefig("figs/"+str(e)+".png")
        torch.save(G.state_dict(), 'weights/model_weights_'+str(e)+'_.pth')

        
        
        
   