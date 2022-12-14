import torch
import math
from torch import nn
import numpy as np
from pathlib import Path
from math import factorial
import matplotlib.pyplot as plt


class SelfONN1DLayer(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,q,sampling_factor=1,idx=-1,dir=[],pad=-1,debug=False,output=False,vis=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.q = q
        self.kernel_size = kernel_size
        self.sampling_factor = sampling_factor
        if isinstance(self.kernel_size,int):
          self.kernel_size = kernel_size
          self.weights = nn.Parameter(torch.Tensor(q,out_channels,in_channels,kernel_size)) # Q x C x K x D
        else:
          raise ValueError("Kernel size must be an integer") 
        
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.dir = dir
        self.pad = pad
        self.debug = debug
        self.idx = idx
        self.output = output
        self.reset_parameters()
                
    def reset_parameters(self):
        bound = 0.01 
        nn.init.uniform_(self.bias,-bound,bound)
        with torch.no_grad():
            for q in range(self.q):
                #torch.nn.init.kaiming_uniform_(self.weights[q], a=math.sqrt(5))
                nn.init.xavier_uniform_(self.weights[q])
                #self.weights.data[q] /= factorial(q+1)

    def forward_slow(self,x): # SEPARABLE FOR POOL OPERATION
        raise NotImplementedError

    def forward(self,x):
        # Input to layer
        if self.pad==-1: padding = int(np.ceil(self.kernel_size/2))-1
        else: padding = self.pad
        x = x.clamp(max=1)
        x = torch.cat([(x**i) for i in range(1,self.q+1)],dim=1)
        w = self.weights.transpose(0,1).reshape(self.out_channels,self.q*self.in_channels,self.kernel_size)
        x = torch.nn.functional.conv1d(x,w,padding=padding)
        
        # Subsampling
        if self.sampling_factor>1:
            x = torch.nn.functional.max_pool1d(x,kernel_size=(int(self.sampling_factor)), padding=0)
        elif self.sampling_factor<1:
            x = torch.nn.functional.interpolate(x,scale_factor=abs(int(self.sampling_factor)))
        
        return x


class SelfONN1DBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,sampling_factor,idx=-1,dir=[],pad=-1,debug=False,output=False,vis=None):
        super().__init__()
        self.q1 = SelfONN1DLayer(in_channels,out_channels,kernel_size,1,sampling_factor=sampling_factor,pad=pad)
        self.q3 = SelfONN1DLayer(in_channels,out_channels,kernel_size,3,sampling_factor=sampling_factor,pad=pad)
        self.q5 = SelfONN1DLayer(in_channels,out_channels,kernel_size,5,sampling_factor=sampling_factor,pad=pad)
        self.q7 = SelfONN1DLayer(in_channels,out_channels,kernel_size,7,sampling_factor=sampling_factor,pad=pad)


    def forward(self,x):
        # Input to layer
        out = torch.stack([self.q1(x), self.q3(x), self.q5(x), self.q7(x)],0).sum(0)
        return out