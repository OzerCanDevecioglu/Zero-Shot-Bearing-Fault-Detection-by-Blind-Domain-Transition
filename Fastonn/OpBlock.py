from .utils import *
from .osl import *
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init

import math
import numpy as np

from tqdm import tqdm

class OpBlock(nn.Module):
    # Initialize block
    def __init__(self,in_channels,out_channels,kernel_size,op_idx,OPLIB):
        super().__init__()

        # Define attributes 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.op_idx = op_idx
        self.OPLIB = OPLIB
        

        # Define Parameters
        self.weights = Parameter(torch.Tensor(in_channels,out_channels,kernel_size*kernel_size))
        self.bias = Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
        #print("Block size: ",self.out_channels)
        
    # Reset parameters 
    def reset_parameters(self):
        bound = 1 / (self.kernel_size**2)
        #init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        for i in range(self.in_channels):
            for o in range(self.out_channels):
                init.uniform_(self.weights[i,o,:],-bound,bound )
        init.uniform_(self.bias,-bound,bound)
        self.conn_stat_now = 0; self.weight_var_now = torch.zeros(self.in_channels)
        self.conn_stat_pre = 0; self.weight_var_pre = torch.zeros(self.in_channels)

    # Set operator set for this block        
    def set_op(self,op_idx): 
        self.op_idx = op_idx
        self.reset_parameters()

        
    def forward(self,x):
        # Load operator from library
        op = self.OPLIB[self.op_idx]

        # Calculate out_size
        out_size = int(math.sqrt(x.shape[-1]))

        # Forward Prop
        x = op['nodal'](x,self.weights) # Nodal
        x = op['pool'](x) # Pool
        x = torch.sum(x,dim=1) # Concat
        x = x.view(x.shape[0],self.out_channels,out_size,out_size)
        x = normPre(x)
        x = op['act'](x,self.bias)

        return x
