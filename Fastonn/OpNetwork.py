from .OpTier import *
import torchvision
from .utils import *
from .osl import *
import matplotlib.pyplot as plt


class OpNetwork(nn.Module):
    def __init__(self,in_channels,tier_sizes,kernel_sizes,operators,sampling_factors,OPLIB,pad=-1,optimize=False):
        """ Constuctor function """
        super().__init__()
        self.in_channels = in_channels
        """ Number of channels in input image """
        self.tier_sizes = tier_sizes
        self.kernel_sizes = kernel_sizes
        self.sampling_factors = sampling_factors
        self.optimize = optimize
        self.OPLIB = OPLIB
        self.pad = pad
        if len(operators)>0: self.set_operators(operators)
        self.opScore = torch.zeros(len(self.tier_sizes)-1,len(OPLIB))
        self.opIdx = torch.zeros(len(self.tier_sizes)-1,len(OPLIB))

    
    def forward(self,x):
        x= self.oper(x)
        return x

    def set_operators(self,operators):
        """Set operators for the network.

        Keyword arguments:
        operators -- Python list consisting of integer indices for operator sets
        """
        assert len(self.tier_sizes)==len(operators), "Operators don't match tier sizes"
        assert len(self.kernel_sizes)==len(operators), "Operators don't match kernel sizes"
        assert len(self.sampling_factors)==len(operators), "Operators don't match sampling factors"
        self.oper = nn.Sequential()
        for i in range(len(self.tier_sizes)):
            if i==0: self.oper.add_module(str(i),OpTier(self.in_channels,self.tier_sizes[i],self.kernel_sizes[i],operators[i],self.sampling_factors[i],self.OPLIB,self.pad,self.optimize,i))
            else: self.oper.add_module(str(i),OpTier(self.tier_sizes[i-1],self.tier_sizes[i],self.kernel_sizes[i],operators[i],self.sampling_factors[i],self.OPLIB,self.pad,self.optimize,i))
        self.operators = operators

    def dump_architecture(self):
        """Print current architecture.
        """
        print("[ ",end='')
        for l in self.oper:
            print("[",end='')
            for b in l.oper:
                print(str(b.out_channels)+" ",end='')
            print("] ",end='')
        print("]")

    def reset_parameters(self):
        """Reset trainable parameters.
        """
        for l in self.oper:
            for b in l.oper:
                b.reset_parameters()

        for ln in range(1,len(self.oper)): 
            self.oper[ln].init_variances(self.oper[ln-1])