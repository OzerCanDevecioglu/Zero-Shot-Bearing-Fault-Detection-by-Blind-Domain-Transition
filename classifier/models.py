import torch
import math
from torch import nn
import numpy as np
from pathlib import Path
from math import factorial
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset
from selfonn import SelfONN1DLayer,SelfONN1DBlock
from scipy.io import loadmat
from tqdm import tqdm

# def get_model_full(q):
       
#     model_full = torch.nn.Sequential(
#             SelfONN1DLayer(in_channels=1,out_channels=16,kernel_size=81,q=q,pad=0,sampling_factor=16),
#             torch.nn.Tanh(),
#             SelfONN1DLayer(in_channels=16,out_channels=8,kernel_size=41,q=q,pad=0,sampling_factor=8),
#             torch.nn.Tanh(),
#             SelfONN1DLayer(in_channels=8,out_channels=8,kernel_size=19,q=q,pad=0,sampling_factor=7),
#             torch.nn.Tanh(),
#             # SelfONN1DLayer(in_channels=16,out_channels=16,kernel_size=7,q=q,pad=0,sampling_factor=2),
#             # torch.nn.Tanh(),
#             # SelfONN1DLayer(in_channels=16,out_channels=16,kernel_size=7,q=q,pad=0,sampling_factor=2),
#             # torch.nn.Tanh(),
#             torch.nn.Flatten(),
#             torch.nn.Linear(in_features=8,out_features=16),
#             torch.nn.Tanh(),
#             torch.nn.Linear(in_features=16,out_features=2),
        
#         )
#     return model_full


# models = dict(
#     model_full=get_model_full
# )





def get_model_full(q):
       
    model_full = torch.nn.Sequential(
            SelfONN1DLayer(in_channels=1,out_channels=16,kernel_size=81,q=q,pad=0,sampling_factor=8),
            torch.nn.Tanh(),
            SelfONN1DLayer(in_channels=16,out_channels=16,kernel_size=41,q=q,pad=0,sampling_factor=4),
            torch.nn.Tanh(),
            SelfONN1DLayer(in_channels=16,out_channels=16,kernel_size=21,q=q,pad=0,sampling_factor=4),
            torch.nn.Tanh(),
            SelfONN1DLayer(in_channels=16,out_channels=16,kernel_size=7,q=q,pad=0,sampling_factor=2),
            torch.nn.Tanh(),
            SelfONN1DLayer(in_channels=16,out_channels=16,kernel_size=7,q=q,pad=0,sampling_factor=2),
            torch.nn.Tanh(),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=16,out_features=32),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=32,out_features=2),
        
        )
    return model_full


models = dict(
    model_full=get_model_full
)