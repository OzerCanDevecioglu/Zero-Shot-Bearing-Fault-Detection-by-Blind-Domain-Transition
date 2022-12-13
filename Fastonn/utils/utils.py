# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 17:00:21 2019

@author: HM17901
"""

import torch
from torch.utils.data import Dataset,DataLoader,Subset
from torchvision import transforms
from torchvision.utils import make_grid
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
from PIL import Image
import glob
from pathlib import Path
from torch.autograd import Function
import h5py
import pkg_resources
from .adam import AdamFast


############################################# optimizers ################################################

# Returns the specified optimizer
def get_optimizer(model,opt_name,lr,momentum=0.9,betas=(0.9,0.999)):
    if opt_name=="vanilla_adam":
        return torch.optim.Adam(model.parameters(),lr=lr,betas=betas),False 
    elif opt_name=="adam":
        from adam import Adam
        return Adam(model.parameters(),lr=lr,betas=betas),True
    elif opt_name=="adamfast":
        from .adam import AdamFast
        return AdamFast(model.parameters(),lr=lr,betas=(0.9,0.999)),True
    elif opt_name=="sgd_momentum":
        return torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum),False
    elif opt_name=="sgd":
        return torch.optim.SGD(model.parameters(),lr=lr),False
    elif opt_name=="cgd":
        from .cgd import CGD
        return CGD(model.parameters(),lr=lr),True 

############################################# utility functions ################################################

# reset_function_generic
def reset_function_generic(m):
    if hasattr(m,'reset_parameters'):
        m.reset_parameters()


# Show images in a batch. Input: B x C x H x W
def imshow_all(images,num_rows):
    for b in range(images.shape[0]):
        for c in range(images.shape[1]):
            images[b,c,:,:] = (images[b,c,:,:]+1) / 2
    images = make_grid(images, nrow=num_rows)
    images = images.permute(1,2,0).detach().cpu().numpy()
    plt.imshow(images)

# Dump contents of a tensor in txt file
def dump(fname,xx):
    with open(fname, "w+") as f:
        for ii in range(xx.shape[0]):
            x = xx[ii].data.flatten()
            for i in range(len(x)):
                f.write("%f," % (x[i]))
            f.write("\n")
    f.close()

# Hook for dumping gradients
def save_grad(name):
    def hook(grad): dump(name,grad)
    return hook

# naive normalizing. TODO: Improve
def normalize(image):
    norm = 127.5
    return (image-norm)/norm

# reverse the naive normalizing
def denormalize(image):
    for b in range(image.shape[0]):
        for c in range(image.shape[1]):
            image[b,c,:,:] *= 127.5
            image[b,c,:,:] += 127.5
    return image

def normPre(x,thr=4):
    with torch.no_grad(): factor = (thr/(x.view(x.shape[0],-1).abs().max(1)[0])).clamp_max(1)
    x = x.mul(factor[:,None,None,None])
    return x

############################################# spm functions ################################################


# Get top operators from the specified mon file
def getTopOperatorsFromMon(fname,val_idx=0):
    best_ops = []
    from_cpp = np.loadtxt(fname.format(val_idx))
    for idx,hfs in enumerate(from_cpp):
        sorted_hf = np.argsort(-hfs[1:])
        best_ops.append(sorted_hf.tolist())
    return best_ops

def getMonOperators(fname,top_mon_op,layer_sizes):
    with open(fname) as f: lines=f.readlines()
    ops = []

    for idx,line in enumerate(lines):
        line = line.strip()
        if line=='': continue
        line = np.array(line.split('\t')).astype(float)
        opTopScore = line[1:]
        opRank = np.argsort(-np.array(opTopScore))
        opTopScore = opTopScore[opRank] #Sort(opTopScore, opLibNo, opRank); // Sort in descending order..

        sum = 0
        for op in range(top_mon_op): sum += opTopScore[op] #for (int op = 0; op < OpSelNo; ++op) sum += opTopScore[op]; // Total health score..
        opSel = np.zeros(len(opTopScore)) #memset(opSel, 0, opLibNo * sizeof(float)); // init..
        total = 0 
        for op in range(top_mon_op - 1,0,-1):
            opSel[opRank[op]] = int(layer_sizes[idx] * opTopScore[op] / sum) #(int)(size * opTopScore[op] / sum); // no of neurons with operator set: opRank[op]
            total += opSel[opRank[op]] #; // total no of neurons assigned so far..
        opSel[opRank[0]] = layer_sizes[idx] - total # ; // highest ranked op. will get the rest of the neurons..
        local_ops = []
        for op in range(top_mon_op): 
            #print("+++++ >>Rank = {0}, OpInd = {1}, H-Factor = {2}, No = {3} ***".format(op, opRank[op], opTopScore[op], opSel[opRank[op]]))
            for count in range(int(opSel[opRank[op]])): local_ops.append(opRank[op])
        ops.append(local_ops)
    return ops

def RandCDF(cdf, n):
    #// Returns a random no. between [0, n-1] based on the cdf..
	#// prefix[n-1] is sum of all frequencies. Generate a random number
	#// with value from 1 to this sum
	r = (np.random.randint(32767) % cdf[n - 1]) + 1
	#// Find index of ceiling of r in prefix arrat
	indexc = findCeil(cdf, r, 0, n - 1)
	return indexc    


#// Utility function to find ceiling of r in arr[l..h]
def findCeil(arr, r, l, h):
    while (l < h):
        mid = l + ((h - l) >> 1)
        if (r > arr[mid]):
            l = mid + 1
        else:
            h = mid
    ret = l if (arr[l] >= r) else -1
    return ret

# one hot encoding of labels
def one_hot(g): return torch.Tensor(label_binarize([g.item()], classes=[0,1,2,3,4,5,6,7,8,9]))

############################################# datasets ################################################
# Transformation Datasets
class TransformationDataset(Dataset):
  def __init__(self,fold_idx=0):
    super().__init__()
    filename = pkg_resources.resource_filename(__name__, "data/transformation/transformation.h5")
    print(filename)
    self.filename = filename
    self.fold_idx = fold_idx
  
  def __len__(self):
    return 4
  
  def transform(self,x):
    x = torch.tensor(x).float().sub_(127.5).div_(127.5).view(1,60,60)
    return x
  
  def __getitem__(self,index):
    index = (self.fold_idx*self.__len__())+index
    with h5py.File(self.filename, 'r') as file: return self.transform(file['noisy'][index]),self.transform(file['clean'][index])


# Top level dataset
class ONNDataset(Dataset):
    def __init__(self, ds, input_pad=0,label_pad=0,input_transform=[],label_transform=[],input_norm_params=[0.5,0.5],label_norm_params=[0.5,0.5]):
        self.ds = ds
        self.input_pad = input_pad
        self.label_pad = label_pad
        self.input_transform = input_transform
        self.label_transform = label_transform
        self.input_norm_params = input_norm_params
        self.label_norm_params = label_norm_params

    def __len__(self):
        return len(self.ds)
    
    def transform(self,x,t,n,p):
        # Image transforms
        if not isinstance(x,int): 
            x = transforms.ToTensor()(x)
            if 'grayscale' in t: x = transforms.Grayscale()(x)
            if 'normalize' in t: x = transforms.Normalize([n[0] for _ in range(x.shape[0])],[n[1] for _ in range(x.shape[0])],inplace=True)(x)
            if 'pad' in t: x = torch.nn.functional.pad(x,[p for _ in range(4)])        
            return x
        else:
            return torch.tensor(x)

    def __getitem__(self,index):
        image,label = self.ds[index]
        image = self.transform(image,self.input_transform,self.input_norm_params,self.input_pad)
        label = self.transform(label,self.label_transform,self.label_norm_params,self.label_pad)
        return (image,label)

# Samples 2 datasets in parallel
class ParallelDataset(Dataset):
    def __init__(self, ds1,ds2):
        self.ds1 = ds1
        self.ds2 = ds2
        assert len(ds1) == len(ds2),"Datasets lengths do not match!"

    def __len__(self): return len(self.ds1)
    
    def __getitem__(self,index): return self.ds1[index],self.ds2[index]

# Loads images from a folder
class ImageIndexed(Dataset):
    def __init__(self, root,num_images=[],device="cpu",image_size=[]):
        self.root = root
        self.num_images = num_images if len(num_images)!=0 else len(glob.glob(self.root.format('*')))

    def __len__(self): return self.num_images
    
    def __getitem__(self,index): return Image.open(self.root.format(index))

# Loads images along with their class
class ImageWithLabel(Dataset):
    def __init__(self, path_class_list):
        self.path_class_list = path_class_list

    def __len__(self): return len(self.path_class_list)
    
    def __getitem__(self,index): return plt.imread(self.path_class_list[index][0]),int(self.path_class_list[index][1])
    
############################################# performance metrics ################################################

# Signal to Noise Ratio
def calc_snr(output,target):
    snr = 0
    for i in range(target.shape[0]):
        mse = torch.mean((target[i].data-output[i].data).pow(2))
        snr += 10*torch.log10(torch.var(target.data[i],unbiased=False) / mse)
    return (snr/target.shape[0]).item()

# Calculate analytical snr
def calc_asnr(output,target):
    snr = 0
    for i in range(target.shape[0]):
        denom = torch.var(output[i].data - target[i].data)
        num = torch.var(target.data[i])
        snr += 10*torch.log10(num/denom)
    return (snr/target.shape[0]).item()

# Peak signal to noise ratio
def calc_psnr(output,target):
    psnr = 0
    for i in range(target.shape[0]):
        mse = torch.mean((target[i].data-output[i].data).pow(2))
        psnr +=  10*torch.log10(4./mse)
    return (psnr/target.shape[0]).item()
    
def calc_acc(output,target,reduce_fn=torch.mean):
    tn,fp,fn,tp = confusion_matrix(target,output)
    accuracy = (tp+tn).float() / (tp+tn+fp+fn).float()
    return reduce_fn(accuracy)

def calc_precision(output,target,reduce_fn=torch.mean):
    _,fp,_,tp = confusion_matrix(target,output)
    precision = tp / (tp+fp+1e-9)
    return reduce_fn(precision)

def calc_recall(output,target,reduce_fn=torch.mean):
    _,_,fn,tp = confusion_matrix(target,output)
    recall = tp / (tp+fn+1e-9)
    return reduce_fn(recall)    

def calc_f1(output,target,reduce_fn=torch.mean):
    p = calc_precision(output,target)
    r = calc_recall(output,target)
    f1 = 2*p*r / (p+r+1e-9)
    return reduce_fn(f1)

def confusion_matrix(target,output,thresh=0,positive_class=1):
    # segmentation
    if target.shape==output.shape:
        target = target>=thresh
        output = output>=thresh
        target = target.flatten(1)
        output = output.flatten(1)
    
    # classification
    else:
        output = output.view(output.shape[0],output.shape[1],-1)
        target = target.view(target.shape[0],-1)
        if len(output.shape) is not len(target.shape): output = (target == torch.argmax(output,dim=1)).float()
        target = (target==target).float()

    target_positive = (target==positive_class)
    target_negative = (target!=positive_class)
    output_positive = (output==positive_class)
    output_negative = (output!=positive_class)

    tp = (target_positive & output_positive).float().sum(-1)
    tn = (target_negative & output_negative).float().sum(-1)
    fp = (target_negative & output_positive).float().sum(-1)
    fn = (target_positive & output_negative).float().sum(-1)

    return  tn,fp,fn,tp 

############################################# loss functions ################################################

def cross_entropy_loss(output,target):
    batch_size = output.shape[0]
    return torch.nn.CrossEntropyLoss()(output.view(batch_size,-1),target)

def mse_loss(output,target):
    return torch.nn.MSELoss()(output,target)

class MyMSE(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input,target):
        ctx.save_for_backward(input,target)
        return torch.mean((input-target).pow(2))
    
    @staticmethod
    def backward(ctx, grad_output):
        input,target = ctx.saved_tensors
        grad_input = grad_output*(input-target)
        grad_target = grad_output*(-1*(input-target))
        return grad_input,grad_target

############################################# datasets ################################################

def getCVIndices(num_images,train_rate,classes=[]):
    from sklearn.model_selection import KFold,StratifiedKFold
    val_rate = 1-train_rate if train_rate < 1 else 0
    indices = np.arange(0,num_images)

    if val_rate is 0: return [[np.reshape(np.array([indices]),(len(indices),)),np.array([])]]
    num_folds = int(1/min(train_rate,val_rate)) 
    
    if len(classes)>0: 
        assert len(classes)==num_images, "ROla"
        kf = StratifiedKFold(n_splits=num_folds)
        kf.get_n_splits(indices,classes)
        folds = np.array(list(kf.split(indices,classes)))
    else: 
        kf = KFold(n_splits=num_folds)
        kf.get_n_splits(indices)
        folds = np.array(list(kf.split(indices)))
    
    
    if train_rate<val_rate: return folds[:,[1,0]]
    else: return folds

def get_dataset_with_folds(problem,input_path,num_images,gt_mask_path=[],split_ratio=1.0,xfold=True,num_classes=0):
    # Dataset
    if problem == "image2image":
        # Parse
        dataset = ParallelDataset(
            ImageIndexed(input_path),
            ImageIndexed(gt_mask_path)
        )
        classes = []
        print("Found {0} image pairs".format(len(dataset)))
    
    elif problem == "classification":
        images_per_class = round(num_images/num_classes)
        path_class_list = [] 
        for idx,class_name in enumerate(Path(input_path).iterdir()):
            for count,image in enumerate(Path(class_name).iterdir()):
                path_class_list.append((str(image),class_name.name))
                if (count+1)==images_per_class: break

        dataset = ImageWithLabel(path_class_list)
        classes = [x[1] for x in path_class_list ]
        print("Found {0} images belonging to {1} classes".format(len(dataset),idx+1))
    
    assert len(dataset)>0,"ERROR: Please check input/gt image paths"
    

    # Folds
    folds = getCVIndices(num_images,split_ratio,classes)
    if not xfold: folds=[folds[0]]

    return dataset,folds    