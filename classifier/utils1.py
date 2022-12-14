import torch
from torch.utils.data import DataLoader,Dataset
from pathlib import Path
from scipy.io import loadmat
import numpy as np
def get_patient_ids(p):
    patient_ids = []
    parent_path = Path(p)
    for file in parent_path.iterdir():
        file_name_parts = file.stem.split('_')
        if file_name_parts[-1] == "train": patient_ids.append(file_name_parts[1])
    return patient_ids
def calc_accuracy(output,target): return (output.squeeze().argmax(dim=-1)==target).float().mean()
def calc_accuracyMSE(output,target): return ((output.squeeze()-target)*(output.squeeze()-target)).float().mean()
# Dataset
class ECGDataset(Dataset):
    def __init__(self, path, patient_id, split, channel_idx=0):
        # self.path = path.format(patient_id=patient_id, split=split)
        # data = loadmat(self.path)
        # self.num_channels = data['d'].shape[0]
        # self.num_samples = data['d'].shape[1]
        # self.channel_idx = channel_idx
        # self.inputt = torch.tensor(data['d']).float()
        # self.labels = torch.tensor(data['class'],dtype=torch.long)
        
        
        # path_data_ch1="C:/Users/ozerc/Desktop/motordata/classifier_mats/ch1data_{split}_{patient_id}"
        # path_data_ch2="C:/Users/ozerc/Desktop/motordata/classifier_mats/ch2data_{split}_{patient_id}"
        # label="C:/Users/ozerc/Desktop/motordata/classifier_mats/labels_{split}_{patient_id}"
        
        # path_data_ch1="C:/Users/ozerc/Desktop/MotorFinalCOde - Kopya/det_per/ch1data_{split}_{patient_id}"
        # path_data_ch2="C:/Users/ozerc/Desktop/MotorFinalCOde - Kopya/test_data/ch2data_{split}_{patient_id}"
        
        path_data_ch1="C:/Users/ozerc/Desktop/MotorFinalCOde - Kopya/test_data/ch1data_{split}_{patient_id}"
        path_data_ch2="C:/Users/ozerc/Desktop/MotorFinalCOde - Kopya/test_data/ch2data_{split}_{patient_id}"
        label="C:/Users/ozerc/Desktop/MotorFinalCOde - Kopya/test_data/labels_{split}_{patient_id}"
        
        self.path_data_ch1 = path_data_ch1.format(patient_id=patient_id, split=split)
        # self.path_data_ch2 = path_data_ch2.format(patient_id=patient_id, split=split)
        self.label = label.format(patient_id=patient_id, split=split)
        path_data_ch1 = loadmat(self.path_data_ch1)
        # path_data_ch2 = loadmat(self.path_data_ch2)
        label = loadmat(self.label)
        self.data_ch1=path_data_ch1['ch1array']
        # self.data_ch1=self.data_ch1.reshape((int(len(self.data_ch1)/4096),4096))
        # self.data_ch2=path_data_ch2['ch2array']
        # self.data_ch2=self.data_ch2.reshape((int(len(self.data_ch2)/128),128))
        
        self.label=label['labelarray']
        # self.label=self.label.reshape((int(len(self.label)/2),2))
        # self.label=np.argmax(self.label, axis=1)
        # self.label=self.label.reshape((1,len(self.label)))

        self.data=np.zeros((1,int(len(self.data_ch1)),4096))
        self.data[0,:,:]=self.data_ch1
        # self.data[1,:,:]=self.data_ch2
        self.num_channels = self.data.shape[0]
        self.num_samples = self.data.shape[1]
        self.channel_idx = channel_idx
        self.inputt = torch.tensor(self.data).float()
        self.labels = torch.tensor(self.label).float()
        print("{0} samples with {1} channels each and {2} classes".format(self.num_samples,self.num_channels,self.labels.unique().numel()))

    def normalize(self,x):
        for idx,data in enumerate(x):
            # data =(data- data.mean(dim=-1).unsqueeze(-1))/data.std(dim=-1).unsqueeze(-1)
            data -= data.min(dim=-1)[0].unsqueeze(-1)
            data /= (data.max(-1)[0]-data.min(-1)[0]).unsqueeze(-1)
            data *= 2
            data -= 1
            
            x[idx] = data
        return x

   
        
    def __len__(self): return self.num_samples
    
    def __getitem__(self,index): 
        data = self.inputt[:1,index,:]#.unsqueeze(0)
        # data = self.standartize(data)
        data = self.normalize(data)
        label = self.labels[index,:].squeeze()
        return data,label