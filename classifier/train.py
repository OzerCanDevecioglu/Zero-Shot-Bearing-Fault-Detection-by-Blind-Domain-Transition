import torch
import math
from torch import nn
import numpy as np
from pathlib import Path
from math import factorial
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset,Subset,random_split
from selfonn import SelfONN1DLayer
from tqdm import tqdm
from models import models
from utils1 import ECGDataset,get_patient_ids,calc_accuracy
from sklearn.metrics import classification_report,confusion_matrix
from cgd import CGD
from torch import nn, optim
def calc_accuracyMSE(output,target): return ((output.squeeze()-target)*(output.squeeze()-target)).float().mean()
# parse paths

patient_ids=[]
for i in range(0,89):
    patient_ids.append(str((i+1)))
data_path = "mats/sub128v4_{patient_id}_{split}"
num_epochs = 100
q_range= [3]
validation_loss=np.zeros((1,num_epochs+1))
training_loss=[]
predicted = []
actual = []

for q in q_range:
 for mode in ['model_full']:
    model_path = Path("q"+str(q)+'_'+mode+'_ch12_sgdonly')
    out_path = Path("q"+str(q)+'_'+mode+'_ch12_sgdonly'); out_path.mkdir(exist_ok=True)
    print(out_path)
    #print(model._modules['0'].q)
    count_train = 0
    count_test = 0
    for patient_id in patient_ids:
        print("Patient: ", patient_id)
        train_ds = ECGDataset(data_path,patient_id,"train")
        train_ds = random_split(train_ds,[int(0.8*len(train_ds)),len(train_ds)-int(0.8*len(train_ds))])
        test_ds = ECGDataset(data_path,patient_id,"test")
        train_dl = DataLoader(train_ds[0],batch_size=10,shuffle=True)
        val_dl = DataLoader(train_ds[1],batch_size=10)
        # train_dl = DataLoader(train_ds,batch_size=8,shuffle=True)
        test_dl = DataLoader(test_ds,batch_size=10)
        # TRAINING 
        best_val_loss = 5e9
        best_train_loss = 1e9
        # training_loss[:,0]=1e9
        predicted =[]
        actual =[]
        for run in range(1):
            # define model
            model = models[mode](q)
            model = model.cuda()
            # break
            optim1 =  optim.Adam(model.parameters(),lr=0.0001)
            epochs = tqdm(range(num_epochs))
            # learning_rate=0.1
            for epoch in epochs:
                # optim = torch.optim.CGD(model.parameters(), lr=learning_rate, momentum=0.9)
                train_acc = []
                val_acc = []

                train_loss = []
                model.train()
                for batch in (train_dl):
                    optim1.zero_grad()
                    data = batch[0].cuda()
                    label = batch[1].cuda()
                    output = model(data)
                    # loss = torch.nn.MSELoss()(output.squeeze(-1),label)
                    loss = nn.MSELoss()
                    outputs = loss(label, output.squeeze(-1))
                    outputs.backward()
                    # loss.backward()
                    optim1.step()
                    # train_loss.append(loss.item())
                    train_acc.append(torch.nn.MSELoss()(output.data,label.data).item())
                train_loss  = np.mean(train_acc)
                # optim.setLR(loss_now)               
                for batch in (test_dl):
                    data2 = batch[0].cuda()
                    label2 = batch[1].cuda()
                    output2 = model(data2)
                    # loss = torch.nn.MSELoss()(output.squeeze(-1),label)
                    outputs = loss(label2, output2.squeeze(-1))
                    val_acc.append(torch.nn.MSELoss()(output2.data,label2.data).item())
                loss_now = np.mean(val_acc)
                epochs.set_postfix({'loss':loss_now}) 
                training_loss.append(loss_now)
                if loss_now<best_val_loss:
                    # print("Ep")
                    best_val_loss = loss_now
                    torch.save(model,out_path.joinpath('patient_{0}.pth'.format(patient_id)))                  


