
import torch
import math
from torch import nn
import numpy as np
from pathlib import Path
from math import factorial
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset
from selfonn import SelfONN1DLayer
from scipy.io import loadmat
from tqdm import tqdm
from models import models
from utils1 import ECGDataset,get_patient_ids,calc_accuracy
from sklearn.metrics import classification_report,confusion_matrix
patient_ids=["1"]

q_range= [3]

import pandas as pd
mod_name="full"
acc=[]
f11=[]
f12=[]
for q in q_range:
    aaa=[]
    aaa=pd.DataFrame(data=aaa)
    ppp=[]
    ppp=pd.DataFrame(data=ppp)
    for mode in [mod_name]:
        # model_path = Path("q"+str(q)+'_paper_struct_ch12_sgdonly')
        model_path = Path("q"+str(q)+'_model_full_ch12_sgdonly')
        print("==== {0} ====".format(model_path))
        for patient_id in patient_ids:
            print(patient_id)
            with np.load(model_path.joinpath("predictions{0}.npz".format(patient_id))) as data:
                actual = data['arr_0']
                actual=actual.argmax(-1)
                actual=pd.DataFrame(data=actual)

                predicted = data['arr_1']
                predicted=pd.DataFrame(data=predicted)
                a=classification_report(actual,predicted,digits=4,output_dict=True)
                aa=a["accuracy"]
                acc.append(aa)
                f111=a["0"]["recall"]
                f11.append(f111)
                f112=a["1"]["recall"]
                f12.append(f112)
                # # aaa=pd.concat([aaa,actual])
                # # ppp=pd.concat([ppp,predicted])
                # print(patient_id)
                print(classification_report(actual,predicted,digits=4))
                print(confusion_matrix(actual,predicted))   
                
acc=pd.DataFrame(data=acc)
f11=pd.DataFrame(data=f11)
f12=pd.DataFrame(data=f12)

plt.subplot(311)
plt.plot(f11)
plt.title("f1 healthy")

plt.subplot(312)
plt.plot(f12)
plt.title("f1 faulthy")

plt.subplot(313)
plt.plot(acc)
plt.title("accuracy")

data=pd.concat([acc,f12,f11],axis=1)

import scipy.io as sio

sio.savemat("data.mat", {'data':data.values})
