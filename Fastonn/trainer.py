import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm
import random, string,time
from copy import deepcopy
from .utils import *
import h5py

# Helper class to train, validate and evaluate torch models
class Trainer:
    # Constructor function
    def __init__(self,model,train_dl,val_dl,test_dl,loss,optim,lr,metrics,device,reset_fn=[],model_name='model',verbose=2):
        """Initialize the trainer instance
        - **model** -- progress bar object to update  
        - **train_dl** -- training dataloader     
        - **val_dl** -- validation dataloader  
        - **test_dl** -- test dataloader  
        - **loss** -- loss function, must return a PyTorch tensor, not scalar  a
        - **opt_name** -- name of the optimizer. Either of  ['adamfast','cgd','adam','vanilla_adam']  
        - **lr** -- initial learning rate    
        - **metrics** -- Python dictionary with format 'metric_name':(func,max/min), where func is any function with inputs target,output and should return a scalar value for accuracy. max/min defines the desired optimization of this metric  
        - **device** -- device on which to train, either of ['cpu','cuda:x'] where x is the index of gpu. Multi-GPU training is not supported yet.    
        - **reset_fn** -- function to reset network parameters. Called at the start of each run  
        - **track** -- metric to track in format ['mode','metric_name','max/min']  
        - **model_name** -- filename of the saved model
        - **verbose** -- extent of debug output: 0=No output, 1=show only run progress bar, 2=show run and epoch progress bar  
        """
        
        # Initialize variables
        self.model = model.to(device)
        self.model_name = model_name
        self.dl = {'train':train_dl,'val':val_dl,'test':test_dl}
        self.device = device
        self.loss = loss
        self.metrics = metrics
        self.metrics['loss'] = (loss,'min')
        if reset_fn == []:
            print('WARNING: No reset function provided. Generic function will be used') 
            self.reset_fn = reset_function_generic
        else: 
            self.reset_fn = reset_fn
        self.optim = optim
        self.optimizer = self.optim(model.parameters(),lr=lr)
        self.lr = lr
        self.verbose = verbose

        self.r = 0
        self.e = 0
        
        # Container for statistics
        self.stats = {'train':{},'val':{},'test':{}}
        
        # best stats
        self.best_metrics = {'train':{},'val':{},'test':{}}

        # best states
        self.best_states = {'train':{},'val':{},'test':{}}
        
        self.init_best_state()
    
    def init_stats(self,num_epochs,num_runs):
        """Initialize containers for storing statistics and models
 
        num_train_batches -- total number of training batches  
        num_val_batches -- total number of validation batches  
        num_test_batches -- total number of testing batches  
        num_epochs -- number of epochs per run  
        num_runs -- number of randomly initialized runs  
        """
        
        # Running stats
        for key,_ in self.metrics.items():
            self.stats['train'][key] = torch.zeros(num_runs,num_epochs,len(self.dl['train']))
            if len(self.dl['val'])>0: self.stats['val'][key] = torch.zeros(num_runs,num_epochs,len(self.dl['val']))
            if len(self.dl['test'])>0: self.stats['test'][key] = torch.zeros(num_runs,1,len(self.dl['test']))

    def init_best_state(self):
        """Initial best state containers 
        """
        for mode in ['train','val','test']:
            # Metrics
            for key,(func,criteria) in self.metrics.items():
                self.best_metrics[mode][key] = 1e9 if criteria=='min' else -1e9
                self.best_states[mode][key] =  self.get_model_state()
             
    def track_all_stats(self,r,e,modes):
        for mode in modes:
            if len(self.dl[mode])==0: continue
            if mode=='test': 
                for key,(func,criteria) in self.metrics.items():
                    self.update_best_metric(mode,key,criteria,torch.mean(self.stats[mode][key][r][0]))
            else:
                for key,(func,criteria) in self.metrics.items():
                    self.update_best_metric(mode,key,criteria,torch.mean(self.stats[mode][key][r][e]))

    def update_best_metric(self,mode,metric,criteria,now):
        # check if metric improved
        before = self.best_metrics[mode][metric]
        if (criteria=='min' and now<before) or (criteria=='max' and now>before): 
            self.best_metrics[mode][metric] = now
            self.best_states[mode][metric] = self.get_model_state()
        
    def update_metrics(self,output,target,run,epoch,batch,mode):
        """Update accuracy metrics 
        - output -- batch output  
        - target -- batch groundtruth  
        - run -- current run number  
        - epoch -- current epoch number  
        - batch -- current batch number  
        - mode -- one of ['train','val','loss']  
        """
        for key,(func,_) in self.metrics.items(): 
            self.stats[mode][key][run][epoch][batch] = func(output.data,target.data)

    def _log_to_pbar(self,pbar,idx,modes):
        """Calculate and show statistics to show in progress bar 
        - pbar -- progress bar object to update  
        - idx -- a list [r,e] corresponding to run, and epoch index. e and b are optional.     
        - modes -- some of ['train','val','test']. Default: train  
        """
        log = {}
        for mode in modes:
            # Raincheck
            if len(self.dl[mode])==0: continue

            # Show progress
            if pbar.desc=='Epoch': # on epoch complete
                if mode=='test': 
                    for key,_ in self.metrics.items(): log[mode+'_'+key] = torch.mean(self.stats[mode][key][idx[0]][0]).item()
                else: 
                    for key,_ in self.metrics.items(): log[mode+'_'+key] = torch.mean(self.stats[mode][key][idx[0]][idx[1]]).item()
            else: # on run complete
                for key,_ in self.metrics.items(): log[mode+'_'+key] = self.best_metrics[mode][key].item()
        try:
            log['lr'] = self.optimizer.getLR()
        except:
            pass
        pbar.set_postfix(log)
                   
    def get_model_state(self,include_model=True):
        """ Returns loadable current state of the model
        """
        return  {
            'model':self.model if include_model else [],
            'optimizer':self.optimizer,
            'optimizer_state':deepcopy(self.optimizer.state_dict()),
            'current_state': deepcopy(self.model.state_dict()),
            'stats':self.stats,
            'run':self.r,
            'epoch':self.e
        }

    def load_model_state(self,state): 
        if state['model'] != []: self.model = state['model']
        self.optimizer = state['optimizer']
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.model.load_state_dict(state['current_state'])
        print("Rolled back to run {0} epoch {1}".format(state['run'],state['epoch']))


    def plot_stats(self,modes=['train','val'],r=-1,e=-1,now=False):
        num_plots = int(len(self.metrics))
        num_rows = np.floor(np.sqrt(num_plots)).astype(int)
        num_cols = np.ceil(num_plots/num_rows).astype(int)
        self.fig,self.ax = plt.subplots(num_rows,num_cols)

        # Custom metrics
        for idx,key_item in enumerate(self.metrics.items()):
            cur_ax = self.ax[idx]
            cur_ax.cla()
            for _,mode in enumerate(modes):
                if len(self.dl[mode])==0: continue
                cur_ax.plot(torch.mean(self.stats[mode][key_item[0]][r][:e],dim=-1)[:],label=mode)
                cur_ax.set_title(key_item[0])
                cur_ax.set_xlabel('Epochs')
                cur_ax.set_ylabel(key_item[0])
            cur_ax.legend()
            cur_ax.grid()

    
        
        #plt.show()
        #plt.pause(0.1)
        if now: plt.show()
        else: self.fig.savefig(self.model_name+'run_'+str(r)+'.png')
        plt.close(self.fig)
    
    def _init_plot(self):
        num_plots = len(self.metrics)
        num_rows = np.floor(np.sqrt(num_plots))
        num_cols = np.ceil(num_plots/num_rows)
        self.fig,self.ax = plt.subplots(1,len(self.metrics)+1)
        #plt.pause(0.001)
    
    def save_all(self,include_model=True):
        torch.save({
                'last_known_state':self.get_model_state(include_model=include_model),
                'best_states':self.best_states,
                'best_metrics':self.best_metrics,
                'metrics':self.metrics
            },
            self.model_name+".pth"
        )


    def train(self,num_epochs=50,num_runs=1):
        """Initialize the trainer instance
        - **num_epochs** -- Number of epochs to train. Default: 50   
        - **num_runs** -- number of randomly initialized runs. Default: 1  
        """

        # Initialize Statistics
        self.model.to(self.device) # just in case
        self.init_stats(num_epochs,num_runs)
        runs = range(num_runs)
        if self.verbose>0: runs = tqdm(runs,desc='Run')
        for r in runs:
            self.r = r
            self.model.apply(self.reset_fn)
            self.model.to(self.device)
            self.optimizer = self.optim(self.model.parameters(),lr=self.lr) #get_optimizer(self.model,self.opt_name,self.lr)
            
            epochs = range(num_epochs)
            if self.verbose>1: epochs = tqdm(epochs,desc='Epoch')
            for e in epochs:
                self.e = e
                self.fit(r,e) # training
                self.evaluate(r,e,pbar=epochs,modes=['train','val'])
                if hasattr(self.optimizer,'setLR'): self.optimizer.setLR(torch.mean(self.stats['train']['loss'][r][e]))
                
            self.evaluate(r,e,runs,modes=['train','val','test'])


             
        print('\n\n')


    def fit(self,r,e):
        batches = self.dl['train'] #tqdm(self.dl['train'],desc='Train',leave=False)
        # Fit all training data
        for b_i,b in enumerate(batches):
            # Forward propagate
            i=b[0].to(self.device) # Input
            g=b[1].to(self.device) # GT
            o = self.model(i) # Output
            
            l = self.loss(o,g) # Calculate loss
            
            # Backpropagation
            l.backward() # Backpropagate loss

            self.optimizer.step() # Update
            self.optimizer.zero_grad() # Reset gradients
        
    
    def evaluate(self,r,e,pbar,modes):
        self.model.eval()
        with torch.no_grad():
            for mode in modes:
                # Select appropriate dataloader
                batches = self.dl[mode]
                
                # Raincheck
                if (len(batches)==0): continue

                # Evaluate now
                for b_i,b in enumerate(batches):
                    # Forward propagate
                    i=b[0].to(self.device) # Input
                    g=b[1].to(self.device) # GT
                    o = self.model(i) # Output
                    #imshow_all(o,4); plt.pause(0.1)
                    # Statistics
                    if mode=='test': 
                        self.update_metrics(o.data,g.data,r,0,b_i,mode)
                    else: self.update_metrics(o.data,g.data,r,e,b_i,mode)

        self.track_all_stats(r,e,modes)     
        if self.verbose>1: self._log_to_pbar(pbar,[r,e],modes)
        self.model.train()
        
        
    def predict_dl(self,dl=[]):
        if dl==[]: dl = self.train_dl
        outputs = []
        with torch.no_grad():
            for b in dl:
                i=b[0].to(self.device) # Input
                g=b[1].to(self.device) # GT
                o = self.model(i) # Output
                outputs.append(o)
        return outputs
