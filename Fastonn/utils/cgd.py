import torch
from torch.optim.optimizer import Optimizer


def normGrad(x,thr=10):
    maxx = torch.max(abs(x.data))
    factor = 1 if maxx<thr else thr/maxx 
    x.data *= factor
    return x

class CGD(Optimizer):
    def __init__(self, params, lr=0.001,alpha=1.05,beta=0.7):
        if lr < 0.0: raise ValueError("Invalid learning rate: {}".format(lr))
        self.loss_prev = 1e9
        self.alpha = alpha
        self.beta = beta
        defaults = dict(lr=lr)
        super(CGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(CGD, self).__setstate__(state)

    def getLearningRate(self,loss_now,lr):
        alpha = 1.05
        beta = 0.7
        
        if ((loss_now<=self.loss_prev)):
            if((lr*alpha) < 0.2): new_lr = alpha*lr
            else: new_lr = 0.2
        else:
            if((beta*lr) > 1e-6): new_lr = beta*lr
            else: new_lr = 1e-6
        self.loss_prev = loss_now
        return new_lr
        
    def step(self):
        loss = None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue

                # Clip gradients
                if len(p.grad.size())==3:  # weight, TODO: find a better way to check this
                    for in_c in range(p.shape[0]):
                        for out_c in range(p.shape[1]):
                            p.grad[in_c,out_c,:] = normGrad(p.grad[in_c,out_c,:],10)
                else:
                    for out_c in range(p.shape[0]): 
                        p.grad[out_c].clamp_(min=-0.1,max=0.1)

                # Update
                d_p = p.grad.data
                p.data.add_(-group['lr'], d_p)

        return loss

    def getLR(self):
        for group in self.param_groups: return group['lr']

    def setLR(self,l):
        for group in self.param_groups:
            group['lr'] = self.getLearningRate(l.item(),group['lr'])
