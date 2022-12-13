import torch
from torch.autograd import Function

########### NODAL FUNCTIONS ################
def mul(x,w): 
    return x[:,:,None,:,:].mul(w[None,:,:,:,None])

def cubic(x,w,K_CUB=100): 
    return K_CUB*mul(x.pow(3),w)

def sine(x,w,K_SIN=100): 
    return torch.sin(K_SIN*mul(x,w))

def expp(x,w): return (torch.exp(mul(x,w)) - 1)

def sinh(x,w,K_SINH = 0.100): return torch.sinh(K_SINH*mul(x,w))

def sinc(x,w,K_SINC=1000,K_SINC1=0.01):
    eps = 1e-10 
    x = mul(x,w)
    return (K_SINC1*(torch.sin(K_SINC*(x+eps))/(x+eps)))

def logg(x,w):
    AA =  1.54308063482  # (e+1/e)/2 
    BB =  1.17520119364
    return mul(torch.log(AA+BB*x),w)


class sinc2(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, x,w):
        K_SINC = 1000
        K_SINC1 = 0.01
        EPS_SINC = 0.01
        idx = (x.data<EPS_SINC) & (x.data>-EPS_SINC)
        y = x.clone()
        y[idx] = 0 
        z = K_SINC1 * torch.sin(K_SINC*mul(y,w)) / mul(y,torch.ones_like(w))
        idx_now = torch.isnan(z)
        z[idx_now] = K_SINC1
        ctx.save_for_backward(x, w,idx_now)
        return z

    @staticmethod
    def backward(ctx, grad_output):
        K_SINC = 1000
        K_SINC1 = 0.01
        EPS_SINC = 0.01
        x,w,idx_now = ctx.saved_tensors
        

        # for grad_w
        coef = K_SINC*K_SINC1
        grad_w = grad_output*coef*torch.cos(K_SINC*mul(x,w))
        grad_w = torch.sum(torch.sum(grad_w,dim=-1),dim=0)
        assert w.size()==(grad_w.size()), "gradw is messed up"

        # for grad_x
        
        temp1 = torch.sin(K_SINC*mul(x,w))
        temp2 = K_SINC*torch.cos(K_SINC*mul(x,w))*w[None,:,:,:,None]
        
        grad_x = (K_SINC1*temp1*temp2) / mul(x,torch.ones_like(w))
        grad_x = grad_x.squeeze(2)

        grad_x.clamp_(min=-1,max=1)
        grad_output[idx_now] = 0
        
        grad_x = grad_output*grad_x
        
        
        return grad_x,grad_w



def sincs2(x,w,K_SINC=1000,K_SINC1=0.01):
    eps_sinc = 0.01
    idx = (x.data<eps_sinc) & (x.data>-eps_sinc)
    y = x.clone()
    y[idx] = 0 
    z = K_SINC1 * torch.sin(K_SINC*mul(y,w)) / y
    idx_now = torch.isnan(z)
    z[idx_now] = K_SINC1
    
    return z

def chirp(x,w,K_CHIRP=10000): return (torch.sin(K_CHIRP*mul(x.pow(2),w)))


########### POOL FUNCTIONS ################
def summ(x): 
    return torch.sum(x,dim=3)

def medd(x): 
    return (x.shape[2]*torch.median(x,dim=3))[0]

def maxx(x): 
    return (x.shape[2]*torch.max(x,dim=3))[0]

########### ACTIVATION FUNCTIONS ################
def tanh(x,b): 
    return torch.tanh(x-b[None,:,None,None])

class tanh2(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, x,b):
        result = torch.tanh(x-b[None,:,None,None])
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result = ctx.saved_tensors
        result = result[0]
        grad_x = grad_output*(1-result.pow(2))
        return grad_x,-torch.sum(torch.sum(grad_x,dim=-1),dim=0)

def lincut(x,b,CUT=10): return torch.clamp((x-b[None,:,None,None])/CUT,min=-1,max=1)

def myrelu(x,b): return torch.nn.functional.relu(x-b[None,:,None,None])


def getOPLIB(NODAL,POOL,ACTIVATION):
    OPLIB = []

    for pool_idx in range(len(POOL)):
        for act_idx in range(len(ACTIVATION)):
            for node_idx in range(len(NODAL)):
                OPLIB.append({"nodal":NODAL[node_idx],"pool":POOL[pool_idx],"act":ACTIVATION[act_idx]})
    
    return OPLIB