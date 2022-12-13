from .osl import *
from .utils import *
from .OpBlock import *

class OpTier(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,operators,OPLIB,padding=-1,sampling_factor=1,layer_idx=-1,optimize=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sampling_factor = sampling_factor
        self.layer_idx = layer_idx
        self.padding = padding
        if(len(operators)==1): operators = [operators[0] for i in range(out_channels)]
        self.operators = operators
        
        unique_ops = np.unique(operators)
        num_blocks = len(unique_ops)
        #print("Number of blocks: ",num_blocks)
        
        self.oper = nn.ModuleList()
        if optimize:
            for op_idx_now in unique_ops:
                self.oper.append(OpBlock(in_channels,np.sum(operators==op_idx_now),kernel_size,op_idx_now,OPLIB))
        else:
            for op_idx_now in operators:
                self.oper.append(OpBlock(in_channels,1,kernel_size,op_idx_now,OPLIB))

    def init_variances(self,prev):
        #print("Pre: ")
        for n in range(len(self.oper)):
            for pn in range(len(prev.oper)):
                self.oper[n].weight_var_pre[pn] = 1000 * self.oper[n].weights[pn].data.var().item()
                prev.oper[pn].conn_stat_pre += self.oper[n].weight_var_pre[pn]
                #print(prev.oper[pn].conn_stat_pre,end=',')
        #print('')
                
    
    def update_variances(self,prev):
        #print("Now: ")
        for n in range(len(self.oper)):
            for pn in range(len(prev.oper)):
                self.oper[n].weight_var_now[pn] = 1000 * self.oper[n].weights[pn].data.var().item()
                prev.oper[pn].conn_stat_now += self.oper[n].weight_var_now[pn]
                #print(prev.oper[pn].conn_stat_now,end=',')
        #print('')    

    def reset_parameters(self):
        for n in self.oper:
            n.reset_parameters()

    def forward(self,x):
        self.output = []
        # Forward Propagation
        if self.padding==-1: padding = int(np.ceil(self.kernel_size/2))-1
        else: padding = self.padding
        x = F.unfold(x,kernel_size=self.kernel_size,padding=padding)
        x = x.view(x.shape[0],self.in_channels,self.kernel_size**2,-1) # -1 = number of receptive fields
        for block in self.oper: self.output.append(block.forward(x))
        self.output = torch.cat(self.output,dim=1)

        # Subsampling
        if self.sampling_factor>1:
            self.output = torch.nn.functional.max_pool2d(self.output,kernel_size=(int(self.sampling_factor)), padding=0)
        elif self.sampling_factor<1:
            self.output = torch.nn.functional.interpolate(self.output,scale_factor=abs(int(self.sampling_factor)))

        return self.output
