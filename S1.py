import torch
from DiffeoOTDomain import Domain
import torch.nn.functional as tnn
use_cuda = 0
torchdeviceId = torch.device('cuda:0') if use_cuda else 'cpu'
torchdtype = torch.float64
from torch.autograd import grad
import numpy as np
import scipy.interpolate
from PIL import Image
import matplotlib.pyplot as plt

class S1(Domain):    
    def __init__(self, m):
        d=1
        vols=(2*np.pi/m)*torch.ones(m).to(dtype=torchdtype, device=torchdeviceId)
        x =2*np.pi*torch.arange(0,m)/m
        self.grid = torch.stack([torch.cos(x),torch.sin(x)], dim=1).T
        super().__init__(d,m,m,vols)

    def div(self,p,X):
        pX= p*X[:,0]
        pdot=torch.diff(pX,prepend=pX[-1:],dim=0)
        return pdot

    def inner_prod(self,p,X):
        pX= p*X[:,0]
        return (((pX)**2)/p).sum()  

    def plot_pair(self,mu1,mu2):        
        print("Needs to be copied from previous version")

    def plot_sequence(self,ps,vecs):
        xs=torch.cumsum(ps*self.grid[0],dim=1)
        ys=torch.cumsum(ps*self.grid[1],dim=1)
        xmin=min(xs.min().item(),0)-.25
        xmax=xs.max().item()+.25
        ymin=min(ys.min().item(),0)-.25
        ymax=ys.max().item()+.25
        T=vecs.shape[0]
        fig, axes = plt.subplots(ncols=T, figsize=(10*T, 10))
        x=torch.cat([torch.zeros(1),torch.cumsum(ps[0]*self.grid[0],dim=0)],dim=0)
        y=torch.cat([torch.zeros(1),torch.cumsum(ps[0]*self.grid[1],dim=0)],dim=0)
        axes[0].plot(x.cpu(),y.cpu())
        axes[0].set_xlim([xmin,xmax])
        axes[0].set_ylim([ymin,ymax])
        for i in range(1,vecs.shape[0]):
            x=torch.cat([torch.zeros(1),torch.cumsum(ps[i+1]*self.grid[0],dim=0)],dim=0)
            y=torch.cat([torch.zeros(1),torch.cumsum(ps[i+1]*self.grid[1],dim=0)],dim=0)
            axes[i].plot(x.cpu(),y.cpu())
            axes[i].set_xlim([xmin,xmax])
            axes[i].set_ylim([ymin,ymax])
        plt.show()
        
        
    def save_gif(self,ps,filename="results/array.gif"):        
        print("Needs to be implemented")

    def gaussian(self,a,b,G):        
        print("Needs to be copied from previous version")
