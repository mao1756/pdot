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

class SquareGrid(Domain):    
    def __init__(self, l, L=None):
        self.l=l
        self.L=L
        x = torch.linspace(-1,1,l)
        y = torch.linspace(-1,1,l)
        self.grid = torch.zeros((1,l,l,2)).to(dtype=torchdtype, device=torchdeviceId)
        for i in range(0,l):
            for j in range(0,l):
                self.grid[0,i,j,0]= x[i]
                self.grid[0,i,j,1]= y[j]
        m=l*l
        n=l*l
        d=2
        vols=4/(m)*torch.ones(m).to(dtype=torchdtype, device=torchdeviceId)
        super().__init__(d,m,n,vols)

    def div(self,p,X):
        pX= p*X.transpose(0,1)
        pX=pX.reshape(2,self.l,self.l)
        divpX=torch.zeros(self.l,self.l).to(dtype=torchdtype, device=torchdeviceId)
        dpXdx=pX[0].diff(dim=0)    
        dpXdy=pX[1].diff(dim=1)
        divpX[1:-1]+=dpXdx[1:]
        divpX[:,1:-1]+=dpXdy[:,1:]
        return divpX.flatten()

    def inner_prod(self,p,X):
        pX= p*X.transpose(0,1)
        pX=pX.reshape(self.d,self.l,self.l)
        if self.L is not None:
            return ((self.L(pX))**2).sum(dim=0).flatten()
        return ((pX)**2).sum(dim=0).flatten()

    def plot_pair(self,mu1,mu2):
        vmax= 1.2*max(mu1.max(),mu2.max()).item()
        fig, axes = plt.subplots(ncols=2, figsize=(20, 10))
        axes[0].matshow(mu1.reshape(self.l,self.l).cpu().numpy(), vmin=0, vmax=vmax)
        axes[1].matshow(mu2.reshape(self.l,self.l).cpu().numpy(), vmin=0, vmax=vmax)
        plt.show()

    def plot_sequence(self,ps,vecs):
        vmax= ps.max().item()
        T=vecs.shape[0]
        ps=ps.reshape(T+1,self.l,self.l)
        vecs=vecs.reshape(T,self.l,self.l,self.d)
        fig, axes = plt.subplots(ncols=2*T+1, figsize=(20*T+10, 10))
        axes[0].pcolormesh(self.grid[0,:,:,0].cpu(),self.grid[0,:,:,1].cpu(), ps[0].cpu(), vmin=0, vmax=vmax)
        axes[0].set_xlim([-1, 1])
        axes[0].set_ylim([-1, 1])
        for i in range(0,T):
            X=vecs[i]
            axes[2*i+1].quiver(self.grid[0,:,:,0].cpu(),self.grid[0,:,:,1].cpu(),(1/T)*X[:,:,0].cpu(),(1/T)*X[:,:,1].cpu())
            axes[2*i+1].set_xlim([-1, 1])
            axes[2*i+1].set_ylim([-1, 1])
            axes[2*i+2].pcolormesh(self.grid[0,:,:,0].cpu(),self.grid[0,:,:,1].cpu(), ps[i+1].cpu(), vmin=0, vmax=vmax)
            axes[2*i+2].set_xlim([-1, 1])
            axes[2*i+2].set_ylim([-1, 1])
        plt.show()

    def save_gif(self,ps,filename="results/array.gif"):
        vmax= ps.max().item()
        xp=np.linspace(0,1,ps.shape[0],endpoint=True)
        x=np.linspace(0,1,120,endpoint=True)    
        f=scipy.interpolate.interp1d(xp,ps,axis=0)
        pss=f(x).reshape(120,self.l,self.l)
        imgs = [Image.fromarray(np.floor(((255/vmax)*img))) for img in pss]
        imgs[0].save(filename, save_all=True, append_images=imgs[1:], duration=50, loop=0)

    def load_dist_inv(self,file_name):
        im = Image.open(file_name)
        im = im.convert("L")
        pix = 280-np.array(im.getdata()).reshape(1,1,im.size[0], im.size[1])
        mu=tnn.grid_sample(torch.from_numpy(pix).to(dtype=torchdtype, device=torchdeviceId), self.grid)
        mu=mu.transpose(2,3).flatten()
        mu=self.normalize(mu)
        return mu

    def load_dist(self,file_name):
        im = Image.open(file_name)
        im = im.convert("L")
        pix = 25+np.array(im.getdata()).reshape(1,1,im.size[0], im.size[1])
        mu=tnn.grid_sample(torch.from_numpy(pix).to(dtype=torchdtype, device=torchdeviceId), self.grid)
        mu=mu.transpose(2,3).flatten()
        mu=self.normalize(mu)
        return mu

    def gaussian(self,a,b,G):
        dists = self.grid.clone()
        dists[:,:,:,0]-=a
        dists[:,:,:,1]-=b    
        mu=torch.exp(-1*torch.einsum('...kji,...kji->...kj',torch.einsum('...kji,mi->...kjm',dists,G),dists)).flatten()       
        mu=self.normalize(mu)
        return mu.to(dtype=torchdtype, device=torchdeviceId)
