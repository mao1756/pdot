import torch
import torch.nn.functional as tnn
use_cuda = 0
torchdeviceId = torch.device('cuda:0') if use_cuda else 'cpu'
torchdtype = torch.float64
from torch.autograd import grad
import numpy as np
import scipy.interpolate
from PIL import Image
import matplotlib.pyplot as plt

def gaussian(a,b,G,grid):
    m=grid.shape[1]
    dists = grid.clone()
    dists[:,:,:,0]-=a
    dists[:,:,:,1]-=b    
    mu=torch.exp(-1*torch.einsum('...kji,...kji->...kj',torch.einsum('...kji,mi->...kjm',dists,G),dists))        
    mu=mu/mu.sum()
    return mu.reshape(1,1,m,m).to(dtype=torchdtype, device=torchdeviceId)

def resample_density(p,grid,X,N):
    X=X/N
    m=grid.shape[1]
    pm=p[0,0]
    pX= pm*X.transpose(0,2)
    pdot=torch.zeros(m,m).to(dtype=torchdtype, device=torchdeviceId)
    dpXdx=pX[0].diff(dim=0)    
    dpXdy=pX[1].diff(dim=1)
    pdot[1:-1]-=dpXdx[1:]
    pdot[:,1:-1]-=dpXdy[:,1:]
    return p+pdot

def resample_densityS1(p,grid,X,N):
    X=X/N
    m=grid.shape[1]
    pX= p*X
    pdot=torch.diff(pX,prepend=pX[-1:m],dim=0)
    return p-pdot

def upsampleT(vecs,N):
    N0=vecs.shape[0]
    xp=np.linspace(0,1,N0,endpoint=True)
    x=np.linspace(0,1,N,endpoint=True)    
    f=scipy.interpolate.interp1d(xp,vecs.cpu().numpy(),axis=0)
    nvecs=f(x)
    return torch.from_numpy(nvecs).to(dtype=torchdtype, device=torchdeviceId)

def L2_error(mu_1,mu_2):
    m=mu_1.shape[2]
    return (m*m*((mu_1-mu_2))**2).sum()

def plot_pair(mu_1,mu_2):
    vmax= 1.2*max(mu_1.max(),mu_2.max()).item()
    fig, axes = plt.subplots(ncols=2, figsize=(20, 10))
    axes[0].matshow(mu_1[0,0].cpu().numpy(), vmin=0, vmax=vmax)
    axes[1].matshow(mu_2[0,0].cpu().numpy(), vmin=0, vmax=vmax)
    plt.show()

def make_and_plot_sequence(mu_1,mu_2,vecs, grid):
    vmax= 1.2*max(mu_1.max(),mu_2.max()).item()
    m=grid.shape[1]
    p=mu_1
    ls=[p.cpu().numpy()]
    T=vecs.shape[0]
    fig, axes = plt.subplots(ncols=2*T-1, figsize=(20*T-10, 10))
    axes[0].pcolormesh(grid[0,:,:,0].cpu(),grid[0,:,:,1].cpu(), p[0,0].transpose(0,1).cpu(), vmin=0, vmax=vmax)
    axes[0].set_xlim([-1, 1])
    axes[0].set_ylim([-1, 1])
    for i in range(1,vecs.shape[0]):
        X=vecs[i]
        axes[2*i-1].quiver(grid[0,:,:,0].cpu(),grid[0,:,:,1].cpu(),-(1/T-1)*X[:,:,0].cpu(),-(1/T-1)*X[:,:,1].cpu())
        axes[2*i-1].set_xlim([-1, 1])
        axes[2*i-1].set_ylim([-1, 1])
        p=resample_density(p,grid,X,T-1)
        ls+=[p.cpu().numpy()]
        axes[2*i].pcolormesh(grid[0,:,:,0].cpu(),grid[0,:,:,1].cpu(), p[0,0].transpose(0,1).cpu(), vmin=0, vmax=vmax)
        axes[2*i].set_xlim([-1, 1])
        axes[2*i].set_ylim([-1, 1])
    plt.show()
    return ls

def make_and_plot_sequenceS1(mu_1,mu_2,vecs, grid):
    vmax= 1.2*max(mu_1.max(),mu_2.max()).item()
    m=grid.shape[1]
    p=mu_1
    ls=[p.cpu().numpy()]
    T=vecs.shape[0]
    fig, axes = plt.subplots(ncols=T, figsize=(10*T, 10))
    axes[0].plot(p.cpu())
    axes[0].set_xlim([0, m-1])
    axes[0].set_ylim([0, vmax])
    for i in range(1,vecs.shape[0]):
        X=vecs[i]
        p=resample_densityS1(p,grid,X,T-1)
        ls+=[p.cpu().numpy()]
        axes[i].plot(p.cpu())
        axes[i].set_xlim([0, m-1])
        axes[i].set_ylim([0, vmax])
    
    axes[i].plot(mu_2.cpu())
    plt.show()
    return ls

def make_and_plot_sequenceS1_c(mu_1,mu_2,vecs, grid):
    vmax= 4*max(mu_1.max(),mu_2.max()).item()
    m=grid.shape[1]
    p=mu_1
    ls=[p.cpu().numpy()]
    T=vecs.shape[0]
    fig, axes = plt.subplots(ncols=T, figsize=(10*T, 10))
    x=torch.cat([torch.zeros(1),torch.cumsum(p*grid[0],dim=0)],dim=0)
    y=torch.cat([torch.zeros(1),torch.cumsum(p*grid[1],dim=0)],dim=0)
    axes[0].plot(x.cpu(),y.cpu())
    axes[0].set_xlim([-1*vmax, vmax])
    axes[0].set_ylim([-1*vmax, vmax])
    for i in range(1,vecs.shape[0]):
        X=vecs[i]
        p=resample_densityS1(p,grid,X,T-1)
        ls+=[p.cpu().numpy()]
        x=torch.cat([torch.zeros(1),torch.cumsum(p*grid[0],dim=0)],dim=0)
        y=torch.cat([torch.zeros(1),torch.cumsum(p*grid[1],dim=0)],dim=0)
        axes[i].plot(x.cpu(),y.cpu())
        axes[i].set_xlim([-1*vmax, vmax])
        axes[i].set_ylim([-1*vmax, vmax])
    
    x=torch.cat([torch.zeros(1),torch.cumsum(mu_2*grid[0],dim=0)],dim=0)
    y=torch.cat([torch.zeros(1),torch.cumsum(mu_2*grid[1],dim=0)],dim=0)
    axes[i].plot(x.cpu(),y.cpu())
    plt.show()
    return ls

def make_and_plot_sequence_u(mu_1,mu_2,vecs,funs, grid):
    vmax= 1.2*max(mu_1.max(),mu_2.max()).item()
    m=grid.shape[1]
    p=mu_1
    ls=[p.cpu().numpy()]
    T=vecs.shape[0]
    fig, axes = plt.subplots(ncols=2*T-1, figsize=(20*T-10, 10))
    axes[0].pcolormesh(grid[0,:,:,0].cpu(),grid[0,:,:,1].cpu(), p[0,0].transpose(0,1).cpu(), vmin=0, vmax=vmax)
    axes[0].set_xlim([-1, 1])
    axes[0].set_ylim([-1, 1])
    for i in range(1,vecs.shape[0]):
        X=vecs[i]
        axes[2*i-1].quiver(grid[0,:,:,0].cpu(),grid[0,:,:,1].cpu(),-(1/T-1)*X[:,:,0].cpu(),-(1/T-1)*X[:,:,1].cpu())
        axes[2*i-1].set_xlim([-1, 1])
        axes[2*i-1].set_ylim([-1, 1])
        p=resample_density(p,grid,X,T-1)+funs[i,:,:,0]
        ls+=[p.cpu().numpy()]
        axes[2*i].pcolormesh(grid[0,:,:,0].cpu(),grid[0,:,:,1].cpu(), p[0,0].transpose(0,1).cpu(), vmin=0, vmax=vmax)
        axes[2*i].set_xlim([-1, 1])
        axes[2*i].set_ylim([-1, 1])
    plt.show()
    return ls

def save_gif(mu_1,mu_2,ls, filename="array.gif"):
    vmax= 1.2*max(mu_1.max(),mu_2.max()).item()
    ps= np.concatenate(ls,axis=0)
    xp=np.linspace(0,1,ps.shape[0],endpoint=True)
    x=np.linspace(0,1,120,endpoint=True)    
    f=scipy.interpolate.interp1d(xp,ps,axis=0)
    pss=f(x)
    imgs = [Image.fromarray(np.floor(((255/vmax)*img[0]))) for img in pss]
    imgs[0].save(filename, save_all=True, append_images=imgs[1:], duration=50, loop=0)


def load_dist(file_name,grid):
    im = Image.open(file_name)
    im = im.convert("L")
    pix = 280-np.array(im.getdata()).reshape(1,1,im.size[0], im.size[1])
    mu=tnn.grid_sample(torch.from_numpy(pix).to(dtype=torchdtype, device=torchdeviceId), grid)
    mu=mu/mu.sum()
    mu=mu.transpose(2,3)
    return mu

def normalize(mu):
    return mu/mu.sum()