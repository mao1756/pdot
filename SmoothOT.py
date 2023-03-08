import torch
import torch.nn.functional as tnn
from scipy.optimize import minimize,fmin_l_bfgs_b
use_cuda = 0
torchdeviceId = torch.device('cuda:0') if use_cuda else 'cpu'
torchdtype = torch.float64
from torch.autograd import grad
import numpy as np
import scipy.interpolate

def Gaussian(a,b,G,grid):
    m=grid.shape[1]
    dists = grid.clone()
    dists[:,:,:,0]-=a
    dists[:,:,:,1]-=b    
    mu=torch.exp(-1*torch.einsum('...kji,...kji->...kj',torch.einsum('...kji,mi->...kjm',dists,G),dists))        
    mu=mu/mu.sum()
    return mu.reshape(1,1,m,m).to(dtype=torchdtype, device=torchdeviceId)


def L2Error(mu_1,mu_2):
    m=mu_1.shape[2]
    return (m*m*((mu_1-mu_2))**2).sum()

def L(vf):
    weight = torch.Tensor([[[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]]]).to(dtype=torchdtype, device=torchdeviceId)
    out = tnn.conv2d(vf, weight.repeat(2,1,1,1), bias=None,groups=2, stride=1, padding='same') 
    out = tnn.conv2d(out, weight.repeat(2,1,1,1), bias=None,groups=2, stride=1, padding='same') 
    out = tnn.conv2d(out, weight.repeat(2,1,1,1), bias=None,groups=2, stride=1, padding='same') 
    return out
    


def InnerProd(X,source,p,m):
    pm=p[0,0]
    pX= pm*X.transpose(0,2)
    return ((L(pX)**2).sum(dim=0)/(pm)).sum()    


def enr_OT(source,target,grid,match_coeff,path_coeff):
    m=grid.shape[1]
    def energy(vecs):
        p=source
        path_enr=0
        for i in range(0,vecs.shape[0]):
            X=vecs[i]
            path_enr += InnerProd(X,source,p,m)
            p=tnn.grid_sample(p,grid+X).transpose(2,3)
            p=p/p.sum()
        return path_coeff*path_enr + match_coeff*L2Error(p,target)
    return energy
    

def SmoothOT_OPT(mu_1,mu_2,grid,N,match_coeff,path_coeff,vecs,max_iter=10000):
    m=grid.shape[1]
    n=m
    
    vecs = vecs[1:]    
    energy = enr_OT(mu_1,mu_2,grid,match_coeff,path_coeff)

    def gradE(vecs):
        qvecs = vecs.clone().requires_grad_(True)
        return grad(energy(qvecs), qvecs, create_graph=True)

    def funopt(vecs):
        vecs=torch.from_numpy(vecs.reshape(N-1,m,n,2)).to(dtype=torchdtype, device=torchdeviceId)
        return float(energy(vecs).detach().cpu().numpy())

    def dfunopt(vecs):
        vecs = torch.from_numpy(vecs.reshape(N-1,m,n,2)).to(dtype=torchdtype, device=torchdeviceId)
        [Gvecs] = gradE(vecs)
        Gvecs = Gvecs.detach().cpu().numpy().flatten().astype('float64')
        return Gvecs

    xopt,fopt,Dic=fmin_l_bfgs_b(funopt, vecs.cpu().numpy().flatten(), fprime=dfunopt, pgtol=1e-09, epsilon=1e-15, maxiter=max_iter, iprint = 1, maxls=20,maxfun=150000)
    vecs = torch.cat([torch.zeros((1,m,n,2)),torch.from_numpy(xopt.reshape(N-1,m,n,2))], dim=0).to(dtype=torchdtype, device=torchdeviceId)
    return vecs

def upsampleN(vecs,N):
    N0=vecs.shape[0]
    xp=np.linspace(0,1,N0,endpoint=True)
    x=np.linspace(0,1,N,endpoint=True)    
    f=scipy.interpolate.interp1d(xp,vecs.cpu().numpy(),axis=0)
    nvecs=f(x)
    return torch.from_numpy(nvecs).to(dtype=torchdtype, device=torchdeviceId)


def SmoothOT(mu_1,mu_2,grid,params):
    m=grid.shape[1]
    n=grid.shape[2]
    
    vecs = torch.zeros((2,m,n,2)).to(dtype=torchdtype, device=torchdeviceId)      
    for param in params:
        N=param['N']
        match_coeff=param['match_coeff']
        path_coeff=param['path_coeff']
        
        if vecs.shape[0]<N:
            vecs=upsampleN(vecs,N)
        
        if ('match_coeff' in param):
            max_iter=param['match_coeff']
            
        vecs = SmoothOT_OPT(mu_1,mu_2,grid,N,match_coeff,path_coeff,vecs)        
    return vecs
