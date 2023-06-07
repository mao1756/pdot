import torch
import torch.nn as tnn
from scipy.optimize import minimize,fmin_l_bfgs_b
from utils import *
use_cuda = 0
torchdeviceId = torch.device('cuda:0') if use_cuda else 'cpu'
torchdtype = torch.float64
from torch.autograd import grad
import numpy as np
import scipy.interpolate

def enr_OT(source,target,grid,match_coeff,path_coeff,inner_prod):
    m=grid.shape[1]
    def energy(vecs):
        vecs=constraint_integrator(source,grid,vecs)
        p=source
        path_enr=0
        for i in range(0,vecs.shape[0]):
            X=vecs[i]
            path_enr += inner_prod(X,source,p,grid)
            p=resample_densityS1(p,grid,X,vecs.shape[0])
        return path_coeff*path_enr + match_coeff*(((p-target)**2).sum())
    return energy
    
def path_length(source,vecs,grid,inner_prod):    
    path_enr=[]
    p=source
    for i in range(1,vecs.shape[0]):
        path_enr += [inner_prod(vecs[i],source,p,grid).sqrt().item()]
        p=resample_densityS1(p,grid,vecs[i],vecs.shape[0])
    return np.array(path_enr)    

def constraint_integrator(source,grid,vecs):
    first_vecs=torch.zeros(vecs.shape[0],1).to(dtype=torchdtype, device=torchdeviceId)
    nvecs=torch.cat([first_vecs,vecs],dim=1)
    p=source
    for i in range(0,vecs.shape[0]):
        nvecs[i,0]=-1*(p[1:]*vecs[i]*grid[0,1:]).sum()/p[0]
        p=resample_densityS1(p,grid,nvecs[i],vecs.shape[0])
    return nvecs
        
    
def DiffeoOT(mu_1,mu_2,grid,T,match_coeff,path_coeff,vecs,inner_prod,max_iter=10000):
    m=grid.shape[1]
    
    vecs = vecs[1:]    
    energy = enr_OT(mu_1,mu_2,grid,match_coeff,path_coeff,inner_prod)

    def gradE(vecs):
        qvecs = vecs.clone().requires_grad_(True)
        return grad(energy(qvecs), qvecs, create_graph=True)

    def funopt(vecs):
        vecs=torch.from_numpy(vecs.reshape(T-1,m-1)).to(dtype=torchdtype, device=torchdeviceId)
        return float(energy(vecs).detach().cpu().numpy())

    def dfunopt(vecs):
        vecs = torch.from_numpy(vecs.reshape(T-1,m-1)).to(dtype=torchdtype, device=torchdeviceId)
        [Gvecs] = gradE(vecs)
        Gvecs = Gvecs.detach().cpu().numpy().flatten().astype('float64')
        return Gvecs

    xopt,fopt,Dic=fmin_l_bfgs_b(funopt, vecs.cpu().numpy().flatten(), fprime=dfunopt, pgtol=1e-09, epsilon=1e-15, maxiter=max_iter, iprint = 1, maxls=20,maxfun=150000)
    vecs=torch.from_numpy(xopt.reshape(T-1,m-1))
    vecs=constraint_integrator(mu_1,grid,vecs)
    vecs = torch.cat([torch.zeros((1,m)),vecs], dim=0).to(dtype=torchdtype, device=torchdeviceId)
    return vecs

def DiffeoOT_multires(mu_1,mu_2,grid,params,inner_prod):
    m=grid.shape[1]
    vecs = torch.zeros((2,m-1)).to(dtype=torchdtype, device=torchdeviceId)      
    for param in params:
        T=param['T']
        match_coeff=param['match_coeff']
        path_coeff=param['path_coeff']        
        if vecs.shape[0]<T:
            vecs=upsampleT(vecs,T)        
        if ('match_coeff' in param):
            max_iter=param['match_coeff']            
        vecs = DiffeoOT(mu_1,mu_2,grid,T,match_coeff,path_coeff,vecs,inner_prod)        
    return vecs
