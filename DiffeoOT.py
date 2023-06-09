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

def enr_OT(source,target,grid,match_coeff,path_coeff,inner_prod,delta=1):
    m=grid.shape[1]
    def energy(vecs,funs=None):
        ps=get_ps(source, grid, inner_prod, vecs, funs)
        T=vecs.shape[0]
        path_enr=0
        for i in range(0,T):
            X=vecs[i]
            if funs is not None:
                f=funs[i]
                path_enr += inner_prod(X,source,ps[i],grid)/2 + (delta**2/2)*((f[:,:,0])**2/ps[i]).sum()
            else:
                path_enr += inner_prod(X,source,ps[i],grid)
                
        return path_coeff*path_enr + match_coeff*L2_error(ps[-1],target)
    return energy

def get_ps(source, grid, inner_prod, vecs, funs=None):
    m=grid.shape[1]
    T=vecs.shape[0]
    ls=[source]
    for i in range(0,T):
        X=vecs[i]
        if funs is not None:
            f=funs[i]
            ls+=[torch.clamp(resample_density(ls[i],grid,X,T)+f[:,:,0]/T,min=1e-6)]
        else:
            ls+=[torch.clamp(resample_density(ls[i],grid,X,T),min=1e-6)]
    return torch.stack(ls,dim=0)
    
def path_length(grid,source,inner_prod,vecs,funs=None,delta=1): 
    m=grid.shape[1]
    ps=get_ps(source, grid, inner_prod, vecs, funs)
    T=vecs.shape[0]
    path_enr=[]
    for i in range(0,T):
        X=vecs[i]
        if funs is not None:
            f=funs[i]
            path_enr += [(inner_prod(X,source,ps[i],grid)/2 + (delta**2/2)*(f[:,:,0]**2/ps[i]).sum()).sqrt().item()]
        else:
            path_enr += [(inner_prod(X,source,ps[i],grid)).sqrt().item()]
    return np.array(path_enr)    
    
def DiffeoOUT(mu_1,mu_2,grid,T,match_coeff,path_coeff,vecs,funs,inner_prod,delta,max_iter=10000):
    m=grid.shape[1]
    n=m    
    vecs = vecs[1:]    
    funs = funs[1:]
    inputs = torch.cat([vecs,funs], dim=3)
    energy = enr_OT(mu_1,mu_2,grid,match_coeff,path_coeff,inner_prod,delta)

    def gradE(vecs,funs):
        qvecs = vecs.clone().requires_grad_(True)
        qfuns = funs.clone().requires_grad_(True)
        return grad(energy(qvecs,qfuns), [qvecs,qfuns], create_graph=True)

    def funopt(vecs):
        inputs=torch.from_numpy(vecs.reshape(T-1,m,n,3)).to(dtype=torchdtype, device=torchdeviceId)
        vecs=inputs[:,:,:,0:2]
        funs=inputs[:,:,:,2:3]
        return float(energy(vecs,funs).detach().cpu().numpy())

    def dfunopt(vecs):
        inputs=torch.from_numpy(vecs.reshape(T-1,m,n,3)).to(dtype=torchdtype, device=torchdeviceId)
        vecs=inputs[:,:,:,0:2]
        funs=inputs[:,:,:,2:3]
        [Gvecs,Gfuns] = gradE(vecs,funs)
        Ginputs = torch.cat([Gvecs,Gfuns], dim=3)
        Ginputs = Ginputs.detach().cpu().numpy().flatten().astype('float64')
        return Ginputs

    xopt,fopt,Dic=fmin_l_bfgs_b(funopt, inputs.cpu().numpy().flatten(), fprime=dfunopt, pgtol=1e-09, epsilon=1e-15, maxiter=max_iter, iprint = 1, maxls=20,maxfun=150000)
    outputs = torch.from_numpy(xopt.reshape(T-1,m,n,3)).to(dtype=torchdtype, device=torchdeviceId)
    vecs=outputs[:,:,:,0:2]
    funs=outputs[:,:,:,2:3]
    ps=get_ps(mu_1, grid, inner_prod, vecs, funs)
    return vecs,funs,ps

def DiffeoOUT_multires(mu_1,mu_2,grid,params,inner_prod,delta):
    m=grid.shape[1]
    n=grid.shape[2]    
    vecs = torch.zeros((2,m,n,2)).to(dtype=torchdtype, device=torchdeviceId)    
    funs = torch.zeros((2,m,n,1)).to(dtype=torchdtype, device=torchdeviceId)      
    for param in params:
        T=param['T']
        match_coeff=param['match_coeff']
        path_coeff=param['path_coeff']        
        if vecs.shape[0]<T:
            vecs=upsampleT(vecs,T)  
            funs=upsampleT(funs,T)        
        if ('match_coeff' in param):
            max_iter=param['match_coeff']            
        vecs,funs,ps = DiffeoOUT(mu_1,mu_2,grid,T,match_coeff,path_coeff,vecs,funs,inner_prod,delta)        
    return vecs,funs,ps

def DiffeoOT(mu_1,mu_2,grid,T,match_coeff,path_coeff,vecs,inner_prod,max_iter=10000):
    m=grid.shape[1]
    n=m
    
    vecs = vecs[1:]    
    energy = enr_OT(mu_1,mu_2,grid,match_coeff,path_coeff,inner_prod)

    def gradE(vecs):
        qvecs = vecs.clone().requires_grad_(True)
        return grad(energy(qvecs), qvecs, create_graph=True)

    def funopt(vecs):
        vecs=torch.from_numpy(vecs.reshape(T-1,m,n,2)).to(dtype=torchdtype, device=torchdeviceId)
        return float(energy(vecs).detach().cpu().numpy())

    def dfunopt(vecs):
        vecs = torch.from_numpy(vecs.reshape(T-1,m,n,2)).to(dtype=torchdtype, device=torchdeviceId)
        [Gvecs] = gradE(vecs)
        Gvecs = Gvecs.detach().cpu().numpy().flatten().astype('float64')
        return Gvecs

    xopt,fopt,Dic=fmin_l_bfgs_b(funopt, vecs.cpu().numpy().flatten(), fprime=dfunopt, pgtol=1e-09, epsilon=1e-15, maxiter=max_iter, iprint = 1, maxls=20,maxfun=150000)
    vecs = torch.from_numpy(xopt.reshape(T-1,m,n,2)).to(dtype=torchdtype, device=torchdeviceId)
    ps=get_ps(mu_1, grid, inner_prod, vecs)
    return vecs,ps

def DiffeoOT_multires(mu_1,mu_2,grid,params,inner_prod):
    m=grid.shape[1]
    n=grid.shape[2]    
    vecs = torch.zeros((2,m,n,2)).to(dtype=torchdtype, device=torchdeviceId)      
    for param in params:
        T=param['T']
        match_coeff=param['match_coeff']
        path_coeff=param['path_coeff']        
        if vecs.shape[0]<T:
            vecs=upsampleT(vecs,T)        
        if ('match_coeff' in param):
            max_iter=param['match_coeff']            
        vecs,ps = DiffeoOT(mu_1,mu_2,grid,T,match_coeff,path_coeff,vecs,inner_prod)        
    return vecs,ps