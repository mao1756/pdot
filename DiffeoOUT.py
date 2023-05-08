import torch
import torch.nn.functional as tnn
from scipy.optimize import minimize,fmin_l_bfgs_b
from utils import *
use_cuda = 0
torchdeviceId = torch.device('cuda:0') if use_cuda else 'cpu'
torchdtype = torch.float64
from torch.autograd import grad
import numpy as np
import scipy.interpolate

def enr_OUT(source,target,grid,match_coeff,path_coeff,inner_prod,delta):
    m=grid.shape[1]
    def energy(vecs,funs):
        p=source
        path_enr=0
        for i in range(0,vecs.shape[0]):
            X=vecs[i]
            f=funs[i]
            path_enr += inner_prod(X,source,p,grid)/2 + (delta**2)*(f[:,:,0]**2/p[0,0]).sum()/2
            p=resample_density(p,grid,X)+f[:,:,0]
        return path_coeff*path_enr + match_coeff*L2_error(p,target)
    return energy
    
def path_length(source,vecs,funs,grid,inner_prod,delta):    
    path_enr=[]
    p=source
    for i in range(1,vecs.shape[0]):
        path_enr += [(inner_prod(vecs[i],source,p,grid)/2 + (delta**2)*(funs[i,:,:,0]**2/p[0,0]).sum()/2).sqrt().item()]
        p=resample_density(p,grid,vecs[i])+funs[i,:,:,0]
    return np.array(path_enr)    
    
def DiffeoOUT(mu_1,mu_2,grid,T,match_coeff,path_coeff,vecs,funs,inner_prod,delta,max_iter=10000):
    m=grid.shape[1]
    n=m
    
    vecs = vecs[1:]    
    funs = funs[1:]
    inputs = torch.cat([vecs,funs], dim=3)
    energy = enr_OUT(mu_1,mu_2,grid,match_coeff,path_coeff,inner_prod,delta)

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
    outputs = torch.cat([torch.zeros((1,m,n,3)),torch.from_numpy(xopt.reshape(T-1,m,n,3))], dim=0).to(dtype=torchdtype, device=torchdeviceId)
    vecs=outputs[:,:,:,0:2]
    funs=outputs[:,:,:,2:3]
    return vecs,funs

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
        vecs,funs = DiffeoOUT(mu_1,mu_2,grid,T,match_coeff,path_coeff,vecs,funs,inner_prod,delta)        
    return vecs,funs
