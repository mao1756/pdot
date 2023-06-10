import torch
import torch.nn as tnn
from scipy.optimize import minimize,fmin_l_bfgs_b
use_cuda = 0
torchdeviceId = torch.device('cuda:0') if use_cuda else 'cpu'
torchdtype = torch.float64
from torch.autograd import grad
import numpy as np
import scipy.interpolate

def enr_OT(source,target,match_coeff,path_coeff,inner_prod,div,vols,delta=1):
    def energy(vecs,funs=None):
        ps=get_ps(source,div, vecs, funs)
        T=vecs.shape[0]
        path_enr=0
        for i in range(0,T):
            X=vecs[i]
            if funs is not None:
                f=funs[i]
                path_enr += ((inner_prod(ps[i],X)/2 + (delta**2/2)*((f[:,0])**2/ps[i]))*vols).sum()
            else:
                path_enr += (inner_prod(ps[i],X)*vols).sum()                
        return path_coeff*path_enr + match_coeff*L2_error(ps[-1],target,vols)
    return energy

def L2_error(mu_1,mu_2,vols):
    return (vols*((mu_1-mu_2))**2).sum()

def get_ps(source,div,vecs,funs=None):
    T=vecs.shape[0]
    ls=[source]
    for i in range(0,T):
        X=vecs[i]
        if funs is not None:
            f=funs[i]
            ls+=[torch.clamp(ls[i]-div(ls[i],X/T)+f[:,0]/T,min=1e-6)]
        else:
            ls+=[torch.clamp(ls[i]-div(ls[i],X/T),min=1e-6)]
            
    return torch.stack(ls,dim=0)
    
def path_length(source,vecs,inner_prod,div,vols,funs=None,delta=1): 
    ps=get_ps(source,div, vecs, funs)
    T=vecs.shape[0]
    ls=[]
    for i in range(0,T):
        X=vecs[i]
        if funs is not None:
            f=funs[i]
            ls += [((inner_prod(ps[i],X)/2 + (delta**2/2)*((f[:,0])**2/ps[i]))*vols).sum().sqrt()]
        else:
            ls += [(inner_prod(ps[i],X)*vols).sum().sqrt()]
    return torch.stack(ls,dim=0)/T

def DiffeoOUT(source,target,T,match_coeff,path_coeff,inner_prod,m,n,d,div,vols,delta,max_iter=10000):    
    funs = torch.zeros(T-1,n,1)    
    vecs = torch.zeros(T-1,m,d)
    energy = enr_OT(source,target,match_coeff,path_coeff,inner_prod,div,vols,delta)
    inputs= torch.cat([funs.flatten(),vecs.flatten()],dim=0)
    def gradE(vecs,funs):
        qvecs = vecs.clone().requires_grad_(True)
        qfuns = funs.clone().requires_grad_(True)
        return grad(energy(qvecs,qfuns), [qvecs,qfuns], create_graph=True)

    def funopt(vecs):
        funs=torch.from_numpy(vecs[:(T-1)*n].reshape(T-1,n,1))
        vecs=torch.from_numpy(vecs[(T-1)*n:].reshape(T-1,m,d))
        return float(energy(vecs,funs).detach().cpu().numpy())

    def dfunopt(vecs):
        funs=torch.from_numpy(vecs[:(T-1)*n].reshape(T-1,n,1))
        vecs=torch.from_numpy(vecs[(T-1)*n:].reshape(T-1,m,d))
        [Gvecs,Gfuns] = gradE(vecs,funs)
        Ginputs = torch.cat([Gfuns.flatten(),Gvecs.flatten()], dim=0)
        Ginputs = Ginputs.detach().cpu().numpy().astype('float64')
        return Ginputs

    xopt,fopt,Dic=fmin_l_bfgs_b(funopt, inputs.cpu().numpy().flatten(), fprime=dfunopt, pgtol=1e-09, epsilon=1e-15, maxiter=max_iter, iprint = 1, maxls=20,maxfun=150000)
    
    funs=torch.from_numpy(xopt[:(T-1)*n].reshape(T-1,n,1))
    vecs=torch.from_numpy(xopt[(T-1)*n:].reshape(T-1,m,d))
    ps=get_ps(source,div, vecs, funs)
    return vecs,funs,ps



def DiffeoOT(source,target,T,match_coeff,path_coeff,inner_prod,m,d,div,vols,max_iter=10000):
    vecs = torch.zeros(T-1,m,d)    
    energy = enr_OT(source,target,match_coeff,path_coeff,inner_prod,div,vols)

    def gradE(vecs):
        qvecs = vecs.clone().requires_grad_(True)
        return grad(energy(qvecs), qvecs, create_graph=True)

    def funopt(vecs):
        vecs=torch.from_numpy(vecs.reshape(T-1,m,d)).to(dtype=torchdtype, device=torchdeviceId)
        return float(energy(vecs).detach().cpu().numpy())

    def dfunopt(vecs):
        vecs = torch.from_numpy(vecs.reshape(T-1,m,d)).to(dtype=torchdtype, device=torchdeviceId)
        [Gvecs] = gradE(vecs)
        Gvecs = Gvecs.detach().cpu().numpy().flatten().astype('float64')
        return Gvecs

    xopt,fopt,Dic=fmin_l_bfgs_b(funopt, vecs.cpu().numpy().flatten(), fprime=dfunopt, pgtol=1e-09, epsilon=1e-15, maxiter=max_iter, iprint = 1, maxls=20,maxfun=150000)
    vecs = torch.from_numpy(xopt.reshape(T-1,m,d)).to(dtype=torchdtype, device=torchdeviceId)
    ps=get_ps(source,div, vecs)
    return vecs,ps


def upsampleT(vecs,N):
    N0=vecs.shape[0]
    xp=np.linspace(0,1,N0,endpoint=True)
    x=np.linspace(0,1,N,endpoint=True)    
    f=scipy.interpolate.interp1d(xp,vecs.cpu().numpy(),axis=0)
    nvecs=f(x)
    return torch.from_numpy(nvecs).to(dtype=torchdtype, device=torchdeviceId)
