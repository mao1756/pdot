import abc
import torch
from DiffeoOT import *

class Domain(abc.ABC):
    #d is the dimension of the tangent vectors used in the OT computation
    #n is the number of vectors (it may not be the same as the number of supports of the measure)
    #m is the number of supports
    def __init__(self,d,m,n,vols):
        self.m=m
        self.n=n
        self.d=d
        self.vols=vols

    @abc.abstractmethod
    def div(self,p,X):
        """
        """
        
    @abc.abstractmethod
    def inner_prod(self,p,X):
        """
        """
        
    @abc.abstractmethod
    def plot_pair(self,mu1,mu2):
        """
        """
        
    @abc.abstractmethod
    def plot_sequence(self,ps,vecs):
        """
        """
        
    @abc.abstractmethod
    def save_gif(self,ps,filename="results/array.gif"):
        """
        """
        
    def normalize(self,mu):
        return mu/(mu*self.vols).sum()
    
    def OT(self,mu1,mu2,T=6,match_coeff=1e6,path_coeff=1,max_iter=1000):
        return DiffeoOT(mu1.flatten(),mu2.flatten(),T,match_coeff,path_coeff,self.inner_prod,self.m,self.d,self.div,self.vols,max_iter)
        
    def OUT(self,mu1,mu2,T=6,match_coeff=1e6,path_coeff=1,delta=1,max_iter=1000):
        return DiffeoOUT(mu1.flatten(),mu2.flatten(),T,match_coeff,path_coeff,self.inner_prod,self.m,self.n,self.d,self.div,self.vols,delta,max_iter)
    
    def path_length(self,mu1,vecs, funs=None,delta=1):
        return path_length(mu1,vecs,self.inner_prod,self.div,self.vols,funs,delta)