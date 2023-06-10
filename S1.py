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
        m=l*l
        n=m
        d=1
        vols=(2*np.pi/m)*torch.ones(m).to(dtype=torchdtype, device=torchdeviceId)
        super().__init__(d,m,n,vols)

    def div(self,p,X):
        print("Needs to be copied from previous version")

    def inner_prod(self,p,X):        
        print("Needs to be copied from previous version")

    def plot_pair(self,mu1,mu2):        
        print("Needs to be copied from previous version")

    def plot_sequence(self,ps,vecs):        
        print("Needs to be copied from previous version")

    def save_gif(self,ps,filename="results/array.gif"):        
        print("Needs to be implemented")

    def gaussian(self,a,b,G):        
        print("Needs to be copied from previous version")
