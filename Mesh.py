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
import open3d as o3d

def batchDot(dv1,dv2):
    return torch.einsum('bi,bi->b', dv1,dv2)

def loadMesh(file_name): 
    mesh = o3d.io.read_triangle_mesh(file_name)
    V = torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float64)).to(dtype=torchdtype,device=torchdeviceId) 
    F = torch.from_numpy(np.asarray(mesh.triangles, np.int64)).to(device=torchdeviceId)  
    color=np.zeros((int(V.shape[0]),0))
    if mesh.has_vertex_colors():
        color=torch.from_numpy(np.asarray(255*np.asarray(mesh.vertex_colors,dtype=np.float64), dtype=np.int)).to( device=torchdeviceId)
    return V, F, color

class Mesh(Domain):    
    def __init__(self, V,F, file_name=None):
        if file_name is not None:
            V, F, _ = loadMesh(file_name)
        self.V=V
        self.F=F
        self.N = self.getNormal()
        face_coordinates = V[F]
        v0, v1, v2 = face_coordinates[:, 0], face_coordinates[:, 1], face_coordinates[:, 2]
        self.E=torch.stack([torch.cross(self.N,v2-v1),torch.cross(self.N,v0-v2),torch.cross(self.N,v1-v0)],dim=1)
        
        self.m=V.shape[0]
        self.n=F.shape[0]
        self.d=3
        self.vols=self.getVertAreas()
        super().__init__(self.d,self.m,self.n,self.vols)

    def div(self,p,X):
        print("Requires HodgeStar02 to compute momentum. This needs to be added")
        #pX= (p*X.transpose(0,1)).transpose(0,1)
        pX=X
        idx = self.F.reshape(-1)
        val = torch.stack([batchDot(pX,self.E[:,0,:]),batchDot(pX,self.E[:,1,:]),batchDot(pX,self.E[:,2,:])],dim=1).reshape(-1)
        incident_f = torch.zeros(self.n, dtype=torchdtype, device=torchdeviceId)
        return -1*incident_f.scatter_add_(0, idx, val)        

    def inner_prod(self,p,X):
        print("Needs to be implemented. Will require HodgeStar02 to compute momentum")

    def plot_pair(self,mu1,mu2):        
        print("Needs to be implemented")

    def plot_sequence(self,ps,vecs):        
        print("Needs to be implemented")

    def save_gif(self,ps,filename="results/array.gif"):                
        print("Needs to be implemented")

    def load_dist(self,file_name):        
        print("Needs to be implemented")

    def gaussian(self,a,b,G):        
        print("Needs to be implemented")
        
    def getVertAreas(self):
        # Number of vertices
        # Get x,y,z coordinates of each face
        face_coordinates = self.V[self.F]
        v0, v1, v2 = face_coordinates[:, 0], face_coordinates[:, 1], face_coordinates[:, 2]

        # Compute the area of each face using Heron's formula
        A = (v1 - v2).norm(dim=1)
        B = (v0 - v2).norm(dim=1) # lengths of each side of the faces
        C = (v0 - v1).norm(dim=1)
        s = 0.5 * (A + B + C) # semi-perimeter
        area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-6).sqrt() # Apply Heron's formula and clamp areas of small faces for numerical stability
        return self.HodgeStar20(area)
    
    def HodgeStar20(self,f):
        f=f.to(dtype=torchdtype,device=torchdeviceId)
        idx = self.F.reshape(-1)
        incident_f = torch.zeros(self.n, dtype=torchdtype, device=torchdeviceId)
        val = torch.stack([f] * 3, dim=1).reshape(-1)
        incident_f.scatter_add_(0, idx, val)    
        vf = 2*incident_f/6.0 
        return vf
    
    def HodgeStar02(self,f):
        print("Needs to be implemented")
        
    def getNormal(self):
        V0, V1, V2 = self.V.index_select(0, self.F[:, 0]), self.V.index_select(0, self.F[:, 1]), self.V.index_select(0, self.F[:, 2])
        N = .5 * torch.cross(V1 - V0, V2 - V0)
        return N
