import torch
from DiffeoOTDomain import Domain
import torch.nn.functional as tnn
use_cuda = 0
torchdeviceId = torch.device('cuda:0') if use_cuda else 'cpu'
torchdtype = torch.float64
from torch.autograd import grad
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import open3d as o3d
from pylab import get_cmap
import copy

def batchDot(dv1,dv2):
    return torch.einsum('bi,bi->b', dv1,dv2)

def loadMesh(file_name): 
    mesh = o3d.io.read_triangle_mesh(file_name)
    mesh = mesh.subdivide_midpoint(number_of_iterations=2)
    #mesh = mesh.simplify_quadric_decimation(120)
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
        self.mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V),o3d.utility.Vector3iVector(F[:,[0,2,1]]))
        self.mesh.compute_vertex_normals()
        self.mesh.normalize_normals()
        self.N = self.getNormal()
        face_coordinates=V[F]
        v0, v1, v2 = face_coordinates[:, 0], face_coordinates[:, 1], face_coordinates[:, 2]
        self.E=torch.stack([torch.cross(self.N,v2-v1),torch.cross(self.N,v0-v2),torch.cross(self.N,v1-v0)],dim=1)
        
        self.n=V.shape[0]
        self.m=F.shape[0]
        self.d=3
        self.vols=self.getVertAreas()
        super().__init__(self.d,self.m,self.n,self.vols)

    def div(self,p,X):
        pX=self.HodgeStar02(p)*X
        idx = self.F.reshape(-1)
        val = torch.stack([batchDot(pX,self.E[:,0,:]),batchDot(pX,self.E[:,1,:]),batchDot(pX,self.E[:,2,:])],dim=1).reshape(-1)
        incident_f = torch.zeros(self.n, dtype=torchdtype, device=torchdeviceId)
        return -1*incident_f.scatter_add_(0, idx, val)        

    def inner_prod(self,p,X):
        pX=(p.unsqueeze(dim=1))*self.HodgeStar20(X)
        return (pX**2).sum(dim=1).unsqueeze(dim=1)

    def plot_pair(self,mu1,mu2):
        vmax=max(mu1.max(),mu2.max()).item()
        cm=get_cmap('viridis')
        c1=cm(mu1/vmax)
        print(c1.shape)
        c2=cm(mu2/vmax)
        mesh1=copy.deepcopy(self.mesh)    
        mesh1.vertex_colors =  o3d.utility.Vector3dVector(c1[:,0:3])
        mesh1.translate((-2.5,0,0), relative=False)
        mesh2=copy.deepcopy(self.mesh)    
        mesh2.vertex_colors =  o3d.utility.Vector3dVector(c2[:,0:3])
        mesh2.translate((0,0,0), relative=False)
        o3d.visualization.draw_geometries([mesh1,mesh2])

    def plot_sequence(self,ps,vecs):        
        vmax= ps.max().item()
        cm=get_cmap('viridis')
        T=vecs.shape[0]
        ps=ps.reshape(T+1,self.V.shape[0])
        ls=[]
        for i in range(0,T+1):
            c1=cm(ps[i]/vmax)
            mesh1=copy.deepcopy(self.mesh)    
            mesh1.vertex_colors =  o3d.utility.Vector3dVector(c1[:,0:3])
            mesh1.translate((2.5*i,0,0), relative=False)
            ls+=[mesh1]
        o3d.visualization.draw_geometries(ls)

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
        self.f_vols=area
        return self.HodgeStar20(area)
    
    def HodgeStar20(self,f):
        if len(f.shape)==1:
            fs=f.unsqueeze(dim=1)
        else:
            fs=f
        vfs=[]
        for i in range(fs.shape[1]):
            f=fs[:,i]
            idx = self.F.view(-1)
            incident_f = torch.zeros(self.V.shape[0], dtype=torchdtype, device=torchdeviceId)
            val = torch.stack([f] * 3, dim=1).view(-1)
            incident_f.scatter_add_(0, idx, val)    
            vfs += [2*incident_f/6.0] 
        return torch.stack(vfs,dim=1)
    
    def HodgeStar02(self,f):
        if len(f.shape)==1:
            fs=f.unsqueeze(dim=1)
        else:
            fs=f
        ffs=torch.zeros(self.F.shape[0],fs.shape[1]).to(dtype=torchdtype, device=torchdeviceId)
        for i in range(self.V.shape[0]):
            idx = (self.F == i).nonzero(as_tuple=False)[:,0]
            add=fs[i]*1.0/idx.shape[0]
            ffs[idx]+=add
        return ffs
    
        
    def getNormal(self):
        V0, V1, V2 = self.V.index_select(0, self.F[:, 0]), self.V.index_select(0, self.F[:, 1]), self.V.index_select(0, self.F[:, 2])
        N = .5 * torch.cross(V1 - V0, V2 - V0)
        return N
