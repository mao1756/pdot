import abc
import torch

class BCMSolver: 
    """Class for a BCD solver for Kantorovich formulations of OUT.
    
    Parameters
    ----------
    CoM : Metric Space Object
        cone space object used for the initialization of the BCD solver
    use_cuda : bool 
        parameter that defines whether to use cuda.
    """ 
    def __init__(self,use_cuda):
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        
    def _contractionRowCol(self,P,Q,Omega,a,b,m,n):
        P = self._rowNormalize(Q*Omega,a,n)
        Q = self._colNormalize(P*Omega,b,m)
        return P,Q  
    
    def _rowNormalize(self,Pnew,a,n):
        sums = torch.sum(Pnew*Pnew,dim=0)
        zeros = sums==0
        RowNormPnew = torch.sqrt(sums.reshape(1,-1)/a.reshape(1,-1))
        RowNormMatrix = RowNormPnew.repeat([n,1])
        
        Pnew[:,zeros]=0
        RowNormMatrix[:,zeros]=1
        PnewNormalized = Pnew/RowNormMatrix
        return PnewNormalized
    
    def _colNormalize(self,Qnew,b,m):
        sums = torch.sum(Qnew*Qnew,dim=1)
        zeros = sums==0
        ColumnNormQnew = torch.sqrt(sums.reshape(1,-1)/b.reshape(1,-1))
        ColumnNormMatrix = ColumnNormQnew.repeat([m,1]).transpose(0,1)
        Qnew[zeros,:]=0
        ColumnNormMatrix[zeros,:]=1
        QnewNormalized = Qnew/ColumnNormMatrix
        return QnewNormalized
    
    def _calcF(self,P,Q,Omega):
        cost=torch.sum(P*Q*Omega)
        return cost

    
    def getOptimalPQ(self,a,b,Omega,max_steps,eps):
        m = a.shape[0]
        n = b.shape[0]
        Omega.to(self.device)
        P = Omega
        Q = Omega
        
        cost=torch.zeros((max_steps+1,1))
        for k in range(0,max_steps):
            P,Q=self._contractionRowCol(P,Q,Omega,a,b,m,n)
            cost[k+1,:]=self._calcF(P,Q,Omega).cpu()
            ind=k+1
            if (cost[k+1]-cost[k,:])/cost[k+1]<eps:
                break   
                
        return P,Q
    
    def getDistFromPQ(self,P,Q,a,b,Omega,delta):
        return 2*delta*torch.sqrt((a.sum()+b.sum()-2*torch.sum(P*Q*Omega)))