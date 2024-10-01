import time,itertools
import numpy as np
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
np.set_printoptions(suppress=True,precision=4,linewidth=2000)
#from pympler import muppy,summary
# set tensor symmetry
import sys
import autoray as ar
import torch
torch.autograd.set_detect_anomaly(False)
from .torch_utils import SVD,QR,set_max_bond
ar.register_function('torch','linalg.svd',SVD.apply) # basically substitute torch.linalg.svd with our own SVD
ar.register_function('torch','linalg.qr',QR.apply)
this = sys.modules[__name__]

def set_options(pbc=False,deterministic=False,**compress_opts): 
    # By calling this function we define the global variable 'pbc' and 'deterministic'
    this.pbc = pbc
    this.deterministic = True if pbc else deterministic
    this.compress_opts = compress_opts

    set_max_bond(compress_opts.get('max_bond',None))

# Cuurent VMC code works for Cubic lattice geometry only.
# TODO: Implement other lattice geometries.

#####################################################################################
"""
Lattice Numerics:
1. Coordinate to Index
2. Index to Coordinate
3. NOTE: Here implicitly use periodic boundary condition by modulo lattice size.
"""
#####################################################################################

def flatten(i,j,k,Lx,Ly,Lz): # coordinate (x,y,z) to index order.
    return (k%Lz)*Lx*Ly+(i%Lx)*Ly+(j%Ly)
def flat2site(ix,Lx,Ly,Lz): # ix: index
    return (ix%(Lx*Ly))//Ly, (ix%(Lx*Ly))%Ly, ix//(Lx*Ly)


#####################################################################################
# READ/WRITE FTN FUNCS
#####################################################################################
import pickle,uuid

def load_tn_from_disc(fname, delete_file=False):
    if type(fname) != str:
        data = fname
    else:
        with open(fname,'rb') as f:
            data = pickle.load(f)
    return data
def write_tn_to_disc(tn, fname, provided_filename=False):
    with open(fname, 'wb') as f:
        pickle.dump(tn, f)
    return fname

#####################################################################################
# PEPS in 3D and PBC correction to the TNS
#####################################################################################

from .tensor_3d import PEPS3D
from .tensor_core import Tensor,TensorNetwork,rand_uuid,group_inds
import copy

def get_spin_config_state(Lx,Ly,Lz, config=None, bdim=3, eps=None):
    """
        Return product state 3DPEPS of a specific spin configuration.
    """
    arrays = []
    for k in range(Lz):
        layer = []
        for i in range(Lx):
            row = []
            for j in range(Ly):
                shape = [bdim] * 6
                if i==0 or i==Lx-1:
                    shape.pop()
                if j==0 or j==Ly-1:
                    shape.pop()
                if k==0 or k==Lz-1:
                    shape.pop()
                shape = tuple(shape) + (2,) # add physical bond. Desired tensor shape: (bdim*6, 2)

                if config is None:
                    data = np.ones(shape) 
                else:
                    data = np.zeros(shape)
                    ix = flatten(i,j,k,Lx,Ly,Lz)
                    config_loc = config[ix]
                    data[(0,)*(len(shape)-1)+(config_loc,)] = 1.
                if eps is not None:
                    data += eps * np.random.rand(*shape)
                row.append(data)
            layer.append(row)
        arrays.append(layer)
    return PEPS3D(arrays)


def peps2pbc3D_old(opeps): # Basically just link the boundary bonds
    peps = opeps.copy()

    xbonds = [rand_uuid() for i in range(peps.Ly*peps.Lz)] #new xbonds

    ybonds = [rand_uuid() for i in range(peps.Lx*peps.Lz)] #new ybonds

    zbonds = [rand_uuid() for i in range(peps.Lx*peps.Ly)] #new ybonds

    # Attach x-bonds for boundary yz-plane
    for j in range(peps.Ly):
        for k in range(peps.Lz):

            # x = 0 plane (0,y,z)

            tensor = peps[0,j,k]
            bdim,pdim = tensor.data.shape[0],tensor.data.shape[-1]
            X_m = xbonds[j*peps.Lz+k]

            if j==0:
                if k==0:
                    data = np.random.rand(*((bdim,)*4+(pdim,)))
                    x_p,y_p,z_p,phy = tensor.inds
                    inds = x_p,y_p,z_p,X_m,phy
                    tensor.modify(data=data, inds=inds)
                elif k==peps.Lz-1:
                    data = np.random.rand(*((bdim,)*4+(pdim,)))
                    x_p,y_p,z_m,phy = tensor.inds
                    inds = x_p,y_p,X_m,z_m,phy
                    tensor.modify(data=data, inds=inds)
                else:
                    data = np.random.rand(*((bdim,)*5+(pdim,)))
                    x_p,y_p,z_p,z_m,phy = tensor.inds
                    inds = x_p,y_p,z_p,X_m,z_m,phy
                    tensor.modify(data=data, inds=inds)
            elif j==peps.Ly-1:
                if k==0:
                    data = np.random.rand(*((bdim,)*4+(pdim,)))
                    x_p,z_p,y_m,phy = tensor.inds
                    inds = x_p,z_p,X_m,y_m,phy
                    tensor.modify(data=data, inds=inds)
                elif k==peps.Lz-1:
                    data = np.random.rand(*((bdim,)*4+(pdim,)))
                    x_p,y_m,z_m,phy = tensor.inds
                    inds = x_p,X_m,y_m,z_m,phy
                    tensor.modify(data=data, inds=inds)
                else:
                    data = np.random.rand(*((bdim,)*5+(pdim,)))
                    x_p,z_p,y_m,z_m,phy = tensor.inds
                    inds = x_p,z_p,X_m,y_m,z_m,phy
                    tensor.modify(data=data, inds=inds)
            else:
                if k==0:
                    data = np.random.rand(*((bdim,)*5+(pdim,)))
                    x_p,y_p,z_p,y_m,phy = tensor.inds
                    inds = x_p,y_p,z_p,X_m,y_m,phy
                    tensor.modify(data=data, inds=inds)
                elif k==peps.Lz-1:
                    data = np.random.rand(*((bdim,)*5+(pdim,)))
                    x_p,y_p,y_m,z_m,phy = tensor.inds
                    inds = x_p,y_p,X_m,y_m,z_m,phy
                    tensor.modify(data=data, inds=inds)
                else:
                    data = np.random.rand(*((bdim,)*6+(pdim,)))
                    x_p,y_p,z_p,y_m,z_m,phy = tensor.inds
                    inds = x_p,y_p,z_p,X_m,y_m,z_m,phy
                    tensor.modify(data=data, inds=inds)
            
            # x = Lx-1 plane (Lx-1,y,z)
            
            tensor = peps[peps.Lx-1,j,k]
            bdim,pdim = tensor.data.shape[0],tensor.data.shape[-1]
            X_p = xbonds[j*peps.Lz+k] # X_p = X_m
            if j==0:
                if k==0:
                    data = np.random.rand(*((bdim,)*4+(pdim,)))
                    y_p,z_p,x_m,phy = tensor.inds
                    inds = X_p,y_p,z_p,x_m,phy
                    tensor.modify(data=data, inds=inds)
                elif k==peps.Lz-1:
                    data = np.random.rand(*((bdim,)*4+(pdim,)))
                    y_p,x_m,z_m,phy = tensor.inds
                    inds = X_p,y_p,x_m,z_m,phy
                    tensor.modify(data=data, inds=inds)
                else:
                    data = np.random.rand(*((bdim,)*5+(pdim,)))
                    y_p,z_p,x_m,z_m,phy = tensor.inds
                    inds = X_p,y_p,z_p,x_m,z_m,phy
                    tensor.modify(data=data, inds=inds)
            elif j==peps.Ly-1:
                if k==0:
                    data = np.random.rand(*((bdim,)*4+(pdim,)))
                    z_p,x_m,y_m,phy = tensor.inds
                    inds = X_p,z_p,x_m,y_m,phy
                    tensor.modify(data=data, inds=inds)
                elif k==peps.Lz-1:
                    data = np.random.rand(*((bdim,)*4+(pdim,)))
                    x_m,y_m,z_m,phy = tensor.inds
                    inds = X_p,x_m,y_m,z_m,phy
                    tensor.modify(data=data, inds=inds)
                else:
                    data = np.random.rand(*((bdim,)*5+(pdim,)))
                    z_p,x_m,y_m,z_m,phy = tensor.inds
                    inds = X_p,z_p,x_m,y_m,z_m,phy
                    tensor.modify(data=data, inds=inds)
            else:
                if k==0:
                    data = np.random.rand(*((bdim,)*5+(pdim,)))
                    y_p,z_p,x_m,y_m,phy = tensor.inds
                    inds = X_p,y_p,z_p,x_m,y_m,phy
                    tensor.modify(data=data, inds=inds)
                elif k==peps.Lz-1:
                    data = np.random.rand(*((bdim,)*5+(pdim,)))
                    y_p,x_m,y_m,z_m,phy = tensor.inds
                    inds = X_p,y_p,x_m,y_m,z_m,phy
                    tensor.modify(data=data, inds=inds)
                else:
                    data = np.random.rand(*((bdim,)*6+(pdim,)))
                    y_p,z_p,x_m,y_m,z_m,phy = tensor.inds
                    inds = X_p,y_p,z_p,x_m,y_m,z_m,phy
                    tensor.modify(data=data, inds=inds)


    # Attach y-bonds for boundary xz-plane
    for i in range(peps.Lx):
        for k in range(peps.Lz):

            # y = 0 plane (x,0,z)
            
            tensor = peps[i,0,k]
            bdim,pdim = tensor.data.shape[0],tensor.data.shape[-1]
            Y_m = ybonds[i*peps.Lz+k]

            if i==0:
                if k==0:
                    data = np.random.rand(*((bdim,)*5+(pdim,)))
                    x_p,y_p,z_p,x_m,phy = tensor.inds
                    inds = x_p,y_p,z_p,x_m,Y_m,phy
                    tensor.modify(data=data, inds=inds)
                elif k==peps.Lz-1:
                    data = np.random.rand(*((bdim,)*5+(pdim,)))
                    x_p,y_p,x_m,z_m,phy = tensor.inds
                    inds = x_p,y_p,x_m,Y_m,z_m,phy
                    tensor.modify(data=data, inds=inds)
                else:
                    data = np.random.rand(*((bdim,)*6+(pdim,)))
                    x_p,y_p,z_p,x_m,z_m,phy = tensor.inds
                    inds = x_p,y_p,z_p,x_m,Y_m,z_m,phy
                    tensor.modify(data=data, inds=inds)
            elif i==peps.Lx-1: # Can be merged in i==0 case since they are identical when x-direction is already periodic.
                if k==0:
                    data = np.random.rand(*((bdim,)*5+(pdim,)))
                    x_p,y_p,z_p,x_m,phy = tensor.inds
                    inds = x_p,y_p,z_p,x_m,Y_m,phy
                    tensor.modify(data=data, inds=inds)
                elif k==peps.Lz-1:
                    data = np.random.rand(*((bdim,)*5+(pdim,)))
                    x_p,y_p,x_m,z_m,phy = tensor.inds
                    inds = x_p,y_p,x_m,Y_m,z_m,phy
                    tensor.modify(data=data, inds=inds)
                else:
                    data = np.random.rand(*((bdim,)*6+(pdim,)))
                    x_p,y_p,z_p,x_m,z_m,phy = tensor.inds
                    inds = x_p,y_p,z_p,x_m,Y_m,z_m,phy
                    tensor.modify(data=data, inds=inds)
            else:
                if k==0:
                    data = np.random.rand(*((bdim,)*5+(pdim,)))
                    x_p,y_p,z_p,x_m,phy = tensor.inds
                    inds = x_p,y_p,z_p,x_m,Y_m,phy
                    tensor.modify(data=data, inds=inds)
                elif k==peps.Lz-1:
                    data = np.random.rand(*((bdim,)*5+(pdim,)))
                    x_p,y_p,x_m,z_m,phy = tensor.inds
                    inds = x_p,y_p,x_m,Y_m,z_m,phy
                    tensor.modify(data=data, inds=inds)
                else:
                    data = np.random.rand(*((bdim,)*6+(pdim,)))
                    x_p,y_p,z_p,x_m,z_m,phy = tensor.inds
                    inds = x_p,y_p,z_p,x_m,Y_m,z_m,phy
                    tensor.modify(data=data, inds=inds)
            
            # y = Ly-1 plane (Lx-1,y,z)
            
            tensor = peps[i,peps.Ly-1,k]
            bdim,pdim = tensor.data.shape[0],tensor.data.shape[-1]
            Y_p = ybonds[i*peps.Lz+k] # Y_p = XY_m
            if i==0:
                if k==0:
                    data = np.random.rand(*((bdim,)*5+(pdim,)))
                    x_p,z_p,x_m,y_m,phy = tensor.inds
                    inds = x_p,Y_p,z_p,x_m,y_m,phy
                    tensor.modify(data=data, inds=inds)
                elif k==peps.Lz-1:
                    data = np.random.rand(*((bdim,)*5+(pdim,)))
                    x_p,x_m,y_m,z_m,phy = tensor.inds
                    inds = x_p,Y_p,x_m,y_m,z_m,phy
                    tensor.modify(data=data, inds=inds)
                else:
                    data = np.random.rand(*((bdim,)*6+(pdim,)))
                    x_p,z_p,x_m,y_m,z_m,phy = tensor.inds
                    inds = x_p,Y_p,z_p,x_m,y_m,z_m,phy
                    tensor.modify(data=data, inds=inds)
            elif i==peps.Lx-1: # Can be merged in i==0 case since they are identical when x-direction is already periodic.
                if k==0:
                    data = np.random.rand(*((bdim,)*5+(pdim,)))
                    x_p,z_p,x_m,y_m,phy = tensor.inds
                    inds = x_p,Y_p,z_p,x_m,y_m,phy
                    tensor.modify(data=data, inds=inds)
                elif k==peps.Lz-1:
                    data = np.random.rand(*((bdim,)*5+(pdim,)))
                    x_p,x_m,y_m,z_m,phy = tensor.inds
                    inds = x_p,Y_p,x_m,y_m,z_m,phy
                    tensor.modify(data=data, inds=inds)
                else:
                    data = np.random.rand(*((bdim,)*6+(pdim,)))
                    x_p,z_p,x_m,y_m,z_m,phy = tensor.inds
                    inds = x_p,Y_p,z_p,x_m,y_m,z_m,phy
                    tensor.modify(data=data, inds=inds)
            else:
                if k==0:
                    data = np.random.rand(*((bdim,)*5+(pdim,)))
                    x_p,z_p,x_m,y_m,phy = tensor.inds
                    inds = x_p,Y_p,z_p,x_m,y_m,phy
                    tensor.modify(data=data, inds=inds)
                elif k==peps.Lz-1:
                    data = np.random.rand(*((bdim,)*5+(pdim,)))
                    x_p,x_m,y_m,z_m,phy = tensor.inds
                    inds = x_p,Y_p,x_m,y_m,z_m,phy
                    tensor.modify(data=data, inds=inds)
                else:
                    data = np.random.rand(*((bdim,)*6+(pdim,)))
                    x_p,z_p,x_m,y_m,z_m,phy = tensor.inds
                    inds = x_p,Y_p,z_p,x_m,y_m,z_m,phy
                    tensor.modify(data=data, inds=inds)
    

    # Attach z-bonds for boundary xy-plane (Now xy-directions are already periodic)
    for i in range(peps.Lx):
        for j in range(peps.Ly):

            # z = 0 plane (x,y,0)
            
            tensor = peps[i,j,0]
            bdim,pdim = tensor.data.shape[0],tensor.data.shape[-1]
            Z_m = zbonds[i*peps.Ly+j]

            data = np.random.rand(*((bdim,)*6+(pdim,)))
            x_p,y_p,z_p,x_m,y_m,phy = tensor.inds
            inds = x_p,y_p,z_p,x_m,y_m,Z_m,phy
            tensor.modify(data=data, inds=inds)

            # z = Lz-1 plane (x,y,Lz-1)
            
            tensor = peps[i,j,peps.Lz-1]
            bdim,pdim = tensor.data.shape[0],tensor.data.shape[-1]
            Z_p = zbonds[i*peps.Ly+j]

            data = np.random.rand(*((bdim,)*6+(pdim,)))
            x_p,y_p,x_m,y_m,z_m,phy = tensor.inds
            inds = x_p,y_p,Z_p,x_m,y_m,z_m,phy
            tensor.modify(data=data, inds=inds)

    return peps

def peps2pbc3D(opeps): # Directly construct PBC tensor networks by treating each tensor the same way.
    peps = opeps.copy()

    xbonds = [rand_uuid() for i in range(peps.Lx*peps.Ly*peps.Lz)] #new xbonds

    ybonds = [rand_uuid() for i in range(peps.Lx*peps.Ly*peps.Lz)] #new ybonds

    zbonds = [rand_uuid() for i in range(peps.Lx*peps.Ly*peps.Lz)] #new ybonds

    # Attach x-bonds for boundary yz-plane
    for i in range(peps.Lx):
        for j in range(peps.Ly):
            for k in range(peps.Lz):
                tensor = peps[i,j,k]
                bdim,pdim = tensor.data.shape[0],tensor.data.shape[-1]
                index = flatten(i,j,k,peps.Lx,peps.Ly,peps.Lz)
                X_p = xbonds[index]
                Y_p = ybonds[index]
                Z_p = zbonds[index]
                X_m = xbonds[flatten(i-1,j,k,peps.Lx,peps.Ly,peps.Lz)]
                Y_m = ybonds[flatten(i,j-1,k,peps.Lx,peps.Ly,peps.Lz)]
                Z_m = zbonds[flatten(i,j,k-1,peps.Lx,peps.Ly,peps.Lz)]
                phy = tensor.inds[-1]
                inds = X_p,Y_p,Z_p,X_m,Y_m,Z_m,phy
                data = np.random.rand(*((bdim,)*6+(pdim,)))
                tensor.modify(data=data, inds=inds)
    return peps
####################################################################################
# Contraction & Amplitude
####################################################################################


class ContractionEngine:

    def init_contraction(self,Lx,Ly,Lz,phys_dim=2):
        """
            Parameters Initialization
        """
        self.Lx,self.Ly,self.Lz = Lx,Ly,Lz
        self.pbc = pbc
        self.deterministic = deterministic
        if self.deterministic:
            self.riz1,self.riz2 = (self.Lz-1) // 2, (self.Lz+1) // 2 
            self.rix1,self.rix2 = (self.Lx-1) // 2, (self.Lx+1) // 2
        self.compress_opts = compress_opts # control how to compress the boundary MPS bonds

        self.data_map = dict()
        for i in range(phys_dim): # On-site physical Hilbert space basis states.
            data = np.zeros(phys_dim)
            data[i] = 1.
            self.data_map[i] = data
    
    """#----------------------------------------------------------------
        Index functions for cubic lattice. Index <---> Site coordinate.
    """#----------------------------------------------------------------
    def flatten(self,i,j):
        return flatten(i,j,self.Lx,self.Ly,self.Lz)
    def flat2site(self,ix):
        return flat2site(ix,self.Lx,self.Ly,self.Lz)
    

    """#----------------------------------------------------------------
        Functions for auto-diff using Pytorch. 
        Gradiant calculation through back-propagation for one specific spin config.
        Final gradiant used for variational update should be the gradiant from auto-diff averaged over many configs.
    """#----------------------------------------------------------------
    def _2backend(self,data,requires_grad):
        if self.backend=='torch':
            data = torch.tensor(data,requires_grad=requires_grad)
        return data
    def _torch2numpy(self,data,backend=None):
        backend = self.backend if backend is None else backend
        if backend=='torch':
            data = data.detach().numpy()
            if data.size==1:
                data = data.reshape(-1)[0]
        return data
    def _2numpy(self,data,backend=None):
        return self._torch2numpy(data,backend=backend)
    def tsr_grad(self,tsr,set_zero=True):
        grad = tsr.grad
        if set_zero:
            tsr.grad = None
        return grad 
    