import numpy as np
from quimb.tensor.tensor_2d_vmc import (
    J1J2,
    AmplitudeFactory,
    ExchangeSampler1,
    ExchangeSampler2,
    DenseSampler,
    set_options,
    write_tn_to_disc,load_tn_from_disc,peps2pbc,
)
from quimb.tensor.vmc import TNVMC
import quimb.tensor as qtn
import sys

import itertools
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
Lx,Ly = 8,8
nspin = 32,32
D = int(sys.argv[1])
chi = int(sys.argv[2])
step = int(sys.argv[3])
J1 = 1.
J2 = 0.5

set_options(deterministic=True,pbc=True,max_bond=chi)

if step == 0:
    fpeps = load_tn_from_disc(f'../tmpdir/su_{Lx},{Ly}_rand')
elif step != 0:
    fpeps = load_tn_from_disc(f'./psi{step}')

# Modify fpeps to fit the newest version of quimb
new_peps = qtn.PEPS.empty(fpeps.Lx,fpeps.Ly,bond_dim=D)
new_peps = peps2pbc(new_peps)
for x,y in itertools.product(range(fpeps.Lx),range(fpeps.Ly)):
    new_peps[x,y].modify(data=fpeps[x,y].data)
fpeps = new_peps

scale = 1.01
for tid in fpeps.tensor_map:
   tsr = fpeps.tensor_map[tid]
   tsr.modify(data = tsr.data * scale)

tmpdir = './' 
amp_fac = AmplitudeFactory(fpeps)
ham = J1J2(J1,J2,Lx,Ly)

burn_in = 10 # burn-in SIZE!!
sampler = ExchangeSampler2(Lx,Ly,burn_in=burn_in) 
sampler.amplitude_factory = amp_fac

tnvmc = TNVMC(ham,sampler,normalize=True,optimizer='sr',solve_full=True,solve_dense=False)
#tnvmc.tmpdir = tmpdir 
start = step
stop = step + 50
tnvmc.rate2 = .5
tnvmc.cond2 = 1e-3
tnvmc.rate1 = float(sys.argv[4])
tnvmc.cond1 = 1e-3
tnvmc.check = 'energy'
tnvmc.debug = False
tnvmc.progbar = True

tnvmc.batchsize = int(float(sys.argv[5]) + .1)
tnvmc.config = 1,0,1,0,1,0,1,0,\
         0,1,0,1,0,1,0,1,\
         1,0,1,0,1,0,1,0,\
         0,1,0,1,0,1,0,1,\
         1,0,1,0,1,0,1,0,\
         0,1,0,1,0,1,0,1,\
         1,0,1,0,1,0,1,0,\
         0,1,0,1,0,1,0,1
sampler.config = 1,0,1,0,1,0,1,0,\
         0,1,0,1,0,1,0,1,\
         1,0,1,0,1,0,1,0,\
         0,1,0,1,0,1,0,1,\
         1,0,1,0,1,0,1,0,\
         0,1,0,1,0,1,0,1,\
         1,0,1,0,1,0,1,0,\
         0,1,0,1,0,1,0,1
config = 1,0,1,0,1,0,1,0,\
         0,1,0,1,0,1,0,1,\
         1,0,1,0,1,0,1,0,\
         0,1,0,1,0,1,0,1,\
         1,0,1,0,1,0,1,0,\
         0,1,0,1,0,1,0,1,\
         1,0,1,0,1,0,1,0,\
         0,1,0,1,0,1,0,1
if RANK==0:
    print('RANK=',RANK)
    print('SIZE=',SIZE)
    print('Lx,Ly=',Lx,Ly)
    print('D=',D)
    print('chi=',chi)
    print('rate=',tnvmc.rate1,tnvmc.rate2)
    print('cond=',tnvmc.cond1,tnvmc.cond2)
    print('nparams=',len(amp_fac.get_x()))
tnvmc.run(start,stop,tmpdir=tmpdir)
# tnvmc._run_energy_expectation(start,stop,tmpdir=tmpdir)
# tnvmc._run_energy_gradient(start,stop,tmpdir=tmpdir)
# total_time = ham.compute_local_energy_time(config=config,amplitude_factory=amp_fac)