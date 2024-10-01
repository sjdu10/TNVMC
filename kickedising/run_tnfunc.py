from funcs import *
import numpy as np
from mpi4py import MPI
import sys
from quimb.utils import progbar as Progbar
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

import os

N_MPI = 20
N_OMP = N_MKL = N_OPENBLAS = 1
# Number of cores used when performing linear algebra operations: N_MPI * N_OMP. Make sure this is less or equal to the number of total threadings.

os.environ['OMP_NUM_THREADS'] = f'{N_OMP}'
os.environ['MKL_NUM_THREADS'] = f'{N_MKL}'
os.environ['OPENBLAS_NUM_THREADS'] = f'{N_OPENBLAS}'
os.environ['MKL_DOMAIN_NUM_THREADS'] = f'{N_MKL}'
os.environ['MKL_DYNAMIC'] = 'FALSE'
os.environ['OMP_DYNAMIC'] = 'FALSE'


# Parameters
n = int(sys.argv[1])
depth = int(sys.argv[2])
chi = int(sys.argv[3])
direction = 'xmax'
J = float(sys.argv[4])
g = float(sys.argv[5])
h = float(sys.argv[6])

def enumerate_bitstrings_gen(configs):
    for config in configs:
        yield config


# Generate PEPS
circ, peps_psi = generate_Kicked_Ising_peps(n, depth, J=J, g=g, h=h)

# Parallel computation to sample all configurations

if RANK == 0:
    collected_config = 0
    configs = enumerate_bitstrings(n)
    configs_gen = enumerate_bitstrings_gen(configs)
    total_config_no = 2**n
    amplitude_dict = {}
    normalizations = 0
    pg = Progbar(total=total_config_no)
    
    while len(amplitude_dict) < total_config_no:
        for i in range(1, SIZE):
            send_config = next(configs_gen, None)
            COMM.send((send_config, False), dest=i, tag=0)
            
        config, amp = COMM.recv(source=MPI.ANY_SOURCE, tag=1)
        amplitude_dict[tuple(config)] = amp
        normalizations += abs(amp)**2
    
        pg.set_description(f'Collected configurations: {len(amplitude_dict)}')
        pg.update()

    for i in range(1, SIZE):
        COMM.send((None, True), dest=i, tag=0)
    
    print('All configurations sampled')

    for config in amplitude_dict.keys():
        amplitude_dict[config] /= np.sqrt(normalizations)
        
    # Compute the entanglement entropy
    psi = state_vector(amplitude_dict, n)
    rho_A = partial_trace_vec(psi)
    S_A = von_neumann_entropy(rho_A)
    
    # Save the results (depth, S_A, chi) to a txt file
    with open(f'./results/TNfunc_n={n}_J={J}_g={g}_h={h}.txt', 'a') as f:
        f.write(f'depth={depth}, S_A={S_A}, chi={chi}\n')
    
else:
    terminate = False
    while not terminate:
        config, terminate = COMM.recv(source=0, tag=0)
        if config is not None:
            amp = amplitude(peps_psi, chi, config, direction=direction)
            COMM.send((config, amp), dest=0, tag=1)
        
        
    
    

    