import os
import numpy as np

N_OMP = N_MKL = N_OPENBLAS = 1

os.environ['OMP_NUM_THREADS'] = f'{N_OMP}'
os.environ['MKL_NUM_THREADS'] = f'{N_MKL}'
os.environ['OPENBLAS_NUM_THREADS'] = f'{N_OPENBLAS}'
os.environ['MKL_DYNAMIC'] = 'FALSE'
os.environ['OMP_DYNAMIC'] = 'FALSE'

n_list = [10]
depth_list = list(range(20))
chi_list = [2,4,8,16,32,64]
J_list = [np.pi/4]
g_list = [np.pi/4]
h_list = [0.5]
cd = '/home/sdu2/TNVMC/kickedising/'

if __name__ == "__main__":
    for chi in chi_list:
        for n in n_list:
            for h in h_list:
                    for depth in depth_list:
                        for J in J_list:
                            for g in g_list:
                                os.system(f"mpirun -np 4 python run_tnfunc_y.py {n} {depth} {chi} {J} {g} {h}")
                                os.system(f"mpirun -np 4 python run_mps.py {n} {depth} {chi} {J} {g} {h}")
                                # os.system(f'sbatch -J n{n}c{chi}d{depth} run.job {n} {depth} {chi} {J} {g} {h} {cd}')
                                os.system(f'mpirun -np 4 python run_exact.py {n} {depth} {chi} {J} {g} {h}')