import numpy as np
import h5py
import os

data = []
if os.path.exists('./summary.out'):
    out = open(f'summary.out', 'r').readlines()
    for l in out:
        if l[:5] == "step=":
            ls = l.split(',')
            data.append(float(ls[1].split('=')[-1]))
out = open(f'out0.out', 'r').readlines()
for l in out:
    if l[:5] == "step=":
        ls = l.split(',')
        data.append(float(ls[1].split('=')[-1]))
out = open(f'out150.out', 'r').readlines()
for l in out:
    if l[:5] == "step=":
        ls = l.split(',')
        data.append(float(ls[1].split('=')[-1]))
data = np.array(data)
print(len(data))
f = h5py.File('energy.hdf5','w')
f.create_dataset('data',data=data)
f.close()

data = []
if os.path.exists('./summary.out'):
    out = open(f'summary.out', 'r').readlines()
    for l in out:
        if l[:5] == "step=":
            ls = l.split(',')
            data.append(float(ls[3].split('=')[-1]))
out0 = open(f'out0.out', 'r').readlines()
for l in out0:
    if l[:5] == "step=":
        ls = l.split(',')
        data.append(float(ls[3].split('=')[-1]))
out150 = open(f'out150.out', 'r').readlines()
for l in out150:
    if l[:5] == "step=":
        ls = l.split(',')
        data.append(float(ls[3].split('=')[-1]))
data = np.array(data)
f = h5py.File('err.hdf5','w')
f.create_dataset('data',data=data)
f.close()